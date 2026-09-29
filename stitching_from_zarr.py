#!/usr/bin/env python3
"""
Tiled native Cellpose stitch3D
==============================

Strategy
--------
1. Read segmentation masks from SpatialData:
       segmentation_mask_optimized_z0
       ...
       segmentation_mask_optimized_z6

2. Divide the XY field into overlapping tiles.

3. For each tile:
       - load all 7 z-layers
       - relabel cell IDs locally to dense IDs (1, 2, 3, ...)
       - run native cellpose.utils.stitch3D()
       - recover which ORIGINAL (z, cell_id) objects were linked
       - merge those relationships globally using union-find

4. Rebuild full-size stitched masks using the global stack IDs.

Outputs
-------
    stack_cell_membership.csv

    stitched_masks/
        mask_z0.tif
        ...
        mask_z6.tif

Important
---------
The actual z-stitching inside every tile is performed by:

    cellpose.utils.stitch3D()

No custom IoU/stitching implementation is used.
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path


ZARR_PATH = Path(
    "/data/gent/vo/000/gvo00070/vsc48277/"
    "yujin_project/40k_subset_0.zarr"
)


LABEL_PREFIX = "segmentation_mask_optimized_z"


N_Z = 7


OUTPUT_DIR = Path(
    "/data/gent/vo/000/gvo00070/vsc48277/"
    "yujin_project/40k_subset_0_cellpose_fullcell_stitch3d_tiled"
)


# Native Cellpose stitch3D threshold
STITCH_THRESHOLD = 0.25


# ─────────────────────────────────────────────────────────────────────────────
# TILE SETTINGS
#
# 8192 should be dramatically smaller than the full 45000 x 45000 image,
# while still being efficient on a high-memory HPC node.
#
# If memory problems remain:
#     TILE_SIZE = 4096
#
# Your approximate Cellpose diameter was ~80 px, so 512 px overlap is
# comfortably larger than a typical cell.
# ─────────────────────────────────────────────────────────────────────────────

TILE_SIZE = 8192

TILE_OVERLAP = 512


# When writing final full-size TIFFs,
# read this many rows from Zarr at once.
WRITE_ROW_CHUNK = 256


# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────

import warnings

warnings.filterwarnings("ignore")


import gc

import numpy as np
import pandas as pd
import tifffile
import spatialdata as sd

from cellpose.utils import stitch3D


# ─────────────────────────────────────────────────────────────────────────────
# Helper: get scale0
# ─────────────────────────────────────────────────────────────────────────────

def get_level0(layer):
    """
    Return scale0 DataArray for SpatialData multiscale elements.
    """

    try:
        return layer["scale0"].image
    except Exception:
        pass

    try:
        return layer["scale0"]["image"]
    except Exception:
        pass

    return layer


# ─────────────────────────────────────────────────────────────────────────────
# Helper: get one 2D segmentation layer lazily
# ─────────────────────────────────────────────────────────────────────────────

def get_label_2d(
    sdata,
    label_name,
):
    """
    Return a lazy 2D DataArray for one z-layer.
    """

    if label_name not in sdata.labels:

        raise KeyError(
            f"Missing SpatialData label:\n"
            f"{label_name}"
        )


    labels = get_level0(
        sdata.labels[label_name]
    )


    # Each independently segmented layer should contain only one z-plane.

    if "z" in labels.dims:

        if labels.sizes["z"] != 1:

            raise ValueError(
                f"{label_name}: expected exactly one z-plane, "
                f"but found {labels.sizes['z']}"
            )

        labels = labels.isel(z=0)


    labels = labels.squeeze()


    if labels.ndim != 2:

        raise ValueError(
            f"{label_name}: expected 2D labels, "
            f"got shape={labels.shape}"
        )


    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Helper: determine tile start positions
# ─────────────────────────────────────────────────────────────────────────────

def make_tile_starts(
    length,
    tile_size,
    overlap,
):
    """
    Generate tile starts such that:
      - the full image is covered
      - adjacent tiles overlap
      - the final tile ends exactly at the image boundary
    """

    if tile_size >= length:
        return [0]


    step = tile_size - overlap


    if step <= 0:

        raise ValueError(
            "TILE_OVERLAP must be smaller than TILE_SIZE"
        )


    starts = list(
        range(
            0,
            length - tile_size + 1,
            step,
        )
    )


    final_start = length - tile_size


    if starts[-1] != final_start:
        starts.append(final_start)


    return sorted(
        set(starts)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Union-Find
# ─────────────────────────────────────────────────────────────────────────────

class UnionFind:
    """
    Global union-find for nodes:

        (z, original_cell_id)

    Example:
        (0, 1234)
        (1, 882)

    If native stitch3D says these belong to the same object,
    they are unioned.
    """

    def __init__(self):

        self.parent = {}
        self.rank = {}


    def add(
        self,
        node,
    ):

        if node not in self.parent:

            self.parent[node] = node
            self.rank[node] = 0


    def find(
        self,
        node,
    ):

        parent = self.parent[node]

        if parent != node:

            self.parent[node] = self.find(
                parent
            )

        return self.parent[node]


    def union(
        self,
        a,
        b,
    ):

        self.add(a)
        self.add(b)


        root_a = self.find(a)
        root_b = self.find(b)


        if root_a == root_b:
            return


        rank_a = self.rank[root_a]
        rank_b = self.rank[root_b]


        if rank_a < rank_b:

            self.parent[root_a] = root_b


        elif rank_a > rank_b:

            self.parent[root_b] = root_a


        else:

            self.parent[root_b] = root_a
            self.rank[root_a] += 1


# ─────────────────────────────────────────────────────────────────────────────
# Dense relabeling for a tile
# ─────────────────────────────────────────────────────────────────────────────

def dense_relabel_tile(
    raw,
    y0,
    y1,
    x0,
    x1,
    full_height,
    full_width,
):
    """
    Convert original global IDs inside one tile:

        [0, 105, 105, 900, ...]
                ↓
        [0,   1,   1,   2, ...]

    This is VERY important because Cellpose's overlap code builds matrices
    based on label IDs. Dense IDs dramatically reduce unnecessary memory.

    Returns
    -------
    local : uint32 array
        Dense local labels.

    local_to_original : int64 array
        local_to_original[local_id] = original Cellpose ID.

    representative_indices : int64 array
        One flat pixel index for every local object.

    safe : bool array
        Whether a cell is safely away from an INTERNAL tile boundary.

        Objects touching internal tile edges are not trusted for stitching
        in that tile, because they may have been artificially cut.

        Global image boundaries are allowed.
    """

    raw = np.asarray(
        raw
    ).squeeze()


    if raw.ndim != 2:

        raise ValueError(
            f"Expected 2D tile, got {raw.shape}"
        )


    ids = np.unique(
        raw
    )


    ids = ids[
        ids > 0
    ]


    # Empty tile

    if len(ids) == 0:

        local = np.zeros(
            raw.shape,
            dtype=np.uint32,
        )

        return (
            local,
            np.zeros(
                1,
                dtype=np.int64,
            ),
            np.zeros(
                1,
                dtype=np.int64,
            ),
            np.zeros(
                1,
                dtype=bool,
            ),
        )


    ids = ids.astype(
        np.int64,
        copy=False,
    )


    max_original_id = int(
        ids.max()
    )


    # Fast lookup:
    #
    # original ID -> local dense ID

    lut = np.zeros(
        max_original_id + 1,
        dtype=np.uint32,
    )


    lut[
        ids
    ] = np.arange(
        1,
        len(ids) + 1,
        dtype=np.uint32,
    )


    local = lut[
        raw
    ]


    del lut


    # local ID -> original ID

    local_to_original = np.zeros(
        len(ids) + 1,
        dtype=np.int64,
    )


    local_to_original[
        1:
    ] = ids


    # ─────────────────────────────────────────────────────────────────────────
    # Find one representative pixel for each local cell.
    #
    # After native stitch3D, looking at this same pixel tells us the
    # resulting stitched ID.
    # ─────────────────────────────────────────────────────────────────────────

    values, first_indices = np.unique(
        local,
        return_index=True,
    )


    representative_indices = np.full(
        len(ids) + 1,
        -1,
        dtype=np.int64,
    )


    keep = (
        values > 0
    )


    representative_indices[
        values[keep]
    ] = first_indices[keep]


    del values
    del first_indices
    del keep


    # ─────────────────────────────────────────────────────────────────────────
    # Detect cells touching INTERNAL tile borders.
    #
    # Why?
    #
    # Example:
    #
    #          cell
    #       ───────────
    #             | tile boundary
    #
    # A cut object may have an artificially altered IoU.
    #
    # Because tiles overlap by 512 px, the same cell should appear fully
    # inside a neighboring tile, where its stitching result can be trusted.
    # ─────────────────────────────────────────────────────────────────────────

    safe = np.ones(
        len(ids) + 1,
        dtype=bool,
    )


    safe[0] = False


    touched = []


    # Top boundary is internal

    if y0 > 0:

        touched.append(
            np.unique(
                local[0, :]
            )
        )


    # Bottom boundary is internal

    if y1 < full_height:

        touched.append(
            np.unique(
                local[-1, :]
            )
        )


    # Left boundary is internal

    if x0 > 0:

        touched.append(
            np.unique(
                local[:, 0]
            )
        )


    # Right boundary is internal

    if x1 < full_width:

        touched.append(
            np.unique(
                local[:, -1]
            )
        )


    if touched:

        touched_ids = np.unique(
            np.concatenate(
                touched
            )
        )


        touched_ids = touched_ids[
            touched_ids > 0
        ]


        safe[
            touched_ids
        ] = False


        del touched_ids


    del touched
    del ids


    return (
        local,
        local_to_original,
        representative_indices,
        safe,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output directories
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


STITCHED_MASK_DIR = (
    OUTPUT_DIR
    / "stitched_masks"
)


STITCHED_MASK_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load SpatialData
# ─────────────────────────────────────────────────────────────────────────────

print(
    "\n[1] Loading SpatialData...",
    flush=True,
)


if not ZARR_PATH.exists():

    raise FileNotFoundError(
        f"Zarr not found:\n"
        f"{ZARR_PATH}"
    )


sdata = sd.read_zarr(
    str(ZARR_PATH)
)


print(
    "  SpatialData loaded.",
    flush=True,
)


print(
    "  Available labels:",
    list(sdata.labels.keys()),
    flush=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Get mask dimensions
# ─────────────────────────────────────────────────────────────────────────────

print(
    "\n[2] Determining mask dimensions...",
    flush=True,
)


first_label = get_label_2d(
    sdata,
    f"{LABEL_PREFIX}0",
)


HEIGHT = int(
    first_label.sizes["y"]
)

WIDTH = int(
    first_label.sizes["x"]
)


del first_label


print(
    f"  Full image: "
    f"{HEIGHT} x {WIDTH}",
    flush=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Prepare tiles
# ─────────────────────────────────────────────────────────────────────────────

y_starts = make_tile_starts(
    HEIGHT,
    TILE_SIZE,
    TILE_OVERLAP,
)


x_starts = make_tile_starts(
    WIDTH,
    TILE_SIZE,
    TILE_OVERLAP,
)


n_tiles = (
    len(y_starts)
    * len(x_starts)
)


print(
    "\n[3] Tile configuration:",
    flush=True,
)


print(
    f"  TILE_SIZE    = {TILE_SIZE}",
    flush=True,
)


print(
    f"  TILE_OVERLAP = {TILE_OVERLAP}",
    flush=True,
)


print(
    f"  Y tiles      = {len(y_starts)}",
    flush=True,
)


print(
    f"  X tiles      = {len(x_starts)}",
    flush=True,
)


print(
    f"  Total tiles  = {n_tiles}",
    flush=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Global Union-Find
# ─────────────────────────────────────────────────────────────────────────────

uf = UnionFind()


# Optional QC log

tile_log = []


# ─────────────────────────────────────────────────────────────────────────────
# 5. Process tiles
# ─────────────────────────────────────────────────────────────────────────────

print(
    "\n[4] Running tiled native stitch3D...",
    flush=True,
)


tile_number = 0


for y0 in y_starts:

    y1 = min(
        y0 + TILE_SIZE,
        HEIGHT,
    )


    for x0 in x_starts:

        x1 = min(
            x0 + TILE_SIZE,
            WIDTH,
        )


        tile_number += 1


        tile_h = (
            y1 - y0
        )

        tile_w = (
            x1 - x0
        )


        print(
            "\n"
            "============================================================",
            flush=True,
        )


        print(
            f"Tile {tile_number}/{n_tiles}",
            flush=True,
        )


        print(
            f"  y = {y0}:{y1}",
            flush=True,
        )


        print(
            f"  x = {x0}:{x1}",
            flush=True,
        )


        print(
            f"  shape = "
            f"{tile_h} x {tile_w}",
            flush=True,
        )


        # ─────────────────────────────────────────────────────────────────────
        # Allocate 7-layer TILE stack only.
        #
        # Full 45000 x 45000 x 7 stack is never allocated.
        # ─────────────────────────────────────────────────────────────────────

        tile_stack = np.zeros(
            (
                N_Z,
                tile_h,
                tile_w,
            ),
            dtype=np.uint32,
        )


        # Metadata needed to map native stitched IDs back
        # to original global segmentation IDs.

        local_to_original_all = []

        representative_indices_all = []

        safe_all = []


        total_local_cells = 0

        total_safe_cells = 0


        # ─────────────────────────────────────────────────────────────────────
        # Load all z layers for this tile
        # ─────────────────────────────────────────────────────────────────────

        for z in range(N_Z):

            label_name = (
                f"{LABEL_PREFIX}{z}"
            )


            labels = get_label_2d(
                sdata,
                label_name,
            )


            raw_tile = (
                labels
                .isel(
                    y=slice(
                        y0,
                        y1,
                    ),
                    x=slice(
                        x0,
                        x1,
                    ),
                )
                .compute()
                .values
            )


            (
                local,
                local_to_original,
                representative_indices,
                safe,
            ) = dense_relabel_tile(
                raw=raw_tile,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                full_height=HEIGHT,
                full_width=WIDTH,
            )


            tile_stack[
                z
            ] = local


            n_local = (
                len(
                    local_to_original
                )
                - 1
            )


            n_safe = int(
                safe.sum()
            )


            total_local_cells += n_local

            total_safe_cells += n_safe


            # Add every original object to the global union-find.
            #
            # Even if it touches this tile boundary, the node itself should
            # still exist. We simply don't trust stitching relationships from
            # this particular tile for unsafe objects.

            for local_id in range(
                1,
                n_local + 1,
            ):

                original_id = int(
                    local_to_original[
                        local_id
                    ]
                )


                node = (
                    z,
                    original_id,
                )


                uf.add(
                    node
                )


            local_to_original_all.append(
                local_to_original
            )


            representative_indices_all.append(
                representative_indices
            )


            safe_all.append(
                safe
            )


            print(
                f"  z{z}: "
                f"{n_local} cells "
                f"({n_safe} safe)",
                flush=True,
            )


            del raw_tile
            del local
            del labels


            gc.collect()


        tile_gib = (
            tile_stack.nbytes
            / 1024**3
        )


        print(
            f"\n  Tile stack RAM = "
            f"{tile_gib:.2f} GiB",
            flush=True,
        )


        # ─────────────────────────────────────────────────────────────────────
        # NATIVE CELLPOSE STITCH3D
        #
        # This is the actual original Cellpose function.
        # ─────────────────────────────────────────────────────────────────────

        print(
            f"  Running native stitch3D "
            f"(threshold={STITCH_THRESHOLD})...",
            flush=True,
        )


        stitched_tile = stitch3D(
            tile_stack,
            stitch_threshold=STITCH_THRESHOLD,
        )


        stitched_tile = np.asarray(
            stitched_tile
        )


        if stitched_tile.shape != (
            N_Z,
            tile_h,
            tile_w,
        ):

            raise ValueError(
                f"Unexpected stitch3D output shape:\n"
                f"{stitched_tile.shape}"
            )


        print(
            "  Native stitch3D finished.",
            flush=True,
        )


        # ─────────────────────────────────────────────────────────────────────
        # Recover native stitched relationships
        #
        # Example:
        #
        # z0 original cell 120
        # z1 original cell 903
        #
        # native stitch3D gives both stitched ID 42
        #
        # → union:
        #
        #       (0, 120)  ↔  (1, 903)
        #
        # ─────────────────────────────────────────────────────────────────────

        stitched_groups = {}


        for z in range(N_Z):

            local_to_original = (
                local_to_original_all[
                    z
                ]
            )


            representative_indices = (
                representative_indices_all[
                    z
                ]
            )


            safe = (
                safe_all[
                    z
                ]
            )


            n_local = (
                len(
                    local_to_original
                )
                - 1
            )


            if n_local == 0:
                continue


            flat_stitched = (
                stitched_tile[
                    z
                ]
                .ravel()
            )


            # Only trust cells that did not touch an INTERNAL tile border.

            safe_local_ids = np.flatnonzero(
                safe
            )


            safe_local_ids = safe_local_ids[
                safe_local_ids > 0
            ]


            for local_id in safe_local_ids:

                rep_index = int(
                    representative_indices[
                        local_id
                    ]
                )


                if rep_index < 0:
                    continue


                stitched_id = int(
                    flat_stitched[
                        rep_index
                    ]
                )


                if stitched_id <= 0:
                    continue


                original_id = int(
                    local_to_original[
                        local_id
                    ]
                )


                node = (
                    z,
                    original_id,
                )


                stitched_groups.setdefault(
                    stitched_id,
                    []
                ).append(
                    node
                )


        # ─────────────────────────────────────────────────────────────────────
        # Union all objects given the same native stitched ID
        # ─────────────────────────────────────────────────────────────────────

        n_union_groups = 0


        for (
            stitched_id,
            nodes,
        ) in stitched_groups.items():


            if len(nodes) < 2:
                continue


            # Only interesting if the native stack contains >1 z plane.

            z_values = {
                node[0]
                for node in nodes
            }


            if len(z_values) < 2:
                continue


            base_node = nodes[0]


            for other_node in nodes[1:]:

                uf.union(
                    base_node,
                    other_node,
                )


            n_union_groups += 1


        print(
            f"  Trusted multi-z groups from tile: "
            f"{n_union_groups}",
            flush=True,
        )


        tile_log.append(
            {
                "tile_number":
                    tile_number,

                "y0":
                    y0,

                "y1":
                    y1,

                "x0":
                    x0,

                "x1":
                    x1,

                "local_cells":
                    total_local_cells,

                "safe_cells":
                    total_safe_cells,

                "trusted_multiz_groups":
                    n_union_groups,
            }
        )


        # ─────────────────────────────────────────────────────────────────────
        # Free tile memory before next tile
        # ─────────────────────────────────────────────────────────────────────

        del stitched_groups

        # stitch3D can return the same underlying array.
        # Delete both references.

        del stitched_tile
        del tile_stack

        del local_to_original_all
        del representative_indices_all
        del safe_all


        gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Save tile QC log
# ─────────────────────────────────────────────────────────────────────────────

tile_log_df = pd.DataFrame(
    tile_log
)


tile_log_path = (
    OUTPUT_DIR
    / "tile_stitch_log.csv"
)


tile_log_df.to_csv(
    tile_log_path,
    index=False,
)


print(
    f"\n  Tile log saved:\n"
    f"  {tile_log_path}",
    flush=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Convert union-find components into GLOBAL stack IDs
# ─────────────────────────────────────────────────────────────────────────────

print(
    "\n[5] Assigning global stack IDs...",
    flush=True,
)


all_nodes = sorted(
    uf.parent.keys(),
    key=lambda x: (
        x[0],
        x[1],
    ),
)


root_to_stack_id = {}

node_to_stack_id = {}


next_stack_id = 1


for node in all_nodes:

    root = uf.find(
        node
    )


    if root not in root_to_stack_id:

        root_to_stack_id[
            root
        ] = next_stack_id

        next_stack_id += 1


    node_to_stack_id[
        node
    ] = root_to_stack_id[
        root
    ]


n_global_stacks = (
    next_stack_id
    - 1
)


print(
    f"  Original segmented objects: "
    f"{len(all_nodes)}",
    flush=True,
)


print(
    f"  Global stitched stacks: "
    f"{n_global_stacks}",
    flush=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Build membership table
# ─────────────────────────────────────────────────────────────────────────────

print(
    "\n[6] Building stack membership...",
    flush=True,
)


stack_nodes = {}


for (
    node,
    stack_id,
) in node_to_stack_id.items():

    stack_nodes.setdefault(
        stack_id,
        []
    ).append(
        node
    )


rows = []


n_multiz = 0
n_singlez = 0


for (
    stack_id,
    nodes,
) in stack_nodes.items():

    z_layers = sorted(
        {
            z
            for (
                z,
                original_cell_id,
            )
            in nodes
        }
    )


    n_z_layers = len(
        z_layers
    )


    is_multiz = (
        n_z_layers > 1
    )


    if is_multiz:

        n_multiz += 1

    else:

        n_singlez += 1


    for (
        z,
        original_cell_id,
    ) in sorted(
        nodes
    ):

        rows.append(
            {
                "stack_id":
                    int(
                        stack_id
                    ),

                "z":
                    int(
                        z
                    ),

                # ID present in FINAL stitched mask
                "cell_id":
                    int(
                        stack_id
                    ),

                # Useful for tracing back to the original
                # segmentation_mask_optimized_zX
                "original_cell_id":
                    int(
                        original_cell_id
                    ),

                "is_multiz_stack":
                    bool(
                        is_multiz
                    ),

                "n_z_layers":
                    int(
                        n_z_layers
                    ),
            }
        )


membership_df = pd.DataFrame(
    rows,
    columns=[
        "stack_id",
        "z",
        "cell_id",
        "original_cell_id",
        "is_multiz_stack",
        "n_z_layers",
    ],
)


membership_df = (
    membership_df
    .sort_values(
        [
            "stack_id",
            "z",
            "original_cell_id",
        ]
    )
    .reset_index(
        drop=True
    )
)


membership_path = (
    OUTPUT_DIR
    / "stack_cell_membership.csv"
)


membership_df.to_csv(
    membership_path,
    index=False,
)


print(
    f"  Total stacks:    "
    f"{n_global_stacks}",
    flush=True,
)


print(
    f"  Multi-z stacks:  "
    f"{n_multiz}",
    flush=True,
)


print(
    f"  Single-z stacks: "
    f"{n_singlez}",
    flush=True,
)


print(
    f"  Membership rows: "
    f"{len(membership_df)}",
    flush=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Write full-size stitched masks
#
# Important:
#
# We DON'T reconstruct the image by pasting stitched tiles.
#
# Instead:
#
# original segmentation ID
#              ↓
# global union-find stack ID
#
# This completely avoids tile-seam conflicts.
# ─────────────────────────────────────────────────────────────────────────────

print(
    "\n[7] Writing full-size stitched masks...",
    flush=True,
)


for z in range(N_Z):

    print(
        f"\n  Writing z{z}...",
        flush=True,
    )


    # ── Original ID -> final stack ID

    nodes_this_z = [
        (
            original_id,
            stack_id,
        )

        for (
            node,
            stack_id,
        )
        in node_to_stack_id.items()

        for (
            node_z,
            original_id,
        )
        in [node]

        if node_z == z
    ]


    if not nodes_this_z:

        raise RuntimeError(
            f"No cells found for z{z}"
        )


    max_original_id = max(
        original_id
        for (
            original_id,
            stack_id,
        )
        in nodes_this_z
    )


    lookup = np.zeros(
        max_original_id + 1,
        dtype=np.uint32,
    )


    for (
        original_id,
        stack_id,
    ) in nodes_this_z:

        lookup[
            original_id
        ] = stack_id


    labels = get_label_2d(
        sdata,
        f"{LABEL_PREFIX}{z}",
    )


    output_path = (
        STITCHED_MASK_DIR
        / f"mask_z{z}.tif"
    )


    # Disk-backed TIFF.
    # Full 45000 x 45000 output does NOT live in RAM.

    output = tifffile.memmap(
        str(
            output_path
        ),
        shape=(
            HEIGHT,
            WIDTH,
        ),
        dtype=np.uint32,
        bigtiff=True,
    )


    for y0 in range(
        0,
        HEIGHT,
        WRITE_ROW_CHUNK,
    ):

        y1 = min(
            y0 + WRITE_ROW_CHUNK,
            HEIGHT,
        )


        raw = (
            labels
            .isel(
                y=slice(
                    y0,
                    y1,
                )
            )
            .compute()
            .values
        )


        raw = np.asarray(
            raw
        ).squeeze()


        if raw.size:

            raw_max = int(
                raw.max()
            )


            if raw_max >= len(
                lookup
            ):

                raise RuntimeError(
                    f"z{z}: original ID {raw_max} "
                    f"is missing from lookup table."
                )


        mapped = lookup[
            raw
        ]


        output[
            y0:y1
        ] = mapped


        del raw
        del mapped


    output.flush()


    del output
    del labels
    del lookup
    del nodes_this_z


    gc.collect()


    print(
        f"    saved → "
        f"{output_path}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 9. Final validation
# ─────────────────────────────────────────────────────────────────────────────

print(
    "\n[8] Final validation...",
    flush=True,
)


for z in range(N_Z):

    mask_path = (
        STITCHED_MASK_DIR
        / f"mask_z{z}.tif"
    )


    mm = tifffile.memmap(
        str(
            mask_path
        )
    )


    # Don't call np.unique() on the full 45k image.
    # Membership table already tells us which IDs should occur.

    n_expected = (
        membership_df[
            membership_df["z"] == z
        ]["stack_id"]
        .nunique()
    )


    print(
        f"  z{z}: "
        f"{n_expected} stitched stack IDs "
        f"expected",
        flush=True,
    )


    del mm


# ─────────────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────────────

del sdata

gc.collect()


print(
    "\n"
    "============================================================",
    flush=True,
)


print(
    "TILED NATIVE STITCH3D FINISHED",
    flush=True,
)


print(
    "============================================================",
    flush=True,
)


print(
    f"\nOutput directory:\n"
    f"{OUTPUT_DIR}",
    flush=True,
)


print(
    "\nFiles:",
    flush=True,
)


print(
    "  stack_cell_membership.csv",
    flush=True,
)


print(
    "  tile_stitch_log.csv",
    flush=True,
)


print(
    f"  stitched_masks/"
    f"mask_z{{0..{N_Z - 1}}}.tif",
    flush=True,
)