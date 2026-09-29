#!/usr/bin/env python3

"""
Extract fixed maximum-projection-cell PolyT z-intensity profiles.

For each stitched stack:

1. Find all linked segmentation masks across the member z-layers.
2. Form their XY union footprint.
3. Find the maximum-projection segmentation cell that overlaps most
   with that linked footprint.
4. Use the COMPLETE maximum-projection cell mask as one fixed XY ROI.
5. Apply that exact ROI to every PolyT z-layer.
6. Calculate mean PolyT intensity for z0-z6.
7. Save the ordered z-intensity profiles.

The 45k maximum-projection segmentation mask is already stored inside:
    40k_subset_0.zarr
as:
    labels/segmentation_mask_optimized_maxproj_45k
"""

from collections import defaultdict
from pathlib import Path
import gc
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tifffile
import zarr
import spatialdata as sd

from scipy import ndimage


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(
    "/data/gent/vo/000/gvo00070/vsc48277/yujin_project"
)


# ── Stitching output ──────────────────────────────────────────────────────────

STITCH_OUTPUT_DIR = (
    PROJECT_DIR
    / "40k_subset_0_cellpose_fullcell_stitch3d_tiled"
)

MEMBERSHIP_CSV = (
    STITCH_OUTPUT_DIR
    / "stack_cell_membership.csv"
)

STITCHED_MASK_DIR = (
    STITCH_OUTPUT_DIR
    / "stitched_masks"
)

MASK_PATTERN = "mask_z{z}.tif"


# ── Current 45k SpatialData ───────────────────────────────────────────────────

ZARR_PATH = (
    PROJECT_DIR
    / "40k_subset_0.zarr"
)

POLYT_ARRAY_PATH = "images/clahe_DAPI_PolyT/0"

POLYT_LAYOUT = "czyx"

POLYT_CHANNEL = 1

N_Z = 7


# Already cropped 45k max-projection segmentation
MAXPROJ_LABEL_LAYER = (
    "segmentation_mask_optimized_maxproj_45k"
)


# ── Output ────────────────────────────────────────────────────────────────────

OUTPUT_DIR = (
    STITCH_OUTPUT_DIR
    / "polyt_maxproj_mask_z_intensity_profiles"
)

OUTPUT_FILENAME = (
    "stack_z_intensity_maxproj_mask_polyt.csv"
)

PROGRESS_EVERY_CHUNKS = 1


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
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


def read_polyt_window(
    array,
    z,
    y_min,
    y_max,
    x_min,
    x_max,
):
    """
    Read one 2D PolyT window.
    """

    indexing = []

    for axis_name in POLYT_LAYOUT:

        if axis_name == "c":
            indexing.append(
                POLYT_CHANNEL
            )

        elif axis_name == "z":
            indexing.append(
                z
            )

        elif axis_name == "y":
            indexing.append(
                slice(
                    y_min,
                    y_max,
                )
            )

        elif axis_name == "x":
            indexing.append(
                slice(
                    x_min,
                    x_max,
                )
            )

    window = np.squeeze(
        np.asarray(
            array[
                tuple(indexing)
            ]
        )
    )

    if window.ndim != 2:
        raise ValueError(
            f"Expected a 2D PolyT window at z{z}, "
            f"got {window.shape}"
        )

    return window


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD MEMBERSHIP TABLE
# ─────────────────────────────────────────────────────────────────────────────

print(
    "\n[1] Loading stack membership table ...",
    flush=True,
)

membership_df = pd.read_csv(
    MEMBERSHIP_CSV
)

required = {
    "stack_id",
    "z",
    "n_z_layers",
}

missing = (
    required
    - set(
        membership_df.columns
    )
)

if missing:
    raise ValueError(
        f"Missing columns in membership CSV: "
        f"{missing}"
    )


membership_df["stack_id"] = (
    membership_df["stack_id"]
    .astype(np.int64)
)

membership_df["z"] = (
    membership_df["z"]
    .astype(np.int16)
)

membership_df["n_z_layers"] = (
    membership_df["n_z_layers"]
    .astype(np.int16)
)


stack_ids = np.sort(
    membership_df[
        "stack_id"
    ].unique()
).astype(
    np.int64
)


print(
    f"  Number of stacks: "
    f"{len(stack_ids)}",
    flush=True,
)


present_z_lookup = (
    membership_df
    .groupby(
        "stack_id",
        sort=False,
    )["z"]
    .apply(
        lambda values:
            set(
                values.astype(int)
            )
    )
    .to_dict()
)


n_z_lookup = (
    membership_df
    .drop_duplicates(
        "stack_id"
    )
    .set_index(
        "stack_id"
    )["n_z_layers"]
    .astype(int)
    .to_dict()
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD STITCHED MASKS
# ─────────────────────────────────────────────────────────────────────────────

print(
    "\n[2] Loading stitched masks ...",
    flush=True,
)


stitched_masks = []

expected_shape = None


for z in range(N_Z):

    path = (
        STITCHED_MASK_DIR
        / MASK_PATTERN.format(
            z=z
        )
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Mask not found:\n"
            f"{path}"
        )


    mask = np.squeeze(
        tifffile.imread(
            str(path)
        )
    ).astype(
        np.int32,
        copy=False,
    )


    if mask.ndim != 2:
        raise ValueError(
            f"Expected a 2D mask at z{z}, "
            f"got {mask.shape}"
        )


    if expected_shape is None:

        expected_shape = (
            mask.shape
        )

    elif mask.shape != expected_shape:

        raise ValueError(
            f"Mask shape mismatch at z{z}: "
            f"expected {expected_shape}, "
            f"got {mask.shape}"
        )


    stitched_masks.append(
        mask
    )


    print(
        f"  z{z}: "
        f"shape={mask.shape}, "
        f"dtype={mask.dtype}",
        flush=True,
    )


IMAGE_HEIGHT, IMAGE_WIDTH = (
    expected_shape
)


print(
    f"  Current subset size: "
    f"{IMAGE_HEIGHT} × {IMAGE_WIDTH}",
    flush=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. OPEN 40k ZARR + PolyT
# ─────────────────────────────────────────────────────────────────────────────

print(
    "\n[3] Opening 40k SpatialData / PolyT ...",
    flush=True,
)


if not ZARR_PATH.exists():
    raise FileNotFoundError(
        f"Zarr store not found:\n"
        f"{ZARR_PATH}"
    )


# Raw Zarr access for efficient PolyT chunk reading
zarr_root = zarr.open(
    str(
        ZARR_PATH
    ),
    mode="r",
)


try:

    polyt_array = zarr_root[
        POLYT_ARRAY_PATH
    ]

except KeyError as error:

    try:
        print(
            zarr_root.tree(),
            flush=True,
        )
    except Exception:
        pass

    raise KeyError(
        f"Zarr array not found:\n"
        f"{POLYT_ARRAY_PATH}"
    ) from error


print(
    f"  PolyT path   : {POLYT_ARRAY_PATH}",
    flush=True,
)

print(
    f"  PolyT shape  : {polyt_array.shape}",
    flush=True,
)

print(
    f"  PolyT dtype  : {polyt_array.dtype}",
    flush=True,
)

print(
    f"  PolyT chunks : {polyt_array.chunks}",
    flush=True,
)


if (
    len(POLYT_LAYOUT) != polyt_array.ndim
    or set(POLYT_LAYOUT) != set("czyx")
):

    raise ValueError(
        "POLYT_LAYOUT must contain exactly "
        "c, z, y and x"
    )


axis_size = {

    axis:
        polyt_array.shape[
            index
        ]

    for (
        index,
        axis,
    )
    in enumerate(
        POLYT_LAYOUT
    )
}


if axis_size["z"] < N_Z:

    raise ValueError(
        f"Zarr has "
        f"{axis_size['z']} z-layers, "
        f"but N_Z={N_Z}"
    )


if (
    axis_size["y"],
    axis_size["x"],
) != expected_shape:

    raise ValueError(
        f"Spatial mismatch: "
        f"PolyT="
        f"{(axis_size['y'], axis_size['x'])}, "
        f"stitched masks={expected_shape}"
    )


if POLYT_CHANNEL >= axis_size["c"]:

    raise ValueError(
        f"POLYT_CHANNEL="
        f"{POLYT_CHANNEL}, "
        f"but only "
        f"{axis_size['c']} channels exist"
    )


y_chunk_size = int(
    polyt_array.chunks[
        POLYT_LAYOUT.index(
            "y"
        )
    ]
)

x_chunk_size = int(
    polyt_array.chunks[
        POLYT_LAYOUT.index(
            "x"
        )
    ]
)


print(
    f"  Spatial PolyT chunk size: "
    f"{y_chunk_size} × "
    f"{x_chunk_size}",
    flush=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. LOAD SAVED 45k MAX-PROJECTION SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

print(
    "\n[4] Loading saved 45k "
    "maximum-projection segmentation ...",
    flush=True,
)


sdata = sd.read_zarr(
    str(
        ZARR_PATH
    )
)


if (
    MAXPROJ_LABEL_LAYER
    not in sdata.labels
):

    raise KeyError(
        f"'{MAXPROJ_LABEL_LAYER}' not found.\n"
        f"Available labels:\n"
        f"{list(sdata.labels.keys())}"
    )


maxproj_da = get_level0(
    sdata.labels[
        MAXPROJ_LABEL_LAYER
    ]
)


maxproj_da = (
    maxproj_da
    .squeeze()
)


if maxproj_da.ndim != 2:

    raise ValueError(
        f"Maximum-projection label must be 2D.\n"
        f"dims={maxproj_da.dims}\n"
        f"shape={maxproj_da.shape}"
    )


if maxproj_da.shape != (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
):

    raise ValueError(
        f"Max-projection mask shape does not match "
        f"stitched masks.\n"
        f"Maxproj={maxproj_da.shape}\n"
        f"Stitched={(IMAGE_HEIGHT, IMAGE_WIDTH)}"
    )


print(
    f"  Layer: "
    f"{MAXPROJ_LABEL_LAYER}",
    flush=True,
)

print(
    f"  Shape: "
    f"{maxproj_da.shape}",
    flush=True,
)


# Load the already-cropped 45k label into memory.

maxproj_45k = (
    maxproj_da
    .compute()
    .values
)


maxproj_45k = (
    np.asarray(
        maxproj_45k
    )
    .squeeze()
    .astype(
        np.int32,
        copy=False,
    )
)


print(
    "  Max-projection mask loaded.",
    flush=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 5. FIND PER-Z BOUNDING BOXES OF STITCHED STACKS
# ─────────────────────────────────────────────────────────────────────────────

print(
    "\n[5] Finding stitched-object bounding boxes ...",
    flush=True,
)


per_z_bboxes = []


for (
    z,
    mask,
) in enumerate(
    stitched_masks
):

    print(
        f"  z{z} ...",
        flush=True,
    )


    object_slices = (
        ndimage.find_objects(
            mask
        )
    )


    bbox_dict = {}


    for (
        label,
        object_slice,
    ) in enumerate(
        object_slices,
        start=1,
    ):

        if object_slice is None:
            continue


        y_slice, x_slice = (
            object_slice
        )


        y0 = int(
            y_slice.start
        )

        y1 = int(
            y_slice.stop
        )

        x0 = int(
            x_slice.start
        )

        x1 = int(
            x_slice.stop
        )


        object_crop = (
            mask[
                y0:y1,
                x0:x1,
            ]
        )


        if not np.any(
            object_crop == label
        ):
            continue


        bbox_dict[
            label
        ] = (
            y0,
            y1,
            x0,
            x1,
        )


    per_z_bboxes.append(
        bbox_dict
    )


    print(
        f"    Found "
        f"{len(bbox_dict)} objects",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. MATCH EACH STITCHED STACK TO MAX-PROJECTION CELL
# ─────────────────────────────────────────────────────────────────────────────

print(
    "\n[6] Matching stitched stacks "
    "to max-projection cells ...",
    flush=True,
)


maxproj_object_slices = (
    ndimage.find_objects(
        maxproj_45k
    )
)


fixed_mask_info = {}


n_matched = 0
n_no_overlap = 0
n_missing_bbox = 0


for (
    stack_number,
    stack_id,
) in enumerate(
    stack_ids,
    start=1,
):

    sid = int(
        stack_id
    )


    present_z = sorted(
        present_z_lookup.get(
            sid,
            set(),
        )
    )


    if not present_z:
        continue


    # ── bounding box containing all linked masks

    bboxes = []


    for z in present_z:

        z = int(
            z
        )


        if not (
            0 <= z < N_Z
        ):
            continue


        bbox = (
            per_z_bboxes[
                z
            ].get(
                sid
            )
        )


        if bbox is not None:
            bboxes.append(
                bbox
            )


    if not bboxes:

        n_missing_bbox += 1
        continue


    union_y0 = min(
        bbox[0]
        for bbox in bboxes
    )

    union_y1 = max(
        bbox[1]
        for bbox in bboxes
    )

    union_x0 = min(
        bbox[2]
        for bbox in bboxes
    )

    union_x1 = max(
        bbox[3]
        for bbox in bboxes
    )


    # ── union footprint of linked z-plane masks

    linked_union = np.zeros(
        (
            union_y1
            - union_y0,

            union_x1
            - union_x0,
        ),
        dtype=bool,
    )


    for z in present_z:

        z = int(
            z
        )


        if not (
            0 <= z < N_Z
        ):
            continue


        linked_union |= (
            stitched_masks[
                z
            ][
                union_y0:union_y1,
                union_x0:union_x1,
            ]
            == sid
        )


    linked_union_area = int(
        linked_union.sum()
    )


    if linked_union_area <= 0:
        continue


    # ── max-projection cells overlapping linked footprint

    maxproj_crop = (
        maxproj_45k[
            union_y0:union_y1,
            union_x0:union_x1,
        ]
    )


    overlapping_labels = (
        maxproj_crop[
            linked_union
            & (
                maxproj_crop > 0
            )
        ]
    )


    if (
        overlapping_labels.size
        == 0
    ):

        n_no_overlap += 1
        continue


    (
        candidate_ids,
        overlap_counts,
    ) = np.unique(
        overlapping_labels,
        return_counts=True,
    )


    # Cell with greatest pixel overlap.

    best_index = int(
        np.argmax(
            overlap_counts
        )
    )


    maxproj_cell_id = int(
        candidate_ids[
            best_index
        ]
    )


    overlap_pixels = int(
        overlap_counts[
            best_index
        ]
    )


    # ── retrieve COMPLETE max-projection cell mask

    object_index = (
        maxproj_cell_id
        - 1
    )


    if (
        object_index < 0
        or object_index
        >= len(
            maxproj_object_slices
        )
    ):
        continue


    object_slice = (
        maxproj_object_slices[
            object_index
        ]
    )


    if object_slice is None:
        continue


    y_slice, x_slice = (
        object_slice
    )


    y0 = int(
        y_slice.start
    )

    y1 = int(
        y_slice.stop
    )

    x0 = int(
        x_slice.start
    )

    x1 = int(
        x_slice.stop
    )


    fixed_mask = (
        maxproj_45k[
            y0:y1,
            x0:x1,
        ]
        == maxproj_cell_id
    )


    maxproj_area = int(
        fixed_mask.sum()
    )


    if maxproj_area <= 0:
        continue


    fixed_mask_info[
        sid
    ] = {

        "maxproj_cell_id":
            maxproj_cell_id,

        "area":
            maxproj_area,

        "bbox":
            (
                y0,
                y1,
                x0,
                x1,
            ),

        "mask":
            fixed_mask,

        "linked_union_area":
            linked_union_area,

        "overlap_pixels":
            overlap_pixels,

        "overlap_fraction_linked":
            (
                overlap_pixels
                / linked_union_area
            ),

        "overlap_fraction_maxproj":
            (
                overlap_pixels
                / maxproj_area
            ),
    }


    n_matched += 1


    if (
        stack_number
        % 5000
        == 0
    ):

        print(
            f"  processed "
            f"{stack_number}/"
            f"{len(stack_ids)}",
            flush=True,
        )


print(
    f"\n  Matched stacks: "
    f"{n_matched}",
    flush=True,
)

print(
    f"  No max-projection overlap: "
    f"{n_no_overlap}",
    flush=True,
)

print(
    f"  Missing stitched bbox: "
    f"{n_missing_bbox}",
    flush=True,
)


del per_z_bboxes

gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# 7. PREPARE CHUNK-WISE FIXED-MASK JOBS
# ─────────────────────────────────────────────────────────────────────────────

print(
    "\n[7] Preparing PolyT extraction jobs ...",
    flush=True,
)


processed_stack_ids = np.asarray(
    sorted(
        fixed_mask_info
    ),
    dtype=np.int64,
)


n_stacks = len(
    processed_stack_ids
)


stack_id_to_index = {

    int(sid):
        index

    for (
        index,
        sid,
    )
    in enumerate(
        processed_stack_ids
    )
}


chunk_jobs = defaultdict(
    list
)


for (
    sid,
    info,
) in fixed_mask_info.items():

    stack_index = (
        stack_id_to_index[
            sid
        ]
    )


    (
        y0,
        y1,
        x0,
        x1,
    ) = info[
        "bbox"
    ]


    fixed_mask = (
        info[
            "mask"
        ]
    )


    y_first = (
        y0
        // y_chunk_size
    )

    y_last = (
        (y1 - 1)
        // y_chunk_size
    )

    x_first = (
        x0
        // x_chunk_size
    )

    x_last = (
        (x1 - 1)
        // x_chunk_size
    )


    for cy in range(
        y_first,
        y_last + 1,
    ):

        chunk_y_min = (
            cy
            * y_chunk_size
        )

        chunk_y_max = min(
            IMAGE_HEIGHT,
            chunk_y_min
            + y_chunk_size,
        )


        global_y0 = max(
            y0,
            chunk_y_min,
        )

        global_y1 = min(
            y1,
            chunk_y_max,
        )


        if (
            global_y0
            >= global_y1
        ):
            continue


        for cx in range(
            x_first,
            x_last + 1,
        ):

            chunk_x_min = (
                cx
                * x_chunk_size
            )

            chunk_x_max = min(
                IMAGE_WIDTH,
                chunk_x_min
                + x_chunk_size,
            )


            global_x0 = max(
                x0,
                chunk_x_min,
            )

            global_x1 = min(
                x1,
                chunk_x_max,
            )


            if (
                global_x0
                >= global_x1
            ):
                continue


            mask_y0 = (
                global_y0
                - y0
            )

            mask_y1 = (
                global_y1
                - y0
            )

            mask_x0 = (
                global_x0
                - x0
            )

            mask_x1 = (
                global_x1
                - x0
            )


            mask_fragment = (
                fixed_mask[
                    mask_y0:mask_y1,
                    mask_x0:mask_x1,
                ]
            )


            if not (
                mask_fragment.any()
            ):
                continue


            local_y0 = (
                global_y0
                - chunk_y_min
            )

            local_y1 = (
                global_y1
                - chunk_y_min
            )

            local_x0 = (
                global_x0
                - chunk_x_min
            )

            local_x1 = (
                global_x1
                - chunk_x_min
            )


            chunk_jobs[
                (
                    cy,
                    cx,
                )
            ].append(
                (
                    stack_index,
                    int(
                        local_y0
                    ),
                    int(
                        local_y1
                    ),
                    int(
                        local_x0
                    ),
                    int(
                        local_x1
                    ),
                    mask_fragment.copy(),
                )
            )


chunk_keys = sorted(
    chunk_jobs
)


print(
    f"  Matched stacks: "
    f"{n_stacks}",
    flush=True,
)

print(
    f"  Occupied chunks: "
    f"{len(chunk_keys)}",
    flush=True,
)

print(
    f"  Planned PolyT chunk reads: "
    f"{N_Z * len(chunk_keys)}",
    flush=True,
)


# We no longer need the full stitched masks.

del stitched_masks

gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# 8. EXTRACT PolyT INTENSITY
# ─────────────────────────────────────────────────────────────────────────────

print(
    "\n[8] Calculating fixed max-projection-mask "
    "PolyT z-intensity profiles ...",
    flush=True,
)


intensity_sums = np.zeros(
    (
        n_stacks,
        N_Z,
    ),
    dtype=np.float64,
)


intensity_counts = np.zeros(
    (
        n_stacks,
        N_Z,
    ),
    dtype=np.int64,
)


total_reads = (
    N_Z
    * len(
        chunk_keys
    )
)


read_counter = 0


for z in range(
    N_Z
):

    print(
        f"\n  Processing z{z} ...",
        flush=True,
    )


    for (
        cy,
        cx,
    ) in chunk_keys:

        jobs = (
            chunk_jobs[
                (
                    cy,
                    cx,
                )
            ]
        )


        chunk_y_min = (
            cy
            * y_chunk_size
        )

        chunk_y_max = min(
            IMAGE_HEIGHT,
            chunk_y_min
            + y_chunk_size,
        )


        chunk_x_min = (
            cx
            * x_chunk_size
        )

        chunk_x_max = min(
            IMAGE_WIDTH,
            chunk_x_min
            + x_chunk_size,
        )


        polyt_chunk = read_polyt_window(
            polyt_array,
            z,
            chunk_y_min,
            chunk_y_max,
            chunk_x_min,
            chunk_x_max,
        )


        for (
            stack_index,
            local_y0,
            local_y1,
            local_x0,
            local_x1,
            mask_fragment,
        ) in jobs:

            polyt_fragment = (
                polyt_chunk[
                    local_y0:local_y1,
                    local_x0:local_x1,
                ]
            )


            valid_mask = (
                mask_fragment
                & np.isfinite(
                    polyt_fragment
                )
            )


            if not (
                valid_mask.any()
            ):
                continue


            values = (
                polyt_fragment[
                    valid_mask
                ]
            )


            intensity_sums[
                stack_index,
                z,
            ] += values.sum(
                dtype=np.float64
            )


            intensity_counts[
                stack_index,
                z,
            ] += int(
                values.size
            )


        del polyt_chunk


        read_counter += 1


        if (
            read_counter
            % PROGRESS_EVERY_CHUNKS
            == 0
            or read_counter
            == total_reads
        ):

            print(
                f"    chunk "
                f"{read_counter}/"
                f"{total_reads}: "
                f"y={chunk_y_min}:"
                f"{chunk_y_max}, "
                f"x={chunk_x_min}:"
                f"{chunk_x_max}, "
                f"jobs={len(jobs)}",
                flush=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# 9. CALCULATE MEANS
# ─────────────────────────────────────────────────────────────────────────────

print(
    "\n[9] Building output table ...",
    flush=True,
)


mean_intensities = np.full(
    (
        n_stacks,
        N_Z,
    ),
    np.nan,
    dtype=np.float64,
)


valid = (
    intensity_counts
    > 0
)


mean_intensities[
    valid
] = (
    intensity_sums[
        valid
    ]
    / intensity_counts[
        valid
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# 10. OUTPUT METADATA
# ─────────────────────────────────────────────────────────────────────────────

maxproj_cell_ids = np.empty(
    n_stacks,
    dtype=np.int64,
)

maxproj_mask_area = np.empty(
    n_stacks,
    dtype=np.int32,
)

linked_union_area = np.empty(
    n_stacks,
    dtype=np.int32,
)

maxproj_overlap_pixels = np.empty(
    n_stacks,
    dtype=np.int32,
)

overlap_fraction_linked = np.empty(
    n_stacks,
    dtype=np.float64,
)

overlap_fraction_maxproj = np.empty(
    n_stacks,
    dtype=np.float64,
)


roi_y_min = np.empty(
    n_stacks,
    dtype=np.int32,
)

roi_y_max = np.empty(
    n_stacks,
    dtype=np.int32,
)

roi_x_min = np.empty(
    n_stacks,
    dtype=np.int32,
)

roi_x_max = np.empty(
    n_stacks,
    dtype=np.int32,
)


n_z_layers = np.empty(
    n_stacks,
    dtype=np.int16,
)


present_matrix = np.zeros(
    (
        n_stacks,
        N_Z,
    ),
    dtype=bool,
)


for (
    index,
    sid_value,
) in enumerate(
    processed_stack_ids
):

    sid = int(
        sid_value
    )


    info = (
        fixed_mask_info[
            sid
        ]
    )


    maxproj_cell_ids[
        index
    ] = int(
        info[
            "maxproj_cell_id"
        ]
    )


    maxproj_mask_area[
        index
    ] = int(
        info[
            "area"
        ]
    )


    linked_union_area[
        index
    ] = int(
        info[
            "linked_union_area"
        ]
    )


    maxproj_overlap_pixels[
        index
    ] = int(
        info[
            "overlap_pixels"
        ]
    )


    overlap_fraction_linked[
        index
    ] = float(
        info[
            "overlap_fraction_linked"
        ]
    )


    overlap_fraction_maxproj[
        index
    ] = float(
        info[
            "overlap_fraction_maxproj"
        ]
    )


    (
        y0,
        y1,
        x0,
        x1,
    ) = info[
        "bbox"
    ]


    roi_y_min[
        index
    ] = y0

    roi_y_max[
        index
    ] = y1

    roi_x_min[
        index
    ] = x0

    roi_x_max[
        index
    ] = x1


    n_z_layers[
        index
    ] = int(
        n_z_lookup.get(
            sid,
            len(
                present_z_lookup.get(
                    sid,
                    set(),
                )
            ),
        )
    )


    for z in (
        present_z_lookup.get(
            sid,
            set(),
        )
    ):

        z = int(
            z
        )


        if (
            0
            <= z
            < N_Z
        ):

            present_matrix[
                index,
                z,
            ] = True


# ─────────────────────────────────────────────────────────────────────────────
# 11. BUILD OUTPUT TABLE
# ─────────────────────────────────────────────────────────────────────────────

output_df = pd.DataFrame(
    {

        "stack_id":
            np.repeat(
                processed_stack_ids,
                N_Z,
            ),

        "z":
            np.tile(
                np.arange(
                    N_Z,
                    dtype=np.int16,
                ),
                n_stacks,
            ),

        "stack_present_in_z":
            present_matrix.reshape(
                -1
            ),

        "n_z_layers":
            np.repeat(
                n_z_layers,
                N_Z,
            ),

        "maxproj_cell_id":
            np.repeat(
                maxproj_cell_ids,
                N_Z,
            ),

        "linked_union_area_pixels":
            np.repeat(
                linked_union_area,
                N_Z,
            ),

        "maxproj_mask_area_pixels":
            np.repeat(
                maxproj_mask_area,
                N_Z,
            ),

        "maxproj_overlap_pixels":
            np.repeat(
                maxproj_overlap_pixels,
                N_Z,
            ),

        "maxproj_overlap_fraction_of_linked_stack":
            np.repeat(
                overlap_fraction_linked,
                N_Z,
            ),

        "maxproj_overlap_fraction_of_cell":
            np.repeat(
                overlap_fraction_maxproj,
                N_Z,
            ),

        "fixed_mask_mean_polyt_intensity":
            mean_intensities.reshape(
                -1
            ),

        "fixed_mask_valid_pixels":
            intensity_counts.reshape(
                -1
            ),

        "roi_type":
            "maximum_projection_segmentation_mask",

        "roi_y_min":
            np.repeat(
                roi_y_min,
                N_Z,
            ),

        "roi_y_max":
            np.repeat(
                roi_y_max,
                N_Z,
            ),

        "roi_x_min":
            np.repeat(
                roi_x_min,
                N_Z,
            ),

        "roi_x_max":
            np.repeat(
                roi_x_max,
                N_Z,
            ),

        "roi_bbox_height":
            np.repeat(
                roi_y_max
                - roi_y_min,
                N_Z,
            ),

        "roi_bbox_width":
            np.repeat(
                roi_x_max
                - roi_x_min,
                N_Z,
            ),
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# 12. SAVE
# ─────────────────────────────────────────────────────────────────────────────

output_path = (
    OUTPUT_DIR
    / OUTPUT_FILENAME
)


output_df.to_csv(
    output_path,
    index=False,
)


print(
    "\nDone!",
    flush=True,
)

print(
    f"  Original stitched stacks: "
    f"{len(stack_ids)}",
    flush=True,
)

print(
    f"  Matched max-projection cells: "
    f"{n_stacks}",
    flush=True,
)

print(
    f"  Rows written: "
    f"{len(output_df)}",
    flush=True,
)

print(
    "  Fixed ROI: complete max-projection "
    "segmentation cell with greatest overlap "
    "with each stitched stack",
    flush=True,
)

print(
    f"  Saved:\n"
    f"  {output_path}",
    flush=True,
)