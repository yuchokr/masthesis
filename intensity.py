#!/usr/bin/env python3
"""Fast chunk-wise extraction of fixed largest-mask DAPI z-intensity profiles.

For each stitched stack, this script:

1. Finds the segmentation area of that stack in every member z-layer.
2. Selects the z-layer with the largest segmentation area.
3. Uses that exact binary segmentation mask as one fixed XY ROI for all z-layers.
4. Reads each spatial DAPI Zarr chunk only once per z-layer.
5. Calculates mean DAPI intensity inside the fixed largest-mask ROI for every
   stack × z-layer.
6. Saves the ordered z-intensity profiles. No bimodality scoring is done.

The bounding box stored in the output only encloses the fixed largest mask and
is used for indexing. Pixels outside the binary mask are never included in the
intensity calculation.
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
from scipy import ndimage


# ── CONFIG ────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path("/data/gent/vo/000/gvo00070/vsc48277/yujin_project")

STITCH_OUTPUT_DIR = (
    PROJECT_DIR
    / "40k_subset_0_cellpose_preprocessed"
    / "overlap_output_stitch3d"
)

MEMBERSHIP_CSV = STITCH_OUTPUT_DIR / "stack_cell_membership.csv"
STITCHED_MASK_DIR = STITCH_OUTPUT_DIR / "stitched_masks"
MASK_PATTERN = "mask_z{z}.tif"

ZARR_PATH = PROJECT_DIR / "40k_subset_0.zarr"
DAPI_ARRAY_PATH = "images/clahe/0"
DAPI_LAYOUT = "czyx"
DAPI_CHANNEL = 0
N_Z = 7

OUTPUT_DIR = STITCH_OUTPUT_DIR / "dapi_largest_mask_z_intensity_profiles"
OUTPUT_FILENAME = "stack_z_intensity.csv"

# Print after every N expensive DAPI chunk reads.
PROGRESS_EVERY_CHUNKS = 1
# ─────────────────────────────────────────────────────────────────────────────


def read_dapi_window(array, z, y_min, y_max, x_min, x_max):
    """Read one 2D DAPI window while respecting DAPI_LAYOUT."""

    indexing = []

    for axis_name in DAPI_LAYOUT:
        if axis_name == "c":
            indexing.append(DAPI_CHANNEL)
        elif axis_name == "z":
            indexing.append(z)
        elif axis_name == "y":
            indexing.append(slice(y_min, y_max))
        elif axis_name == "x":
            indexing.append(slice(x_min, x_max))

    window = np.squeeze(
        np.asarray(array[tuple(indexing)])
    )

    if window.ndim != 2:
        raise ValueError(
            f"Expected a 2D DAPI window at z{z}, got {window.shape}"
        )

    return window


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Membership table ───────────────────────────────────────────────────────
print("\n[1] Loading stack membership table ...", flush=True)

membership_df = pd.read_csv(MEMBERSHIP_CSV)
required = {"stack_id", "z", "n_z_layers"}
missing = required - set(membership_df.columns)

if missing:
    raise ValueError(
        f"Missing columns in membership CSV: {missing}"
    )

membership_df["stack_id"] = membership_df["stack_id"].astype(np.int64)
membership_df["z"] = membership_df["z"].astype(np.int16)
membership_df["n_z_layers"] = membership_df["n_z_layers"].astype(np.int16)

stack_ids = np.sort(
    membership_df["stack_id"].unique()
).astype(np.int64)

print(
    f"  Number of stacks: {len(stack_ids)}",
    flush=True,
)

present_z_lookup = (
    membership_df
    .groupby("stack_id", sort=False)["z"]
    .apply(lambda values: set(values.astype(int)))
    .to_dict()
)

n_z_lookup = (
    membership_df
    .drop_duplicates("stack_id")
    .set_index("stack_id")["n_z_layers"]
    .astype(int)
    .to_dict()
)


# ── 2. Stitched masks ─────────────────────────────────────────────────────────
print("\n[2] Loading stitched masks ...", flush=True)

stitched_masks = []
expected_shape = None

for z in range(N_Z):
    path = STITCHED_MASK_DIR / MASK_PATTERN.format(z=z)

    if not path.exists():
        raise FileNotFoundError(
            f"Mask not found: {path}"
        )

    mask = np.squeeze(
        tifffile.imread(str(path))
    ).astype(
        np.int32,
        copy=False,
    )

    if mask.ndim != 2:
        raise ValueError(
            f"Expected a 2D mask at z{z}, got {mask.shape}"
        )

    if expected_shape is None:
        expected_shape = mask.shape
    elif mask.shape != expected_shape:
        raise ValueError(
            f"Mask shape mismatch at z{z}: "
            f"expected {expected_shape}, got {mask.shape}"
        )

    stitched_masks.append(mask)

    print(
        f"  z{z}: shape={mask.shape}, dtype={mask.dtype}",
        flush=True,
    )

IMAGE_HEIGHT, IMAGE_WIDTH = expected_shape


# ── 3. DAPI Zarr ──────────────────────────────────────────────────────────────
print("\n[3] Opening DAPI Zarr array ...", flush=True)

if not ZARR_PATH.exists():
    raise FileNotFoundError(
        f"Zarr store not found: {ZARR_PATH}"
    )

zarr_root = zarr.open(
    str(ZARR_PATH),
    mode="r",
)

try:
    dapi_array = zarr_root[DAPI_ARRAY_PATH]
except KeyError as error:
    try:
        print(zarr_root.tree(), flush=True)
    except Exception:
        pass

    raise KeyError(
        f"Zarr array not found: {DAPI_ARRAY_PATH}"
    ) from error

print(f"  path={DAPI_ARRAY_PATH}", flush=True)
print(f"  shape={dapi_array.shape}", flush=True)
print(f"  dtype={dapi_array.dtype}", flush=True)
print(f"  chunks={dapi_array.chunks}", flush=True)
print(f"  configured layout={DAPI_LAYOUT}", flush=True)

if (
    len(DAPI_LAYOUT) != dapi_array.ndim
    or set(DAPI_LAYOUT) != set("czyx")
):
    raise ValueError(
        "DAPI_LAYOUT must contain exactly c, z, y and x"
    )

axis_size = {
    axis: dapi_array.shape[index]
    for index, axis in enumerate(DAPI_LAYOUT)
}

if axis_size["z"] < N_Z:
    raise ValueError(
        f"Zarr has {axis_size['z']} z-layers, but N_Z={N_Z}"
    )

if (
    axis_size["y"],
    axis_size["x"],
) != expected_shape:
    raise ValueError(
        f"Spatial mismatch: "
        f"DAPI={(axis_size['y'], axis_size['x'])}, "
        f"mask={expected_shape}"
    )

if DAPI_CHANNEL >= axis_size["c"]:
    raise ValueError(
        f"DAPI_CHANNEL={DAPI_CHANNEL}, but only "
        f"{axis_size['c']} channels exist"
    )

y_chunk_size = int(
    dapi_array.chunks[DAPI_LAYOUT.index("y")]
)

x_chunk_size = int(
    dapi_array.chunks[DAPI_LAYOUT.index("x")]
)

print(
    f"  Spatial chunk size: "
    f"{y_chunk_size} × {x_chunk_size}",
    flush=True,
)


# ── 4. Per-z bounding boxes and segmentation areas ────────────────────────────
print(
    "\n[4] Calculating per-z object bounding boxes and areas ...",
    flush=True,
)

# Per z-layer:
#   stack_id -> (y_min, y_max, x_min, x_max)
per_z_bboxes = []

# Per z-layer:
#   stack_id -> segmentation pixel area
per_z_areas = []

for z, mask in enumerate(stitched_masks):
    print(
        f"  Finding objects at z{z} ...",
        flush=True,
    )

    object_slices = ndimage.find_objects(mask)
    bbox_dict = {}
    area_dict = {}

    for label, object_slice in enumerate(
        object_slices,
        start=1,
    ):
        if object_slice is None:
            continue

        y_slice, x_slice = object_slice

        y0 = int(y_slice.start)
        y1 = int(y_slice.stop)
        x0 = int(x_slice.start)
        x1 = int(x_slice.stop)

        object_crop = mask[
            y0:y1,
            x0:x1,
        ]

        area = int(
            np.count_nonzero(
                object_crop == label
            )
        )

        if area <= 0:
            continue

        bbox_dict[label] = (
            y0,
            y1,
            x0,
            x1,
        )

        area_dict[label] = area

    per_z_bboxes.append(bbox_dict)
    per_z_areas.append(area_dict)

    print(
        f"    Found {len(bbox_dict)} objects",
        flush=True,
    )


# ── 5. Select the largest segmentation for every stack ───────────────────────
print(
    "\n[5] Selecting the largest segmentation mask per stack ...",
    flush=True,
)

# stack_id -> metadata for the selected largest mask
largest_mask_info = {}

for stack_id in stack_ids:
    sid = int(stack_id)

    best_z = None
    best_area = -1
    best_bbox = None

    for z in present_z_lookup.get(sid, set()):
        z = int(z)

        if not (0 <= z < N_Z):
            continue

        area = per_z_areas[z].get(sid)
        bbox = per_z_bboxes[z].get(sid)

        if area is None or bbox is None:
            continue

        # If two layers have the same area, the lower z-index is retained.
        if area > best_area:
            best_area = int(area)
            best_z = z
            best_bbox = bbox

    if (
        best_z is None
        or best_bbox is None
        or best_area <= 0
    ):
        continue

    y0, y1, x0, x1 = best_bbox

    fixed_mask_crop = (
        stitched_masks[best_z][
            y0:y1,
            x0:x1,
        ]
        == sid
    )

    actual_area = int(
        fixed_mask_crop.sum()
    )

    if actual_area <= 0:
        continue

    largest_mask_info[sid] = {
        "source_z": int(best_z),
        "area": actual_area,
        "bbox": (
            int(y0),
            int(y1),
            int(x0),
            int(x1),
        ),
        "mask": fixed_mask_crop,
    }

print(
    f"  Selected largest masks for "
    f"{len(largest_mask_info)} stacks",
    flush=True,
)

# Per-z metadata is no longer needed.
del per_z_bboxes
del per_z_areas
gc.collect()


# ── 6. Build chunk jobs from exact largest-mask fragments ─────────────────────
print(
    "\n[6] Preparing chunk-wise fixed-mask extraction jobs ...",
    flush=True,
)

processed_stack_ids = np.asarray(
    sorted(largest_mask_info),
    dtype=np.int64,
)

n_stacks = len(processed_stack_ids)

stack_id_to_index = {
    int(sid): index
    for index, sid in enumerate(processed_stack_ids)
}

# Each job is:
#   (
#       stack_index,
#       local_y_min_in_chunk,
#       local_y_max_in_chunk,
#       local_x_min_in_chunk,
#       local_x_max_in_chunk,
#       fixed_binary_mask_fragment,
#   )
#
# The same jobs are reused for every DAPI z-layer because the largest mask is a
# fixed XY ROI.
chunk_jobs = defaultdict(list)

for sid, info in largest_mask_info.items():
    stack_index = stack_id_to_index[sid]
    y0, y1, x0, x1 = info["bbox"]
    fixed_mask = info["mask"]

    y_first = y0 // y_chunk_size
    y_last = (y1 - 1) // y_chunk_size
    x_first = x0 // x_chunk_size
    x_last = (x1 - 1) // x_chunk_size

    for cy in range(y_first, y_last + 1):
        chunk_y_min = cy * y_chunk_size
        chunk_y_max = min(
            IMAGE_HEIGHT,
            chunk_y_min + y_chunk_size,
        )

        global_y0 = max(y0, chunk_y_min)
        global_y1 = min(y1, chunk_y_max)

        if global_y0 >= global_y1:
            continue

        for cx in range(x_first, x_last + 1):
            chunk_x_min = cx * x_chunk_size
            chunk_x_max = min(
                IMAGE_WIDTH,
                chunk_x_min + x_chunk_size,
            )

            global_x0 = max(x0, chunk_x_min)
            global_x1 = min(x1, chunk_x_max)

            if global_x0 >= global_x1:
                continue

            # Coordinates relative to the stack mask crop.
            mask_y0 = global_y0 - y0
            mask_y1 = global_y1 - y0
            mask_x0 = global_x0 - x0
            mask_x1 = global_x1 - x0

            mask_fragment = fixed_mask[
                mask_y0:mask_y1,
                mask_x0:mask_x1,
            ]

            if not mask_fragment.any():
                continue

            # Coordinates relative to the spatial DAPI chunk.
            local_y0 = global_y0 - chunk_y_min
            local_y1 = global_y1 - chunk_y_min
            local_x0 = global_x0 - chunk_x_min
            local_x1 = global_x1 - chunk_x_min

            chunk_jobs[(cy, cx)].append(
                (
                    stack_index,
                    int(local_y0),
                    int(local_y1),
                    int(local_x0),
                    int(local_x1),
                    mask_fragment.copy(),
                )
            )

chunk_keys = sorted(chunk_jobs)

print(
    f"  Number of stacks: {n_stacks}",
    flush=True,
)

print(
    f"  Occupied spatial chunks: {len(chunk_keys)}",
    flush=True,
)

print(
    f"  Planned DAPI chunk reads: "
    f"{N_Z * len(chunk_keys)}",
    flush=True,
)

# The full 45k × 45k stitched masks are no longer needed.
del stitched_masks
gc.collect()


# ── 7. Chunk-wise fixed-mask intensity extraction ─────────────────────────────
print(
    "\n[7] Calculating per-stack fixed-mask z-intensity profiles ...",
    flush=True,
)

intensity_sums = np.zeros(
    (n_stacks, N_Z),
    dtype=np.float64,
)

intensity_counts = np.zeros(
    (n_stacks, N_Z),
    dtype=np.int64,
)

total_reads = N_Z * len(chunk_keys)
read_counter = 0

for z in range(N_Z):
    print(
        f"\n  Processing z{z} ...",
        flush=True,
    )

    for cy, cx in chunk_keys:
        jobs = chunk_jobs[(cy, cx)]

        chunk_y_min = cy * y_chunk_size
        chunk_y_max = min(
            IMAGE_HEIGHT,
            chunk_y_min + y_chunk_size,
        )

        chunk_x_min = cx * x_chunk_size
        chunk_x_max = min(
            IMAGE_WIDTH,
            chunk_x_min + x_chunk_size,
        )

        dapi_chunk = read_dapi_window(
            dapi_array,
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
            dapi_fragment = dapi_chunk[
                local_y0:local_y1,
                local_x0:local_x1,
            ]

            valid_mask = (
                mask_fragment
                & np.isfinite(dapi_fragment)
            )

            if not valid_mask.any():
                continue

            values = dapi_fragment[
                valid_mask
            ]

            intensity_sums[
                stack_index,
                z,
            ] += values.sum(
                dtype=np.float64
            )

            intensity_counts[
                stack_index,
                z,
            ] += int(values.size)

        del dapi_chunk
        gc.collect()

        read_counter += 1

        if (
            read_counter % PROGRESS_EVERY_CHUNKS == 0
            or read_counter == total_reads
        ):
            print(
                f"    chunk {read_counter}/{total_reads}: "
                f"y={chunk_y_min}:{chunk_y_max}, "
                f"x={chunk_x_min}:{chunk_x_max}, "
                f"jobs={len(jobs)}",
                flush=True,
            )


# ── 8. Means and output table ─────────────────────────────────────────────────
print(
    "\n[8] Building output table ...",
    flush=True,
)

mean_intensities = np.full(
    (n_stacks, N_Z),
    np.nan,
    dtype=np.float64,
)

valid = intensity_counts > 0

mean_intensities[valid] = (
    intensity_sums[valid]
    / intensity_counts[valid]
)

largest_mask_source_z = np.empty(
    n_stacks,
    dtype=np.int16,
)

largest_mask_area = np.empty(
    n_stacks,
    dtype=np.int32,
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
    (n_stacks, N_Z),
    dtype=bool,
)

for index, sid_value in enumerate(processed_stack_ids):
    sid = int(sid_value)
    info = largest_mask_info[sid]

    largest_mask_source_z[index] = int(
        info["source_z"]
    )

    largest_mask_area[index] = int(
        info["area"]
    )

    y0, y1, x0, x1 = info["bbox"]

    roi_y_min[index] = y0
    roi_y_max[index] = y1
    roi_x_min[index] = x0
    roi_x_max[index] = x1

    n_z_layers[index] = int(
        n_z_lookup.get(
            sid,
            len(present_z_lookup.get(sid, set())),
        )
    )

    for z in present_z_lookup.get(sid, set()):
        z = int(z)

        if 0 <= z < N_Z:
            present_matrix[index, z] = True


output_df = pd.DataFrame(
    {
        "stack_id": np.repeat(
            processed_stack_ids,
            N_Z,
        ),
        "z": np.tile(
            np.arange(N_Z, dtype=np.int16),
            n_stacks,
        ),
        "stack_present_in_z": (
            present_matrix.reshape(-1)
        ),
        "n_z_layers": np.repeat(
            n_z_layers,
            N_Z,
        ),
        "largest_mask_source_z": np.repeat(
            largest_mask_source_z,
            N_Z,
        ),
        "largest_mask_area_pixels": np.repeat(
            largest_mask_area,
            N_Z,
        ),
        "fixed_mask_mean_dapi_intensity": (
            mean_intensities.reshape(-1)
        ),
        "fixed_mask_valid_pixels": (
            intensity_counts.reshape(-1)
        ),
        "roi_type": "largest_segmentation_mask",
        "roi_y_min": np.repeat(
            roi_y_min,
            N_Z,
        ),
        "roi_y_max": np.repeat(
            roi_y_max,
            N_Z,
        ),
        "roi_x_min": np.repeat(
            roi_x_min,
            N_Z,
        ),
        "roi_x_max": np.repeat(
            roi_x_max,
            N_Z,
        ),
        "roi_bbox_height": np.repeat(
            roi_y_max - roi_y_min,
            N_Z,
        ),
        "roi_bbox_width": np.repeat(
            roi_x_max - roi_x_min,
            N_Z,
        ),
    }
)


# ── 9. Save ───────────────────────────────────────────────────────────────────
output_path = OUTPUT_DIR / OUTPUT_FILENAME

output_df.to_csv(
    output_path,
    index=False,
)

print("\nDone!", flush=True)
print(
    f"  Stacks processed: {n_stacks}",
    flush=True,
)
print(
    f"  Rows written: {len(output_df)}",
    flush=True,
)
print(
    "  ROI: exact largest segmentation mask, "
    "fixed across all z-layers",
    flush=True,
)
print(
    f"  Saved: {output_path}",
    flush=True,
)