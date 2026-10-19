#!/usr/bin/env python3

"""
Memory-efficient geometry metrics for stitched cell stacks.

Per stack_id this script calculates:
  1) surface-area variation across z
  2) center-of-mass / centroid shift across consecutive z-layers
  3) IoU to the matched maximum-projection cell

It also saves loss/inflation relative to the max-projection reference because
these come from the same intersection calculation at essentially no extra cost.

Geometry is calculated from the ORIGINAL z-specific stitched masks.
The max-projection mask is used only as the reference for IoU/loss/inflation.
Missing z-layers are not treated as zero-area layers.

Outputs are written to:
  .../stack_geometry_metrics/
      stack_geometry_metrics.csv
      stack_geometry_metrics_per_z.csv
      top20_surface_area_change.csv
      top20_com_shift.csv
      top20_low_iou.csv
      distributions/*.png
      rankings/*.png
      top20_surface_area/rank_*.png
      top20_com_shift/rank_*.png
      top20_low_iou/rank_*.png
"""

from pathlib import Path
import warnings
import gc

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff
import zarr


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_DIR = Path(
    "/data/gent/vo/000/gvo00070/vsc48277/yujin_project"
)

STITCH_OUTPUT_DIR = (
    PROJECT_DIR
    / "40k_subset_0_cellpose_fullcell_stitch3d_tiled_filtered"
)

STITCHED_MASK_DIR = STITCH_OUTPUT_DIR / "stitched_masks"

VSI_CSV = (
    STITCH_OUTPUT_DIR
    / "ovrlpy_vsi_maxproj_masks"
    / "stack_vsi_summary_maxproj_mask.csv"
)

ZARR_PATH = PROJECT_DIR / "40k_subset_0.zarr"

MAXPROJ_ARRAY_PATH = (
    "labels/segmentation_mask_optimized_maxproj_45k/0"
)

# CLAHE DAPI + PolyT image used for Top-20 diagnostic figures.
# Expected shape: (channel, z, y, x)
IMAGE_ARRAY_PATH = (
    "images/clahe_DAPI_PolyT/0"
)

DAPI_CHANNEL = 0
POLYT_CHANNEL = 1

# Padding around the matched max-projection cell for visualization.
DISPLAY_PADDING = 100

OUT_DIR = STITCH_OUTPUT_DIR / "stack_geometry_metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

Z_VALUES = list(range(7))
PIXEL_SIZE_UM = 0.108
CHUNK_SIZE = 1024
TOP_N = 20


# =============================================================================
# PLOT STYLE
# =============================================================================

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.15)


def require_columns(df, cols, name):
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} is missing columns: {sorted(missing)}\n"
            f"Available: {df.columns.tolist()}"
        )


def agg_nan(series, func):
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(func(arr))


# =============================================================================
# 1. LOAD STACK -> MAXPROJ MAPPING
# =============================================================================

print("\n[1] Loading stack-to-maxproj mapping ...", flush=True)

if not VSI_CSV.exists():
    raise FileNotFoundError(VSI_CSV)

vsi_df = pd.read_csv(VSI_CSV)
require_columns(
    vsi_df,
    ["stack_id", "maxproj_cell_id", "n_vsi_pixels"],
    "VSI table",
)

vsi_df["stack_id"] = pd.to_numeric(
    vsi_df["stack_id"], errors="raise"
).astype(np.int64)

vsi_df["maxproj_cell_id"] = pd.to_numeric(
    vsi_df["maxproj_cell_id"], errors="coerce"
)

vsi_df["n_vsi_pixels"] = pd.to_numeric(
    vsi_df["n_vsi_pixels"], errors="coerce"
)

# Geometry metrics are calculated independently of VSI quality.
# VSI_CSV is used here only as a convenient stack_id -> maxproj_cell_id mapping.
mapping_source = (
    vsi_df.loc[
        vsi_df["maxproj_cell_id"].notna(),
        ["stack_id", "maxproj_cell_id"],
    ]
    .copy()
)

# Do not silently choose one mapping if a stack_id has conflicting
# max-projection assignments.
mapping_conflicts = (
    mapping_source
    .groupby("stack_id")["maxproj_cell_id"]
    .nunique()
)

mapping_conflicts = mapping_conflicts[
    mapping_conflicts > 1
]

if len(mapping_conflicts) > 0:
    raise ValueError(
        "Some stack_id values map to more than one maxproj_cell_id. "
        f"Examples: {mapping_conflicts.head(10).to_dict()}"
    )

mapping_df = (
    mapping_source
    .drop_duplicates("stack_id")
    .copy()
)

mapping_df["maxproj_cell_id"] = mapping_df[
    "maxproj_cell_id"
].astype(np.int64)

if mapping_df.empty:
    raise ValueError("No valid stack-to-maxproj mappings after QC.")

stack_ids = mapping_df["stack_id"].to_numpy(dtype=np.int64)
max_stack_id = int(stack_ids.max())
max_maxproj_id = int(mapping_df["maxproj_cell_id"].max())

stack_to_maxproj = np.zeros(max_stack_id + 1, dtype=np.int64)
stack_to_maxproj[stack_ids] = mapping_df[
    "maxproj_cell_id"
].to_numpy(dtype=np.int64)

valid_stack_lookup = np.zeros(max_stack_id + 1, dtype=bool)
valid_stack_lookup[stack_ids] = True

print(f"  Mapped stacks: {len(mapping_df):,}")
print(
    "  Unique matched maxproj cells: "
    f"{mapping_df['maxproj_cell_id'].nunique():,}"
)


# =============================================================================
# 2. OPEN MASKS
# =============================================================================

print("\n[2] Opening masks ...", flush=True)

root = zarr.open(str(ZARR_PATH), mode="r")
if MAXPROJ_ARRAY_PATH not in root:
    raise KeyError(MAXPROJ_ARRAY_PATH)

maxproj = root[MAXPROJ_ARRAY_PATH]
if maxproj.ndim != 2:
    raise ValueError(f"Expected 2D maxproj mask, got {maxproj.shape}")

if IMAGE_ARRAY_PATH not in root:
    raise KeyError(IMAGE_ARRAY_PATH)

image_array = root[IMAGE_ARRAY_PATH]

if image_array.ndim != 4:
    raise ValueError(
        f"Expected image array (c,z,y,x), got {image_array.shape}"
    )

if image_array.shape[0] <= max(DAPI_CHANNEL, POLYT_CHANNEL):
    raise ValueError(
        f"Requested DAPI/PolyT channels are not available: "
        f"image shape={image_array.shape}"
    )

if image_array.shape[1] <= max(Z_VALUES):
    raise ValueError(
        f"Image does not contain all requested z-layers: "
        f"image shape={image_array.shape}, Z_VALUES={Z_VALUES}"
    )

stitched_masks = {}
for z in Z_VALUES:
    p = STITCHED_MASK_DIR / f"mask_z{z}.tif"
    if not p.exists():
        raise FileNotFoundError(p)
    stitched_masks[z] = tiff.memmap(p)

shape = tuple(stitched_masks[Z_VALUES[0]].shape)
if tuple(maxproj.shape) != shape:
    raise ValueError(
        f"Shape mismatch: stitched={shape}, maxproj={maxproj.shape}"
    )

for z in Z_VALUES:
    if tuple(stitched_masks[z].shape) != shape:
        raise ValueError(f"mask_z{z}.tif has inconsistent shape")

HEIGHT, WIDTH = shape

if tuple(image_array.shape[-2:]) != shape:
    raise ValueError(
        f"Spatial shape mismatch: image={image_array.shape[-2:]}, "
        f"stitched={shape}"
    )

print(f"  Shape: {HEIGHT} x {WIDTH}")
print(f"  Image shape: {image_array.shape}")
print(f"  Chunk size: {CHUNK_SIZE}")


# =============================================================================
# 3. MAXPROJ AREAS
# =============================================================================

print("\n[3] Computing maxproj-cell areas ...", flush=True)

maxproj_area = np.zeros(max_maxproj_id + 1, dtype=np.int64)

maxproj_min_x = np.full(
    max_maxproj_id + 1,
    WIDTH,
    dtype=np.int32,
)
maxproj_min_y = np.full(
    max_maxproj_id + 1,
    HEIGHT,
    dtype=np.int32,
)
maxproj_max_x = np.full(
    max_maxproj_id + 1,
    -1,
    dtype=np.int32,
)
maxproj_max_y = np.full(
    max_maxproj_id + 1,
    -1,
    dtype=np.int32,
)

total_tiles = (
    int(np.ceil(HEIGHT / CHUNK_SIZE))
    * int(np.ceil(WIDTH / CHUNK_SIZE))
)

tile_n = 0
for y0 in range(0, HEIGHT, CHUNK_SIZE):
    y1 = min(y0 + CHUNK_SIZE, HEIGHT)
    for x0 in range(0, WIDTH, CHUNK_SIZE):
        x1 = min(x0 + CHUNK_SIZE, WIDTH)
        tile_n += 1

        mp = np.asarray(maxproj[y0:y1, x0:x1]).astype(
            np.int64, copy=False
        )

        keep = (mp > 0) & (mp <= max_maxproj_id)
        if np.any(keep):
            labels = mp[keep]

            bc = np.bincount(
                labels, minlength=max_maxproj_id + 1
            )
            maxproj_area += bc[: max_maxproj_id + 1]

            yy, xx = np.nonzero(keep)
            gx = xx.astype(np.int32) + x0
            gy = yy.astype(np.int32) + y0

            np.minimum.at(
                maxproj_min_x,
                labels,
                gx,
            )
            np.minimum.at(
                maxproj_min_y,
                labels,
                gy,
            )
            np.maximum.at(
                maxproj_max_x,
                labels,
                gx,
            )
            np.maximum.at(
                maxproj_max_y,
                labels,
                gy,
            )

        del mp

        if tile_n % 100 == 0 or tile_n == total_tiles:
            print(
                f"  maxproj tiles: {tile_n:,}/{total_tiles:,}",
                flush=True,
            )

missing_reference_ids = mapping_df.loc[
    maxproj_area[
        mapping_df["maxproj_cell_id"].to_numpy(dtype=np.int64)
    ] == 0,
    "maxproj_cell_id",
].unique()

if len(missing_reference_ids) > 0:
    raise ValueError(
        "Some mapped maxproj_cell_id values do not exist in the "
        "max-projection label mask. Examples: "
        f"{missing_reference_ids[:10].tolist()}"
    )


# =============================================================================
# 4. PER-Z AREA, CENTROID, INTERSECTION
# =============================================================================

print("\n[4] Computing per-z geometry ...", flush=True)

per_z_frames = []

for z in Z_VALUES:
    print(f"\n  z={z}", flush=True)

    area = np.zeros(max_stack_id + 1, dtype=np.int64)
    sum_x = np.zeros(max_stack_id + 1, dtype=np.float64)
    sum_y = np.zeros(max_stack_id + 1, dtype=np.float64)
    intersection = np.zeros(max_stack_id + 1, dtype=np.int64)

    # z-specific stack bounding boxes, accumulated without loading a whole mask.
    stack_min_x = np.full(
        max_stack_id + 1,
        WIDTH,
        dtype=np.int32,
    )
    stack_min_y = np.full(
        max_stack_id + 1,
        HEIGHT,
        dtype=np.int32,
    )
    stack_max_x = np.full(
        max_stack_id + 1,
        -1,
        dtype=np.int32,
    )
    stack_max_y = np.full(
        max_stack_id + 1,
        -1,
        dtype=np.int32,
    )

    tile_n = 0

    for y0 in range(0, HEIGHT, CHUNK_SIZE):
        y1 = min(y0 + CHUNK_SIZE, HEIGHT)
        for x0 in range(0, WIDTH, CHUNK_SIZE):
            x1 = min(x0 + CHUNK_SIZE, WIDTH)
            tile_n += 1

            stitched = np.asarray(
                stitched_masks[z][y0:y1, x0:x1]
            ).astype(np.int64, copy=False)

            flat = stitched.ravel()
            in_range = (
                (flat > 0)
                & (flat <= max_stack_id)
            )

            if np.any(in_range):
                flat_idx = np.flatnonzero(in_range)
                labels = flat[flat_idx]
                mapped = valid_stack_lookup[labels]

                if np.any(mapped):
                    flat_idx = flat_idx[mapped]
                    labels = labels[mapped]

                    # area
                    bc = np.bincount(
                        labels, minlength=max_stack_id + 1
                    )
                    area += bc[: max_stack_id + 1]

                    # centroid coordinate sums
                    w = x1 - x0
                    local_y = flat_idx // w
                    local_x = flat_idx % w

                    gx = local_x.astype(np.float64) + x0
                    gy = local_y.astype(np.float64) + y0

                    # Bounding boxes for visualization of the complete
                    # z-specific segmented object.
                    np.minimum.at(
                        stack_min_x,
                        labels,
                        gx.astype(np.int32),
                    )
                    np.minimum.at(
                        stack_min_y,
                        labels,
                        gy.astype(np.int32),
                    )
                    np.maximum.at(
                        stack_max_x,
                        labels,
                        gx.astype(np.int32),
                    )
                    np.maximum.at(
                        stack_max_y,
                        labels,
                        gy.astype(np.int32),
                    )

                    bx = np.bincount(
                        labels,
                        weights=gx,
                        minlength=max_stack_id + 1,
                    )
                    by = np.bincount(
                        labels,
                        weights=gy,
                        minlength=max_stack_id + 1,
                    )

                    sum_x += bx[: max_stack_id + 1]
                    sum_y += by[: max_stack_id + 1]

                    # intersection with matched maxproj cell
                    mp = np.asarray(
                        maxproj[y0:y1, x0:x1]
                    ).astype(np.int64, copy=False)

                    mp_selected = mp.ravel()[flat_idx]
                    target = stack_to_maxproj[labels]
                    same = mp_selected == target

                    if np.any(same):
                        bi = np.bincount(
                            labels[same],
                            minlength=max_stack_id + 1,
                        )
                        intersection += bi[: max_stack_id + 1]

                    del mp

            del stitched

            if tile_n % 100 == 0 or tile_n == total_tiles:
                print(
                    f"    tiles: {tile_n:,}/{total_tiles:,}",
                    flush=True,
                )

    present = stack_ids[area[stack_ids] > 0]

    zdf = pd.DataFrame(
        {
            "stack_id": present,
            "z": z,
            "area_px": area[present],
            "intersection_px": intersection[present],
        }
    )

    zdf["centroid_x_px"] = sum_x[present] / area[present]
    zdf["centroid_y_px"] = sum_y[present] / area[present]

    zdf["bbox_x_min"] = stack_min_x[present]
    zdf["bbox_x_max"] = stack_max_x[present]
    zdf["bbox_y_min"] = stack_min_y[present]
    zdf["bbox_y_max"] = stack_max_y[present]

    zdf["maxproj_cell_id"] = stack_to_maxproj[present]
    zdf["maxproj_area_px"] = maxproj_area[
        zdf["maxproj_cell_id"].to_numpy(dtype=np.int64)
    ]

    zdf["union_px"] = (
        zdf["area_px"]
        + zdf["maxproj_area_px"]
        - zdf["intersection_px"]
    )

    zdf["iou"] = np.where(
        zdf["union_px"] > 0,
        zdf["intersection_px"] / zdf["union_px"],
        np.nan,
    )

    zdf["inflation_fraction"] = np.where(
        zdf["maxproj_area_px"] > 0,
        (zdf["area_px"] - zdf["intersection_px"])
        / zdf["maxproj_area_px"],
        np.nan,
    )

    zdf["loss_fraction"] = np.where(
        zdf["maxproj_area_px"] > 0,
        (zdf["maxproj_area_px"] - zdf["intersection_px"])
        / zdf["maxproj_area_px"],
        np.nan,
    )

    zdf["area_um2"] = zdf["area_px"] * (PIXEL_SIZE_UM ** 2)
    per_z_frames.append(zdf)

    del (
        area,
        sum_x,
        sum_y,
        intersection,
        stack_min_x,
        stack_min_y,
        stack_max_x,
        stack_max_y,
        zdf,
    )
    gc.collect()


# =============================================================================
# 5. ADJACENT-Z METRICS, VECTORIZED
# =============================================================================

print("\n[5] Building per-z table ...", flush=True)

per_z_df = pd.concat(per_z_frames, ignore_index=True)
per_z_df = per_z_df.sort_values(
    ["stack_id", "z"]
).reset_index(drop=True)

by_stack = per_z_df.groupby("stack_id", sort=False)

per_z_df["previous_z"] = by_stack["z"].shift(1)
per_z_df["previous_area_px"] = by_stack["area_px"].shift(1)
per_z_df["previous_centroid_x_px"] = by_stack[
    "centroid_x_px"
].shift(1)
per_z_df["previous_centroid_y_px"] = by_stack[
    "centroid_y_px"
].shift(1)

consecutive = (
    per_z_df["z"] - per_z_df["previous_z"]
) == 1

area_denominator = np.maximum(
    per_z_df["area_px"].to_numpy(dtype=float),
    per_z_df["previous_area_px"].to_numpy(dtype=float),
)

per_z_df["adjacent_area_change"] = np.nan
valid_area = consecutive & (area_denominator > 0)
per_z_df.loc[
    valid_area,
    "adjacent_area_change",
] = (
    np.abs(
        per_z_df.loc[valid_area, "area_px"].to_numpy(dtype=float)
        - per_z_df.loc[valid_area, "previous_area_px"].to_numpy(dtype=float)
    )
    / area_denominator[valid_area.to_numpy()]
)

per_z_df["centroid_shift_px"] = np.nan

dx = (
    per_z_df["centroid_x_px"]
    - per_z_df["previous_centroid_x_px"]
)
dy = (
    per_z_df["centroid_y_px"]
    - per_z_df["previous_centroid_y_px"]
)

per_z_df.loc[
    consecutive,
    "centroid_shift_px",
] = np.hypot(
    dx.loc[consecutive],
    dy.loc[consecutive],
)

per_z_df["centroid_shift_um"] = (
    per_z_df["centroid_shift_px"] * PIXEL_SIZE_UM
)


# =============================================================================
# 6. ONE ROW PER STACK
# =============================================================================

print("\n[6] Summarizing one row per stack ...", flush=True)

rows = []

for stack_id, sub in per_z_df.groupby("stack_id", sort=False):
    areas = sub["area_px"].to_numpy(dtype=float)
    max_area = np.nanmax(areas)
    min_area = np.nanmin(areas)

    if max_area > 0:
        relative_area_range = (max_area - min_area) / max_area
    else:
        relative_area_range = np.nan

    finite_shift_px = sub["centroid_shift_px"].dropna().to_numpy(dtype=float)
    finite_shift_um = sub["centroid_shift_um"].dropna().to_numpy(dtype=float)

    rows.append(
        {
            "stack_id": int(stack_id),
            "maxproj_cell_id": int(sub["maxproj_cell_id"].iloc[0]),
            "n_z_layers": int(len(sub)),
            "z_min": int(sub["z"].min()),
            "z_max": int(sub["z"].max()),
            "n_adjacent_pairs": int(sub["adjacent_area_change"].notna().sum()),

            # Surface-area metrics
            "mean_area_px": float(np.mean(areas)),
            "median_area_px": float(np.median(areas)),
            "min_area_px": float(min_area),
            "max_area_px": float(max_area),
            "mean_area_um2": float(np.mean(areas) * PIXEL_SIZE_UM ** 2),
            "relative_area_range": float(relative_area_range),
            "mean_adjacent_area_change": agg_nan(
                sub["adjacent_area_change"], np.mean
            ),
            "max_adjacent_area_change": agg_nan(
                sub["adjacent_area_change"], np.max
            ),

            # COM shift
            "mean_centroid_shift_px": agg_nan(
                sub["centroid_shift_px"], np.mean
            ),
            "max_centroid_shift_px": agg_nan(
                sub["centroid_shift_px"], np.max
            ),
            "total_centroid_path_px": (
                float(finite_shift_px.sum())
                if finite_shift_px.size
                else np.nan
            ),
            "mean_centroid_shift_um": agg_nan(
                sub["centroid_shift_um"], np.mean
            ),
            "max_centroid_shift_um": agg_nan(
                sub["centroid_shift_um"], np.max
            ),
            "total_centroid_path_um": (
                float(finite_shift_um.sum())
                if finite_shift_um.size
                else np.nan
            ),

            # IoU / shape deviation
            "mean_iou": agg_nan(sub["iou"], np.mean),
            "median_iou": agg_nan(sub["iou"], np.median),
            "min_iou": agg_nan(sub["iou"], np.min),
            "std_iou": agg_nan(sub["iou"], np.std),
            "mean_inflation": agg_nan(
                sub["inflation_fraction"], np.mean
            ),
            "max_inflation": agg_nan(
                sub["inflation_fraction"], np.max
            ),
            "mean_loss": agg_nan(
                sub["loss_fraction"], np.mean
            ),
            "max_loss": agg_nan(
                sub["loss_fraction"], np.max
            ),
        }
    )

geometry_df = pd.DataFrame(rows).sort_values(
    "stack_id"
).reset_index(drop=True)


# =============================================================================
# 7. SAVE TABLES
# =============================================================================

per_z_csv = OUT_DIR / "stack_geometry_metrics_per_z.csv"
geometry_csv = OUT_DIR / "stack_geometry_metrics.csv"

per_z_df.to_csv(per_z_csv, index=False)
geometry_df.to_csv(geometry_csv, index=False)

print(f"  Saved: {per_z_csv}")
print(f"  Saved: {geometry_csv}")


# =============================================================================
# 8. TOP-20 TABLES
# =============================================================================

surface_rank = (
    geometry_df.dropna(subset=["max_adjacent_area_change"])
    .sort_values("max_adjacent_area_change", ascending=False)
    .head(TOP_N)
    .copy()
)

com_rank = (
    geometry_df.dropna(subset=["max_centroid_shift_um"])
    .sort_values("max_centroid_shift_um", ascending=False)
    .head(TOP_N)
    .copy()
)

iou_rank = (
    geometry_df.dropna(subset=["min_iou"])
    .sort_values("min_iou", ascending=True)
    .head(TOP_N)
    .copy()
)

for df in (surface_rank, com_rank, iou_rank):
    df.insert(0, "rank", np.arange(1, len(df) + 1))

surface_rank.to_csv(
    OUT_DIR / "top20_surface_area_change.csv", index=False
)
com_rank.to_csv(
    OUT_DIR / "top20_com_shift.csv", index=False
)
iou_rank.to_csv(
    OUT_DIR / "top20_low_iou.csv", index=False
)


# =============================================================================
# 9. DISTRIBUTION PLOTS
# =============================================================================

dist_dir = OUT_DIR / "distributions"
dist_dir.mkdir(parents=True, exist_ok=True)

plot_specs = [
    (
        "max_adjacent_area_change",
        "Maximum adjacent area change",
        "Distribution of maximum adjacent area change",
        "01_max_adjacent_area_change_distribution.png",
    ),
    (
        "max_centroid_shift_um",
        "Maximum centroid shift (µm)",
        "Distribution of maximum centroid shift",
        "02_max_centroid_shift_distribution.png",
    ),
    (
        "min_iou",
        "Minimum IoU with matched max-projection cell",
        "Distribution of minimum IoU",
        "03_min_iou_distribution.png",
    ),
]

for column, xlabel, title, filename in plot_specs:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(geometry_df[column].dropna(), bins=60)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cell-stack count")
    ax.set_title(title)
    clean_axis(ax)
    fig.tight_layout()
    fig.savefig(
        dist_dir / filename,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


# =============================================================================
# 10. TOP-20 RANK SUMMARY PLOTS
# =============================================================================

rank_dir = OUT_DIR / "rankings"
rank_dir.mkdir(parents=True, exist_ok=True)


def save_rank_plot(df, value_column, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["rank"], df[value_column], marker="o")
    ax.set_xticks(df["rank"])
    ax.set_xlabel("Rank")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    clean_axis(ax)
    fig.tight_layout()
    fig.savefig(
        rank_dir / filename,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


save_rank_plot(
    surface_rank,
    "max_adjacent_area_change",
    "Maximum adjacent area change",
    "Top 20 cell stacks by cross-sectional area change",
    "01_top20_surface_area_change.png",
)

save_rank_plot(
    com_rank,
    "max_centroid_shift_um",
    "Maximum centroid shift (µm)",
    "Top 20 cell stacks by centroid shift",
    "02_top20_com_shift.png",
)

save_rank_plot(
    iou_rank,
    "min_iou",
    "Minimum IoU",
    "Top 20 cell stacks with lowest IoU",
    "03_top20_low_iou.png",
)


# =============================================================================
# 11. INDIVIDUAL DIAGNOSTIC PLOTS FOR EACH TOP-20 STACK
# =============================================================================

surface_dir = OUT_DIR / "top20_surface_area"
com_dir = OUT_DIR / "top20_com_shift"
iou_dir = OUT_DIR / "top20_low_iou"

for d in (surface_dir, com_dir, iou_dir):
    d.mkdir(parents=True, exist_ok=True)


def robust_limits(image):
    arr = np.asarray(
        image,
        dtype=np.float32,
    )

    arr = arr[
        np.isfinite(
            arr
        )
    ]

    if arr.size == 0:
        return 0.0, 1.0

    vmin, vmax = np.percentile(
        arr,
        [
            1.0,
            99.5,
        ],
    )

    if vmax <= vmin:
        vmax = vmin + 1.0

    return (
        float(vmin),
        float(vmax),
    )


def add_mask_contour(
    ax,
    mask,
    color,
    linestyle="-",
    linewidth=1.2,
):
    mask = np.asarray(
        mask,
        dtype=bool,
    )

    if np.any(
        mask
    ):
        ax.contour(
            mask.astype(
                np.uint8
            ),
            levels=[
                0.5
            ],
            colors=[
                color
            ],
            linestyles=linestyle,
            linewidths=linewidth,
        )


def plot_stack_diagnostic(
    stack_id,
    rank,
    metric_name,
    metric_value,
    output_path,
):
    """
    Top-20 diagnostic figure.

    For every z-layer in which the stack exists:
      left  = DAPI with segmentation contours
      right = PolyT with segmentation contours

    Cyan solid:
        z-specific stitched segmentation for this stack

    Red dashed:
        matched maximum-projection segmentation

    Bottom row:
        surface area, center-of-mass shift, and IoU profiles.
    """

    sub = (
        per_z_df.loc[
            per_z_df[
                "stack_id"
            ]
            == stack_id
        ]
        .sort_values(
            "z"
        )
        .copy()
    )

    if sub.empty:
        return


    maxproj_cell_id = int(
        sub[
            "maxproj_cell_id"
        ].iloc[
            0
        ]
    )


    # --------------------------------------------------------------
    # Crop around the UNION of:
    #   1) the matched max-projection cell, and
    #   2) every z-specific stitched mask in this stack.
    #
    # This is important for COM-shift / low-IoU outliers: cropping only
    # around the maxproj cell could hide the displaced part of the stack.
    # --------------------------------------------------------------

    if (
        maxproj_cell_id > max_maxproj_id
        or maxproj_max_x[maxproj_cell_id] < 0
    ):
        return

    union_x_min = min(
        int(maxproj_min_x[maxproj_cell_id]),
        int(sub["bbox_x_min"].min()),
    )
    union_x_max = max(
        int(maxproj_max_x[maxproj_cell_id]),
        int(sub["bbox_x_max"].max()),
    )
    union_y_min = min(
        int(maxproj_min_y[maxproj_cell_id]),
        int(sub["bbox_y_min"].min()),
    )
    union_y_max = max(
        int(maxproj_max_y[maxproj_cell_id]),
        int(sub["bbox_y_max"].max()),
    )

    x_min = max(
        0,
        union_x_min - DISPLAY_PADDING,
    )
    x_max = min(
        WIDTH,
        union_x_max + 1 + DISPLAY_PADDING,
    )
    y_min = max(
        0,
        union_y_min - DISPLAY_PADDING,
    )
    y_max = min(
        HEIGHT,
        union_y_max + 1 + DISPLAY_PADDING,
    )


    maxproj_crop = np.asarray(
        maxproj[
            y_min:y_max,
            x_min:x_max,
        ]
    )

    maxproj_binary = (
        maxproj_crop
        == maxproj_cell_id
    )


    n_z = len(
        sub
    )


    fig = plt.figure(
        figsize=(
            11,
            2.7 * n_z + 3.5,
        )
    )


    gs = fig.add_gridspec(
        nrows=n_z + 1,
        ncols=6,
        height_ratios=(
            [1.0] * n_z
            + [1.2]
        ),
        hspace=0.32,
        wspace=0.35,
    )


    # --------------------------------------------------------------
    # DAPI + PolyT rows
    # --------------------------------------------------------------

    for row_index, (_, z_row) in enumerate(
        sub.iterrows()
    ):

        z = int(
            z_row[
                "z"
            ]
        )


        dapi_crop = np.asarray(
            image_array[
                DAPI_CHANNEL,
                z,
                y_min:y_max,
                x_min:x_max,
            ]
        )


        polyt_crop = np.asarray(
            image_array[
                POLYT_CHANNEL,
                z,
                y_min:y_max,
                x_min:x_max,
            ]
        )


        stitched_crop = np.asarray(
            stitched_masks[
                z
            ][
                y_min:y_max,
                x_min:x_max,
            ]
        )


        stitched_binary = (
            stitched_crop
            == stack_id
        )


        dapi_vmin, dapi_vmax = robust_limits(
            dapi_crop
        )

        polyt_vmin, polyt_vmax = robust_limits(
            polyt_crop
        )


        # DAPI
        ax = fig.add_subplot(
            gs[
                row_index,
                0:3,
            ]
        )


        ax.imshow(
            dapi_crop,
            cmap="gray",
            vmin=dapi_vmin,
            vmax=dapi_vmax,
        )


        add_mask_contour(
            ax,
            stitched_binary,
            color="cyan",
            linestyle="-",
            linewidth=1.4,
        )


        add_mask_contour(
            ax,
            maxproj_binary,
            color="red",
            linestyle="--",
            linewidth=1.0,
        )


        ax.set_title(
            (
                f"z{z} | DAPI | "
                f"area={z_row['area_um2']:.1f} µm²"
            )
        )

        ax.set_xticks(
            []
        )

        ax.set_yticks(
            []
        )


        # PolyT
        ax = fig.add_subplot(
            gs[
                row_index,
                3:6,
            ]
        )


        ax.imshow(
            polyt_crop,
            cmap="gray",
            vmin=polyt_vmin,
            vmax=polyt_vmax,
        )


        add_mask_contour(
            ax,
            stitched_binary,
            color="cyan",
            linestyle="-",
            linewidth=1.4,
        )


        add_mask_contour(
            ax,
            maxproj_binary,
            color="red",
            linestyle="--",
            linewidth=1.0,
        )


        ax.set_title(
            (
                f"z{z} | PolyT | "
                f"IoU={z_row['iou']:.3f}"
            )
        )

        ax.set_xticks(
            []
        )

        ax.set_yticks(
            []
        )


    # --------------------------------------------------------------
    # Metric profiles
    # --------------------------------------------------------------

    ax = fig.add_subplot(
        gs[
            n_z,
            0:2,
        ]
    )

    ax.plot(
        sub[
            "z"
        ],
        sub[
            "area_um2"
        ],
        marker="o",
    )

    ax.set_xlabel(
        "z-layer"
    )

    ax.set_ylabel(
        "Area (µm²)"
    )

    ax.set_title(
        "Cross-sectional area"
    )

    ax.set_xticks(
        sub[
            "z"
        ]
    )

    clean_axis(
        ax
    )


    ax = fig.add_subplot(
        gs[
            n_z,
            2:4,
        ]
    )

    shift = sub.dropna(
        subset=[
            "centroid_shift_um"
        ]
    )

    if not shift.empty:

        ax.plot(
            shift[
                "z"
            ],
            shift[
                "centroid_shift_um"
            ],
            marker="o",
        )

        ax.set_xticks(
            shift[
                "z"
            ]
        )

    ax.set_xlabel(
        "Current z-layer"
    )

    ax.set_ylabel(
        "Shift (µm)"
    )

    ax.set_title(
        "Center-of-mass shift"
    )

    clean_axis(
        ax
    )


    ax = fig.add_subplot(
        gs[
            n_z,
            4:6,
        ]
    )

    ax.plot(
        sub[
            "z"
        ],
        sub[
            "iou"
        ],
        marker="o",
    )

    ax.set_xlabel(
        "z-layer"
    )

    ax.set_ylabel(
        "IoU"
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_title(
        "IoU to max-projection"
    )

    ax.set_xticks(
        sub[
            "z"
        ]
    )

    clean_axis(
        ax
    )


    fig.suptitle(
        (
            f"Rank {rank} | stack {stack_id} | "
            f"maxproj cell {maxproj_cell_id} | "
            f"{metric_name} = {metric_value:.4f}\n"
            f"cyan solid = stitched z-specific mask; "
            f"red dashed = matched max-projection mask"
        ),
        y=0.995,
    )


    fig.subplots_adjust(
        top=0.965,
    )


    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        fig
    )


for _, row in surface_rank.iterrows():
    rank = int(row["rank"])
    sid = int(row["stack_id"])
    value = float(row["max_adjacent_area_change"])
    plot_stack_diagnostic(
        sid,
        rank,
        "max adjacent area change",
        value,
        surface_dir / f"rank_{rank:02d}_stack_{sid}.png",
    )

for _, row in com_rank.iterrows():
    rank = int(row["rank"])
    sid = int(row["stack_id"])
    value = float(row["max_centroid_shift_um"])
    plot_stack_diagnostic(
        sid,
        rank,
        "max centroid shift (µm)",
        value,
        com_dir / f"rank_{rank:02d}_stack_{sid}.png",
    )

for _, row in iou_rank.iterrows():
    rank = int(row["rank"])
    sid = int(row["stack_id"])
    value = float(row["min_iou"])
    plot_stack_diagnostic(
        sid,
        rank,
        "minimum IoU",
        value,
        iou_dir / f"rank_{rank:02d}_stack_{sid}.png",
    )


# =============================================================================
# 12. SUMMARY
# =============================================================================

print("\nDone!", flush=True)
print(f"  Stacks summarized: {len(geometry_df):,}")
print(f"  Per-z rows: {len(per_z_df):,}")

print("\nMain outputs:")
print(f"  {geometry_csv}")
print(f"  {per_z_csv}")

print("\nTop-20 folders:")
print(f"  {surface_dir}")
print(f"  {com_dir}")
print(f"  {iou_dir}")

print("\nRanking direction:")
print("  Cross-sectional area: max_adjacent_area_change  (higher = more suspicious)")
print("  COM shift:    max_centroid_shift_um     (higher = more suspicious)")
print("  IoU:          min_iou                   (lower  = more suspicious)")
