#!/usr/bin/env python3

"""
Plot top ordered-bimodal cell stacks using PolyT intensity profiles
measured inside fixed maximum-projection segmentation masks.

Selection:
    profile_status == "valid_peak_pair"
    sorted by ordered_bimodality_score descending
    top N stacks

For each selected stack, save:
    1. stack_<ID>_dapi_zlayers.png
    2. stack_<ID>_intensity_profile.png

Image overlay:
    Background      = DAPI image at each z-layer
    Red boundary    = fixed maximum-projection segmentation mask
                      used for PolyT intensity calculation
    Cyan boundary   = actual stitched segmentation of the selected stack
                      at the current z-layer
    Green boundary  = all other stitched segmentation masks in the crop

Intensity profile:
    Mean PolyT intensity measured inside the same fixed
    maximum-projection segmentation mask across z0-z6.
"""

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import zarr
import spatialdata as sd

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from skimage.segmentation import find_boundaries


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(
    "/data/gent/vo/000/gvo00070/vsc48277/yujin_project"
)

STITCH_OUTPUT_DIR = (
    PROJECT_DIR
    / "40k_subset_0_cellpose_fullcell_stitch3d_tiled"
)


# ── Input tables ──────────────────────────────────────────────────────────────

SCORES_CSV = (
    STITCH_OUTPUT_DIR
    / "ordered_bimodality_scores_polyt_maxproj"
    / "stack_ordered_bimodality_scores_polyt_maxproj.csv"
)

INTENSITY_CSV = (
    STITCH_OUTPUT_DIR
    / "polyt_maxproj_mask_z_intensity_profiles"
    / "stack_z_intensity_maxproj_mask_polyt.csv"
)


# ── Stitched masks ────────────────────────────────────────────────────────────

STITCHED_MASK_DIR = (
    STITCH_OUTPUT_DIR
    / "stitched_masks"
)

MASK_PATTERN = "mask_z{z}.tif"


# ── SpatialData ───────────────────────────────────────────────────────────────

ZARR_PATH = (
    PROJECT_DIR
    / "40k_subset_0.zarr"
)

# DAPI is used only as the grayscale background for visualization.
DAPI_ARRAY_PATH = "images/clahe_DAPI_PolyT/0"

DAPI_LAYOUT = "czyx"

DAPI_CHANNEL = 1


# Already-cropped maximum-projection segmentation stored in 40k_subset_0.zarr.
MAXPROJ_LABEL_LAYER = (
    "segmentation_mask_optimized_maxproj_45k"
)


# ── PolyT intensity ───────────────────────────────────────────────────────────

INTENSITY_COLUMN = (
    "fixed_mask_mean_polyt_intensity"
)


# ── Plot settings ─────────────────────────────────────────────────────────────

N_Z = 7

TOP_N = 20

DISPLAY_PADDING = 25

DISPLAY_LOW_PERCENTILE = 1

DISPLAY_HIGH_PERCENTILE = 99.8


# Same parameters used by the bimodality scoring script.
PEAK_SMOOTHING_SIGMA = 0.5

MIN_PEAK_PROMINENCE = 0.20

MIN_PEAK_DISTANCE = 2


# ── Output ────────────────────────────────────────────────────────────────────

OUTPUT_DIR = (
    STITCH_OUTPUT_DIR
    / "top_ordered_bimodality_plots_polyt_maxproj"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_level0(layer):
    """
    Return highest-resolution DataArray from a SpatialData element.
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


def read_dapi_crop(
    array,
    z,
    y_start,
    y_stop,
    x_start,
    x_stop,
):
    """
    Read one 2D DAPI crop.
    """

    indexing = []

    for axis_name in DAPI_LAYOUT:

        if axis_name == "c":

            indexing.append(
                DAPI_CHANNEL
            )

        elif axis_name == "z":

            indexing.append(
                z
            )

        elif axis_name == "y":

            indexing.append(
                slice(
                    y_start,
                    y_stop,
                )
            )

        elif axis_name == "x":

            indexing.append(
                slice(
                    x_start,
                    x_stop,
                )
            )


    crop = np.squeeze(
        np.asarray(
            array[
                tuple(indexing)
            ]
        )
    )


    if crop.ndim != 2:

        raise ValueError(
            f"Expected 2D DAPI crop at z{z}, "
            f"got {crop.shape}"
        )


    return crop


def normalize_profile(
    values,
):
    """
    Min-max normalize profile to [0, 1].
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )


    if not np.all(
        np.isfinite(
            values
        )
    ):

        raise ValueError(
            "Intensity profile contains "
            "non-finite values."
        )


    value_min = float(
        values.min()
    )

    value_max = float(
        values.max()
    )


    value_range = (
        value_max
        - value_min
    )


    if value_range <= 0:

        return np.zeros_like(
            values,
            dtype=np.float64,
        )


    return (
        values
        - value_min
    ) / value_range


def detect_peaks_with_edges(
    profile,
    prominence,
    distance,
):
    """
    Detect peaks including possible peaks at z0 and z6.
    """

    profile = np.asarray(
        profile,
        dtype=np.float64,
    )


    pad_value = float(
        profile.min()
    )


    padded = np.concatenate(
        [
            [pad_value],
            profile,
            [pad_value],
        ]
    )


    padded_peaks, properties = find_peaks(
        padded,
        prominence=prominence,
        distance=distance,
    )


    peaks = (
        padded_peaks
        - 1
    )


    valid = (
        (peaks >= 0)
        & (peaks < profile.size)
    )


    peaks = (
        peaks[
            valid
        ]
        .astype(int)
    )


    properties = {
        key:
            np.asarray(
                values
            )[valid]

        for (
            key,
            values,
        )
        in properties.items()
    }


    return (
        peaks,
        properties,
    )


def calculate_display_limits(
    image_crops,
):
    """
    Calculate one common DAPI display range across z0-z6.
    """

    finite_values = [

        crop[
            np.isfinite(
                crop
            )
        ].reshape(-1)

        for crop in image_crops
    ]


    finite_values = [

        values

        for values in finite_values

        if values.size > 0
    ]


    if not finite_values:

        return (
            0.0,
            1.0,
        )


    all_values = np.concatenate(
        finite_values
    )


    positive = (
        all_values[
            all_values > 0
        ]
    )


    display_values = (
        positive
        if positive.size > 0
        else all_values
    )


    (
        vmin,
        vmax,
    ) = np.percentile(
        display_values,
        [
            DISPLAY_LOW_PERCENTILE,
            DISPLAY_HIGH_PERCENTILE,
        ],
    )


    vmin = float(
        vmin
    )

    vmax = float(
        vmax
    )


    if vmax <= vmin:

        vmax = (
            vmin
            + 1.0
        )


    return (
        vmin,
        vmax,
    )


def first_unique_value(
    dataframe,
    column,
):
    """
    Retrieve one metadata value expected to be constant
    for all z rows of one stack.
    """

    if column not in dataframe.columns:

        raise ValueError(
            f"Required column '{column}' "
            f"is missing."
        )


    values = (
        dataframe[
            column
        ]
        .dropna()
        .unique()
    )


    if len(values) == 0:

        raise ValueError(
            f"No valid value found "
            f"for '{column}'."
        )


    if len(values) > 1:

        raise ValueError(
            f"Multiple values found for "
            f"'{column}' within one stack: "
            f"{values}"
        )


    return values[0]


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD SCORE AND INTENSITY TABLES
# ─────────────────────────────────────────────────────────────────────────────

print(
    "\n[1] Loading score and intensity tables ...",
    flush=True,
)


if not SCORES_CSV.exists():

    raise FileNotFoundError(
        f"Scores CSV not found:\n"
        f"{SCORES_CSV}"
    )


if not INTENSITY_CSV.exists():

    raise FileNotFoundError(
        f"Intensity CSV not found:\n"
        f"{INTENSITY_CSV}"
    )


scores_df = pd.read_csv(
    SCORES_CSV
)


intensity_df = pd.read_csv(
    INTENSITY_CSV
)


scores_df[
    "stack_id"
] = pd.to_numeric(
    scores_df[
        "stack_id"
    ],
    errors="raise",
).astype(
    np.int64
)


intensity_df[
    "stack_id"
] = pd.to_numeric(
    intensity_df[
        "stack_id"
    ],
    errors="raise",
).astype(
    np.int64
)


intensity_df[
    "z"
] = pd.to_numeric(
    intensity_df[
        "z"
    ],
    errors="raise",
).astype(
    int
)


# Required intensity-table columns
required_intensity_columns = {

    "stack_id",
    "z",

    INTENSITY_COLUMN,

    "maxproj_cell_id",

    "roi_y_min",
    "roi_y_max",

    "roi_x_min",
    "roi_x_max",
}


missing_intensity_columns = (
    required_intensity_columns
    - set(
        intensity_df.columns
    )
)


if missing_intensity_columns:

    raise ValueError(
        "Missing columns from intensity CSV:\n"
        f"{missing_intensity_columns}\n\n"
        "Available columns:\n"
        f"{intensity_df.columns.tolist()}"
    )


intensity_df[
    INTENSITY_COLUMN
] = pd.to_numeric(
    intensity_df[
        INTENSITY_COLUMN
    ],
    errors="coerce",
)


print(
    f"  Intensity column: "
    f"{INTENSITY_COLUMN}",
    flush=True,
)


print(
    f"  Score rows: "
    f"{len(scores_df)}",
    flush=True,
)


print(
    f"  Intensity rows: "
    f"{len(intensity_df)}",
    flush=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SELECT TOP VALID BIMODAL STACKS
# ─────────────────────────────────────────────────────────────────────────────

print(
    "\n[2] Selecting top valid_peak_pair stacks ...",
    flush=True,
)


required_score_columns = {

    "stack_id",

    "profile_status",

    "ordered_bimodality_score",

    "peak_1_z",

    "peak_2_z",

    "valley_z",
}


missing_score_columns = (
    required_score_columns
    - set(
        scores_df.columns
    )
)


if missing_score_columns:

    raise ValueError(
        f"Missing score columns: "
        f"{missing_score_columns}"
    )


valid_df = scores_df.loc[
    scores_df[
        "profile_status"
    ]
    == "valid_peak_pair"
].copy()


valid_df[
    "ordered_bimodality_score"
] = pd.to_numeric(
    valid_df[
        "ordered_bimodality_score"
    ],
    errors="coerce",
)


valid_df = (
    valid_df
    .dropna(
        subset=[
            "ordered_bimodality_score"
        ]
    )
    .sort_values(
        by=[
            "ordered_bimodality_score",
            "stack_id",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .head(
        TOP_N
    )
    .reset_index(
        drop=True
    )
)


if valid_df.empty:

    raise ValueError(
        "No profile_status == "
        "'valid_peak_pair' stacks found."
    )


selection_path = (
    OUTPUT_DIR
    / "top_valid_peak_pair_stacks.csv"
)


valid_df.to_csv(
    selection_path,
    index=False,
)


print(
    "  Selected stacks:",
    flush=True,
)


for (
    rank,
    row,
) in valid_df.iterrows():

    print(
        f"    rank {rank + 1}: "
        f"stack_id="
        f"{int(row['stack_id'])}, "
        f"score="
        f"{float(row['ordered_bimodality_score']):.4f}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. OPEN STITCHED MASKS
# ─────────────────────────────────────────────────────────────────────────────

print(
    "\n[3] Opening stitched masks ...",
    flush=True,
)


stitched_masks = []

expected_shape = None


for z in range(
    N_Z
):

    mask_path = (
        STITCHED_MASK_DIR
        / MASK_PATTERN.format(
            z=z
        )
    )


    if not mask_path.exists():

        raise FileNotFoundError(
            f"Mask not found:\n"
            f"{mask_path}"
        )


    # Memory-map instead of loading all 45k masks into RAM.
    mask = tifffile.memmap(
        str(
            mask_path
        )
    )


    mask = np.squeeze(
        mask
    )


    if mask.ndim != 2:

        raise ValueError(
            f"Expected 2D mask at z{z}, "
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
        f"  Opened z{z}: "
        f"{mask.shape}",
        flush=True,
    )


IMAGE_HEIGHT, IMAGE_WIDTH = (
    expected_shape
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. OPEN DAPI IMAGE
# ─────────────────────────────────────────────────────────────────────────────

print(
    "\n[4] Opening DAPI Zarr ...",
    flush=True,
)


zarr_root = zarr.open(
    str(
        ZARR_PATH
    ),
    mode="r",
)


if (
    DAPI_ARRAY_PATH
    not in zarr_root
):

    raise KeyError(
        f"DAPI array not found:\n"
        f"{DAPI_ARRAY_PATH}"
    )


dapi_array = (
    zarr_root[
        DAPI_ARRAY_PATH
    ]
)


print(
    f"  shape="
    f"{dapi_array.shape}",
    flush=True,
)


print(
    f"  chunks="
    f"{dapi_array.chunks}",
    flush=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 5. OPEN SAVED MAX-PROJECTION SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

print(
    "\n[5] Opening saved 45k "
    "max-projection segmentation ...",
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
        f"'{MAXPROJ_LABEL_LAYER}' "
        f"not found.\n"
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
        f"Maximum-projection segmentation "
        f"must be 2D.\n"
        f"shape={maxproj_da.shape}"
    )


if (
    int(
        maxproj_da.sizes["y"]
    ),
    int(
        maxproj_da.sizes["x"]
    ),
) != (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
):

    raise ValueError(
        "Max-projection mask and stitched "
        "masks have different shapes.\n"
        f"maxproj={maxproj_da.shape}\n"
        f"stitched="
        f"{(IMAGE_HEIGHT, IMAGE_WIDTH)}"
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


# ─────────────────────────────────────────────────────────────────────────────
# 6. CREATE PLOTS
# ─────────────────────────────────────────────────────────────────────────────

print(
    "\n[6] Creating plots ...",
    flush=True,
)


for (
    rank,
    score_row,
) in valid_df.iterrows():

    stack_id = int(
        score_row[
            "stack_id"
        ]
    )


    ordered_score = float(
        score_row[
            "ordered_bimodality_score"
        ]
    )


    peak_1_z = int(
        score_row[
            "peak_1_z"
        ]
    )


    peak_2_z = int(
        score_row[
            "peak_2_z"
        ]
    )


    valley_z = int(
        score_row[
            "valley_z"
        ]
    )


    print(
        f"\n  Rank {rank + 1}: "
        f"stack_id={stack_id}",
        flush=True,
    )


    # ─────────────────────────────────────────────────────────────────────────
    # 6A. Get this stack's PolyT profile and fixed max-projection ROI metadata
    # ─────────────────────────────────────────────────────────────────────────

    stack_profile = (
        intensity_df.loc[
            intensity_df[
                "stack_id"
            ]
            == stack_id
        ]
        .sort_values(
            "z"
        )
        .reset_index(
            drop=True
        )
    )


    if len(
        stack_profile
    ) != N_Z:

        raise ValueError(
            f"stack_id={stack_id} has "
            f"{len(stack_profile)} rows; "
            f"expected {N_Z}."
        )


    z_values = (
        stack_profile[
            "z"
        ]
        .to_numpy(
            dtype=int
        )
    )


    if not np.array_equal(
        z_values,
        np.arange(
            N_Z
        ),
    ):

        raise ValueError(
            f"stack_id={stack_id} does not "
            f"contain complete z0-z{N_Z - 1}."
        )


    intensities = (
        stack_profile[
            INTENSITY_COLUMN
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    if not np.all(
        np.isfinite(
            intensities
        )
    ):

        raise ValueError(
            f"stack_id={stack_id} contains "
            f"non-finite PolyT intensity."
        )


    maxproj_cell_id = int(
        first_unique_value(
            stack_profile,
            "maxproj_cell_id",
        )
    )


    mask_y_min = int(
        first_unique_value(
            stack_profile,
            "roi_y_min",
        )
    )


    mask_y_max = int(
        first_unique_value(
            stack_profile,
            "roi_y_max",
        )
    )


    mask_x_min = int(
        first_unique_value(
            stack_profile,
            "roi_x_min",
        )
    )


    mask_x_max = int(
        first_unique_value(
            stack_profile,
            "roi_x_max",
        )
    )


    # ── Display crop around the fixed max-projection cell ────────────────────

    crop_y_min = max(
        0,
        mask_y_min
        - DISPLAY_PADDING,
    )


    crop_y_max = min(
        IMAGE_HEIGHT,
        mask_y_max
        + DISPLAY_PADDING,
    )


    crop_x_min = max(
        0,
        mask_x_min
        - DISPLAY_PADDING,
    )


    crop_x_max = min(
        IMAGE_WIDTH,
        mask_x_max
        + DISPLAY_PADDING,
    )


    # Read only the local region from max-projection labels.
    maxproj_crop = (
        maxproj_da
        .isel(
            y=slice(
                crop_y_min,
                crop_y_max,
            ),
            x=slice(
                crop_x_min,
                crop_x_max,
            ),
        )
        .compute()
        .values
    )


    maxproj_crop = np.asarray(
        maxproj_crop
    ).squeeze()


    fixed_roi_mask = (
        maxproj_crop
        == maxproj_cell_id
    )


    if not fixed_roi_mask.any():

        raise RuntimeError(
            f"Max-projection cell "
            f"{maxproj_cell_id} was not found "
            f"inside ROI for stack {stack_id}."
        )


    fixed_boundary = find_boundaries(
        fixed_roi_mask,
        connectivity=1,
        mode="inner",
    )


    # ─────────────────────────────────────────────────────────────────────────
    # 6B. Load DAPI and stitched-segmentation crops at z0-z6
    # ─────────────────────────────────────────────────────────────────────────

    dapi_crops = []

    current_masks = []

    label_crops = []


    for z in range(
        N_Z
    ):

        dapi_crop = read_dapi_crop(
            array=dapi_array,
            z=z,
            y_start=crop_y_min,
            y_stop=crop_y_max,
            x_start=crop_x_min,
            x_stop=crop_x_max,
        )


        label_crop = np.asarray(
            stitched_masks[
                z
            ][
                crop_y_min:crop_y_max,
                crop_x_min:crop_x_max,
            ]
        )


        current_mask = (
            label_crop
            == stack_id
        )


        dapi_crops.append(
            dapi_crop
        )


        current_masks.append(
            current_mask
        )


        label_crops.append(
            label_crop
        )


    (
        common_vmin,
        common_vmax,
    ) = calculate_display_limits(
        dapi_crops
    )


    # ─────────────────────────────────────────────────────────────────────────
    # 6C. Reconstruct normalized/smoothed PolyT profile
    # ─────────────────────────────────────────────────────────────────────────

    normalized_profile = normalize_profile(
        intensities
    )


    if PEAK_SMOOTHING_SIGMA > 0:

        smoothed_profile = gaussian_filter1d(
            normalized_profile,
            sigma=PEAK_SMOOTHING_SIGMA,
            mode="nearest",
        )

    else:

        smoothed_profile = (
            normalized_profile.copy()
        )


    (
        detected_peak_indices,
        _,
    ) = detect_peaks_with_edges(
        profile=smoothed_profile,
        prominence=MIN_PEAK_PROMINENCE,
        distance=MIN_PEAK_DISTANCE,
    )


    # ─────────────────────────────────────────────────────────────────────────
    # 6D. DAPI + segmentation overlays
    # ─────────────────────────────────────────────────────────────────────────

    figure, axes = plt.subplots(
        2,
        4,
        figsize=(
            16,
            8,
        ),
        facecolor="white",
    )


    axes = (
        axes.ravel()
    )


    roi_y, roi_x = np.where(
        fixed_roi_mask
    )


    roi_center_y = float(
        np.mean(
            roi_y
        )
    )


    roi_center_x = float(
        np.mean(
            roi_x
        )
    )


    for z in range(
        N_Z
    ):

        axis = (
            axes[
                z
            ]
        )


        axis.imshow(
            dapi_crops[
                z
            ],
            cmap="gray",
            vmin=common_vmin,
            vmax=common_vmax,
            interpolation="nearest",
        )


        # ── Green: all other stitched objects ────────────────────────────────

        all_label_boundaries = find_boundaries(
            label_crops[
                z
            ],
            connectivity=1,
            mode="inner",
        )


        other_boundary = (
            all_label_boundaries
            & (
                label_crops[
                    z
                ]
                > 0
            )
            & (
                label_crops[
                    z
                ]
                != stack_id
            )
        )


        if other_boundary.any():

            other_overlay = np.zeros(
                (
                    other_boundary.shape[0],
                    other_boundary.shape[1],
                    4,
                ),
                dtype=np.float32,
            )


            other_overlay[
                other_boundary
            ] = [
                0.0,
                1.0,
                0.0,
                0.9,
            ]


            axis.imshow(
                other_overlay,
                interpolation="nearest",
            )


        # ── Red: fixed maximum-projection cell ───────────────────────────────

        fixed_overlay = np.zeros(
            (
                fixed_boundary.shape[0],
                fixed_boundary.shape[1],
                4,
            ),
            dtype=np.float32,
        )


        fixed_overlay[
            fixed_boundary
        ] = [
            1.0,
            0.0,
            0.0,
            1.0,
        ]


        axis.imshow(
            fixed_overlay,
            interpolation="nearest",
        )


        # ── Cyan: selected stitched stack at current z ───────────────────────

        current_boundary = find_boundaries(
            current_masks[
                z
            ],
            connectivity=1,
            mode="inner",
        )


        if current_boundary.any():

            current_overlay = np.zeros(
                (
                    current_boundary.shape[0],
                    current_boundary.shape[1],
                    4,
                ),
                dtype=np.float32,
            )


            current_overlay[
                current_boundary
            ] = [
                0.0,
                1.0,
                1.0,
                1.0,
            ]


            axis.imshow(
                current_overlay,
                interpolation="nearest",
            )


        # ── Stack ID label ───────────────────────────────────────────────────

        axis.text(
            roi_center_x,
            roi_center_y,
            str(
                stack_id
            ),
            ha="center",
            va="center",
            fontsize=9,
            color="yellow",
            bbox={
                "facecolor":
                    "black",

                "alpha":
                    0.65,

                "edgecolor":
                    "none",

                "pad":
                    2,
            },
        )


        # ── Peak/valley role ─────────────────────────────────────────────────

        if z == peak_1_z:

            role_text = (
                "PEAK 1"
            )

        elif z == peak_2_z:

            role_text = (
                "PEAK 2"
            )

        elif z == valley_z:

            role_text = (
                "VALLEY"
            )

        elif z in (
            z_values[
                detected_peak_indices
            ]
        ):

            role_text = (
                "candidate peak"
            )

        else:

            role_text = "-"


        current_area = int(
            current_masks[
                z
            ].sum()
        )


        overlap_pixels = int(
            np.count_nonzero(
                current_masks[
                    z
                ]
                & fixed_roi_mask
            )
        )


        axis.set_title(
            f"z{z} | PolyT mean="
            f"{intensities[z]:.3f}\n"
            f"{role_text} | "
            f"stitched area={current_area}\n"
            f"maxproj/stitched overlap="
            f"{overlap_pixels}",
            fontsize=9,
        )


        axis.axis(
            "off"
        )


    for index in range(
        N_Z,
        len(
            axes
        ),
    ):

        axes[
            index
        ].axis(
            "off"
        )


    figure.suptitle(
        f"Rank {rank + 1} | "
        f"stack_id={stack_id} | "
        f"ordered_bimodality_score="
        f"{ordered_score:.4f}\n"
        f"Red = fixed max-projection cell "
        f"(ID {maxproj_cell_id}); "
        f"cyan = stitched stack at current z; "
        f"green = other stitched cells\n"
        f"Background = DAPI | "
        f"Intensity measurement = PolyT",
        fontsize=13,
    )


    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.86,
        ]
    )


    dapi_output_path = (
        OUTPUT_DIR
        / (
            f"rank_{rank + 1}_"
            f"stack_{stack_id}_"
            f"dapi_zlayers.png"
        )
    )


    figure.savefig(
        dapi_output_path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )


    plt.close(
        figure
    )


    # ─────────────────────────────────────────────────────────────────────────
    # 6E. PolyT intensity profile
    # ─────────────────────────────────────────────────────────────────────────

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(
            14,
            5.5,
        ),
        facecolor="white",
    )


    # ── Raw PolyT profile ─────────────────────────────────────────────────────

    raw_axis = (
        axes[
            0
        ]
    )


    raw_axis.plot(
        z_values,
        intensities,
        marker="o",
        linewidth=1.8,
        label=(
            "Fixed max-projection-mask "
            "mean PolyT"
        ),
    )


    for (
        z,
        intensity,
    ) in zip(
        z_values,
        intensities,
    ):

        if z == peak_1_z:

            label = (
                f"Peak 1\n"
                f"z{z}\n"
                f"{intensity:.2f}"
            )

        elif z == peak_2_z:

            label = (
                f"Peak 2\n"
                f"z{z}\n"
                f"{intensity:.2f}"
            )

        elif z == valley_z:

            label = (
                f"Valley\n"
                f"z{z}\n"
                f"{intensity:.2f}"
            )

        else:

            label = (
                f"z{z}\n"
                f"{intensity:.2f}"
            )


        raw_axis.annotate(
            label,
            xy=(
                z,
                intensity,
            ),
            xytext=(
                0,
                10,
            ),
            textcoords=(
                "offset points"
            ),
            ha="center",
            fontsize=8,
        )


    raw_axis.scatter(
        [
            peak_1_z,
            peak_2_z,
        ],
        intensities[
            [
                peak_1_z,
                peak_2_z,
            ]
        ],
        s=120,
        marker="^",
        zorder=5,
        label="Selected peaks",
    )


    raw_axis.scatter(
        valley_z,
        intensities[
            valley_z
        ],
        s=100,
        marker="v",
        zorder=5,
        label="Selected valley",
    )


    raw_axis.set_xlabel(
        "z-layer"
    )


    raw_axis.set_ylabel(
        "Mean PolyT intensity inside "
        "fixed max-projection mask"
    )


    raw_axis.set_title(
        "Ordered raw PolyT intensity profile"
    )


    raw_axis.set_xticks(
        range(
            N_Z
        )
    )


    raw_axis.grid(
        alpha=0.25
    )


    raw_axis.legend(
        fontsize=8
    )


    # ── Normalized/smoothed profile ───────────────────────────────────────────

    normalized_axis = (
        axes[
            1
        ]
    )


    normalized_axis.plot(
        z_values,
        normalized_profile,
        marker="o",
        linewidth=1.2,
        linestyle="--",
        alpha=0.7,
        label="Normalized raw PolyT profile",
    )


    normalized_axis.plot(
        z_values,
        smoothed_profile,
        marker="o",
        linewidth=2,
        label=(
            "Smoothed profile "
            f"(sigma="
            f"{PEAK_SMOOTHING_SIGMA})"
        ),
    )


    if (
        detected_peak_indices.size
        > 0
    ):

        normalized_axis.scatter(
            z_values[
                detected_peak_indices
            ],
            smoothed_profile[
                detected_peak_indices
            ],
            s=80,
            marker="^",
            label="Detected candidates",
            zorder=5,
        )


    normalized_axis.scatter(
        [
            peak_1_z,
            peak_2_z,
        ],
        smoothed_profile[
            [
                peak_1_z,
                peak_2_z,
            ]
        ],
        s=160,
        marker="^",
        zorder=6,
        label="Selected peaks",
    )


    normalized_axis.scatter(
        valley_z,
        smoothed_profile[
            valley_z
        ],
        s=140,
        marker="v",
        zorder=6,
        label="Selected valley",
    )


    normalized_axis.set_xlabel(
        "z-layer"
    )


    normalized_axis.set_ylabel(
        "Normalized PolyT intensity"
    )


    normalized_axis.set_ylim(
        -0.08,
        1.15,
    )


    normalized_axis.set_xticks(
        range(
            N_Z
        )
    )


    normalized_axis.set_title(
        "Ordered PolyT bimodality profile"
    )


    normalized_axis.grid(
        alpha=0.25
    )


    normalized_axis.legend(
        fontsize=8
    )


    figure.suptitle(
        f"Rank {rank + 1} | "
        f"stack_id={stack_id} | "
        f"maxproj_cell_id={maxproj_cell_id} | "
        f"ordered_bimodality_score="
        f"{ordered_score:.4f}\n"
        f"peak 1=z{peak_1_z}, "
        f"valley=z{valley_z}, "
        f"peak 2=z{peak_2_z}",
        fontsize=13,
    )


    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.87,
        ]
    )


    intensity_output_path = (
        OUTPUT_DIR
        / (
            f"rank_{rank + 1}_"
            f"stack_{stack_id}_"
            f"polyt_intensity_profile.png"
        )
    )


    figure.savefig(
        intensity_output_path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )


    plt.close(
        figure
    )


    print(
        f"    Saved: "
        f"{dapi_output_path.name}",
        flush=True,
    )


    print(
        f"    Saved: "
        f"{intensity_output_path.name}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────────────

print(
    "\nDone!",
    flush=True,
)


print(
    f"  Selection table:\n"
    f"  {selection_path}",
    flush=True,
)


print(
    f"  Plot directory:\n"
    f"  {OUTPUT_DIR}",
    flush=True,
)