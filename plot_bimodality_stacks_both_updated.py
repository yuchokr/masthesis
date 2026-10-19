#!/usr/bin/env python3

"""
Plot top two-maxima bimodality cell stacks with DAPI and PolyT images
shown side-by-side for every z-layer.

For each selected stack:

    z0   DAPI | PolyT
    z1   DAPI | PolyT
    z2   DAPI | PolyT
    ...
    z6   DAPI | PolyT

Image overlays:
    Red boundary   = fixed maximum-projection segmentation mask
    Cyan boundary  = actual stitched segmentation of selected stack
    Green boundary = all other stitched segmentation masks

The intensity profile used for bimodality is read from the supplied
intensity CSV and detected automatically as either DAPI or PolyT.
"""


# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import zarr
import spatialdata as sd

from skimage.segmentation import find_boundaries


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


# -----------------------------------------------------------------------------
# Input tables
# -----------------------------------------------------------------------------

SCORES_CSV = (
    STITCH_OUTPUT_DIR
    / "ordered_bimodality_scores_dapi_maxproj_updated"
    / "stack_ordered_bimodality_scores_dapi_maxproj.csv"
)


INTENSITY_CSV = (
    STITCH_OUTPUT_DIR
    / "dapi_maxproj_mask_z_intensity_profiles"
    / "stack_z_intensity_maxproj_mask_dapi.csv"
)


# -----------------------------------------------------------------------------
# Stitched masks
# -----------------------------------------------------------------------------

STITCHED_MASK_DIR = (
    STITCH_OUTPUT_DIR
    / "stitched_masks"
)

MASK_PATTERN = "mask_z{z}.tif"


# -----------------------------------------------------------------------------
# SpatialData
# -----------------------------------------------------------------------------

ZARR_PATH = (
    PROJECT_DIR
    / "40k_subset_0.zarr"
)


IMAGE_ARRAY_PATH = (
    "images/clahe_DAPI_PolyT/0"
)

IMAGE_LAYOUT = "czyx"

DAPI_CHANNEL = 0
POLYT_CHANNEL = 1


# Already-cropped maximum-projection segmentation
MAXPROJ_LABEL_LAYER = (
    "segmentation_mask_optimized_maxproj_45k"
)


INTENSITY_COLUMN_CANDIDATES = [
    "fixed_mask_mean_dapi_intensity",
    "fixed_mask_mean_polyt_intensity",
]


# -----------------------------------------------------------------------------
# Plot settings
# -----------------------------------------------------------------------------

N_Z = 7

TOP_N = 20

DISPLAY_PADDING = 25

DISPLAY_LOW_PERCENTILE = 1
DISPLAY_HIGH_PERCENTILE = 99.8


# Same settings used by the new two-maxima bimodality scoring
MIN_PEAK_DISTANCE = 3
MIN_RAW_VALLEY_DEPTH = 0.15


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

OUTPUT_DIR = (
    STITCH_OUTPUT_DIR
    / "top_ordered_bimodality_plots_dapi_polyt_updated"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# HELPERS
# =============================================================================

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


# -----------------------------------------------------------------------------

def resolve_intensity_column(dataframe):
    """
    Identify the intensity column actually present in the CSV.
    """

    for column in INTENSITY_COLUMN_CANDIDATES:

        if column in dataframe.columns:

            return column

    raise ValueError(
        "Could not identify intensity column.\n"
        f"Tried:\n{INTENSITY_COLUMN_CANDIDATES}\n\n"
        f"Available columns:\n"
        f"{dataframe.columns.tolist()}"
    )


# -----------------------------------------------------------------------------

def get_intensity_name(column):
    """
    Human-readable channel name from intensity-column name.
    """

    column_lower = column.lower()

    if "polyt" in column_lower:
        return "PolyT"

    if "dapi" in column_lower:
        return "DAPI"

    return "Image"


# -----------------------------------------------------------------------------

def read_image_crop(
    array,
    channel,
    z,
    y_start,
    y_stop,
    x_start,
    x_stop,
):
    """
    Read one 2D crop from the DAPI/PolyT image array.
    """

    indexing = []

    for axis_name in IMAGE_LAYOUT:

        if axis_name == "c":

            indexing.append(
                channel
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
            f"Expected 2D image crop at z{z}, "
            f"got {crop.shape}"
        )


    return crop


# -----------------------------------------------------------------------------

def calculate_display_limits(
    image_crops,
):
    """
    Calculate one common display range across z0-z6
    for one imaging channel.
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


# -----------------------------------------------------------------------------

def first_unique_value(
    dataframe,
    column,
):
    """
    Retrieve one metadata value expected to be constant
    across all z rows of one stack.
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


# =============================================================================
# 1. LOAD SCORE AND INTENSITY TABLES
# =============================================================================

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


# Detect actual intensity column
INTENSITY_COLUMN = resolve_intensity_column(
    intensity_df
)

INTENSITY_NAME = get_intensity_name(
    INTENSITY_COLUMN
)


print(
    f"  Intensity column: "
    f"{INTENSITY_COLUMN}",
    flush=True,
)

print(
    f"  Bimodality signal: "
    f"{INTENSITY_NAME}",
    flush=True,
)


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
    f"  Score rows: "
    f"{len(scores_df)}",
    flush=True,
)


print(
    f"  Intensity rows: "
    f"{len(intensity_df)}",
    flush=True,
)


# =============================================================================
# 2. SELECT TOP VALID BIMODAL STACKS
# =============================================================================

print(
    "\n[2] Selecting top positive two-maxima bimodality-score stacks ...",
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


scores_df[
    "ordered_bimodality_score"
] = pd.to_numeric(
    scores_df[
        "ordered_bimodality_score"
    ],
    errors="coerce",
)

valid_df = (
    scores_df.loc[
        scores_df[
            "ordered_bimodality_score"
        ].fillna(0) > 0
    ]
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
    .head(TOP_N)
    .reset_index(drop=True)
)

if valid_df.empty:
    raise ValueError(
        "No stacks with ordered_bimodality_score > 0 were found."
    )

selection_path = (
    OUTPUT_DIR
    / "top_positive_two_maxima_bimodality_stacks.csv"
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


# =============================================================================
# 3. OPEN STITCHED MASKS
# =============================================================================

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


    # Memory map to avoid loading all 45k masks into RAM
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


# =============================================================================
# 4. OPEN DAPI + POLYT IMAGE ARRAY
# =============================================================================

print(
    "\n[4] Opening DAPI + PolyT Zarr ...",
    flush=True,
)


zarr_root = zarr.open(
    str(
        ZARR_PATH
    ),
    mode="r",
)


if (
    IMAGE_ARRAY_PATH
    not in zarr_root
):

    try:
        print(
            zarr_root.tree(),
            flush=True,
        )
    except Exception:
        pass

    raise KeyError(
        f"Image array not found:\n"
        f"{IMAGE_ARRAY_PATH}"
    )


image_array = (
    zarr_root[
        IMAGE_ARRAY_PATH
    ]
)


print(
    f"  path="
    f"{IMAGE_ARRAY_PATH}",
    flush=True,
)


print(
    f"  shape="
    f"{image_array.shape}",
    flush=True,
)


print(
    f"  chunks="
    f"{image_array.chunks}",
    flush=True,
)


if (
    len(IMAGE_LAYOUT)
    != image_array.ndim
):

    raise ValueError(
        f"IMAGE_LAYOUT={IMAGE_LAYOUT!r} "
        f"does not match image ndim="
        f"{image_array.ndim}"
    )


# =============================================================================
# 5. OPEN SAVED MAX-PROJECTION SEGMENTATION
# =============================================================================

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


# =============================================================================
# 6. CREATE PLOTS
# =============================================================================

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


    # =========================================================================
    # 6A. Intensity profile + fixed max-projection ROI metadata
    # =========================================================================

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
            f"non-finite intensity."
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


    # -------------------------------------------------------------------------
    # Display crop around fixed max-projection cell
    # -------------------------------------------------------------------------

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


    # -------------------------------------------------------------------------
    # Read only local region from max-projection segmentation
    # -------------------------------------------------------------------------

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


    # =========================================================================
    # 6B. Load DAPI + PolyT + stitched segmentation crops
    # =========================================================================

    dapi_crops = []

    polyt_crops = []

    current_masks = []

    label_crops = []


    for z in range(
        N_Z
    ):

        dapi_crop = read_image_crop(
            array=image_array,
            channel=DAPI_CHANNEL,
            z=z,
            y_start=crop_y_min,
            y_stop=crop_y_max,
            x_start=crop_x_min,
            x_stop=crop_x_max,
        )


        polyt_crop = read_image_crop(
            array=image_array,
            channel=POLYT_CHANNEL,
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


        polyt_crops.append(
            polyt_crop
        )


        current_masks.append(
            current_mask
        )


        label_crops.append(
            label_crop
        )


    # Independent display scaling for DAPI and PolyT
    (
        dapi_vmin,
        dapi_vmax,
    ) = calculate_display_limits(
        dapi_crops
    )


    (
        polyt_vmin,
        polyt_vmax,
    ) = calculate_display_limits(
        polyt_crops
    )


    # =========================================================================
    # 6D. DAPI + PolyT SIDE-BY-SIDE PLOT
    # =========================================================================

    figure, axes = plt.subplots(
        N_Z,
        2,
        figsize=(
            10,
            N_Z * 4,
        ),
        facecolor="white",
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

        # ---------------------------------------------------------------------
        # Boundaries
        # ---------------------------------------------------------------------

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


        current_boundary = find_boundaries(
            current_masks[
                z
            ],
            connectivity=1,
            mode="inner",
        )


        # ---------------------------------------------------------------------
        # Peak / valley role
        # ---------------------------------------------------------------------

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


        # ---------------------------------------------------------------------
        # DAPI + PolyT panels
        # ---------------------------------------------------------------------

        panel_settings = [

            (
                dapi_crops[z],
                "DAPI",
                dapi_vmin,
                dapi_vmax,
            ),

            (
                polyt_crops[z],
                "PolyT",
                polyt_vmin,
                polyt_vmax,
            ),
        ]


        for (
            column,
            (
                image_crop,
                channel_name,
                vmin,
                vmax,
            ),
        ) in enumerate(
            panel_settings
        ):

            axis = (
                axes[
                    z,
                    column
                ]
            )


            axis.imshow(
                image_crop,
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )


            # -------------------------------------------------------------
            # Green = other stitched objects
            # -------------------------------------------------------------

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


            # -------------------------------------------------------------
            # Red = fixed max-projection ROI
            # -------------------------------------------------------------

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


            # -------------------------------------------------------------
            # Cyan = selected stitched stack at current z
            # -------------------------------------------------------------

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


            # -------------------------------------------------------------
            # Stack ID
            # -------------------------------------------------------------

            axis.text(
                roi_center_x,
                roi_center_y,
                str(
                    stack_id
                ),
                ha="center",
                va="center",
                fontsize=8,
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


            # -------------------------------------------------------------
            # Title
            # -------------------------------------------------------------

            axis.set_title(
                f"z{z} | {channel_name}\n"
                f"{INTENSITY_NAME} mean="
                f"{intensities[z]:.3f} | "
                f"{role_text}\n"
                f"stitched area="
                f"{current_area} | "
                f"overlap="
                f"{overlap_pixels}",
                fontsize=9,
            )


            axis.axis(
                "off"
            )


    figure.suptitle(
        f"Rank {rank + 1} | "
        f"stack_id={stack_id} | "
        f"ordered_bimodality_score="
        f"{ordered_score:.4f}\n"
        f"Red = fixed max-projection cell "
        f"(ID {maxproj_cell_id}); "
        f"cyan = stitched stack; "
        f"green = other stitched cells\n"
        f"Left = DAPI | "
        f"Right = PolyT | "
        f"Bimodality signal = "
        f"{INTENSITY_NAME}",
        fontsize=13,
    )


    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ]
    )


    image_output_path = (
        OUTPUT_DIR
        / (
            f"rank_{rank + 1}_"
            f"stack_{stack_id}_"
            f"dapi_polyt_zlayers.png"
        )
    )


    figure.savefig(
        image_output_path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )


    plt.close(
        figure
    )


    # =========================================================================
    # 6E. INTENSITY PROFILE
    # =========================================================================

    figure, raw_axis = plt.subplots(
        1,
        1,
        figsize=(9, 5.5),
        facecolor="white",
    )

    raw_axis.plot(
        z_values,
        intensities,
        marker="o",
        linewidth=1.8,
        label=(
            f"Fixed max-projection-mask mean {INTENSITY_NAME}"
        ),
    )

    for z, intensity in zip(z_values, intensities):
        if z == peak_1_z:
            label = f"Peak 1\nz{z}\n{intensity:.2f}"
        elif z == peak_2_z:
            label = f"Peak 2\nz{z}\n{intensity:.2f}"
        elif z == valley_z:
            label = f"Valley\nz{z}\n{intensity:.2f}"
        else:
            label = f"z{z}\n{intensity:.2f}"

        raw_axis.annotate(
            label,
            xy=(z, intensity),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    raw_axis.scatter(
        [peak_1_z, peak_2_z],
        intensities[[peak_1_z, peak_2_z]],
        s=120,
        marker="^",
        zorder=5,
        label="Selected top-two maxima",
    )

    if valley_z >= 0:
        raw_axis.scatter(
            valley_z,
            intensities[valley_z],
            s=100,
            marker="v",
            zorder=5,
            label="Selected intervening valley",
        )

    raw_valley_depth = float(
        score_row.get("raw_valley_depth", ordered_score)
    )
    peak_distance = int(
        score_row.get("peak_distance", peak_2_z - peak_1_z)
    )
    is_ordered_bimodal = bool(
        score_row.get(
            "is_ordered_bimodal",
            raw_valley_depth >= MIN_RAW_VALLEY_DEPTH,
        )
    )

    raw_axis.set_xlabel("z-layer")
    raw_axis.set_ylabel(
        f"Mean {INTENSITY_NAME} intensity inside fixed max-projection mask"
    )
    raw_axis.set_title(
        f"Two-maxima raw {INTENSITY_NAME} profile\n"
        f"distance={peak_distance}, "
        f"valley depth={raw_valley_depth:.3f}, "
        f"is_ordered_bimodal={is_ordered_bimodal}"
    )
    raw_axis.set_xticks(range(N_Z))
    raw_axis.grid(alpha=0.25)
    raw_axis.legend(fontsize=8)

    figure.suptitle(
        f"Rank {rank + 1} | stack_id={stack_id} | "
        f"maxproj_cell_id={maxproj_cell_id} | "
        f"ordered_bimodality_score={ordered_score:.4f}\n"
        f"peak 1=z{peak_1_z}, valley=z{valley_z}, peak 2=z{peak_2_z}",
        fontsize=13,
    )

    figure.tight_layout(rect=[0, 0, 1, 0.90])


    intensity_output_path = (
        OUTPUT_DIR
        / (
            f"rank_{rank + 1}_"
            f"stack_{stack_id}_"
            f"{INTENSITY_NAME.lower()}_"
            f"intensity_profile.png"
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
        f"{image_output_path.name}",
        flush=True,
    )


    print(
        f"    Saved: "
        f"{intensity_output_path.name}",
        flush=True,
    )


# =============================================================================
# DONE
# =============================================================================

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
