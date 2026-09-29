#!/usr/bin/env python3
"""
Plot the top 5 ordered-bimodal cell stacks
===========================================

Selection:
    profile_status == "valid_peak_pair"
    sorted by ordered_bimodality_score descending
    top N stacks

For each selected stack, save:
    1. stack_<ID>_dapi_zlayers.png
    2. stack_<ID>_intensity_profile.png

DAPI plot overlays:
    Red boundary   = fixed largest segmentation mask used for intensity
    Cyan boundary  = actual segmentation of the selected stack at each z-layer
    Green boundary = all other segmentation masks visible in the crop
"""

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import zarr

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from skimage.segmentation import find_boundaries


# ── CONFIG ────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(
    "/data/gent/vo/000/gvo00070/vsc48277/yujin_project"
)

STITCH_OUTPUT_DIR = (
    PROJECT_DIR
    / "40k_subset_0_cellpose_fullcell_stitch3d_tiled"
)

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

STITCHED_MASK_DIR = (
    STITCH_OUTPUT_DIR
    / "stitched_masks"
)

MASK_PATTERN = "mask_z{z}.tif"

ZARR_PATH = PROJECT_DIR / "40k_subset_0.zarr"
DAPI_ARRAY_PATH = "images/clahe_DAPI_PolyT/0"
DAPI_LAYOUT = "czyx"
DAPI_CHANNEL = 0

N_Z = 7
TOP_N = 20

DISPLAY_PADDING = 25
DISPLAY_LOW_PERCENTILE = 1
DISPLAY_HIGH_PERCENTILE = 99.8

# Use the same settings as the scoring script.
PEAK_SMOOTHING_SIGMA = 0.5
MIN_PEAK_PROMINENCE = 0.20
MIN_PEAK_DISTANCE = 2

OUTPUT_DIR = (
    STITCH_OUTPUT_DIR
    / "top_ordered_bimodality_plots"
)
# ─────────────────────────────────────────────────────────────────────────────


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def read_dapi_crop(
    array,
    z,
    y_start,
    y_stop,
    x_start,
    x_stop,
):
    """Read one 2D DAPI crop while respecting DAPI_LAYOUT."""

    indexing = []

    for axis_name in DAPI_LAYOUT:
        if axis_name == "c":
            indexing.append(DAPI_CHANNEL)
        elif axis_name == "z":
            indexing.append(z)
        elif axis_name == "y":
            indexing.append(slice(y_start, y_stop))
        elif axis_name == "x":
            indexing.append(slice(x_start, x_stop))

    crop = np.squeeze(
        np.asarray(
            array[tuple(indexing)]
        )
    )

    if crop.ndim != 2:
        raise ValueError(
            f"Expected a 2D DAPI crop at z{z}, "
            f"got shape {crop.shape}"
        )

    return crop


def resolve_intensity_column(
    dataframe,
):
    candidates = [
        "fixed_mask_mean_dapi_intensity",
        "fixed_roi_mean_dapi_intensity",
        "mean_dapi_intensity",
        "bbox_mean_dapi_intensity",
    ]

    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    raise ValueError(
        "Could not identify the intensity column. "
        f"Available columns: {dataframe.columns.tolist()}"
    )


def normalize_profile(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "Intensity profile contains non-finite values."
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

    peaks = peaks[
        valid
    ].astype(int)

    properties = {
        key: np.asarray(values)[valid]
        for key, values
        in properties.items()
    }

    return peaks, properties


def calculate_display_limits(
    dapi_crops,
):
    finite_values = [
        crop[
            np.isfinite(crop)
        ].reshape(-1)
        for crop in dapi_crops
    ]

    finite_values = [
        values
        for values in finite_values
        if values.size > 0
    ]

    if not finite_values:
        return 0.0, 1.0

    all_values = np.concatenate(
        finite_values
    )

    positive = all_values[
        all_values > 0
    ]

    display_values = (
        positive
        if positive.size > 0
        else all_values
    )

    vmin, vmax = np.percentile(
        display_values,
        [
            DISPLAY_LOW_PERCENTILE,
            DISPLAY_HIGH_PERCENTILE,
        ],
    )

    vmin = float(vmin)
    vmax = float(vmax)

    if vmax <= vmin:
        vmax = vmin + 1.0

    return vmin, vmax


# ── 1. Load score and intensity tables ───────────────────────────────────────
print(
    "\n[1] Loading score and intensity tables ...",
    flush=True,
)

scores_df = pd.read_csv(
    SCORES_CSV
)

intensity_df = pd.read_csv(
    INTENSITY_CSV
)

scores_df["stack_id"] = (
    scores_df["stack_id"]
    .astype(np.int64)
)

intensity_df["stack_id"] = (
    intensity_df["stack_id"]
    .astype(np.int64)
)

intensity_df["z"] = (
    intensity_df["z"]
    .astype(int)
)

intensity_column = resolve_intensity_column(
    intensity_df
)

print(
    f"  Intensity column: {intensity_column}",
    flush=True,
)


# ── 2. Select top-scoring valid peak pairs ───────────────────────────────────
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
    - set(scores_df.columns)
)

if missing_score_columns:
    raise ValueError(
        f"Missing score columns: "
        f"{missing_score_columns}"
    )


valid_df = scores_df.loc[
    scores_df["profile_status"]
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
    .head(TOP_N)
    .reset_index(drop=True)
)


if valid_df.empty:
    raise ValueError(
        "No profile_status == 'valid_peak_pair' stacks were found."
    )


selected_stack_ids = (
    valid_df["stack_id"]
    .astype(np.int64)
    .to_numpy()
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

for rank, row in valid_df.iterrows():
    print(
        f"    rank {rank + 1}: "
        f"stack_id={int(row['stack_id'])}, "
        f"score="
        f"{float(row['ordered_bimodality_score']):.4f}",
        flush=True,
    )


# ── 3. Load stitched masks ───────────────────────────────────────────────────
print(
    "\n[3] Loading stitched masks ...",
    flush=True,
)

stitched_masks = []
expected_shape = None

for z in range(N_Z):
    mask_path = (
        STITCHED_MASK_DIR
        / MASK_PATTERN.format(z=z)
    )

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Mask not found: {mask_path}"
        )

    mask = np.squeeze(
        tifffile.imread(
            str(mask_path)
        )
    ).astype(
        np.int32,
        copy=False,
    )

    if mask.ndim != 2:
        raise ValueError(
            f"Expected a 2D mask at z{z}, "
            f"got shape {mask.shape}"
        )

    if expected_shape is None:
        expected_shape = mask.shape

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
        f"  Loaded z{z}",
        flush=True,
    )


IMAGE_HEIGHT, IMAGE_WIDTH = (
    expected_shape
)


# ── 4. Open DAPI Zarr ────────────────────────────────────────────────────────
print(
    "\n[4] Opening DAPI Zarr ...",
    flush=True,
)

zarr_root = zarr.open(
    str(ZARR_PATH),
    mode="r",
)

dapi_array = zarr_root[
    DAPI_ARRAY_PATH
]

print(
    f"  shape={dapi_array.shape}",
    flush=True,
)

print(
    f"  chunks={dapi_array.chunks}",
    flush=True,
)


# ── 5. Plot selected stacks ──────────────────────────────────────────────────
print(
    "\n[5] Creating plots ...",
    flush=True,
)

for rank, score_row in valid_df.iterrows():
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

    stack_profile = (
        intensity_df.loc[
            intensity_df["stack_id"]
            == stack_id
        ]
        .sort_values("z")
        .reset_index(drop=True)
    )

    if len(stack_profile) != N_Z:
        raise ValueError(
            f"stack_id={stack_id} has "
            f"{len(stack_profile)} z rows; "
            f"expected {N_Z}."
        )

    z_values = stack_profile[
        "z"
    ].to_numpy(
        dtype=int
    )

    intensities = stack_profile[
        intensity_column
    ].to_numpy(
        dtype=np.float64
    )

    if not np.array_equal(
        z_values,
        np.arange(N_Z),
    ):
        raise ValueError(
            f"stack_id={stack_id} does not contain "
            f"the complete ordered z0–z{N_Z - 1} profile."
        )

    if "largest_mask_source_z" in stack_profile.columns:
        largest_z = int(
            stack_profile[
                "largest_mask_source_z"
            ].dropna().iloc[0]
        )
    else:
        # Fallback: identify the largest segmentation directly.
        areas = [
            int(
                np.count_nonzero(
                    stitched_masks[z]
                    == stack_id
                )
            )
            for z in range(N_Z)
        ]

        largest_z = int(
            np.argmax(areas)
        )


    largest_mask_full = (
        stitched_masks[
            largest_z
        ]
        == stack_id
    )

    y_positions, x_positions = np.where(
        largest_mask_full
    )

    if y_positions.size == 0:
        raise RuntimeError(
            f"Could not locate stack_id={stack_id} "
            f"in largest-mask source z{largest_z}."
        )

    mask_y_min = int(
        y_positions.min()
    )

    mask_y_max = int(
        y_positions.max() + 1
    )

    mask_x_min = int(
        x_positions.min()
    )

    mask_x_max = int(
        x_positions.max() + 1
    )

    crop_y_min = max(
        0,
        mask_y_min - DISPLAY_PADDING,
    )

    crop_y_max = min(
        IMAGE_HEIGHT,
        mask_y_max + DISPLAY_PADDING,
    )

    crop_x_min = max(
        0,
        mask_x_min - DISPLAY_PADDING,
    )

    crop_x_max = min(
        IMAGE_WIDTH,
        mask_x_max + DISPLAY_PADDING,
    )


    fixed_roi_mask = largest_mask_full[
        crop_y_min:crop_y_max,
        crop_x_min:crop_x_max
    ]

    fixed_boundary = find_boundaries(
        fixed_roi_mask,
        connectivity=1,
        mode="inner",
    )


    dapi_crops = []
    current_masks = []
    label_crops = []

    for z in range(N_Z):
        dapi_crop = read_dapi_crop(
            array=dapi_array,
            z=z,
            y_start=crop_y_min,
            y_stop=crop_y_max,
            x_start=crop_x_min,
            x_stop=crop_x_max,
        )

        label_crop = stitched_masks[z][
            crop_y_min:crop_y_max,
            crop_x_min:crop_x_max
        ]

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


    common_vmin, common_vmax = (
        calculate_display_limits(
            dapi_crops
        )
    )


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

    detected_peak_indices, _ = (
        detect_peaks_with_edges(
            profile=smoothed_profile,
            prominence=MIN_PEAK_PROMINENCE,
            distance=MIN_PEAK_DISTANCE,
        )
    )


    # ── A. DAPI + segmentation overlays ──────────────────────────────────────
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(16, 8),
        facecolor="white",
    )

    axes = axes.ravel()

    roi_y, roi_x = np.where(
        fixed_roi_mask
    )

    roi_center_y = float(
        np.mean(roi_y)
    )

    roi_center_x = float(
        np.mean(roi_x)
    )

    for z in range(N_Z):
        axis = axes[z]

        axis.imshow(
            dapi_crops[z],
            cmap="gray",
            vmin=common_vmin,
            vmax=common_vmax,
            interpolation="nearest",
        )

        # Green: boundaries of all other segmented objects in this crop.
        all_label_boundaries = find_boundaries(
            label_crops[z],
            connectivity=1,
            mode="inner",
        )

        other_boundary = (
            all_label_boundaries
            & (label_crops[z] > 0)
            & (label_crops[z] != stack_id)
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


        # Red: fixed largest segmentation mask used for intensity.
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


        # Cyan: actual segmentation at the current z-layer.
        current_boundary = find_boundaries(
            current_masks[z],
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


        axis.text(
            roi_center_x,
            roi_center_y,
            str(stack_id),
            ha="center",
            va="center",
            fontsize=9,
            color="yellow",
            bbox={
                "facecolor": "black",
                "alpha": 0.65,
                "edgecolor": "none",
                "pad": 2,
            },
        )


        if z == peak_1_z:
            role_text = "PEAK 1"
        elif z == peak_2_z:
            role_text = "PEAK 2"
        elif z == valley_z:
            role_text = "VALLEY"
        elif z in z_values[
            detected_peak_indices
        ]:
            role_text = "candidate peak"
        else:
            role_text = "-"


        current_area = int(
            current_masks[z].sum()
        )

        overlap_pixels = int(
            np.count_nonzero(
                current_masks[z]
                & fixed_roi_mask
            )
        )

        axis.set_title(
            f"z{z} | mean="
            f"{intensities[z]:.3f}\n"
            f"{role_text} | "
            f"current area={current_area}\n"
            f"fixed/current overlap="
            f"{overlap_pixels}",
            fontsize=9,
        )

        axis.axis("off")


    for index in range(
        N_Z,
        len(axes),
    ):
        axes[index].axis(
            "off"
        )


    figure.suptitle(
        f"Rank {rank + 1} | "
        f"stack_id={stack_id} | "
        f"ordered_bimodality_score="
        f"{ordered_score:.4f}\n"
        f"Red = fixed largest segmentation ROI "
        f"from z{largest_z}; "
        f"cyan = selected stack at current z; "
        f"green = other segmentations",
        fontsize=14,
    )

    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.90,
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


    # ── B. Raw and normalized intensity profiles ─────────────────────────────
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.5),
        facecolor="white",
    )


    raw_axis = axes[0]

    raw_axis.plot(
        z_values,
        intensities,
        marker="o",
        linewidth=1.8,
        label="Fixed largest-mask mean DAPI",
    )

    for z, intensity in zip(
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
        "Mean DAPI intensity "
        "inside fixed largest mask"
    )

    raw_axis.set_title(
        "Ordered raw intensity profile"
    )

    raw_axis.set_xticks(
        range(N_Z)
    )

    raw_axis.grid(
        alpha=0.25
    )

    raw_axis.legend(
        fontsize=8
    )


    normalized_axis = axes[1]

    normalized_axis.plot(
        z_values,
        normalized_profile,
        marker="o",
        linewidth=1.2,
        linestyle="--",
        alpha=0.7,
        label="Normalized raw profile",
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


    if detected_peak_indices.size > 0:
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
        "Normalized intensity"
    )

    normalized_axis.set_ylim(
        -0.08,
        1.15,
    )

    normalized_axis.set_xticks(
        range(N_Z)
    )

    normalized_axis.set_title(
        "Ordered bimodality profile"
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
        f"{dapi_output_path.name}",
        flush=True,
    )

    print(
        f"    Saved: "
        f"{intensity_output_path.name}",
        flush=True,
    )


print(
    "\nDone!",
    flush=True,
)

print(
    f"  Selection table: "
    f"{selection_path}",
    flush=True,
)

print(
    f"  Plot directory: "
    f"{OUTPUT_DIR}",
    flush=True,
)