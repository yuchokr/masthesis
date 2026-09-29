#!/usr/bin/env python3

"""
Calculate ordered z-axis bimodality scores from PolyT z-intensity profiles
measured using fixed maximum-projection segmentation masks.

Input:
    stack_z_intensity_maxproj_mask_polyt.csv

For each stack:
1. Read PolyT mean intensity at z0-z6.
2. Min-max normalize the 7-point profile.
3. Smooth with a 1D Gaussian filter.
4. Detect peaks including edge peaks.
5. Evaluate all valid peak pairs.
6. Calculate valley depth and peak balance.
7. Calculate ordered bimodality score.
8. Save one summary row per stack.
"""

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


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

INPUT_CSV = (
    STITCH_OUTPUT_DIR
    / "polyt_maxproj_mask_z_intensity_profiles"
    / "stack_z_intensity_maxproj_mask_polyt.csv"
)

OUTPUT_DIR = (
    STITCH_OUTPUT_DIR
    / "ordered_bimodality_scores_polyt_maxproj"
)

OUTPUT_FILENAME = (
    "stack_ordered_bimodality_scores_polyt_maxproj.csv"
)


# Intensity column in the new CSV
INTENSITY_COLUMN = (
    "fixed_mask_mean_polyt_intensity"
)


# z-profile settings
N_Z = 7

PEAK_SMOOTHING_SIGMA = 0.5

MIN_PEAK_PROMINENCE = 0.20

MIN_PEAK_DISTANCE = 2

MIN_VALLEY_DEPTH = 0.25

REQUIRE_ALL_Z_LAYERS = True

PROGRESS_EVERY = 5000


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORY
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def normalize_profile(values):
    """
    Min-max normalize intensity profile to [0, 1].
    """

    values = np.asarray(
        values,
        dtype=np.float64,
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


def detect_peaks_with_edges(profile):
    """
    Detect peaks including possible peaks at z0 and z6.

    scipy.signal.find_peaks does not normally detect edge points,
    so the profile is padded on both sides with the minimum value.
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
        prominence=MIN_PEAK_PROMINENCE,
        distance=MIN_PEAK_DISTANCE,
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


def empty_result(
    n_values,
    n_peaks=0,
    status="no_valid_peak_pair",
):
    """
    Return a standard result when a valid bimodal peak pair
    cannot be identified.
    """

    return {

        "profile_status":
            status,

        "n_profile_values":
            int(
                n_values
            ),

        "n_detected_peaks":
            int(
                n_peaks
            ),

        "peak_1_z":
            -1,

        "peak_2_z":
            -1,

        "peak_1_intensity":
            np.nan,

        "peak_2_intensity":
            np.nan,

        "peak_1_normalized_height":
            np.nan,

        "peak_2_normalized_height":
            np.nan,

        "peak_1_prominence":
            np.nan,

        "peak_2_prominence":
            np.nan,

        "minimum_peak_prominence":
            np.nan,

        "peak_distance":
            -1,

        "valley_z":
            -1,

        "valley_intensity":
            np.nan,

        "valley_normalized_height":
            np.nan,

        "valley_depth":
            np.nan,

        "peak_balance":
            np.nan,

        "ordered_bimodality_score":
            0.0,

        "is_ordered_bimodal":
            False,
    }


def calculate_ordered_bimodality(
    z_values,
    intensities,
):
    """
    Calculate ordered bimodality score for one stack.

    Score =
        geometric mean of:
            minimum peak prominence
            valley depth
            peak balance
    """

    z_values = np.asarray(
        z_values,
        dtype=int,
    )

    intensities = np.asarray(
        intensities,
        dtype=np.float64,
    )


    # Remove non-finite intensity values
    finite = np.isfinite(
        intensities
    )

    z_values = (
        z_values[
            finite
        ]
    )

    intensities = (
        intensities[
            finite
        ]
    )


    if intensities.size == 0:

        return empty_result(
            0,
            status="no_finite_intensity",
        )


    # Require exactly z0-z6
    if REQUIRE_ALL_Z_LAYERS:

        expected_z = np.arange(
            N_Z,
            dtype=int,
        )

        if (
            intensities.size
            != N_Z

            or not np.array_equal(
                z_values,
                expected_z,
            )
        ):

            return empty_result(
                intensities.size,
                status="incomplete_z_profile",
            )


    # Constant profile
    if np.allclose(
        intensities,
        intensities[0],
    ):

        return empty_result(
            intensities.size,
            status="constant_profile",
        )


    # ── Normalize ─────────────────────────────────────────────────────────────

    normalized = normalize_profile(
        intensities
    )


    # ── Smooth ────────────────────────────────────────────────────────────────

    if PEAK_SMOOTHING_SIGMA > 0:

        smoothed = gaussian_filter1d(
            normalized,
            sigma=PEAK_SMOOTHING_SIGMA,
            mode="nearest",
        )

    else:

        smoothed = (
            normalized.copy()
        )


    # ── Peak detection ────────────────────────────────────────────────────────

    (
        peak_indices,
        properties,
    ) = detect_peaks_with_edges(
        smoothed
    )


    prominences = np.asarray(
        properties.get(
            "prominences",
            [],
        ),
        dtype=np.float64,
    )


    if peak_indices.size < 2:

        return empty_result(
            intensities.size,
            peak_indices.size,
            status="fewer_than_two_peaks",
        )


    # ── Evaluate all peak pairs ────────────────────────────────────────────────

    best = None


    for i in range(
        peak_indices.size
    ):

        for j in range(
            i + 1,
            peak_indices.size,
        ):

            p1_idx = int(
                peak_indices[
                    i
                ]
            )

            p2_idx = int(
                peak_indices[
                    j
                ]
            )


            p1_z = int(
                z_values[
                    p1_idx
                ]
            )

            p2_z = int(
                z_values[
                    p2_idx
                ]
            )


            distance = (
                p2_z
                - p1_z
            )


            if (
                distance
                < MIN_PEAK_DISTANCE
            ):
                continue


            # Values between the two peaks
            between = (
                smoothed[
                    p1_idx + 1:
                    p2_idx
                ]
            )


            if between.size == 0:
                continue


            # Lowest point between peaks
            valley_idx = (
                p1_idx
                + 1
                + int(
                    np.argmin(
                        between
                    )
                )
            )


            valley_z = int(
                z_values[
                    valley_idx
                ]
            )


            # Heights in smoothed normalized profile
            h1 = float(
                smoothed[
                    p1_idx
                ]
            )

            h2 = float(
                smoothed[
                    p2_idx
                ]
            )

            hv = float(
                smoothed[
                    valley_idx
                ]
            )


            lower = min(
                h1,
                h2,
            )

            higher = max(
                h1,
                h2,
            )


            # ── Valley depth ──────────────────────────────────────────────────

            if lower > 0:

                valley_depth = float(
                    np.clip(
                        (
                            lower
                            - hv
                        )
                        / lower,
                        0.0,
                        1.0,
                    )
                )

            else:

                valley_depth = 0.0


            # ── Peak prominence ────────────────────────────────────────────────

            prom1 = float(
                prominences[
                    i
                ]
            )

            prom2 = float(
                prominences[
                    j
                ]
            )

            min_prom = min(
                prom1,
                prom2,
            )


            # ── Peak balance ──────────────────────────────────────────────────

            if higher > 0:

                peak_balance = float(
                    np.clip(
                        lower
                        / higher,
                        0.0,
                        1.0,
                    )
                )

            else:

                peak_balance = 0.0


            # ── Ordered bimodality score ──────────────────────────────────────

            score = float(
                (
                    min_prom
                    * valley_depth
                    * peak_balance
                )
                ** (
                    1.0
                    / 3.0
                )
            )


            # ── Result ─────────────────────────────────────────────────────────

            result = {

                "profile_status":
                    "valid_peak_pair",

                "n_profile_values":
                    int(
                        intensities.size
                    ),

                "n_detected_peaks":
                    int(
                        peak_indices.size
                    ),

                "peak_1_z":
                    p1_z,

                "peak_2_z":
                    p2_z,

                "peak_1_intensity":
                    float(
                        intensities[
                            p1_idx
                        ]
                    ),

                "peak_2_intensity":
                    float(
                        intensities[
                            p2_idx
                        ]
                    ),

                "peak_1_normalized_height":
                    h1,

                "peak_2_normalized_height":
                    h2,

                "peak_1_prominence":
                    prom1,

                "peak_2_prominence":
                    prom2,

                "minimum_peak_prominence":
                    min_prom,

                "peak_distance":
                    int(
                        distance
                    ),

                "valley_z":
                    valley_z,

                "valley_intensity":
                    float(
                        intensities[
                            valley_idx
                        ]
                    ),

                "valley_normalized_height":
                    hv,

                "valley_depth":
                    valley_depth,

                "peak_balance":
                    peak_balance,

                "ordered_bimodality_score":
                    score,

                "is_ordered_bimodal":
                    bool(
                        min_prom
                        >= MIN_PEAK_PROMINENCE

                        and distance
                        >= MIN_PEAK_DISTANCE

                        and valley_depth
                        >= MIN_VALLEY_DEPTH
                    ),
            }


            if (
                best is None

                or score
                > best[
                    "ordered_bimodality_score"
                ]
            ):

                best = result


    if best is None:

        return empty_result(
            intensities.size,
            peak_indices.size,
            status="no_valid_peak_pair",
        )


    if not best[
        "is_ordered_bimodal"
    ]:

        best[
            "profile_status"
        ] = (
            "peak_pair_below_threshold"
        )


    return best


def first_value(
    df,
    column,
    default=np.nan,
):
    """
    Return first non-NaN value from one metadata column.
    """

    if column not in df.columns:
        return default

    values = (
        df[
            column
        ]
        .dropna()
    )

    if values.empty:
        return default

    return values.iloc[0]


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD INPUT CSV
# ─────────────────────────────────────────────────────────────────────────────

print(
    "\n[1] Loading PolyT z-intensity CSV ...",
    flush=True,
)


if not INPUT_CSV.exists():

    raise FileNotFoundError(
        f"Input CSV not found:\n"
        f"{INPUT_CSV}"
    )


intensity_df = pd.read_csv(
    INPUT_CSV
)


required = {
    "stack_id",
    "z",
    INTENSITY_COLUMN,
}


missing = (
    required
    - set(
        intensity_df.columns
    )
)


if missing:

    raise ValueError(
        f"Missing required columns: "
        f"{missing}\n\n"
        f"Available columns:\n"
        f"{intensity_df.columns.tolist()}"
    )


# ── Data types ────────────────────────────────────────────────────────────────

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


intensity_df[
    INTENSITY_COLUMN
] = pd.to_numeric(
    intensity_df[
        INTENSITY_COLUMN
    ],
    errors="coerce",
)


# ── Check duplicates ──────────────────────────────────────────────────────────

if intensity_df.duplicated(
    [
        "stack_id",
        "z",
    ]
).any():

    duplicated = intensity_df.loc[
        intensity_df.duplicated(
            [
                "stack_id",
                "z",
            ],
            keep=False,
        ),
        [
            "stack_id",
            "z",
        ],
    ]

    raise ValueError(
        "Duplicated stack_id × z rows were found.\n"
        f"{duplicated.head(20)}"
    )


n_stacks = int(
    intensity_df[
        "stack_id"
    ].nunique()
)


print(
    f"  Input: "
    f"{INPUT_CSV}",
    flush=True,
)

print(
    f"  Rows: "
    f"{len(intensity_df)}",
    flush=True,
)

print(
    f"  Stacks: "
    f"{n_stacks}",
    flush=True,
)

print(
    f"  Intensity column: "
    f"{INTENSITY_COLUMN}",
    flush=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CALCULATE ORDERED BIMODALITY
# ─────────────────────────────────────────────────────────────────────────────

print(
    "\n[2] Calculating ordered bimodality scores ...",
    flush=True,
)


rows = []


for (
    index,
    (
        stack_id,
        stack_df,
    ),
) in enumerate(
    intensity_df.groupby(
        "stack_id",
        sort=True,
    ),
    start=1,
):

    stack_df = (
        stack_df
        .sort_values(
            "z"
        )
    )


    result = calculate_ordered_bimodality(
        stack_df[
            "z"
        ].to_numpy(
            dtype=int
        ),

        stack_df[
            INTENSITY_COLUMN
        ].to_numpy(
            dtype=np.float64
        ),
    )


    # Number of original stitched z layers
    if (
        "stack_present_in_z"
        in stack_df.columns
    ):

        fallback_n_z = int(
            stack_df[
                "stack_present_in_z"
            ]
            .fillna(False)
            .sum()
        )

    else:

        fallback_n_z = np.nan


    rows.append(
        {

            "stack_id":
                int(
                    stack_id
                ),

            "intensity_column":
                INTENSITY_COLUMN,

            # ── Bimodality parameters ────────────────────────────────────────

            "peak_smoothing_sigma":
                PEAK_SMOOTHING_SIGMA,

            "minimum_peak_prominence_threshold":
                MIN_PEAK_PROMINENCE,

            "minimum_peak_distance_threshold":
                MIN_PEAK_DISTANCE,

            "minimum_valley_depth_threshold":
                MIN_VALLEY_DEPTH,


            # ── Original stitched-stack metadata ─────────────────────────────

            "n_z_layers":
                first_value(
                    stack_df,
                    "n_z_layers",
                    fallback_n_z,
                ),


            # ── Max-projection matching metadata ─────────────────────────────

            "maxproj_cell_id":
                first_value(
                    stack_df,
                    "maxproj_cell_id",
                ),

            "maxproj_mask_area_pixels":
                first_value(
                    stack_df,
                    "maxproj_mask_area_pixels",
                ),

            "linked_union_area_pixels":
                first_value(
                    stack_df,
                    "linked_union_area_pixels",
                ),

            "maxproj_overlap_pixels":
                first_value(
                    stack_df,
                    "maxproj_overlap_pixels",
                ),

            "maxproj_overlap_fraction_of_linked_stack":
                first_value(
                    stack_df,
                    "maxproj_overlap_fraction_of_linked_stack",
                ),

            "maxproj_overlap_fraction_of_cell":
                first_value(
                    stack_df,
                    "maxproj_overlap_fraction_of_cell",
                ),

            "roi_type":
                first_value(
                    stack_df,
                    "roi_type",
                    "maximum_projection_segmentation_mask",
                ),


            # ── Bimodality result ─────────────────────────────────────────────

            **result,
        }
    )


    if (
        index
        % PROGRESS_EVERY
        == 0

        or index
        == n_stacks
    ):

        print(
            f"  Processed "
            f"{index}/"
            f"{n_stacks} stacks",
            flush=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. BUILD RESULT TABLE
# ─────────────────────────────────────────────────────────────────────────────

scores_df = pd.DataFrame(
    rows
)


scores_df = (
    scores_df
    .sort_values(
        [
            "ordered_bimodality_score",
            "stack_id",
        ],
        ascending=[
            False,
            True,
        ],
        na_position="last",
    )
    .reset_index(
        drop=True
    )
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SAVE
# ─────────────────────────────────────────────────────────────────────────────

output_path = (
    OUTPUT_DIR
    / OUTPUT_FILENAME
)


scores_df.to_csv(
    output_path,
    index=False,
)


# ─────────────────────────────────────────────────────────────────────────────
# 5. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

score_positive_count = int(
    (
        scores_df["ordered_bimodality_score"]
        > 0
    ).sum()
)

ordered_bimodal_count = int(
    scores_df[
        "is_ordered_bimodal"
    ].sum()
)


print(
    "\nDone!",
    flush=True,
)

print(
    f"  Total stacks: {len(scores_df)}",
    flush=True,
)

print(
    f"  Score > 0: {score_positive_count}",
    flush=True,
)

print(
    f"  is_ordered_bimodal=True: {ordered_bimodal_count}",
    flush=True,
)

print(
    "  Profile status counts:",
    flush=True,
)

for (
    status,
    count,
) in (
    scores_df[
        "profile_status"
    ]
    .value_counts(
        dropna=False
    )
    .items()
):

    print(
        f"    {status}: {count}",
        flush=True,
    )


print(
    f"\n  Saved:\n"
    f"  {output_path}",
    flush=True,
)