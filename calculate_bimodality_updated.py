#!/usr/bin/env python3

"""
PolyT bimodality using a simple two-maxima + intervening-valley rule.

For each stack:
1. Use raw fixed-mask mean PolyT intensities at z0-z6.
2. Select the two highest-intensity z-layers.
3. If their z-distance is < MIN_PEAK_DISTANCE, score = 0.
4. Otherwise, find the minimum raw intensity strictly between them.
5. Score = (lower_peak - valley) / lower_peak.
6. is_ordered_bimodal=True when score >= MIN_RAW_VALLEY_DEPTH.

No normalization, smoothing, find_peaks, prominence, or geometric mean.
"""

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


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

INPUT_CSV = (
    STITCH_OUTPUT_DIR
    / "dapi_maxproj_mask_z_intensity_profiles"
    / "stack_z_intensity_maxproj_mask_dapi.csv"
)

OUTPUT_DIR = (
    STITCH_OUTPUT_DIR
    / "ordered_bimodality_scores_dapi_maxproj_updated"
)

OUTPUT_FILENAME = (
    "stack_ordered_bimodality_scores_dapi_maxproj.csv"
)

INTENSITY_COLUMN = "fixed_mask_mean_polyt_intensity"

N_Z = 7

# z1 and z4 -> distance 3 -> valley is evaluated.
MIN_PEAK_DISTANCE = 3

# Binary flag threshold. Continuous score is still the measured valley depth.
MIN_RAW_VALLEY_DEPTH = 0

REQUIRE_ALL_Z_LAYERS = True
PROGRESS_EVERY = 5000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# HELPERS
# =============================================================================

def empty_result(n_values, status):
    return {
        "profile_status": status,
        "n_profile_values": int(n_values),
        "n_selected_peaks": 0,
        "peak_1_z": -1,
        "peak_2_z": -1,
        "peak_1_intensity": np.nan,
        "peak_2_intensity": np.nan,
        "peak_distance": -1,
        "valley_z": -1,
        "valley_intensity": np.nan,
        "raw_valley_drop": np.nan,
        "raw_valley_depth": np.nan,
        "peak_balance": np.nan,
        "ordered_bimodality_score": 0.0,
        "is_ordered_bimodal": False,
    }


def first_value(df, column, default=np.nan):
    if column not in df.columns:
        return default

    values = df[column].dropna()

    if values.empty:
        return default

    return values.iloc[0]


def calculate_ordered_bimodality(z_values, intensities):
    """
    New two-maxima bimodality definition.

    ordered_bimodality_score =
        (lower of two maxima - minimum between maxima)
        / lower of two maxima

    Score is calculated only when the two selected maxima are at least
    MIN_PEAK_DISTANCE z-indices apart.
    """

    z_values = np.asarray(z_values, dtype=int)
    intensities = np.asarray(intensities, dtype=np.float64)

    # Remove non-finite intensities.
    finite = np.isfinite(intensities)
    z_values = z_values[finite]
    intensities = intensities[finite]

    if intensities.size == 0:
        return empty_result(0, "no_finite_intensity")

    # Require complete z0-z6 profile.
    if REQUIRE_ALL_Z_LAYERS:
        expected_z = np.arange(N_Z, dtype=int)

        if (
            intensities.size != N_Z
            or not np.array_equal(z_values, expected_z)
        ):
            return empty_result(
                intensities.size,
                "incomplete_z_profile",
            )

    # A perfectly constant profile contains no meaningful valley.
    if np.allclose(intensities, intensities[0]):
        return empty_result(
            intensities.size,
            "constant_profile",
        )

    # -------------------------------------------------------------------------
    # Select the two highest RAW intensity values.
    #
    # Tie handling is deterministic:
    #   1) higher intensity first
    #   2) if equal, lower z first
    # -------------------------------------------------------------------------

    ranked_indices = np.lexsort(
        (
            z_values,
            -intensities,
        )
    )

    selected = ranked_indices[:2]

    # Reorder selected maxima by z-position.
    selected = selected[
        np.argsort(z_values[selected])
    ]

    p1_idx = int(selected[0])
    p2_idx = int(selected[1])

    p1_z = int(z_values[p1_idx])
    p2_z = int(z_values[p2_idx])

    p1_intensity = float(intensities[p1_idx])
    p2_intensity = float(intensities[p2_idx])

    peak_distance = int(p2_z - p1_z)

    lower_peak = min(
        p1_intensity,
        p2_intensity,
    )

    higher_peak = max(
        p1_intensity,
        p2_intensity,
    )

    if higher_peak > 0:
        peak_balance = float(
            np.clip(
                lower_peak / higher_peak,
                0.0,
                1.0,
            )
        )
    else:
        peak_balance = 0.0

    # -------------------------------------------------------------------------
    # IMPORTANT:
    # Do not even evaluate a valley unless the two highest values are
    # sufficiently separated.
    # -------------------------------------------------------------------------

    if peak_distance < MIN_PEAK_DISTANCE:
        return {
            "profile_status": "top_two_peaks_too_close",
            "n_profile_values": int(intensities.size),
            "n_selected_peaks": 2,
            "peak_1_z": p1_z,
            "peak_2_z": p2_z,
            "peak_1_intensity": p1_intensity,
            "peak_2_intensity": p2_intensity,
            "peak_distance": peak_distance,
            "valley_z": -1,
            "valley_intensity": np.nan,
            "raw_valley_drop": np.nan,
            "raw_valley_depth": np.nan,
            "peak_balance": peak_balance,
            "ordered_bimodality_score": 0.0,
            "is_ordered_bimodal": False,
        }

    # Find z-layers strictly between the two maxima.
    between_indices = np.where(
        (z_values > p1_z)
        & (z_values < p2_z)
    )[0]

    if between_indices.size == 0:
        return empty_result(
            intensities.size,
            "no_intervening_z_layer",
        )

    # Minimum RAW intensity between selected maxima = valley.
    valley_idx = int(
        between_indices[
            np.argmin(
                intensities[between_indices]
            )
        ]
    )

    valley_z = int(z_values[valley_idx])
    valley_intensity = float(
        intensities[valley_idx]
    )

    raw_valley_drop = float(
        lower_peak - valley_intensity
    )

    if lower_peak > 0:
        raw_valley_depth = float(
            np.clip(
                raw_valley_drop / lower_peak,
                0.0,
                1.0,
            )
        )
    else:
        raw_valley_depth = 0.0

    # New score: directly interpretable fractional valley drop.
    score = raw_valley_depth

    is_ordered_bimodal = bool(
        peak_distance >= MIN_PEAK_DISTANCE
        and raw_valley_depth >= MIN_RAW_VALLEY_DEPTH
    )

    if is_ordered_bimodal:
        status = "valid_two_maxima_valley"
    else:
        status = "valley_below_threshold"

    return {
        "profile_status": status,
        "n_profile_values": int(intensities.size),
        "n_selected_peaks": 2,
        "peak_1_z": p1_z,
        "peak_2_z": p2_z,
        "peak_1_intensity": p1_intensity,
        "peak_2_intensity": p2_intensity,
        "peak_distance": peak_distance,
        "valley_z": valley_z,
        "valley_intensity": valley_intensity,
        "raw_valley_drop": raw_valley_drop,
        "raw_valley_depth": raw_valley_depth,
        "peak_balance": peak_balance,
        "ordered_bimodality_score": score,
        "is_ordered_bimodal": is_ordered_bimodal,
    }


# =============================================================================
# 1. LOAD INPUT
# =============================================================================

print("\n[1] Loading PolyT z-intensity CSV ...", flush=True)

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Input CSV not found:\n{INPUT_CSV}"
    )

intensity_df = pd.read_csv(INPUT_CSV)

required = {
    "stack_id",
    "z",
    INTENSITY_COLUMN,
}

missing = required - set(intensity_df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n\n"
        f"Available columns:\n"
        f"{intensity_df.columns.tolist()}"
    )

intensity_df["stack_id"] = pd.to_numeric(
    intensity_df["stack_id"],
    errors="raise",
).astype(np.int64)

intensity_df["z"] = pd.to_numeric(
    intensity_df["z"],
    errors="raise",
).astype(int)

intensity_df[INTENSITY_COLUMN] = pd.to_numeric(
    intensity_df[INTENSITY_COLUMN],
    errors="coerce",
)

if intensity_df.duplicated(
    ["stack_id", "z"]
).any():
    duplicated = intensity_df.loc[
        intensity_df.duplicated(
            ["stack_id", "z"],
            keep=False,
        ),
        ["stack_id", "z"],
    ]

    raise ValueError(
        "Duplicated stack_id x z rows were found.\n"
        f"{duplicated.head(20)}"
    )

n_stacks = int(
    intensity_df["stack_id"].nunique()
)

print(f"  Input: {INPUT_CSV}", flush=True)
print(f"  Rows: {len(intensity_df)}", flush=True)
print(f"  Stacks: {n_stacks}", flush=True)
print(f"  Intensity column: {INTENSITY_COLUMN}", flush=True)
print(
    "  Method: two highest raw intensities + intervening raw valley",
    flush=True,
)
print(
    f"  Minimum peak distance: {MIN_PEAK_DISTANCE}",
    flush=True,
)
print(
    f"  Binary valley-depth threshold: {MIN_RAW_VALLEY_DEPTH}",
    flush=True,
)


# =============================================================================
# 2. CALCULATE SCORE PER STACK
# =============================================================================

print(
    "\n[2] Calculating two-maxima bimodality scores ...",
    flush=True,
)

rows = []

for index, (stack_id, stack_df) in enumerate(
    intensity_df.groupby(
        "stack_id",
        sort=True,
    ),
    start=1,
):
    stack_df = stack_df.sort_values("z")

    result = calculate_ordered_bimodality(
        stack_df["z"].to_numpy(dtype=int),
        stack_df[INTENSITY_COLUMN].to_numpy(
            dtype=np.float64
        ),
    )

    if "stack_present_in_z" in stack_df.columns:
        fallback_n_z = int(
            stack_df["stack_present_in_z"]
            .fillna(False)
            .sum()
        )
    else:
        fallback_n_z = np.nan

    rows.append(
        {
            "stack_id": int(stack_id),
            "intensity_column": INTENSITY_COLUMN,

            "bimodality_method":
                "two_highest_raw_intensities_with_intervening_valley",

            "minimum_peak_distance_threshold":
                MIN_PEAK_DISTANCE,

            "minimum_raw_valley_depth_threshold":
                MIN_RAW_VALLEY_DEPTH,

            "n_z_layers":
                first_value(
                    stack_df,
                    "n_z_layers",
                    fallback_n_z,
                ),

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

            "roi_y_min":
                first_value(
                    stack_df,
                    "roi_y_min",
                ),

            "roi_y_max":
                first_value(
                    stack_df,
                    "roi_y_max",
                ),

            "roi_x_min":
                first_value(
                    stack_df,
                    "roi_x_min",
                ),

            "roi_x_max":
                first_value(
                    stack_df,
                    "roi_x_max",
                ),

            **result,
        }
    )

    if (
        index % PROGRESS_EVERY == 0
        or index == n_stacks
    ):
        print(
            f"  Processed {index}/{n_stacks} stacks",
            flush=True,
        )


# =============================================================================
# 3. BUILD RESULT TABLE
# =============================================================================

scores_df = pd.DataFrame(rows)

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
    .reset_index(drop=True)
)


# =============================================================================
# 4. SAVE
# =============================================================================

output_path = (
    OUTPUT_DIR
    / OUTPUT_FILENAME
)

scores_df.to_csv(
    output_path,
    index=False,
)


# =============================================================================
# 5. SUMMARY
# =============================================================================

score_positive_count = int(
    (
        scores_df[
            "ordered_bimodality_score"
        ]
        > 0
    ).sum()
)

ordered_bimodal_count = int(
    scores_df[
        "is_ordered_bimodal"
    ].sum()
)

distance_valid_count = int(
    (
        scores_df["peak_distance"]
        >= MIN_PEAK_DISTANCE
    ).sum()
)

print("\nDone!", flush=True)

print(
    f"  Total stacks: {len(scores_df)}",
    flush=True,
)

print(
    f"  Top-two maxima distance >= {MIN_PEAK_DISTANCE}: "
    f"{distance_valid_count}",
    flush=True,
)

print(
    f"  Score > 0: {score_positive_count}",
    flush=True,
)

print(
    f"  is_ordered_bimodal=True "
    f"(score >= {MIN_RAW_VALLEY_DEPTH}): "
    f"{ordered_bimodal_count}",
    flush=True,
)

print(
    "  Profile status counts:",
    flush=True,
)

for status, count in (
    scores_df["profile_status"]
    .value_counts(dropna=False)
    .items()
):
    print(
        f"    {status}: {count}",
        flush=True,
    )

print(
    f"\n  Saved:\n  {output_path}",
    flush=True,
)
