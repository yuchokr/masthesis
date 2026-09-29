#!/usr/bin/env python3
"""Calculate ordered z-axis bimodality scores for every stack in stack_z_intensity.csv."""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path("/data/gent/vo/000/gvo00070/vsc48277/yujin_project")
STITCH_OUTPUT_DIR = (
    PROJECT_DIR
    / "40k_subset_0_cellpose_preprocessed"
    / "overlap_output_stitch3d"
)

INPUT_CSV = (
    STITCH_OUTPUT_DIR
    / "dapi_largest_mask_z_intensity_profiles"
    / "stack_z_intensity.csv"
)

OUTPUT_DIR = STITCH_OUTPUT_DIR / "ordered_bimodality_scores"
OUTPUT_FILENAME = "stack_ordered_bimodality_scores.csv"

INTENSITY_COLUMN = None
INTENSITY_COLUMN_CANDIDATES = [
    "fixed_mask_mean_dapi_intensity",
    "fixed_roi_mean_dapi_intensity",
    "mean_dapi_intensity",
    "bbox_mean_dapi_intensity",
]

N_Z = 7
PEAK_SMOOTHING_SIGMA = 0.5
MIN_PEAK_PROMINENCE = 0.20
MIN_PEAK_DISTANCE = 2
MIN_VALLEY_DEPTH = 0.25
REQUIRE_ALL_Z_LAYERS = True
PROGRESS_EVERY = 5000
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_intensity_column(df):
    if INTENSITY_COLUMN is not None:
        if INTENSITY_COLUMN not in df.columns:
            raise ValueError(
                f"Configured INTENSITY_COLUMN={INTENSITY_COLUMN!r} not found. "
                f"Available columns: {df.columns.tolist()}"
            )
        return INTENSITY_COLUMN

    for column in INTENSITY_COLUMN_CANDIDATES:
        if column in df.columns:
            return column

    raise ValueError(
        "Could not identify the intensity column. "
        f"Tried {INTENSITY_COLUMN_CANDIDATES}. "
        f"Available columns: {df.columns.tolist()}"
    )


def normalize_profile(values):
    values = np.asarray(values, dtype=np.float64)
    value_min = float(values.min())
    value_max = float(values.max())
    value_range = value_max - value_min
    if value_range <= 0:
        return np.zeros_like(values, dtype=np.float64)
    return (values - value_min) / value_range


def detect_peaks_with_edges(profile):
    profile = np.asarray(profile, dtype=np.float64)
    pad_value = float(profile.min())
    padded = np.concatenate([[pad_value], profile, [pad_value]])
    padded_peaks, properties = find_peaks(
        padded,
        prominence=MIN_PEAK_PROMINENCE,
        distance=MIN_PEAK_DISTANCE,
    )
    peaks = padded_peaks - 1
    valid = (peaks >= 0) & (peaks < profile.size)
    peaks = peaks[valid].astype(int)
    properties = {
        key: np.asarray(values)[valid]
        for key, values in properties.items()
    }
    return peaks, properties


def empty_result(n_values, n_peaks=0, status="no_valid_peak_pair"):
    return {
        "profile_status": status,
        "n_profile_values": int(n_values),
        "n_detected_peaks": int(n_peaks),
        "peak_1_z": -1,
        "peak_2_z": -1,
        "peak_1_intensity": np.nan,
        "peak_2_intensity": np.nan,
        "peak_1_normalized_height": np.nan,
        "peak_2_normalized_height": np.nan,
        "peak_1_prominence": np.nan,
        "peak_2_prominence": np.nan,
        "minimum_peak_prominence": np.nan,
        "peak_distance": -1,
        "valley_z": -1,
        "valley_intensity": np.nan,
        "valley_normalized_height": np.nan,
        "valley_depth": np.nan,
        "peak_balance": np.nan,
        "ordered_bimodality_score": 0.0,
        "is_ordered_bimodal": False,
    }


def calculate_ordered_bimodality(z_values, intensities):
    z_values = np.asarray(z_values, dtype=int)
    intensities = np.asarray(intensities, dtype=np.float64)

    finite = np.isfinite(intensities)
    z_values = z_values[finite]
    intensities = intensities[finite]

    if intensities.size == 0:
        return empty_result(0, status="no_finite_intensity")

    if REQUIRE_ALL_Z_LAYERS:
        expected_z = np.arange(N_Z, dtype=int)
        if intensities.size != N_Z or not np.array_equal(z_values, expected_z):
            return empty_result(
                intensities.size,
                status="incomplete_z_profile",
            )

    if np.allclose(intensities, intensities[0]):
        return empty_result(
            intensities.size,
            status="constant_profile",
        )

    normalized = normalize_profile(intensities)
    smoothed = (
        gaussian_filter1d(
            normalized,
            sigma=PEAK_SMOOTHING_SIGMA,
            mode="nearest",
        )
        if PEAK_SMOOTHING_SIGMA > 0
        else normalized.copy()
    )

    peak_indices, properties = detect_peaks_with_edges(smoothed)
    prominences = np.asarray(
        properties.get("prominences", []),
        dtype=np.float64,
    )

    if peak_indices.size < 2:
        return empty_result(
            intensities.size,
            peak_indices.size,
            status="fewer_than_two_peaks",
        )

    best = None

    for i in range(peak_indices.size):
        for j in range(i + 1, peak_indices.size):
            p1_idx = int(peak_indices[i])
            p2_idx = int(peak_indices[j])
            p1_z = int(z_values[p1_idx])
            p2_z = int(z_values[p2_idx])
            distance = p2_z - p1_z

            if distance < MIN_PEAK_DISTANCE:
                continue

            between = smoothed[p1_idx + 1 : p2_idx]
            if between.size == 0:
                continue

            valley_idx = p1_idx + 1 + int(np.argmin(between))
            valley_z = int(z_values[valley_idx])

            h1 = float(smoothed[p1_idx])
            h2 = float(smoothed[p2_idx])
            hv = float(smoothed[valley_idx])
            lower = min(h1, h2)
            higher = max(h1, h2)

            valley_depth = (
                float(np.clip((lower - hv) / lower, 0.0, 1.0))
                if lower > 0
                else 0.0
            )

            prom1 = float(prominences[i])
            prom2 = float(prominences[j])
            min_prom = min(prom1, prom2)
            peak_balance = (
                float(np.clip(lower / higher, 0.0, 1.0))
                if higher > 0
                else 0.0
            )

            score = float(
                (min_prom * valley_depth * peak_balance) ** (1.0 / 3.0)
            )

            result = {
                "profile_status": "valid_peak_pair",
                "n_profile_values": int(intensities.size),
                "n_detected_peaks": int(peak_indices.size),
                "peak_1_z": p1_z,
                "peak_2_z": p2_z,
                "peak_1_intensity": float(intensities[p1_idx]),
                "peak_2_intensity": float(intensities[p2_idx]),
                "peak_1_normalized_height": h1,
                "peak_2_normalized_height": h2,
                "peak_1_prominence": prom1,
                "peak_2_prominence": prom2,
                "minimum_peak_prominence": min_prom,
                "peak_distance": int(distance),
                "valley_z": valley_z,
                "valley_intensity": float(intensities[valley_idx]),
                "valley_normalized_height": hv,
                "valley_depth": valley_depth,
                "peak_balance": peak_balance,
                "ordered_bimodality_score": score,
                "is_ordered_bimodal": bool(
                    min_prom >= MIN_PEAK_PROMINENCE
                    and distance >= MIN_PEAK_DISTANCE
                    and valley_depth >= MIN_VALLEY_DEPTH
                ),
            }

            if best is None or score > best["ordered_bimodality_score"]:
                best = result

    if best is None:
        return empty_result(
            intensities.size,
            peak_indices.size,
            status="no_valid_peak_pair",
        )

    if not best["is_ordered_bimodal"]:
        best["profile_status"] = "peak_pair_below_threshold"

    return best


def first_value(df, column, default=np.nan):
    if column not in df.columns:
        return default
    values = df[column].dropna()
    return values.iloc[0] if not values.empty else default


print("\n[1] Loading stack_z_intensity.csv ...", flush=True)
if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

intensity_df = pd.read_csv(INPUT_CSV)
required = {"stack_id", "z"}
missing = required - set(intensity_df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

intensity_column = resolve_intensity_column(intensity_df)

intensity_df["stack_id"] = pd.to_numeric(
    intensity_df["stack_id"], errors="raise"
).astype(np.int64)
intensity_df["z"] = pd.to_numeric(
    intensity_df["z"], errors="raise"
).astype(int)
intensity_df[intensity_column] = pd.to_numeric(
    intensity_df[intensity_column], errors="coerce"
)

if intensity_df.duplicated(["stack_id", "z"]).any():
    raise ValueError("Duplicated stack_id × z rows were found.")

n_stacks = int(intensity_df["stack_id"].nunique())
print(f"  Rows: {len(intensity_df)}", flush=True)
print(f"  Stacks: {n_stacks}", flush=True)
print(f"  Intensity column: {intensity_column}", flush=True)

print("\n[2] Calculating ordered bimodality scores ...", flush=True)
rows = []

for index, (stack_id, stack_df) in enumerate(
    intensity_df.groupby("stack_id", sort=True),
    start=1,
):
    stack_df = stack_df.sort_values("z")
    result = calculate_ordered_bimodality(
        stack_df["z"].to_numpy(dtype=int),
        stack_df[intensity_column].to_numpy(dtype=np.float64),
    )

    if "stack_present_in_z" in stack_df.columns:
        fallback_n_z = int(stack_df["stack_present_in_z"].fillna(False).sum())
    else:
        fallback_n_z = np.nan

    rows.append(
        {
            "stack_id": int(stack_id),
            "intensity_column": intensity_column,
            "peak_smoothing_sigma": PEAK_SMOOTHING_SIGMA,
            "minimum_peak_prominence_threshold": MIN_PEAK_PROMINENCE,
            "minimum_peak_distance_threshold": MIN_PEAK_DISTANCE,
            "minimum_valley_depth_threshold": MIN_VALLEY_DEPTH,
            "n_z_layers": first_value(stack_df, "n_z_layers", fallback_n_z),
            "largest_mask_source_z": first_value(
                stack_df, "largest_mask_source_z"
            ),
            "largest_mask_area_pixels": first_value(
                stack_df, "largest_mask_area_pixels"
            ),
            "roi_type": first_value(
                stack_df, "roi_type", "largest_segmentation_mask"
            ),
            **result,
        }
    )

    if index % PROGRESS_EVERY == 0 or index == n_stacks:
        print(f"  Processed {index}/{n_stacks} stacks", flush=True)

scores_df = pd.DataFrame(rows)
scores_df = scores_df.sort_values(
    ["ordered_bimodality_score", "stack_id"],
    ascending=[False, True],
    na_position="last",
).reset_index(drop=True)

output_path = OUTPUT_DIR / OUTPUT_FILENAME
scores_df.to_csv(output_path, index=False)

print("\nDone!", flush=True)
print(f"  Total stacks: {len(scores_df)}", flush=True)
print(
    f"  Score > 0: {(scores_df['ordered_bimodality_score'] > 0).sum()}",
    flush=True,
)
print(
    f"  is_ordered_bimodal=True: {scores_df['is_ordered_bimodal'].sum()}",
    flush=True,
)
print("  Profile status counts:", flush=True)
for status, count in scores_df["profile_status"].value_counts(dropna=False).items():
    print(f"    {status}: {count}", flush=True)
print(f"  Saved: {output_path}", flush=True)