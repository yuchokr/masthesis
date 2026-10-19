#!/usr/bin/env python3

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

PROJECT_DIR = Path("/data/gent/vo/000/gvo00070/vsc48277/yujin_project")
STITCH_OUTPUT_DIR = PROJECT_DIR / "40k_subset_0_cellpose_fullcell_stitch3d_tiled_filtered"

VSI_TRANSCRIPT_CSV = (
    STITCH_OUTPUT_DIR
    / "vsi_vs_transcript_counts"
    / "stack_vsi_transcript_counts.csv"
)

BIMODALITY_CSV = (
    STITCH_OUTPUT_DIR
    / "ordered_bimodality_scores_dapi_maxproj_updated"
    / "stack_ordered_bimodality_scores_dapi_maxproj.csv"
)

GEOMETRY_CSV = (
    STITCH_OUTPUT_DIR
    / "stack_geometry_metrics"
    / "stack_geometry_metrics.csv"
)

OUT_DIR = STITCH_OUTPUT_DIR / "metric_correlation_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_VSI_PIXELS = 10

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

def require_columns(df, cols, name):
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} is missing columns: {sorted(missing)}\n"
            f"Available: {df.columns.tolist()}"
        )

def unique_stack_table(df, name):
    dup = df["stack_id"].duplicated(keep=False)
    if dup.any():
        raise ValueError(
            f"{name} has duplicated stack_id values. "
            f"Examples: {df.loc[dup, 'stack_id'].head(10).tolist()}"
        )
    return df

def pairwise_corr(df, columns, method="spearman"):
    """
    Safe pairwise correlation matrix.
    Fixes the previous diagonal bug where selecting df[[a, a]]
    created duplicate columns and scipy returned a matrix.
    """
    n = len(columns)

    corr = np.full((n, n), np.nan, dtype=float)
    pval = np.full((n, n), np.nan, dtype=float)
    count = np.zeros((n, n), dtype=int)

    for i, a in enumerate(columns):
        for j, b in enumerate(columns):

            # Diagonal
            if i == j:
                valid = (
                    pd.to_numeric(df[a], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )

                count[i, j] = len(valid)
                corr[i, j] = 1.0
                pval[i, j] = 0.0
                continue

            # Off-diagonal
            pair = pd.DataFrame({
                "x": pd.to_numeric(df[a], errors="coerce"),
                "y": pd.to_numeric(df[b], errors="coerce"),
            })

            pair = (
                pair
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            count[i, j] = len(pair)

            if len(pair) < 3:
                continue

            x = pair["x"].to_numpy(dtype=float)
            y = pair["y"].to_numpy(dtype=float)

            if np.all(x == x[0]) or np.all(y == y[0]):
                continue

            if method == "spearman":
                result = spearmanr(x, y)
                r = result.statistic
                p = result.pvalue

            elif method == "pearson":
                result = pearsonr(x, y)
                r = result.statistic
                p = result.pvalue

            else:
                raise ValueError(f"Unknown method: {method}")

            corr[i, j] = float(r)
            pval[i, j] = float(p)

    return (
        pd.DataFrame(corr, index=columns, columns=columns),
        pd.DataFrame(pval, index=columns, columns=columns),
        pd.DataFrame(count, index=columns, columns=columns),
    )

def plot_heatmap(
    matrix_df,
    display_names,
    title,
    colorbar_label,
    output_path,
    vmin=-1,
    vmax=1,
    fmt=".2f",
):
    arr = matrix_df.to_numpy(dtype=float)
    n = len(matrix_df)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    im = ax.imshow(
        arr,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)

    labels = [display_names.get(c, c) for c in matrix_df.columns]

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", linewidth=0.5, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(n):
        for j in range(n):
            value = arr[i, j]

            if np.isfinite(value):
                label = format(value, fmt)
                normalized = (value - vmin) / (vmax - vmin)
                txt_color = (
                    "white"
                    if (normalized < 0.25 or normalized > 0.75)
                    else "black"
                )
            else:
                label = "NA"
                txt_color = "black"

            ax.text(
                j, i, label,
                ha="center",
                va="center",
                fontsize=9,
                color=txt_color,
            )

    ax.set_title(title, pad=12)
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

def plot_count_heatmap(
    matrix_df,
    display_names,
    title,
    output_path,
):
    arr = matrix_df.to_numpy(dtype=float)
    n = len(matrix_df)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    im = ax.imshow(arr, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cell stacks")

    labels = [display_names.get(c, c) for c in matrix_df.columns]

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(n):
        for j in range(n):
            ax.text(
                j, i,
                f"{int(arr[i, j])}",
                ha="center",
                va="center",
                fontsize=8,
            )

    ax.set_title(title, pad=12)
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

def run_subset_analysis(
    subset_df,
    subset_name,
    file_prefix,
    raw_cols,
    raw_names,
    anomaly_cols,
    anomaly_names,
):
    print(f"\nSubset: {subset_name}")
    print(f"  n = {len(subset_df):,}")

    if len(subset_df) < 3:
        print("  Skipped: fewer than 3 stacks.")
        return

    # RAW
    spearman_raw_sub, spearman_raw_sub_p, raw_counts_sub = pairwise_corr(
        subset_df,
        raw_cols,
        method="spearman",
    )

    pearson_raw_sub, pearson_raw_sub_p, _ = pairwise_corr(
        subset_df,
        raw_cols,
        method="pearson",
    )

    spearman_raw_sub.to_csv(
        OUT_DIR / f"{file_prefix}_spearman_raw_metrics.csv"
    )
    spearman_raw_sub_p.to_csv(
        OUT_DIR / f"{file_prefix}_spearman_pvalues_raw_metrics.csv"
    )
    pearson_raw_sub.to_csv(
        OUT_DIR / f"{file_prefix}_pearson_raw_metrics.csv"
    )
    pearson_raw_sub_p.to_csv(
        OUT_DIR / f"{file_prefix}_pearson_pvalues_raw_metrics.csv"
    )
    raw_counts_sub.to_csv(
        OUT_DIR / f"{file_prefix}_pairwise_sample_counts.csv"
    )

    plot_heatmap(
        spearman_raw_sub,
        raw_names,
        (
            "Spearman correlation between cell-stack metrics\n"
            f"{subset_name} (n={len(subset_df):,})"
        ),
        "Spearman ρ",
        OUT_DIR / f"{file_prefix}_spearman_raw_metrics.png",
    )

    plot_heatmap(
        pearson_raw_sub,
        raw_names,
        (
            "Pearson correlation between cell-stack metrics\n"
            f"{subset_name} (n={len(subset_df):,})"
        ),
        "Pearson r",
        OUT_DIR / f"{file_prefix}_pearson_raw_metrics.png",
    )

    # ANOMALY-ORIENTED
    spearman_anom_sub, spearman_anom_sub_p, anom_counts_sub = pairwise_corr(
        subset_df,
        anomaly_cols,
        method="spearman",
    )

    spearman_anom_sub.to_csv(
        OUT_DIR / f"{file_prefix}_spearman_anomaly_oriented.csv"
    )
    spearman_anom_sub_p.to_csv(
        OUT_DIR / f"{file_prefix}_spearman_pvalues_anomaly_oriented.csv"
    )

    plot_heatmap(
        spearman_anom_sub,
        anomaly_names,
        (
            "Spearman correlation between anomaly-oriented metrics\n"
            f"{subset_name} (n={len(subset_df):,})"
        ),
        "Spearman ρ",
        OUT_DIR / f"{file_prefix}_spearman_anomaly_oriented.png",
    )

# -------------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------------

for p in [
    VSI_TRANSCRIPT_CSV,
    BIMODALITY_CSV,
    GEOMETRY_CSV,
]:
    if not p.exists():
        raise FileNotFoundError(p)

vsi_tx_df = pd.read_csv(VSI_TRANSCRIPT_CSV)
bim_df = pd.read_csv(BIMODALITY_CSV)
geometry_df = pd.read_csv(GEOMETRY_CSV)

require_columns(
    vsi_tx_df,
    ["stack_id", "mean_vsi", "n_vsi_pixels", "transcript_count"],
    "VSI/transcript table",
)

require_columns(
    bim_df,
    ["stack_id", "ordered_bimodality_score"],
    "Bimodality table",
)

require_columns(
    geometry_df,
    [
        "stack_id",
        "max_adjacent_area_change",
        "max_centroid_shift_um",
        "min_iou",
    ],
    "Geometry table",
)

for df in [
    vsi_tx_df,
    bim_df,
    geometry_df,
]:
    df["stack_id"] = pd.to_numeric(
        df["stack_id"],
        errors="raise",
    ).astype(np.int64)

for c in [
    "mean_vsi",
    "n_vsi_pixels",
    "transcript_count",
]:
    vsi_tx_df[c] = pd.to_numeric(
        vsi_tx_df[c],
        errors="coerce",
    )

bim_df["ordered_bimodality_score"] = pd.to_numeric(
    bim_df["ordered_bimodality_score"],
    errors="coerce",
)

for c in [
    "max_adjacent_area_change",
    "max_centroid_shift_um",
    "min_iou",
]:
    geometry_df[c] = pd.to_numeric(
        geometry_df[c],
        errors="coerce",
    )

vsi_tx_df = unique_stack_table(
    vsi_tx_df,
    "VSI/transcript table",
)

bim_df = unique_stack_table(
    bim_df,
    "Bimodality table",
)

geometry_df = unique_stack_table(
    geometry_df,
    "Geometry table",
)

# -------------------------------------------------------------------------
# QC + merge
# -------------------------------------------------------------------------

vsi_tx_df = vsi_tx_df.loc[
    vsi_tx_df["mean_vsi"].notna()
    & (vsi_tx_df["n_vsi_pixels"] >= MIN_VSI_PIXELS)
].copy()

merged_df = (
    vsi_tx_df[
        [
            "stack_id",
            "mean_vsi",
            "n_vsi_pixels",
            "transcript_count",
        ]
    ]
    .merge(
        bim_df[
            [
                "stack_id",
                "ordered_bimodality_score",
            ]
        ],
        on="stack_id",
        how="inner",
        validate="one_to_one",
    )
    .merge(
        geometry_df[
            [
                "stack_id",
                "max_adjacent_area_change",
                "max_centroid_shift_um",
                "min_iou",
            ]
        ],
        on="stack_id",
        how="inner",
        validate="one_to_one",
    )
)

merged_df["log1p_transcript_count"] = np.log1p(
    merged_df["transcript_count"].clip(lower=0)
)

merged_df["vsi_anomaly"] = (
    1.0
    - merged_df["mean_vsi"]
)

merged_df["iou_anomaly"] = (
    1.0
    - merged_df["min_iou"]
)

merged_df.to_csv(
    OUT_DIR / "merged_cell_stack_metrics.csv",
    index=False,
)

print(f"Merged stacks: {len(merged_df):,}")

# -------------------------------------------------------------------------
# Metric definitions
# -------------------------------------------------------------------------

raw_cols = [
    "mean_vsi",
    "ordered_bimodality_score",
    "max_adjacent_area_change",
    "max_centroid_shift_um",
    "min_iou",
    "log1p_transcript_count",
]

raw_names = {
    "mean_vsi": "Mean VSI",
    "ordered_bimodality_score": "PolyT bimodality",
    "max_adjacent_area_change": "Max area change",
    "max_centroid_shift_um": "Max COM shift",
    "min_iou": "Minimum IoU",
    "log1p_transcript_count": "log(1 + transcripts)",
}

anomaly_cols = [
    "vsi_anomaly",
    "ordered_bimodality_score",
    "max_adjacent_area_change",
    "max_centroid_shift_um",
    "iou_anomaly",
]

anomaly_names = {
    "vsi_anomaly": "1 − Mean VSI",
    "ordered_bimodality_score": "PolyT bimodality",
    "max_adjacent_area_change": "Max area change",
    "max_centroid_shift_um": "Max COM shift",
    "iou_anomaly": "1 − Minimum IoU",
}

# -------------------------------------------------------------------------
# Full dataset correlations
# -------------------------------------------------------------------------

spearman_raw, spearman_raw_p, raw_counts = pairwise_corr(
    merged_df,
    raw_cols,
    method="spearman",
)

pearson_raw, pearson_raw_p, _ = pairwise_corr(
    merged_df,
    raw_cols,
    method="pearson",
)

spearman_raw.to_csv(
    OUT_DIR / "spearman_raw_metrics.csv"
)
spearman_raw_p.to_csv(
    OUT_DIR / "spearman_pvalues_raw_metrics.csv"
)
pearson_raw.to_csv(
    OUT_DIR / "pearson_raw_metrics.csv"
)
pearson_raw_p.to_csv(
    OUT_DIR / "pearson_pvalues_raw_metrics.csv"
)
raw_counts.to_csv(
    OUT_DIR / "pairwise_sample_counts.csv"
)

plot_heatmap(
    spearman_raw,
    raw_names,
    "Spearman correlation between cell-stack metrics",
    "Spearman ρ",
    OUT_DIR / "01_spearman_raw_metrics.png",
)

plot_heatmap(
    pearson_raw,
    raw_names,
    "Pearson correlation between cell-stack metrics",
    "Pearson r",
    OUT_DIR / "02_pearson_raw_metrics.png",
)

plot_count_heatmap(
    raw_counts,
    raw_names,
    "Pairwise sample counts",
    OUT_DIR / "04_pairwise_sample_counts.png",
)

spearman_anom, spearman_anom_p, anom_counts = pairwise_corr(
    merged_df,
    anomaly_cols,
    method="spearman",
)

spearman_anom.to_csv(
    OUT_DIR / "spearman_anomaly_oriented.csv"
)
spearman_anom_p.to_csv(
    OUT_DIR / "spearman_pvalues_anomaly_oriented.csv"
)

plot_heatmap(
    spearman_anom,
    anomaly_names,
    (
        "Spearman correlation between anomaly-oriented metrics\n"
        "(higher values = more suspicious)"
    ),
    "Spearman ρ",
    OUT_DIR / "03_spearman_anomaly_oriented.png",
)

# -------------------------------------------------------------------------
# Subset 1: bimodality > 0
# -------------------------------------------------------------------------

bimodality_nonzero_df = merged_df.loc[
    merged_df["ordered_bimodality_score"] > 0
].copy()

run_subset_analysis(
    bimodality_nonzero_df,
    subset_name="Bimodality score > 0",
    file_prefix="05_bimodality_gt_0",
    raw_cols=raw_cols,
    raw_names=raw_names,
    anomaly_cols=anomaly_cols,
    anomaly_names=anomaly_names,
)

# -------------------------------------------------------------------------
# Subset 2: bimodality >= 0.15
# -------------------------------------------------------------------------

bimodality_strict_df = merged_df.loc[
    merged_df["ordered_bimodality_score"] >= 0.15
].copy()

run_subset_analysis(
    bimodality_strict_df,
    subset_name="Bimodality score ≥ 0.15",
    file_prefix="06_bimodality_ge_0_15",
    raw_cols=raw_cols,
    raw_names=raw_names,
    anomaly_cols=anomaly_cols,
    anomaly_names=anomaly_names,
)

print("\nDone.")
print(f"Saved to:\n{OUT_DIR}")
