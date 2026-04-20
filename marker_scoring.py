#!/usr/bin/env python3

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import anndata as ad
import scanpy as sc
import polars as pl


# --------------------------------------------------
# 1. 파일 경로
# --------------------------------------------------
TOP_CSV = "/data/gent/vo/000/gvo00070/vsc48277/yujin_project/40k_subset_0_result/local_top_bottom/doublet_top_transcripts.csv"
BOTTOM_CSV = "/data/gent/vo/000/gvo00070/vsc48277/yujin_project/40k_subset_0_result/local_top_bottom/doublet_bottom_transcripts.csv"
OUT_CSV = "/data/gent/vo/000/gvo00070/vsc48277/yujin_project/40k_subset_0_result/local_top_bottom/doublet_top_bottom_annotation_markerNormalized.csv"

ZARR_PATH = "/data/gent/vo/000/gvo00070/Yujin_vizgen/vizgen_liver_SPArrOW_optimized.zarr"
TABLE_KEY = "table_SPArrOW_optimized_annotated"
ANNOT_COL = "annot_SPArrOW_marker_gene_lists"

N_MARKERS_PER_TYPE = 10
RANK_METHOD = "wilcoxon"


# --------------------------------------------------
# 2. marker dictionary 자동 생성
# --------------------------------------------------
def build_marker_dict_from_adata(
    zarr_path: str,
    table_key: str,
    annot_col: str,
    n_markers: int = 10,
    method: str = "wilcoxon",
) -> dict[str, list[str]]:
    print("▶ Loading annotated table for marker extraction...")
    adata = ad.read_zarr(f"{zarr_path}/tables/{table_key}")

    if annot_col not in adata.obs.columns:
        raise ValueError(
            f"Annotation column '{annot_col}' not found in adata.obs.\n"
            f"Available columns: {list(adata.obs.columns)}"
        )

    adata = adata.copy()
    adata.obs["cell_type"] = adata.obs[annot_col].astype(str)

    adata = adata[
        (~adata.obs["cell_type"].isna()) &
        (adata.obs["cell_type"] != "nan") &
        (adata.obs["cell_type"] != "unknown_celltype")
    ].copy()

    print(f"▶ Remaining cells after filtering: {adata.n_obs:,}")

    print("▶ Running rank_genes_groups...")
    sc.tl.rank_genes_groups(
        adata,
        groupby="cell_type",
        method=method,
        use_raw=False,
    )

    result = adata.uns["rank_genes_groups"]
    groups = list(result["names"].dtype.names)

    markers: dict[str, list[str]] = {}
    for group in groups:
        genes = [g for g in result["names"][group][:n_markers] if g is not None]
        markers[group] = genes

    print("\n=== Marker dictionary ===")
    for ct, genes in markers.items():
        print(f"{ct}: {genes}")

    return markers


# --------------------------------------------------
# 3. helper
# --------------------------------------------------
def summarize_one_side(
    df: pl.DataFrame,
    side: str,
    markers: dict[str, list[str]],
) -> pl.DataFrame:
    if df.shape[0] == 0:
        return pl.DataFrame(
            schema={
                "doublet_id": pl.Int64,
                f"{side}_n_tx": pl.Int64,
                f"{side}_top_type": pl.String,
                f"{side}_top_score": pl.Float64,
                f"{side}_second_type": pl.String,
                f"{side}_second_score": pl.Float64,
                f"{side}_confidence": pl.String,
                f"{side}_total_marker_counts": pl.Int64,
                f"{side}_top_pair": pl.String,
            }
        )

    total_counts = df.group_by("doublet_id").len().rename({"len": f"{side}_n_tx"})

    gene_counts = (
        df.group_by(["doublet_id", "gene"])
        .len()
        .rename({"len": "count"})
    )

    doublet_ids = (
        df.select("doublet_id")
        .unique()
        .sort("doublet_id")["doublet_id"]
        .to_list()
    )

    rows = []
    for did in doublet_ids:
        sub = gene_counts.filter(pl.col("doublet_id") == did)
        gene_to_count = dict(zip(sub["gene"].to_list(), sub["count"].to_list()))
        n_tx = int(total_counts.filter(pl.col("doublet_id") == did)[f"{side}_n_tx"][0])

        marker_counts_by_type = {}
        for cell_type, marker_genes in markers.items():
            marker_counts_by_type[cell_type] = sum(gene_to_count.get(g, 0) for g in marker_genes)

        total_marker_counts = sum(marker_counts_by_type.values())

        scores = {}
        for cell_type, marker_count in marker_counts_by_type.items():
            score = marker_count / total_marker_counts if total_marker_counts > 0 else 0.0
            scores[cell_type] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        top_type, top_score = ranked[0]
        if len(ranked) > 1:
            second_type, second_score = ranked[1]
        else:
            second_type, second_score = "NA", 0.0

        diff = top_score - second_score

        if n_tx < 5:
            confidence = "low_n"
        elif total_marker_counts == 0:
            confidence = "unknown"
        elif diff >= 0.25:
            confidence = "high"
        elif diff >= 0.10:
            confidence = "medium"
        else:
            confidence = "low"

        rows.append(
            {
                "doublet_id": did,
                f"{side}_n_tx": n_tx,
                f"{side}_total_marker_counts": int(total_marker_counts),
                f"{side}_top_type": top_type,
                f"{side}_top_score": round(float(top_score), 4),
                f"{side}_second_type": second_type,
                f"{side}_second_score": round(float(second_score), 4),
                f"{side}_confidence": confidence,
                f"{side}_top_pair": f"{top_type} + {second_type}",
            }
        )

    return pl.DataFrame(rows)


# --------------------------------------------------
# 4. main
# --------------------------------------------------
def main():
    markers = build_marker_dict_from_adata(
        zarr_path=ZARR_PATH,
        table_key=TABLE_KEY,
        annot_col=ANNOT_COL,
        n_markers=N_MARKERS_PER_TYPE,
        method=RANK_METHOD,
    )

    print("\n▶ Loading top/bottom transcript CSVs...")
    top_df = pl.read_csv(TOP_CSV)
    bottom_df = pl.read_csv(BOTTOM_CSV)

    required = {"doublet_id", "gene"}
    if not required.issubset(set(top_df.columns)):
        raise ValueError(f"Top CSV must contain {required}. Found: {top_df.columns}")
    if not required.issubset(set(bottom_df.columns)):
        raise ValueError(f"Bottom CSV must contain {required}. Found: {bottom_df.columns}")

    print("▶ Scoring top...")
    top_summary = summarize_one_side(top_df, "top", markers)

    print("▶ Scoring bottom...")
    bottom_summary = summarize_one_side(bottom_df, "bottom", markers)

    result = top_summary.join(bottom_summary, on="doublet_id", how="outer").sort("doublet_id")

    # 새 pair 정의
    result = result.with_columns(
        (
            pl.col("top_top_pair") + " || " + pl.col("bottom_top_pair")
        ).alias("combined_pair")
    )

    # 참고용: 기존 top1-bottom1 pair도 남김
    result = result.with_columns(
        pl.when(pl.col("top_top_type").is_not_null() & pl.col("bottom_top_type").is_not_null())
        .then(
            pl.when(pl.col("top_top_type") <= pl.col("bottom_top_type"))
            .then(pl.col("top_top_type") + " | " + pl.col("bottom_top_type"))
            .otherwise(pl.col("bottom_top_type") + " | " + pl.col("top_top_type"))
        )
        .otherwise(None)
        .alias("pair_type_top1_only")
    )

    result.write_csv(OUT_CSV)

    print(f"\nSaved: {OUT_CSV}")
    print("\n=== Preview ===")
    print(result.head(20))


if __name__ == "__main__":
    main()