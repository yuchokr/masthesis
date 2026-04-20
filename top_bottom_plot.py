#!/usr/bin/env python3

import polars as pl
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 경로 설정
# --------------------------------------------------
INPUT = "/data/gent/vo/000/gvo00070/vsc48277/yujin_project/40k_subset_0_result/local_top_bottom/doublet_top_bottom_annotation_markerNormalized.csv"

OUT_BAR = "/data/gent/vo/000/gvo00070/vsc48277/yujin_project/40k_subset_0_result/local_top_bottom/combined_pair_top10.png"
OUT_HEATMAP = "/data/gent/vo/000/gvo00070/vsc48277/yujin_project/40k_subset_0_result/local_top_bottom/top_vs_bottom_heatmap.png"


# --------------------------------------------------
# 데이터 로드
# --------------------------------------------------
print("▶ Loading data...")
df = pl.read_csv(INPUT)


# --------------------------------------------------
# 1. Combined pair frequency (bar plot)
# --------------------------------------------------
print("▶ Plotting combined pair frequency...")

freq = (
    df
    .filter(pl.col("combined_pair").is_not_null())
    .group_by("combined_pair")
    .len()
    .rename({"len": "count"})
    .sort("count", descending=True)
    .head(10)
)

plt.figure(figsize=(8, 5))
plt.barh(freq["combined_pair"], freq["count"])
plt.gca().invert_yaxis()

plt.xlabel("Count")
plt.title("Top Combined Doublet Pairs")

plt.tight_layout()
plt.savefig(OUT_BAR, dpi=200)
plt.close()

print(f"   Saved: {OUT_BAR}")


# --------------------------------------------------
# 2. Top vs Bottom cell-type heatmap
# --------------------------------------------------
print("▶ Plotting top vs bottom heatmap...")

pdf = df.to_pandas()

pivot = pd.crosstab(
    pdf["top_top_type"],
    pdf["bottom_top_type"]
)

plt.figure(figsize=(7, 6))
plt.imshow(pivot, aspect="auto")

plt.colorbar(label="Count")

plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
plt.yticks(range(len(pivot.index)), pivot.index)

plt.xlabel("Bottom cell type")
plt.ylabel("Top cell type")
plt.title("Top vs Bottom Cell-Type Combinations")

plt.tight_layout()
plt.savefig(OUT_HEATMAP, dpi=200)
plt.close()

print(f"   Saved: {OUT_HEATMAP}")


# --------------------------------------------------
# DONE
# --------------------------------------------------
print("\n✅ All plots generated successfully!")