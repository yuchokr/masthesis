#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import anndata as ad


# ─────────────────────────────────────────────
# 0. 설정
# ─────────────────────────────────────────────
INPUT_CSV = "/data/gent/vo/000/gvo00070/vsc48277/yujin_project/40k_subset_0_result/doublet_overlapping_cells_by_z_transformed.csv"

ZARR_PATH = "/data/gent/vo/000/gvo00070/Yujin_vizgen/vizgen_liver_SPArrOW_optimized.zarr"
TABLE_KEY = "table_SPArrOW_optimized_annotated"

CELL_ID_COL = "cell_ID"
ANNOT_COL = "annot_SPArrOW_marker_gene_lists"

OUT_ROW_CSV = "/data/gent/vo/000/gvo00070/vsc48277/yujin_project/40k_subset_0_result/doublet_overlapping_cells_by_z_annotated.csv"
OUT_SUMMARY_CSV = "/data/gent/vo/000/gvo00070/vsc48277/yujin_project/40k_subset_0_result/doublet_hit_annotation_summary.csv"


# ─────────────────────────────────────────────
# 1. annotation table 로드
# ─────────────────────────────────────────────
print("[INFO] loading annotation table...")
adata = ad.read_zarr(f"{ZARR_PATH}/tables/{TABLE_KEY}")

obs = adata.obs.copy()
obs[CELL_ID_COL] = pd.to_numeric(obs[CELL_ID_COL], errors="coerce")
obs = obs.dropna(subset=[CELL_ID_COL]).copy()
obs[CELL_ID_COL] = obs[CELL_ID_COL].astype(int)

obs["cell_type"] = obs[ANNOT_COL].astype(str)

cell_id_to_type = dict(zip(obs[CELL_ID_COL], obs["cell_type"]))

print(f"[INFO] annotation cells: {len(cell_id_to_type):,}")


# ─────────────────────────────────────────────
# 2. overlap csv 로드
# ─────────────────────────────────────────────
print("[INFO] loading overlap csv...")
df = pd.read_csv(INPUT_CSV)

print(f"[INFO] rows: {len(df):,}")

# hit row만
hit_df = df[df["n_cells"] > 0].copy()
print(f"[INFO] hit rows: {len(hit_df):,}")

if len(hit_df) == 0:
    print("[INFO] no hit rows found. nothing to annotate.")
    raise SystemExit


# ─────────────────────────────────────────────
# 3. cell_ids -> cell_types 변환
# ─────────────────────────────────────────────
def parse_cell_ids(cell_ids_str):
    if pd.isna(cell_ids_str):
        return []
    s = str(cell_ids_str).strip()
    if s == "":
        return []
    out = []
    for x in s.split(";"):
        x = x.strip()
        if x == "":
            continue
        try:
            out.append(int(x))
        except ValueError:
            pass
    return out


def ids_to_types(ids):
    types = []
    for cid in ids:
        ct = cell_id_to_type.get(cid, "unknown")
        types.append(ct)
    return types


hit_df["cell_id_list"] = hit_df["cell_ids"].apply(parse_cell_ids)
hit_df["cell_type_list"] = hit_df["cell_id_list"].apply(ids_to_types)
hit_df["cell_types"] = hit_df["cell_type_list"].apply(lambda xs: ";".join(xs))

# pair type 정리
def make_pair_type(type_list):
    if len(type_list) == 0:
        return ""
    uniq = sorted(set(type_list))
    return " / ".join(uniq)

hit_df["pair_type"] = hit_df["cell_type_list"].apply(make_pair_type)

# unknown 포함 여부
hit_df["has_unknown"] = hit_df["cell_type_list"].apply(lambda xs: any(x == "unknown" for x in xs))


# ─────────────────────────────────────────────
# 4. row-level 결과 저장
# ─────────────────────────────────────────────
row_cols = [
    "doublet_index",
    "x_input", "y_input",
    "x_px", "y_px",
    "x_local", "y_local",
    "z",
    "n_cells",
    "cell_ids",
    "cell_types",
    "pair_type",
    "has_unknown",
    "n_z_with_hits",
    "z_with_hits",
]

hit_df[row_cols].to_csv(OUT_ROW_CSV, index=False)
print("[INFO] saved row-level annotation:", OUT_ROW_CSV)


# ─────────────────────────────────────────────
# 5. doublet별 summary 저장
# ─────────────────────────────────────────────
summary_rows = []

for did, sub in hit_df.groupby("doublet_index"):
    sub = sub.sort_values("z").copy()

    hit_z = sub["z"].tolist()
    pair_types_by_z = [f"z{int(z)}:{pt}" for z, pt in zip(sub["z"], sub["pair_type"])]
    cell_types_by_z = [f"z{int(z)}:{ct}" for z, ct in zip(sub["z"], sub["cell_types"])]

    all_types = []
    for xs in sub["cell_type_list"]:
        all_types.extend(xs)

    unique_types = sorted(set(all_types))

    summary_rows.append({
        "doublet_index": did,
        "x_input": sub["x_input"].iloc[0],
        "y_input": sub["y_input"].iloc[0],
        "n_hit_z": len(hit_z),
        "hit_z": ";".join(map(str, hit_z)),
        "unique_cell_types_across_z": ";".join(unique_types),
        "n_unique_cell_types_across_z": len(unique_types),
        "pair_types_by_z": " | ".join(pair_types_by_z),
        "cell_types_by_z": " | ".join(cell_types_by_z),
        "has_unknown_any_z": bool(sub["has_unknown"].any()),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values(
    ["n_hit_z", "n_unique_cell_types_across_z"],
    ascending=[False, False]
)

summary_df.to_csv(OUT_SUMMARY_CSV, index=False)
print("[INFO] saved doublet summary:", OUT_SUMMARY_CSV)

print("\n=== top annotated hit doublets ===")
print(summary_df.head(20).to_string(index=False))