#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import zarr
import anndata as ad


# ─────────────────────────────────────────────
# 0. 설정
# ─────────────────────────────────────────────
ZARR_PATH = "/data/gent/vo/000/gvo00070/Yujin_vizgen/vizgen_liver_SPArrOW_optimized.zarr"
TABLE_KEY = "table_SPArrOW_optimized_annotated"
LABEL_KEY = "segmentation_mask_optimized"
DOUBLETS_CSV = "/data/gent/vo/000/gvo00070/vsc48277/yujin_project/ovrlpy_run/doublets_um.csv"

ANNOT_COL = "annot_SPArrOW_marker_gene_lists"
CELL_ID_COL = "cell_ID"

# micron → pixel affine (중요!)
A_X = 9.259401321411132812
C_X = 385.3059082
A_Y = 9.25933266
C_Y = 999.81787109

DOUBLET_THRESH = 0.5
SEARCH_RADIUS = 30


# ─────────────────────────────────────────────
# 1. table 로드
# ─────────────────────────────────────────────
print("▶ Loading table...")
adata = ad.read_zarr(f"{ZARR_PATH}/tables/{TABLE_KEY}")

obs = adata.obs.copy()
obs["cell_type"] = obs[ANNOT_COL].astype(str)

obs[CELL_ID_COL] = pd.to_numeric(obs[CELL_ID_COL], errors="coerce")
obs = obs.dropna(subset=[CELL_ID_COL])
obs[CELL_ID_COL] = obs[CELL_ID_COL].astype(int)

print(f"   Cells: {len(obs):,}")


# ─────────────────────────────────────────────
# 2. label (global pixel)
# ─────────────────────────────────────────────
print("\n▶ Loading label image...")
label = zarr.open_array(f"{ZARR_PATH}/labels/{LABEL_KEY}/0", mode="r")

if label.ndim == 3:
    label = label[0]

label = np.asarray(label)

print("   Label shape:", label.shape)


# ─────────────────────────────────────────────
# 3. mapping 준비
# ─────────────────────────────────────────────
cell_id_to_type = dict(zip(obs[CELL_ID_COL], obs["cell_type"]))


# ─────────────────────────────────────────────
# 4. doublet 로드 + 좌표 변환 (🔥 핵심)
# ─────────────────────────────────────────────
print("\n▶ Loading doublets...")
doublets = pd.read_csv(DOUBLETS_CSV, usecols=["x", "y", "integrity"])

# 👉 global micron → global pixel
doublets["x_px"] = doublets["x"] * A_X + C_X
doublets["y_px"] = doublets["y"] * A_Y + C_Y

print("   Doublets:", len(doublets))

print("\n[Doublet coord range: global pixel]")
print("x:", float(doublets["x_px"].min()), "to", float(doublets["x_px"].max()))
print("y:", float(doublets["y_px"].min()), "to", float(doublets["y_px"].max()))


# ─────────────────────────────────────────────
# 5. mask 기반 매칭
# ─────────────────────────────────────────────
def find_cells(x, y):
    x = int(round(x))
    y = int(round(y))

    if x < 0 or y < 0 or y >= label.shape[0] or x >= label.shape[1]:
        return []

    patch = label[
        max(0, y-SEARCH_RADIUS):min(label.shape[0], y+SEARCH_RADIUS),
        max(0, x-SEARCH_RADIUS):min(label.shape[1], x+SEARCH_RADIUS)
    ]

    ids = np.unique(patch)
    ids = ids[ids != 0]

    return ids[:2]


pairs = []
cell_vsi = {cid: 1.0 for cid in obs[CELL_ID_COL]}

outside = 0
no_label = 0

for row in doublets.itertuples():

    x = row.x_px
    y = row.y_px
    integrity = row.integrity

    if x < 0 or y < 0 or x >= label.shape[1] or y >= label.shape[0]:
        outside += 1
        continue

    ids = find_cells(x, y)

    if len(ids) == 0:
        no_label += 1
        continue

    # cell-level VSI
    for cid in ids:
        if cid in cell_vsi:
            cell_vsi[cid] = min(cell_vsi[cid], integrity)

    if len(ids) < 2:
        continue

    ct1 = cell_id_to_type.get(ids[0], "unknown")
    ct2 = cell_id_to_type.get(ids[1], "unknown")

    if ct1 > ct2:
        ct1, ct2 = ct2, ct1

    pairs.append((ct1, ct2, integrity))


pairs = pd.DataFrame(pairs, columns=["type_1", "type_2", "integrity"])


# ─────────────────────────────────────────────
# 6. 결과 요약
# ─────────────────────────────────────────────
print("\n▶ Matching stats")
print("Outside:", outside)
print("No label:", no_label)
print("Pairs:", len(pairs))


# cell-level VSI
obs["vsi_score"] = obs[CELL_ID_COL].map(cell_vsi)
obs["is_doublet"] = obs["vsi_score"] < DOUBLET_THRESH


summary = obs.groupby("cell_type").agg(
    n_cells=("vsi_score", "count"),
    mean_vsi=("vsi_score", "mean"),
    doublet_rate=("is_doublet", "mean")
)

print("\n=== Cell summary ===")
print(summary)


pair_summary = pairs.groupby(["type_1", "type_2"]).size().sort_values(ascending=False)

print("\n=== Top pairs ===")
print(pair_summary.head(15))


print("\n✅ DONE")

import matplotlib.pyplot as plt

print("\n▶ Plotting overlay...")

# 일부 샘플링 (속도)
sample = doublets.sample(min(2000, len(doublets)), random_state=42)

# label downsample (속도/메모리)
DS = 10  # 10~20 추천
label_ds = label[::DS, ::DS]

plt.figure(figsize=(7,7))

plt.imshow(label_ds, cmap="gray", alpha=0.3)

plt.scatter(
    sample["x_px"] / DS,
    sample["y_px"] / DS,
    s=2,
    c="red"
)

plt.title("Doublets on segmentation mask (overlay)")
plt.axis("off")

# 저장
fig_path = "doublet_overlay.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")

plt.show()

print(f"   Saved: {fig_path}")