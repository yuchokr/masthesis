#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# 0. 설정
# ─────────────────────────────────────────────
DOUBLETS_CSV = "/data/gent/vo/000/gvo00070/vsc48277/yujin_project/40k_subset_0_result/doublets_um.csv"
MASK_DIR = "/data/gent/vo/000/gvo00070/vsc48277/yujin_project/40k_subset_0_2d_segmentation_result/masks_each_z"

OUT_CSV = "/data/gent/vo/000/gvo00070/vsc48277/yujin_project/40k_subset_0_result/doublet_overlapping_cells_by_z_transformed.csv"
DEBUG_OVERLAY_PATH = "/data/gent/vo/000/gvo00070/vsc48277/yujin_project/40k_subset_0_result/debug_overlay_mask_z3_transformed.png"

# doublet csv 컬럼명
X_COL = "x"
Y_COL = "y"

# micron -> pixel affine
A_X = 9.259401321411132812
C_X = 385.3059082
A_Y = 9.25933266
C_Y = 999.81787109

# full vendor image size used in your previous transform
FULL_H = 45056
FULL_W = 45056

# subset origin
ROI_X0 = 0
ROI_Y0 = 0

SEARCH_RADIUS = 20

# debug plot
DEBUG_Z = 3
N_DEBUG = 2000
DEBUG_DOWNSAMPLE = 10


# ─────────────────────────────────────────────
# 1. 함수
# ─────────────────────────────────────────────
def transform_doublet_coords(df, H, W):
    out = df.copy()

    # raw affine
    out["x_px_raw"] = out[X_COL] * A_X + C_X
    out["y_px_raw"] = out[Y_COL] * A_Y + C_Y

    # 1) y flip
    out["y_px_flip"] = (H - 1) - out["y_px_raw"]

    # 2) y = -x reflection
    out["x_tmp"] = (W - 1) - out["y_px_flip"]
    out["y_tmp"] = (H - 1) - out["x_px_raw"]

    # 3) 90° clockwise rotation
    out["x_px"] = out["y_tmp"]
    out["y_px"] = (W - 1) - out["x_tmp"]

    # 4) additional 180° rotation
    out["x_px"] = (W - 1) - out["x_px"]
    out["y_px"] = (H - 1) - out["y_px"]

    # 5) final shift
    out["x_px"] = out["x_px"] - 100
    out["y_px"] = out["y_px"] - 1000

    # local coords for current subset mask
    out["x_local"] = out["x_px"] - ROI_X0
    out["y_local"] = out["y_px"] - ROI_Y0

    return out


def make_disk_offsets(radius):
    offsets = []
    r2 = radius * radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= r2:
                offsets.append((dy, dx))
    return offsets


def get_labels_near_xy(mask, x, y, offsets):
    H, W = mask.shape
    x = int(round(x))
    y = int(round(y))

    labels = set()
    for dy, dx in offsets:
        yy = y + dy
        xx = x + dx
        if 0 <= yy < H and 0 <= xx < W:
            lab = int(mask[yy, xx])
            if lab > 0:
                labels.add(lab)
    return sorted(labels)


def get_label_at_xy(mask, x, y):
    H, W = mask.shape
    x = int(round(x))
    y = int(round(y))

    if 0 <= yy < H and 0 <= xx < W:
        return int(mask[y, x])
    return 0


# ─────────────────────────────────────────────
# 2. doublet 로드 + 변환
# ─────────────────────────────────────────────
print("[INFO] loading doublets...")
doublets = pd.read_csv(DOUBLETS_CSV)
print("[INFO] n_doublets:", len(doublets))
print("[INFO] columns:", list(doublets.columns))

if X_COL not in doublets.columns or Y_COL not in doublets.columns:
    raise ValueError(f"Missing required columns: {X_COL}, {Y_COL}")

doublets = transform_doublet_coords(doublets, FULL_H, FULL_W)

print("\n[INFO] transformed coordinate ranges")
print("x_px    :", float(doublets["x_px"].min()), "to", float(doublets["x_px"].max()))
print("y_px    :", float(doublets["y_px"].min()), "to", float(doublets["y_px"].max()))
print("x_local :", float(doublets["x_local"].min()), "to", float(doublets["x_local"].max()))
print("y_local :", float(doublets["y_local"].min()), "to", float(doublets["y_local"].max()))


# ─────────────────────────────────────────────
# 3. mask 로드
# ─────────────────────────────────────────────
print("\n[INFO] loading masks...")
mask_files = sorted(Path(MASK_DIR).glob("mask_z*.tif"))
if len(mask_files) == 0:
    raise FileNotFoundError(f"No mask_z*.tif found in {MASK_DIR}")

mask_dict = {}
for f in mask_files:
    z = int(f.stem.replace("mask_z", ""))
    mask_dict[z] = tiff.imread(f)

available_z = sorted(mask_dict.keys())
print("[INFO] available z:", available_z)

# shape check
first_mask = mask_dict[available_z[0]]
H_mask, W_mask = first_mask.shape
print(f"[INFO] mask shape: {H_mask} x {W_mask}")

inside_mask = (
    (doublets["x_local"] >= 0) &
    (doublets["x_local"] < W_mask) &
    (doublets["y_local"] >= 0) &
    (doublets["y_local"] < H_mask)
)
print(f"[INFO] inside current mask: {inside_mask.sum():,} / {len(doublets):,} ({inside_mask.mean()*100:.2f}%)")

offsets = make_disk_offsets(SEARCH_RADIUS)


# ─────────────────────────────────────────────
# 4. z별 overlap 찾기
# ─────────────────────────────────────────────
print("\n[INFO] finding overlapping cells...")
rows = []

for i, r in doublets.iterrows():
    x_local = r["x_local"]
    y_local = r["y_local"]

    for z in available_z:
        mask = mask_dict[z]
        labels = get_labels_near_xy(mask, x_local, y_local, offsets)

        rows.append({
            "doublet_index": i,
            "x_input": r[X_COL],
            "y_input": r[Y_COL],
            "x_px": r["x_px"],
            "y_px": r["y_px"],
            "x_local": x_local,
            "y_local": y_local,
            "z": z,
            "n_cells": len(labels),
            "cell_ids": ";".join(map(str, labels)) if labels else ""
        })

result_df = pd.DataFrame(rows)

summary_rows = []
for did, sub in result_df.groupby("doublet_index"):
    hit_z = sub.loc[sub["n_cells"] > 0, "z"].tolist()
    summary_rows.append({
        "doublet_index": did,
        "n_z_with_hits": len(hit_z),
        "z_with_hits": ";".join(map(str, hit_z))
    })

summary_df = pd.DataFrame(summary_rows)
result_df = result_df.merge(summary_df, on="doublet_index", how="left")

result_df.to_csv(OUT_CSV, index=False)
print("[INFO] saved csv:", OUT_CSV)

print("\n[INFO] n_cells distribution")
print(result_df["n_cells"].value_counts().sort_index())

per_doublet_max = result_df.groupby("doublet_index")["n_cells"].max()
print("\n[INFO] per-doublet max n_cells")
print(per_doublet_max.value_counts().sort_index())


# ─────────────────────────────────────────────
# 5. debug overlay 저장
# ─────────────────────────────────────────────
print("\n[INFO] saving debug overlay...")

if DEBUG_Z not in mask_dict:
    raise ValueError(f"DEBUG_Z={DEBUG_Z} not found in masks: {available_z}")

mask_debug = mask_dict[DEBUG_Z]

# sample for plotting
plot_df = doublets.copy()
if len(plot_df) > N_DEBUG:
    plot_df = plot_df.sample(N_DEBUG, random_state=42)

# inside only
plot_inside = plot_df[
    (plot_df["x_local"] >= 0) &
    (plot_df["x_local"] < W_mask) &
    (plot_df["y_local"] >= 0) &
    (plot_df["y_local"] < H_mask)
].copy()

# downsample mask for faster plotting
ds = DEBUG_DOWNSAMPLE
mask_vis = mask_debug[::ds, ::ds]

plt.figure(figsize=(8, 8))
plt.imshow(mask_vis > 0, cmap="gray", alpha=0.5)
plt.scatter(
    plot_inside["x_local"] / ds,
    plot_inside["y_local"] / ds,
    s=3,
    c="red"
)
plt.title(f"Transformed doublets on mask_z{DEBUG_Z}")
plt.axis("off")
plt.tight_layout()
plt.savefig(DEBUG_OVERLAY_PATH, dpi=200, bbox_inches="tight")
plt.close()

print("[INFO] saved overlay:", DEBUG_OVERLAY_PATH)
print(f"[INFO] plotted inside points: {len(plot_inside):,}")

# 추가로 hit 점만 따로 overlay
hit_debug = result_df[
    (result_df["z"] == DEBUG_Z) &
    (result_df["n_cells"] > 0)
].copy()

HIT_OVERLAY_PATH = str(Path(DEBUG_OVERLAY_PATH).with_name(f"debug_overlay_mask_z{DEBUG_Z}_hits_only.png"))

if len(hit_debug) > 0:
    plt.figure(figsize=(8, 8))
    plt.imshow(mask_vis > 0, cmap="gray", alpha=0.5)
    plt.scatter(
        hit_debug["x_local"] / ds,
        hit_debug["y_local"] / ds,
        s=4,
        c="lime"
    )
    plt.title(f"Hit doublets on mask_z{DEBUG_Z} (n_cells > 0)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(HIT_OVERLAY_PATH, dpi=200, bbox_inches="tight")
    plt.close()
    print("[INFO] saved hits-only overlay:", HIT_OVERLAY_PATH)
    print(f"[INFO] hit points on z={DEBUG_Z}: {len(hit_debug):,}")
else:
    print(f"[INFO] no hit points found on z={DEBUG_Z}")


print("\n[INFO] done.")