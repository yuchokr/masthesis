#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import zarr
import tifffile as tiff
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# 0. 설정
# ─────────────────────────────────────────────
ANNOTATED_CSV = "/data/gent/vo/000/gvo00070/vsc48277/yujin_project/40k_subset_0_result/doublet_overlapping_cells_by_z_annotated.csv"

ZARR_PATH = "/data/gent/vo/000/gvo00070/vsc48277/yujin_project/40k_subset_0.zarr"
IMAGE_KEY = "images/Yujin_vizgen_Liver1Slice1_z_global/0"   # shape: (C, Z, Y, X)
CHANNEL_IDX = 0   # 0=DAPI 추정, 필요시 1로 바꿔보기

MASK_DIR = "/data/gent/vo/000/gvo00070/vsc48277/yujin_project/40k_subset_0_2d_segmentation_result/masks_each_z"

OUT_DIR = "/data/gent/vo/000/gvo00070/vsc48277/yujin_project/40k_subset_0_result/doublet_patch_panels"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# 보고 싶은 후보들
# 예시: clean/interesting 후보 위주
TARGET_DOUBLETS = [16, 85, 170, 187, 440, 504, 506]

# crop 반경 (pixel)
PATCH_RADIUS = 256

# mask overlay 시 label 표시 여부
SHOW_ALL_MASK = False    # True면 crop 내 모든 mask 표시
SHOW_HIT_ONLY = True   # True면 row의 cell_ids만 강조

# 저장 dpi
SAVE_DPI = 200


# ─────────────────────────────────────────────
# 1. 함수
# ─────────────────────────────────────────────
def normalize_img(img):
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, [1, 99])
    if p99 <= p1:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - p1) / (p99 - p1), 0, 1)


def parse_cell_ids(cell_ids_str):
    if pd.isna(cell_ids_str):
        return []
    s = str(cell_ids_str).strip()
    if s == "":
        return []
    out = []
    for x in s.split(";"):
        x = x.strip()
        if x:
            try:
                out.append(int(x))
            except ValueError:
                pass
    return out


def crop_box(x, y, H, W, radius):
    x = int(round(x))
    y = int(round(y))

    x0 = max(0, x - radius)
    x1 = min(W, x + radius)
    y0 = max(0, y - radius)
    y1 = min(H, y + radius)

    return x0, x1, y0, y1


def mask_to_outline(mask_crop):
    """
    간단한 outline 생성
    """
    outline = np.zeros_like(mask_crop, dtype=bool)

    outline[:-1, :] |= (mask_crop[:-1, :] != mask_crop[1:, :]) & (mask_crop[:-1, :] > 0)
    outline[1:, :]  |= (mask_crop[1:, :]  != mask_crop[:-1, :]) & (mask_crop[1:, :]  > 0)
    outline[:, :-1] |= (mask_crop[:, :-1] != mask_crop[:, 1:]) & (mask_crop[:, :-1] > 0)
    outline[:, 1:]  |= (mask_crop[:, 1:]  != mask_crop[:, :-1]) & (mask_crop[:, 1:]  > 0)

    return outline


def get_hit_only_mask(mask_crop, hit_ids):
    if len(hit_ids) == 0:
        return np.zeros_like(mask_crop, dtype=np.uint8)
    return np.isin(mask_crop, hit_ids).astype(np.uint8)


# ─────────────────────────────────────────────
# 2. 데이터 로드
# ─────────────────────────────────────────────
print("[INFO] loading annotated csv...")
df = pd.read_csv(ANNOTATED_CSV)

print("[INFO] loading image zarr...")
root = zarr.open(ZARR_PATH, mode="r")
img_arr = root[IMAGE_KEY]
print("[INFO] image shape:", img_arr.shape)   # (C, Z, Y, X)

C, Z, H, W = img_arr.shape

print("[INFO] loading masks...")
mask_dict = {}
for z in range(Z):
    mask_path = Path(MASK_DIR) / f"mask_z{z}.tif"
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)
    mask_dict[z] = tiff.imread(mask_path)

print("[INFO] loaded masks for z =", list(mask_dict.keys()))


# ─────────────────────────────────────────────
# 3. target 후보만 처리
# ─────────────────────────────────────────────
for did in TARGET_DOUBLETS:
    sub = df[df["doublet_index"] == did].copy()

    if len(sub) == 0:
        print(f"[WARN] doublet {did} not found, skipping")
        continue

    sub = sub.sort_values("z").copy()

    # 대표 좌표
    x = float(sub["x_local"].iloc[0])
    y = float(sub["y_local"].iloc[0])

    # 전체 crop box
    x0, x1, y0, y1 = crop_box(x, y, H, W, PATCH_RADIUS)

    # figure 생성
    fig, axes = plt.subplots(Z, 2, figsize=(10, 4 * Z))
    if Z == 1:
        axes = np.array([axes])

    for z in range(Z):
        row = sub[sub["z"] == z]
        if len(row) == 0:
            row = None
            n_cells = 0
            cell_ids = ""
            cell_types = ""
            pair_type = ""
            hit_ids = []
        else:
            row = row.iloc[0]
            n_cells = int(row["n_cells"])
            cell_ids = str(row["cell_ids"]) if pd.notna(row["cell_ids"]) else ""
            cell_types = str(row["cell_types"]) if pd.notna(row["cell_types"]) else ""
            pair_type = str(row["pair_type"]) if pd.notna(row["pair_type"]) else ""
            hit_ids = parse_cell_ids(cell_ids)

        # image crop
        img = np.asarray(img_arr[CHANNEL_IDX, z, y0:y1, x0:x1])
        img_norm = normalize_img(img)

        # mask crop
        mask_crop = mask_dict[z][y0:y1, x0:x1]

        # 점 위치 (crop local)
        px = x - x0
        py = y - y0

        # ── col 1: raw image
        ax = axes[z, 0]
        ax.imshow(img_norm, cmap="gray")
        ax.scatter(px, py, s=20, c="red")
        ax.set_title(f"Doublet {did} | z={z} | raw")
        ax.axis("off")

        # ── col 2: overlay
        ax = axes[z, 1]
        ax.imshow(img_norm, cmap="gray")

        if SHOW_ALL_MASK:
            outline = mask_to_outline(mask_crop)
            yy, xx = np.where(outline)
            ax.scatter(xx, yy, s=0.2, c="cyan")

        if SHOW_HIT_ONLY and len(hit_ids) > 0:
            hit_mask = get_hit_only_mask(mask_crop, hit_ids)
            yy, xx = np.where(hit_mask > 0)
            ax.scatter(xx, yy, s=0.3, c="yellow")

        ax.scatter(px, py, s=20, c="red")

        title = f"z={z} | n_cells={n_cells}"
        if pair_type not in ["", "nan"]:
            title += f"\n{pair_type}"
        elif cell_types not in ["", "nan"]:
            title += f"\n{cell_types}"

        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    out_path = Path(OUT_DIR) / f"doublet_{did}_z_patch_panel.png"
    plt.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()

    print(f"[INFO] saved: {out_path}")

print("[INFO] done.")