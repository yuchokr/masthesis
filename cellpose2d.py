#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import zarr
import tifffile as tiff
import matplotlib.pyplot as plt

from cellpose import models, utils


# ─────────────────────────────────────────────
# 0. 설정
# ─────────────────────────────────────────────
ZARR_PATH = "/data/gent/vo/000/gvo00070/vsc48277/yujin_project/40k_subset_0.zarr"
IMAGE_KEY = "images/Yujin_vizgen_Liver1Slice1_z_global/0"

OUT_DIR = "/data/gent/vo/000/gvo00070/vsc48277/yujin_project/40k_subset_0_2d_segmentation_result"

# shape = (2, 7, 45000, 45000) = (C, Z, Y, X)
CHANNEL_IDX = 0   # 보통 DAPI=0, 필요시 1로 변경
Z_START = 0
Z_END = 7         # python range처럼 끝은 포함 안 됨

MODEL_TYPE = "nuclei"
DIAMETER = 30
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0
USE_GPU = False

SAVE_OVERLAY = True
OVERLAY_DOWNSAMPLE = 8   # overlay 저장 시 너무 무거우면 키우기


# ─────────────────────────────────────────────
# 1. 함수
# ─────────────────────────────────────────────
def normalize_img(img):
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, [1, 99])

    if p99 <= p1:
        return np.zeros_like(img, dtype=np.float32)

    img = np.clip((img - p1) / (p99 - p1), 0, 1)
    return img


def save_overlay(img, mask, out_png, title="", downsample=8):
    """
    큰 이미지라 downsample해서 저장
    """
    if downsample > 1:
        img_small = img[::downsample, ::downsample]
        mask_small = mask[::downsample, ::downsample]
    else:
        img_small = img
        mask_small = mask

    outlines = utils.masks_to_outlines(mask_small)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_small, cmap="gray")
    yy, xx = np.where(outlines)
    plt.scatter(xx, yy, s=0.1)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────
# 2. 출력 폴더
# ─────────────────────────────────────────────
out_dir = Path(OUT_DIR)
mask_dir = out_dir / "masks_each_z"
overlay_dir = out_dir / "overlay_each_z"

mask_dir.mkdir(parents=True, exist_ok=True)
overlay_dir.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 3. zarr 열기
# ─────────────────────────────────────────────
print("[INFO] opening zarr...")
root = zarr.open(ZARR_PATH, mode="r")
arr = root[IMAGE_KEY]

print("[INFO] shape:", arr.shape)
print("[INFO] dtype:", arr.dtype)
print("[INFO] chunks:", arr.chunks)

# 기대 shape: (C, Z, Y, X)
if arr.ndim != 4:
    raise ValueError(f"Expected 4D array, got {arr.shape}")

C, Z, Y, X = arr.shape
print(f"[INFO] C={C}, Z={Z}, Y={Y}, X={X}")

if not (0 <= CHANNEL_IDX < C):
    raise ValueError(f"CHANNEL_IDX {CHANNEL_IDX} out of range for C={C}")

z_values = list(range(Z_START, min(Z_END, Z)))


# ─────────────────────────────────────────────
# 4. Cellpose 모델 로드
# ─────────────────────────────────────────────
print("[INFO] loading Cellpose model...")
model = models.CellposeModel(gpu=USE_GPU, model_type=MODEL_TYPE)


# ─────────────────────────────────────────────
# 5. z별 독립 2D segmentation
# ─────────────────────────────────────────────
summary = []

for z in z_values:
    print(f"\n[INFO] segmenting z={z} ...")

    # 핵심: 전체 stack을 읽지 말고 필요한 z 하나만 읽기
    img = arr[CHANNEL_IDX, z, :, :]
    img = np.asarray(img)

    print(f"[INFO] loaded slice shape: {img.shape}")

    img_norm = normalize_img(img)

    masks, flows, styles = model.eval(
        img_norm,
        diameter=DIAMETER,
        channels=[0, 0],   # grayscale single channel
        flow_threshold=FLOW_THRESHOLD,
        cellprob_threshold=CELLPROB_THRESHOLD,
        do_3D=False
    )

    masks = masks.astype(np.uint32)

    n_cells = len(np.unique(masks)) - 1
    max_label = int(masks.max())

    print(f"[INFO] z={z}, n_cells={n_cells}, max_label={max_label}")

    # z별 mask 저장
    mask_path = mask_dir / f"mask_z{z}.tif"
    tiff.imwrite(mask_path, masks)

    # overlay 저장
    if SAVE_OVERLAY:
        overlay_path = overlay_dir / f"overlay_z{z}.png"
        save_overlay(
            img_norm,
            masks,
            overlay_path,
            title=f"z={z}, n_cells={n_cells}",
            downsample=OVERLAY_DOWNSAMPLE
        )

    summary.append({
        "z": z,
        "channel": CHANNEL_IDX,
        "n_cells": int(n_cells),
        "max_label": max_label,
        "mask_path": str(mask_path)
    })

    # 메모리 정리 느낌으로 참조 삭제
    del img, img_norm, masks, flows, styles


# ─────────────────────────────────────────────
# 6. summary 저장
# ─────────────────────────────────────────────
summary_df = pd.DataFrame(summary)
summary_csv = out_dir / "segmentation_summary.csv"
summary_df.to_csv(summary_csv, index=False)

print("\n[INFO] done.")
print("[INFO] summary saved:", summary_csv)
print("[INFO] mask dir:", mask_dir)
if SAVE_OVERLAY:
    print("[INFO] overlay dir:", overlay_dir)