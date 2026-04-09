"""
2.5D Segmentation: Cellpose2D + z-linking (stitch_threshold)
- 대용량 zarr (40000x40000) 타일링 + overlap 처리
- HPC 128GB RAM 기준

사전 설치:
    pip install cellpose zarr numpy
"""

import zarr
import numpy as np
from cellpose import models
from pathlib import Path


# ─────────────────────────────────────────────
# 설정값 — 여기만 수정하세요
# ─────────────────────────────────────────────

ZARR_PATH  = "larger_40k_subset.zarr"
INPUT_KEY  = "images/Yujin_vizgen_Liver1Slice1_z_global/0"
OUTPUT_KEY = "seg_cellpose2d_zlink"

CHANNEL_IDX = 1          # DAPI = C=1

# 타일 설정
TILE_SIZE   = 4096       # 타일 크기 (픽셀)
OVERLAP     = 256        # 겹치는 영역 (픽셀) — 경계 세포 보호

# Cellpose 설정
MODEL_TYPE         = "nuclei"
USE_GPU            = True
DIAMETER           = 30.0    # None 이면 자동 추정
FLOW_THRESHOLD     = 0.4
CELLPROB_THRESHOLD = 0.0
STITCH_THRESHOLD   = 0.1     # z-linking 강도

# ─────────────────────────────────────────────


def process_tile(model, tile_zyx, diameter, flow_threshold, cellprob_threshold, stitch_threshold):
    """
    tile_zyx: (Z, Y, X) numpy array (이미 채널 선택된 상태)
    반환: (Z, Y, X) uint32 mask
    """
    print(f"         process_tile input shape={tile_zyx.shape}, dtype={tile_zyx.dtype}")

    result = model.eval(
        tile_zyx,
        diameter=diameter,
        channels=[0, 0],          # 그레이스케일 단일 채널
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        do_3D=False,
        stitch_threshold=stitch_threshold,
        z_axis=0,
        progress=False,
    )

    # Cellpose 버전 차이를 피하기 위해 result[0]만 사용
    masks = result[0]

    return masks.astype(np.uint32)


def crop_overlap(arr, y_start, y_end, x_start, x_end,
                 tile_y0, tile_y1, tile_x0, tile_x1,
                 overlap, img_H, img_W):
    """
    타일 결과에서 overlap 영역을 제거하고 유효한 영역만 반환.
    반환: (cropped_mask, out_y0, out_y1, out_x0, out_x1)
    """
    # overlap 제거: 가장자리가 아닌 경우에만 안쪽으로 자름
    inner_y0 = overlap if tile_y0 > 0 else 0
    inner_y1 = arr.shape[-2] - overlap if tile_y1 < img_H else arr.shape[-2]
    inner_x0 = overlap if tile_x0 > 0 else 0
    inner_x1 = arr.shape[-1] - overlap if tile_x1 < img_W else arr.shape[-1]

    cropped = arr[..., inner_y0:inner_y1, inner_x0:inner_x1]

    out_y0 = tile_y0 + inner_y0
    out_y1 = out_y0 + cropped.shape[-2]
    out_x0 = tile_x0 + inner_x0
    out_x1 = out_x0 + cropped.shape[-1]

    return cropped, out_y0, out_y1, out_x0, out_x1


def run_tiled_segmentation(
    zarr_path, input_key, output_key,
    channel_idx=1,
    tile_size=4096, overlap=256,
    model_type="nuclei", use_gpu=False,
    diameter=30.0, flow_threshold=0.4,
    cellprob_threshold=0.0, stitch_threshold=0.1,
):
    store = zarr.open(zarr_path, mode="r+")
    raw: zarr.Array = store[input_key]   # (C, Z, Y, X)

    C, Z, H, W = raw.shape
    print(f"[입력] shape={raw.shape}  dtype={raw.dtype}")
    print(f"[설정] channel={channel_idx}  tile={tile_size}  overlap={overlap}")

    if not (0 <= channel_idx < C):
        raise ValueError(f"channel_idx={channel_idx} is out of range for C={C}")

    # ── 출력 배열 초기화 ──────────────────────────
    if output_key in store:
        del store[output_key]

    out: zarr.Array = store.create_dataset(
        output_key,
        shape=(Z, H, W),
        dtype=np.uint32,
        chunks=(1, 1024, 1024),
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2),
        fill_value=0,
    )
    out.attrs.update({
        "method":             "Cellpose2D + z-linking (tiled)",
        "model_type":         model_type,
        "diameter":           diameter,
        "flow_threshold":     flow_threshold,
        "cellprob_threshold": cellprob_threshold,
        "stitch_threshold":   stitch_threshold,
        "tile_size":          tile_size,
        "overlap":            overlap,
        "source_channel":     channel_idx,
        "source_array":       input_key,
    })

    # ── 모델 로드 ─────────────────────────────────
    model = models.Cellpose(model_type=model_type, gpu=use_gpu)

    # ── 타일 순회 ─────────────────────────────────
    y_starts = list(range(0, H, tile_size))
    x_starts = list(range(0, W, tile_size))
    total = len(y_starts) * len(x_starts)
    tile_idx = 0

    # 타일별 ID 오프셋 (서로 다른 타일의 세포 ID가 겹치지 않게)
    global_id_offset = 0

    for y0 in y_starts:
        for x0 in x_starts:
            tile_idx += 1

            tile_y1_nominal = min(y0 + tile_size, H)
            tile_x1_nominal = min(x0 + tile_size, W)

            # overlap 포함한 실제 읽기 범위
            read_y0 = max(0, y0 - overlap)
            read_x0 = max(0, x0 - overlap)
            read_y1 = min(H, tile_y1_nominal + overlap)
            read_x1 = min(W, tile_x1_nominal + overlap)

            print(
                f"[{tile_idx}/{total}] 타일 y={y0}~{tile_y1_nominal}, x={x0}~{tile_x1_nominal}  "
                f"(읽기: y={read_y0}~{read_y1}, x={read_x0}~{read_x1})"
            )

            # (C, Z, Y, X) → (Z, Y, X) : 채널 선택
            tile = raw[channel_idx, :, read_y0:read_y1, read_x0:read_x1]

            # numpy array로 명시적 변환
            tile = np.asarray(tile)

            # Cellpose 실행
            masks = process_tile(
                model=model,
                tile_zyx=tile,
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                stitch_threshold=stitch_threshold,
            )

            # 현재 타일의 원래 최대 ID 기록
            tile_max_id = int(masks.max())

            # ID 오프셋 적용 (0은 배경이므로 유지)
            if tile_max_id > 0:
                fg = masks > 0
                masks[fg] += global_id_offset
                global_id_offset = int(masks.max())
            else:
                fg = None

            # overlap 제거 후 유효 영역만 추출
            cropped, out_y0, out_y1, out_x0, out_x1 = crop_overlap(
                masks,
                y_start=y0, y_end=tile_y1_nominal,
                x_start=x0, x_end=tile_x1_nominal,
                tile_y0=read_y0, tile_y1=read_y1,
                tile_x0=read_x0, tile_x1=read_x1,
                overlap=overlap, img_H=H, img_W=W,
            )

            # zarr 에 기록
            out[:, out_y0:out_y1, out_x0:out_x1] = cropped

            print(
                f"         → 이 타일 원래 mask max ID: {tile_max_id}  "
                f"누적 global max ID: {global_id_offset}  "
                f"기록 범위: z=0:{Z}, y={out_y0}:{out_y1}, x={out_x0}:{out_x1}"
            )

    print(f"\n[완료] 총 세포 수(대략, global max ID 기준): {global_id_offset}")
    print(f"[저장] zarr key: '{output_key}'  shape={out.shape}")
    return out


# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_tiled_segmentation(
        zarr_path          = ZARR_PATH,
        input_key          = INPUT_KEY,
        output_key         = OUTPUT_KEY,
        channel_idx        = CHANNEL_IDX,
        tile_size          = TILE_SIZE,
        overlap            = OVERLAP,
        model_type         = MODEL_TYPE,
        use_gpu            = USE_GPU,
        diameter           = DIAMETER,
        flow_threshold     = FLOW_THRESHOLD,
        cellprob_threshold = CELLPROB_THRESHOLD,
        stitch_threshold   = STITCH_THRESHOLD,
    )