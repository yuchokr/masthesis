#!/usr/bin/env python3

from pathlib import Path
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
import spatialdata as sd


# ============================================================
# CONFIG
# ============================================================

ZARR_PATH = Path(
    "/data/gent/vo/000/gvo00070/vsc48277/"
    "yujin_project/40k_subset_0.zarr"
)

STITCH_DIR = Path(
    "/data/gent/vo/000/gvo00070/vsc48277/"
    "yujin_project/"
    "40k_subset_0_cellpose_fullcell_stitch3d_tiled"
)

MASK_DIR = STITCH_DIR / "stitched_masks"

MEMBERSHIP_CSV = (
    STITCH_DIR / "stack_cell_membership.csv"
)

OUTPUT_DIR = (
    STITCH_DIR / "stack_boundary_visualizations"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Image layer
#
# 우선 아래 순서대로 존재하는 layer를 자동 선택
# ------------------------------------------------------------

PREFERRED_IMAGE_LAYERS = [
    "clahe_DAPI_PolyT",
    "clahe",
    "min_max_filtered",
    "Yujin_vizgen_Liver1Slice1_z_global",
]


DAPI_CHANNEL = 0
POLYT_CHANNEL = 1

N_Z = 7

# 저번 plot처럼 cell 주변 여백
PAD = 80

# bounding box 검색할 때 한 번에 읽는 row 수
SCAN_ROWS = 512

# random stack selection
RANDOM_SEED = 12


# ============================================================
# Helper
# ============================================================

def get_level0(layer):
    """
    Get scale0 DataArray from a SpatialData multiscale image.
    """

    try:
        return layer["scale0"].image
    except Exception:
        pass

    try:
        return layer["scale0"]["image"]
    except Exception:
        pass

    return layer


# ============================================================
# 1. Load SpatialData
# ============================================================

print("\n[1] Loading SpatialData...")

sdata = sd.read_zarr(
    str(ZARR_PATH)
)


# ------------------------------------------------------------
# Automatically choose image layer
# ------------------------------------------------------------

IMAGE_LAYER = None

for candidate in PREFERRED_IMAGE_LAYERS:

    if candidate in sdata.images:

        IMAGE_LAYER = candidate
        break


if IMAGE_LAYER is None:

    raise KeyError(
        "No suitable image layer found.\n"
        f"Available images:\n"
        f"{list(sdata.images.keys())}"
    )


print(
    f"  Using image layer: {IMAGE_LAYER}"
)


img = get_level0(
    sdata.images[IMAGE_LAYER]
)


print(
    f"  dims  = {img.dims}"
)

print(
    f"  shape = {img.shape}"
)


# ============================================================
# 2. Load membership table
# ============================================================

print("\n[2] Loading membership CSV...")

membership = pd.read_csv(
    MEMBERSHIP_CSV
)


print(
    f"  rows = {len(membership)}"
)

print(
    f"  stacks = "
    f"{membership['stack_id'].nunique()}"
)


# ============================================================
# 3. Calculate true n_z_layers
#
# CSV에 혹시 같은 stack/z가 중복되어도
# unique z 기준으로 계산
# ============================================================

stack_info = (
    membership
    .groupby("stack_id")
    ["z"]
    .nunique()
    .reset_index(
        name="n_z_layers"
    )
)


print("\nStack counts:")

print(
    stack_info[
        "n_z_layers"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# 4. Select one random stack for each n_z_layers = 1...7
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)


selected_rows = []


for n_layers in range(
    1,
    N_Z + 1
):

    candidates = (
        stack_info[
            stack_info[
                "n_z_layers"
            ] == n_layers
        ]
        ["stack_id"]
        .to_numpy()
    )


    if len(candidates) == 0:

        print(
            f"  No stack with "
            f"n_z_layers={n_layers}"
        )

        continue


    stack_id = int(
        rng.choice(
            candidates
        )
    )


    selected_rows.append(
        {
            "stack_id":
                stack_id,

            "n_z_layers":
                n_layers,
        }
    )


selected_df = pd.DataFrame(
    selected_rows
)


selected_path = (
    OUTPUT_DIR
    / "selected_random_stacks.csv"
)


selected_df.to_csv(
    selected_path,
    index=False,
)


print("\nSelected stacks:")

print(
    selected_df.to_string(
        index=False
    )
)


# ============================================================
# 5. Find bounding box of one stitched stack
#
# Important:
# 전체 45000x45000 boolean array를 만들지 않고
# row chunk 단위로 검색
# ============================================================

def find_stack_bbox(
    stack_id,
    z_layers,
    pad=80,
):

    ymin = None
    ymax = None
    xmin = None
    xmax = None


    for z in z_layers:

        mask_path = (
            MASK_DIR
            / f"mask_z{z}.tif"
        )


        mask = tifffile.memmap(
            str(mask_path)
        )


        height, width = mask.shape


        for y0 in range(
            0,
            height,
            SCAN_ROWS
        ):

            y1 = min(
                y0 + SCAN_ROWS,
                height
            )


            chunk = mask[
                y0:y1
            ]


            ys, xs = np.where(
                chunk == stack_id
            )


            if len(ys) == 0:
                continue


            ys = (
                ys + y0
            )


            chunk_ymin = int(
                ys.min()
            )

            chunk_ymax = int(
                ys.max()
            )

            chunk_xmin = int(
                xs.min()
            )

            chunk_xmax = int(
                xs.max()
            )


            if ymin is None:

                ymin = chunk_ymin
                ymax = chunk_ymax
                xmin = chunk_xmin
                xmax = chunk_xmax

            else:

                ymin = min(
                    ymin,
                    chunk_ymin
                )

                ymax = max(
                    ymax,
                    chunk_ymax
                )

                xmin = min(
                    xmin,
                    chunk_xmin
                )

                xmax = max(
                    xmax,
                    chunk_xmax
                )


        del mask


    if ymin is None:

        raise RuntimeError(
            f"Stack {stack_id} not found."
        )


    ymin = max(
        0,
        ymin - pad
    )

    xmin = max(
        0,
        xmin - pad
    )


    ymax = min(
        img.sizes["y"],
        ymax + pad + 1
    )

    xmax = min(
        img.sizes["x"],
        xmax + pad + 1
    )


    return (
        ymin,
        ymax,
        xmin,
        xmax,
    )


# ============================================================
# Display helper
# ============================================================

def display_image(
    ax,
    image,
):

    image = np.asarray(
        image
    )


    finite = image[
        np.isfinite(image)
    ]


    if finite.size > 0:

        vmin = np.percentile(
            finite,
            1
        )

        vmax = np.percentile(
            finite,
            99.5
        )


        if vmax <= vmin:

            vmax = None

    else:

        vmin = None
        vmax = None


    ax.imshow(
        image,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )


# ============================================================
# 6. Plot each selected stack
# ============================================================

print(
    "\n[3] Creating plots..."
)


for _, selection in (
    selected_df.iterrows()
):

    stack_id = int(
        selection[
            "stack_id"
        ]
    )

    n_z_layers = int(
        selection[
            "n_z_layers"
        ]
    )


    # --------------------------------------------------------
    # z layers where this stack is present
    # --------------------------------------------------------

    present_z = sorted(
        membership.loc[
            membership[
                "stack_id"
            ] == stack_id,
            "z"
        ]
        .unique()
        .astype(int)
        .tolist()
    )


    print(
        f"\nStack {stack_id}: "
        f"{n_z_layers} z-layers "
        f"{present_z}"
    )


    # --------------------------------------------------------
    # Common XY bounding box across all z
    # --------------------------------------------------------

    (
        ymin,
        ymax,
        xmin,
        xmax,
    ) = find_stack_bbox(
        stack_id=stack_id,
        z_layers=present_z,
        pad=PAD,
    )


    print(
        f"  bbox = "
        f"y[{ymin}:{ymax}], "
        f"x[{xmin}:{xmax}]"
    )


    # --------------------------------------------------------
    # Figure:
    #
    # row 0 = DAPI
    # row 1 = PolyT
    #
    # columns = z0 ... z6
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        N_Z,
        figsize=(
            3.0 * N_Z,
            6.2,
        ),
    )


    for z in range(N_Z):

        # ----------------------------------------------------
        # Read only image crop
        # ----------------------------------------------------

        dapi_crop = (
            img
            .isel(
                c=DAPI_CHANNEL,
                z=z,
                y=slice(
                    ymin,
                    ymax
                ),
                x=slice(
                    xmin,
                    xmax
                ),
            )
            .compute()
            .values
        )


        polyt_crop = (
            img
            .isel(
                c=POLYT_CHANNEL,
                z=z,
                y=slice(
                    ymin,
                    ymax
                ),
                x=slice(
                    xmin,
                    xmax
                ),
            )
            .compute()
            .values
        )


        # ----------------------------------------------------
        # Read stitched segmentation crop
        # ----------------------------------------------------

        mask_path = (
            MASK_DIR
            / f"mask_z{z}.tif"
        )


        mask = tifffile.memmap(
            str(mask_path)
        )


        mask_crop = np.asarray(
            mask[
                ymin:ymax,
                xmin:xmax
            ]
        )


        selected_mask = (
            mask_crop
            == stack_id
        )


        present = bool(
            np.any(
                selected_mask
            )
        )


        # ====================================================
        # DAPI row
        # ====================================================

        ax = axes[
            0,
            z
        ]


        display_image(
            ax,
            dapi_crop
        )


        if present:

            ax.contour(
                selected_mask.astype(
                    float
                ),
                levels=[
                    0.5
                ],
                colors="red",
                linewidths=1.3,
            )


        ax.set_title(
            f"z{z}\n"
            + (
                "present"
                if present
                else "absent"
            ),
            fontsize=10,
        )


        ax.axis(
            "off"
        )


        # ====================================================
        # PolyT row
        # ====================================================

        ax = axes[
            1,
            z
        ]


        display_image(
            ax,
            polyt_crop
        )


        if present:

            ax.contour(
                selected_mask.astype(
                    float
                ),
                levels=[
                    0.5
                ],
                colors="red",
                linewidths=1.3,
            )


        ax.axis(
            "off"
        )


        del mask
        del mask_crop
        del selected_mask
        del dapi_crop
        del polyt_crop


    # --------------------------------------------------------
    # Row labels
    # --------------------------------------------------------

    axes[
        0,
        0
    ].set_ylabel(
        "DAPI",
        fontsize=13,
    )


    axes[
        1,
        0
    ].set_ylabel(
        "PolyT",
        fontsize=13,
    )


    fig.suptitle(
        f"Stitched stack {stack_id}  |  "
        f"n_z_layers = {n_z_layers}  |  "
        f"z = {present_z}",
        fontsize=15,
    )


    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.93,
        ]
    )


    output_path = (
        OUTPUT_DIR
        / (
            f"stack_{stack_id}"
            f"_nZ{n_z_layers}.png"
        )
    )


    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    print(
        f"  saved → {output_path}"
    )


    gc.collect()


print(
    "\n======================================"
)

print(
    "PLOTTING FINISHED"
)

print(
    "======================================"
)

print(
    f"\nOutput directory:\n"
    f"{OUTPUT_DIR}"
)

print(
    "\nFiles:"
)

print(
    "  selected_random_stacks.csv"
)

print(
    "  stack_<ID>_nZ<N>.png"
)