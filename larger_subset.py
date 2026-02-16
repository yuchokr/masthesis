#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import sparrow as sp


def infer_z_layers(images_dir: Path):
    pat = re.compile(r"_z(\d+)\.tif$")
    zs = sorted(
        {
            int(m.group(1))
            for p in images_dir.glob("mosaic_*_z*.tif")
            for m in [pat.search(p.name)]
            if m
        }
    )
    if not zs:
        raise ValueError(f"No mosaic_*_z*.tif found under {images_dir}")
    return zs


def expand_roi(roi, pad=0, scale=None):
    # Vizgen random ROI convention: (xmin, xmax, ymin, ymax)
    xmin, xmax, ymin, ymax = roi

    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if ymin > ymax:
        ymin, ymax = ymax, ymin

    if scale is not None:
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        half_w = (xmax - xmin) * scale / 2.0
        half_h = (ymax - ymin) * scale / 2.0
        xmin, xmax = cx - half_w, cx + half_w
        ymin, ymax = cy - half_h, cy + half_h

    if pad:
        xmin -= pad
        ymin -= pad
        xmax += pad
        ymax += pad

    # SpatialData bounding_box expects: [xmin, ymin, xmax, ymax]
    return [int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax))]


def rechunk_image(img, xy=1024, z=1, c=1):
    chunks = {}
    for d in img.dims:
        if d in ("y", "x"):
            chunks[d] = xy
        elif d in ("z",):
            chunks[d] = z
        elif d in ("c", "channel"):
            chunks[d] = c
    return img.chunk(chunks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Vizgen MERSCOPE root dir (e.g., Liver1Slice1)")
    p.add_argument("--output", required=True, help="Output zarr path")
    p.add_argument(
        "--roi",
        type=int,
        nargs=4,
        required=True,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="ROI order: XMIN XMAX YMIN YMAX (Vizgen random ROI format)",
    )
    p.add_argument("--pad", type=int, default=0, help="Expand ROI by padding (pixels)")
    p.add_argument("--scale", type=float, default=None, help="Scale ROI around center (e.g., 1.5)")
    p.add_argument("--coord-system", default="global")

    p.add_argument("--z-layers", default=None, help="Comma list, e.g., 0,1,2,3,4,5,6")
    p.add_argument("--backend", choices=["dask_image", "rioxarray"], default=None)
    p.add_argument("--do-3d", action="store_true", help="Read images/transcripts as 3D (z,y,x)")
    p.add_argument("--z-projection", action="store_true", help="Project z if not using 3D")
    p.add_argument("--with-transcripts", action="store_true", help="Load transcripts (can be huge)")
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--rechunk-xy", type=int, default=1024, help="Rechunk size for x/y before writing zarr")
    p.add_argument("--rechunk-z", type=int, default=1, help="Rechunk size for z before writing zarr")
    p.add_argument("--rechunk-c", type=int, default=1, help="Rechunk size for channel before writing zarr")

    args = p.parse_args()

    if args.z_layers:
        z_layers = [int(x) for x in args.z_layers.split(",")]
    else:
        z_layers = infer_z_layers(Path(args.input) / "images")

    roi = expand_roi(args.roi, pad=args.pad, scale=args.scale)

    sdata = sp.io.merscope(
        args.input,
        to_coordinate_system=args.coord_system,
        z_layers=z_layers,
        backend=args.backend,
        transcripts=args.with_transcripts,
        mosaic_images=True,
        do_3D=args.do_3d,
        z_projection=args.z_projection,
    )

    sdata_crop = sdata.query.bounding_box(
        axes=["x", "y"],
        min_coordinate=roi[:2],
        max_coordinate=roi[2:],
        target_coordinate_system=args.coord_system,
    )

    # Fix irregular chunking before writing
    for name, img in list(sdata_crop.images.items()):
        sdata_crop.images[name] = rechunk_image(
            img, xy=args.rechunk_xy, z=args.rechunk_z, c=args.rechunk_c
        )

    sdata_crop.write(args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
