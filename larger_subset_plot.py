#!/usr/bin/env python3
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import spatialdata as sd


def first_dim(da, candidates):
    for d in candidates:
        if d in da.dims:
            return d
    return None


def parse_indices(spec, size):
    if spec is None or spec.lower() == "all":
        return list(range(size))
    idx = [int(x) for x in spec.split(",") if x.strip() != ""]
    idx = [i for i in idx if 0 <= i < size]
    if not idx:
        raise ValueError(f"No valid indices in '{spec}' for size={size}")
    return idx


def to_2d(da, c_idx=None, z_idx=None, t_idx=0, stride=8):
    t_dim = first_dim(da, ["t", "time"])
    if t_dim is not None:
        da = da.isel({t_dim: t_idx})

    c_dim = first_dim(da, ["c", "channel"])
    if c_dim is not None and c_idx is not None:
        da = da.isel({c_dim: c_idx})

    z_dim = first_dim(da, ["z"])
    if z_dim is not None and z_idx is not None:
        da = da.isel({z_dim: z_idx})

    y_dim = first_dim(da, ["y"])
    x_dim = first_dim(da, ["x"])
    if y_dim is None or x_dim is None:
        raise ValueError(f"Expected x/y dims, got {da.dims}")

    if stride > 1:
        da = da.isel({y_dim: slice(None, None, stride), x_dim: slice(None, None, stride)})

    da = da.transpose(y_dim, x_dim)
    arr = da.data
    if hasattr(arr, "compute"):
        arr = arr.compute()
    arr = np.asarray(arr)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D after slicing, got shape={arr.shape}, dims={da.dims}")
    return arr


def robust_limits(arrays, pmin=1.0, pmax=99.5):
    v = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays if np.isfinite(a).any()])
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(v, [pmin, pmax])
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--image", default=None, help="Single image key (uses channels as rows if c exists)")
    ap.add_argument("--images", default=None, help="Comma-separated image keys (rows). Used when channels are separate keys.")
    ap.add_argument("--channels", default="all", help="Channel indices for --image, e.g. 0,1 or all")
    ap.add_argument("--z", default="all", help="z indices, e.g. 0,1,2 or all")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--pmin", type=float, default=1.0)
    ap.add_argument("--pmax", type=float, default=99.5)
    ap.add_argument("--ncols-max", type=int, default=12)
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sdata = sd.read_zarr(args.zarr)

    if args.list_only:
        print("Available image keys:")
        for k, da in sdata.images.items():
            print(f"- {k}: dims={da.dims}, shape={tuple(da.shape)}")
        return

    # Build row specs: (row_label, dataarray, c_idx_or_none)
    rows = []

    if args.image is not None:
        if args.image not in sdata.images:
            raise KeyError(f"--image '{args.image}' not found in sdata.images")
        da = sdata.images[args.image]
        c_dim = first_dim(da, ["c", "channel"])
        if c_dim is None:
            rows.append((args.image, da, None))
        else:
            c_idx_list = parse_indices(args.channels, da.sizes[c_dim])
            rows.extend([(f"{args.image} | c={ci}", da, ci) for ci in c_idx_list])
    else:
        if args.images is not None:
            keys = [k.strip() for k in args.images.split(",") if k.strip() != ""]
        else:
            keys = list(sdata.images.keys())

        if not keys:
            raise ValueError("No image keys selected")

        for k in keys:
            if k not in sdata.images:
                raise KeyError(f"Image key '{k}' not found")
            da = sdata.images[k]
            c_dim = first_dim(da, ["c", "channel"])
            if c_dim is None:
                rows.append((k, da, None))
            else:
                # In multi-key mode, take first channel unless explicit single index provided.
                c_idx_list = parse_indices(args.channels, da.sizes[c_dim])
                for ci in c_idx_list:
                    rows.append((f"{k} | c={ci}", da, ci))

    if not rows:
        raise ValueError("No rows to plot")

    # z indices from first row's data
    z_dim0 = first_dim(rows[0][1], ["z"])
    if z_dim0 is None:
        z_list = [None]
    else:
        z_list = parse_indices(args.z, rows[0][1].sizes[z_dim0])

    if len(z_list) > args.ncols_max:
        z_list = z_list[: args.ncols_max]

    nrows = len(rows)
    ncols = len(z_list)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.2 * nrows), squeeze=False)

    for r, (label, da, c_idx) in enumerate(rows):
        planes = []
        for z_idx in z_list:
            planes.append(to_2d(da, c_idx=c_idx, z_idx=z_idx, stride=args.stride))
        vmin, vmax = robust_limits(planes, pmin=args.pmin, pmax=args.pmax)

        for c, z_idx in enumerate(z_list):
            ax = axes[r, c]
            ax.imshow(planes[c], cmap="gray", vmin=vmin, vmax=vmax)
            if r == 0:
                ax.set_title(f"z={z_idx}" if z_idx is not None else "2D", fontsize=10)
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=200, bbox_inches="tight")
        print(f"Saved: {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
