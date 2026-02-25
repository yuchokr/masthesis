#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

import ovrlpy



def load_um_coordinate_df(input_csv: Path) -> pl.DataFrame:
    cols = set(pd.read_csv(input_csv, nrows=0).columns)

    if {"global_x", "global_y"}.issubset(cols):
        # Raw Vizgen detected_transcripts.csv (micron)
        df = ovrlpy.io.read_MERSCOPE(input_csv)
        if isinstance(df, pl.DataFrame):
            return df.select(["x", "y", "z", "gene"])
        return pl.from_pandas(df[["x", "y", "z", "gene"]]).with_columns(pl.col("gene").cast(pl.String))

    # Pre-made micron csv
    x_col = "x_um" if "x_um" in cols else "x"
    y_col = "y_um" if "y_um" in cols else "y"
    required = {x_col, y_col, "z", "gene"}
    if not required.issubset(cols):
        raise ValueError(f"Need columns {sorted(required)}. Found: {sorted(cols)}")

    return pl.read_csv(input_csv).select(
        [
            pl.col(x_col).cast(pl.Float64).alias("x"),
            pl.col(y_col).cast(pl.Float64).alias("y"),
            pl.col("z").cast(pl.Float64),
            pl.col("gene").cast(pl.String),
        ]
    )


def main():
    p = argparse.ArgumentParser(description="Minimal ovrlpy run (analyse in micron, optional pixel export)")
    p.add_argument("--input-csv", required=True, help="Micron CSV: x,y,z,gene (or x_um,y_um,z,gene)")
    p.add_argument("--outdir", required=True)
    p.add_argument("--n-components", type=int, default=10)
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--min-signal", type=float, default=2.0)
    p.add_argument("--integrity-sigma", type=float, default=2.0)
    p.add_argument("--signal-threshold", type=float, default=2.0)
    p.add_argument("--doublet-case", type=int, default=0)
    p.add_argument("--window-size", type=float, default=40.0)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    coordinate_df = load_um_coordinate_df(Path(args.input_csv))
    print(f"Number of transcripts: {len(coordinate_df):,}")

    # 01 Tissue overview
    xy = coordinate_df.select(["x", "y"]).to_numpy()
    fig, ax = plt.subplots()
    ax.scatter(xy[:, 0], xy[:, 1], s=0.1)
    ax.set(aspect="equal", title="Tissue overview (micron)")
    fig.savefig(outdir / "01_tissue_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 02/03 ovrlpy
    liver = ovrlpy.Ovrlp(
        coordinate_df,
        n_components=args.n_components,
        n_workers=args.n_workers,
        random_state=args.random_state,
    )
    liver.analyse()

    _ = ovrlpy.plot_pseudocells(liver)
    plt.gcf().savefig(outdir / "02_pseudocells.png", dpi=200, bbox_inches="tight")
    plt.close(plt.gcf())

    fig = ovrlpy.plot_signal_integrity(liver, signal_threshold=args.signal_threshold)
    if fig is None:
        fig = plt.gcf()
    fig.savefig(outdir / "03_signal_integrity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 04/05 doublets (always in micron)
    doublets = liver.detect_doublets(min_signal=args.min_signal, integrity_sigma=args.integrity_sigma)
    n_doublets = len(doublets)
    print(f"Detected doublets: {n_doublets:,}")
    doublets.write_csv(outdir / "doublets_um.csv")

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    if n_doublets > 0:
        sc = ax.scatter(doublets["x"], doublets["y"], c=doublets["integrity"], s=0.2, cmap="viridis")
        fig.colorbar(sc, ax=ax)
    else:
        ax.text(0.5, 0.5, "No doublets detected", ha="center", va="center", transform=ax.transAxes)
    fig.savefig(outdir / "04_doublet_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    if n_doublets > 0:
        case = min(max(0, args.doublet_case), n_doublets - 1)
        x, y = doublets["x", "y"].row(case)
        _ = ovrlpy.plot_region_of_interest(liver, x, y, window_size=args.window_size)
        plt.gcf().savefig(outdir / "05_doublet_roi.png", dpi=200, bbox_inches="tight")
        plt.close(plt.gcf())


    print(f"Saved outputs to: {outdir}")


if __name__ == "__main__":
    main()
