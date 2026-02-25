#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

import ovrlpy


def load_coordinate_df(input_csv: Path):
    # Original Vizgen file: detected_transcripts.csv -> use ovrlpy parser.
    header = set(pd.read_csv(input_csv, nrows=0).columns)
    if {"global_x", "global_y"}.issubset(header):
        return ovrlpy.io.read_MERSCOPE(input_csv)

    # Already prepared ovrlpy input: x,y,z,gene
    required = {"x", "y", "z", "gene"}
    if not required.issubset(header):
        raise ValueError(
            f"Input CSV must be either detected_transcripts.csv or include columns {sorted(required)}. "
            f"Found: {sorted(header)}"
        )
    return pl.read_csv(input_csv).select(["x", "y", "z", "gene"]).with_columns(pl.col("gene").cast(pl.String))


def main():
    p = argparse.ArgumentParser(description="Run ovrlpy MERSCOPE liver tutorial as a Python script")
    p.add_argument("--input-csv", required=True, help="Path to detected_transcripts.csv or x,y,z,gene CSV")
    p.add_argument("--outdir", required=True, help="Directory to save figures and outputs")
    p.add_argument("--n-components", type=int, default=10)
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--scatter-step", type=int, default=5000, help="Plot every Nth transcript for overview")
    p.add_argument("--min-signal", type=float, default=3.0)
    p.add_argument("--integrity-sigma", type=float, default=3.0)
    p.add_argument("--signal-threshold", type=float, default=3.0)
    p.add_argument("--doublet-case", type=int, default=1)
    p.add_argument("--window-size", type=float, default=40.0)
    args = p.parse_args()

    input_csv = Path(args.input_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    coordinate_df = load_coordinate_df(input_csv)
    print(f"Number of transcripts: {len(coordinate_df):,}")
    print(coordinate_df.head())

    # Tissue overview
    fig, ax = plt.subplots()
    ax.scatter(coordinate_df["x"], coordinate_df["y"], s=0.1)
    ax.set(aspect="equal", title=f"Tissue overview")
    fig.savefig(outdir / "01_tissue_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Run ovrlpy pipeline
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
    fig.savefig(outdir / "03_signal_integrity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Doublet probability
    doublets = liver.detect_doublets(min_signal=args.min_signal, integrity_sigma=args.integrity_sigma)
    doublets.write_csv(outdir / "doublets.csv")

    fig, ax = plt.subplots()
    scatter = ax.scatter(doublets["x"], doublets["y"], c=doublets["integrity"], s=0.2, cmap="viridis")
    ax.set_aspect("equal")
    fig.colorbar(scatter, ax=ax)
    fig.savefig(outdir / "04_doublet_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    if len(doublets) > 0:
        case = min(max(0, args.doublet_case), len(doublets) - 1)
        x, y = doublets["x", "y"].row(case)
        _ = ovrlpy.plot_region_of_interest(liver, x, y, window_size=args.window_size)
        plt.gcf().savefig(outdir / "05_doublet_roi.png", dpi=200, bbox_inches="tight")
        plt.close(plt.gcf())

    print(f"Saved outputs to: {outdir}")


if __name__ == "__main__":
    main()
