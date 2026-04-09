#!/usr/bin/env python3
"""

Example:
    python virtual_subslices_from_csv.py \
        --csv transcripts.csv \
        --outdir virtual_subslices_out \
        --gene-col gene --x-col global_x --y-col global_y --z-col global_z
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ovrlpy
import polars as pl
import seaborn as sns
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.sparse import coo_array


# -----------------------------
# Small plotting defaults
# -----------------------------
CM = 1 / 2.54
FONT = {"family": "sans-serif", "size": 8}
matplotlib.rc("font", **FONT)
SAVEFIG_KWARGS = {"bbox_inches": "tight", "dpi": 300}


# -----------------------------
# Helpers
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Virtual subslices analysis from transcript CSV")
    p.add_argument("--csv", required=True, help="Path to transcript CSV file")
    p.add_argument("--outdir", default="virtual_subslices_out", help="Output directory")

    p.add_argument("--gene-col", default="gene", help="Gene column name")
    p.add_argument("--x-col", default="x", help="X coordinate column name")
    p.add_argument("--y-col", default="y", help="Y coordinate column name")
    p.add_argument("--z-col", default="z", help="Z coordinate column name")

    p.add_argument("--gridsize", type=int, default=5, help="Grid size in um for main plots")
    p.add_argument("--n-workers", type=int, default=16, help="Number of workers for ovrlpy.Ovrlp")
    p.add_argument("--random-state", type=int, default=42, help="Random state")
    p.add_argument("--min-transcripts", type=int, default=20, help="Minimum transcripts for fit_transcripts")
    p.add_argument(
        "--doublet-threshold",
        type=float,
        default=1.5,
        help="Threshold used for VSI visualization/doublet panels",
    )
    p.add_argument(
        "--doublet-min-signal",
        type=float,
        default=3,
        help="Minimum signal used in detect_doublets",
    )
    p.add_argument(
        "--integrity-sigma",
        type=float,
        default=2,
        help="integrity_sigma used in detect_doublets",
    )
    p.add_argument(
        "--max-doublets-to-plot",
        type=int,
        default=3,
        help="How many detected doublets to visualize",
    )
    p.add_argument(
        "--window-size",
        type=float,
        default=40,
        help="Half window size for doublet zoom-ins",
    )
    return p.parse_args()


def ensure_required_columns(df: pl.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}\nAvailable columns: {df.columns}")


def load_transcripts(
    csv_path: str,
    *,
    gene_col: str,
    x_col: str,
    y_col: str,
    z_col: str,
) -> pl.DataFrame:
    df = pl.read_csv(csv_path)

    print("Columns:", df.columns)  # 디버깅용

    ensure_required_columns(df, [gene_col, x_col, y_col, z_col])

    df = df.select(
        pl.col(gene_col).cast(pl.String).alias("gene"),
        pl.col(x_col).cast(pl.Float64).alias("x"),
        pl.col(y_col).cast(pl.Float64).alias("y"),
        pl.col(z_col).cast(pl.Float64).alias("z"),
    )

    df = df.with_columns(
        (pl.col("x") - pl.col("x").min()).alias("x"),
        (pl.col("y") - pl.col("y").min()).alias("y"),
    )

    return df

def pixel_gene_count(df: pl.DataFrame, gridsize: int, coord: list[str] = ["x", "y"]) -> np.ndarray:
    pixelcount = (
        df.with_columns(pl.col(c).floordiv(gridsize).cast(int) for c in coord)
        .group_by(coord)
        .len(name="count")
    )
    count = (
        coo_array(
            (
                pixelcount["count"].to_numpy(),
                (pixelcount[coord[0]].to_numpy(), pixelcount[coord[1]].to_numpy()),
            )
        )
        .toarray()
        .astype(float)
    )
    count[count == 0] = np.nan
    return count


def save_pickle(obj, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def plot_counts(arr: np.ndarray, ax: plt.Axes, title: str):
    im = ax.imshow(arr.T, cmap="viridis", origin="lower", vmin=0, vmax=20)
    ax.set_axis_off()
    ax.set_title(title)
    ax.invert_xaxis()
    ax.invert_yaxis()
    return im


def plot_z(arr: np.ndarray, ax: plt.Axes, title: str):
    im = ax.imshow(arr.T, cmap="Spectral", origin="lower", vmin=0, vmax=8)
    ax.set_axis_off()
    ax.set_title(title)
    ax.invert_xaxis()
    ax.invert_yaxis()
    return im


def plot_vsi(ovrlp_obj, ax: plt.Axes, *, threshold: float = 2):
    from ovrlpy._plotting import _plot_signal_integrity

    img = _plot_signal_integrity(ax, ovrlp_obj.integrity_map, ovrlp_obj.signal_map, threshold)
    ax.set(xticks=[], yticks=[], aspect="equal")
    return img


def plot_doublet(axs, ovrlps, x: float, y: float, *, threshold: float = 2, window_size: float = 40):
    for ax, ovrlp_obj in zip(axs, ovrlps):
        _ = plot_vsi(ovrlp_obj, ax, threshold=threshold)
        ax.set_xlim(x - window_size, x + window_size)
        ax.set_ylim(y - window_size, y + window_size)


# -----------------------------
# Main workflow
# -----------------------------
def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    transcripts = load_transcripts(
        args.csv,
        gene_col=args.gene_col,
        x_col=args.x_col,
        y_col=args.y_col,
        z_col=args.z_col,
    )
    print(f"Loaded {transcripts.shape[0]:,} transcripts from {args.csv}")

    # ---- z reorientation / mean elevation ----
    gridsize = args.gridsize
    pixel_elevation = (
        transcripts.lazy()
        .with_columns(pl.col(c).floordiv(gridsize).cast(int) for c in ["x", "y"])
        .group_by(["x", "y"])
        .agg(pl.col("z").mean())
        .collect()
    )

    mean_elevation = coo_array(
        (
            pixel_elevation["z"].to_numpy(),
            (pixel_elevation["x"].to_numpy(), pixel_elevation["y"].to_numpy()),
        )
    ).toarray()
    mean_elevation[mean_elevation == 0] = np.nan

    z_range = (float(np.nanmin(mean_elevation)), float(np.nanmax(mean_elevation)))
    print(f"Mean elevation z-range: {z_range}")

    # Histogram of mean elevation
    fig, ax = plt.subplots(figsize=(6, 3.2))
    sns.histplot(mean_elevation[~np.isnan(mean_elevation)], binwidth=0.5, ax=ax)
    ax.set(xlabel="mean z [um]", ylabel=f"# bins ({gridsize} x {gridsize} um)")
    fig.savefig(outdir / "mean_elevation_hist.png", **SAVEFIG_KWARGS)
    plt.close(fig)

    # Global z heatmap
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(9 * CM, 5.5 * CM))
        im = ax.imshow(mean_elevation.T, cmap="Spectral", origin="lower", vmin=0, vmax=8)
        plt.colorbar(im, label="mean z [um]")
        ax.set_axis_off()
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.add_artist(ScaleBar(gridsize, units="um", frameon=False, color="w", loc="lower right"))
        fig.savefig(outdir / "global_z.png", **SAVEFIG_KWARGS)
        plt.close(fig)

    # Naive vs virtual slicing counts
    naive_threshold = float(np.nanmean(mean_elevation))
    top_count = pixel_gene_count(transcripts.filter(pl.col("z") > naive_threshold), gridsize)
    bottom_count = pixel_gene_count(transcripts.filter(pl.col("z") < naive_threshold), gridsize)

    transcripts_proc = ovrlpy.process_coordinates(transcripts)
    top_count_virtual = pixel_gene_count(
        transcripts_proc.filter(pl.col("z") > pl.col("z_center")), gridsize
    )
    bottom_count_virtual = pixel_gene_count(
        transcripts_proc.filter(pl.col("z") < pl.col("z_center")), gridsize
    )

    with plt.style.context("dark_background"):
        fig = plt.figure(figsize=(9 * CM, 5.5 * CM), constrained_layout=True)
        gs = fig.add_gridspec(nrows=2, ncols=3, width_ratios=[19, 19, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[0, 1])
        ax4 = fig.add_subplot(gs[1, 1])
        cax = fig.add_subplot(gs[:, -1])

        _ = plot_counts(top_count, ax1, "top - naive")
        im = plot_counts(bottom_count, ax2, "bottom - naive")
        _ = plot_counts(top_count_virtual, ax3, "top - virtual")
        _ = plot_counts(bottom_count_virtual, ax4, "bottom - virtual")

        fig.colorbar(im, cax=cax, label="counts")
        ax2.add_artist(ScaleBar(gridsize, units="um", frameon=False, color="w", loc="lower right"))
        fig.savefig(outdir / "slicing_strategies.png", **SAVEFIG_KWARGS)
        plt.close(fig)

    # Different grid sizes
    from ovrlpy._subslicing import _assign_xy, _mean_elevation, _message_passing

    elevations = {}
    for gs in [1, 2, 5, 10]:
        assigned_out = _assign_xy(transcripts_proc, gridsize=gs, shift=False)

        # ovrlpy 버전에 따라 _assign_xy가 tuple을 반환할 수 있음
        if isinstance(assigned_out, tuple):
            assigned_df = None
            for obj in assigned_out:
                if hasattr(obj, "columns") and "x_pixel" in obj.columns:
                    assigned_df = obj
                    break
            if assigned_df is None:
                raise ValueError(
                    f"_assign_xy returned a tuple, but no dataframe with 'x_pixel' was found. "
                    f"Returned types: {[type(x) for x in assigned_out]}"
                )
        else:
            assigned_df = assigned_out
    
        elevation = _mean_elevation(assigned_df, "z", dtype=pl.Float32)
        elevations[gs] = _message_passing(elevation, n_iter=20)

    with plt.style.context("dark_background"):
        fig = plt.figure(figsize=(9 * CM, 5.5 * CM), constrained_layout=True)
        gs = fig.add_gridspec(nrows=2, ncols=3, width_ratios=[19, 19, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[0, 1])
        ax4 = fig.add_subplot(gs[1, 1])
        cax = fig.add_subplot(gs[:, -1])

        for (size, elevation), ax in zip(elevations.items(), [ax1, ax2, ax3, ax4]):
            im = plot_z(elevation, ax, f"gridsize: {size} um")

        fig.colorbar(im, cax=cax, label="z-split [um]")
        ax4.add_artist(ScaleBar(size, units="um", frameon=False, color="w", loc="lower left"))
        fig.savefig(outdir / "grid_sizes.png", **SAVEFIG_KWARGS)
        plt.close(fig)

    # Build whole / top / bottom Ovrlp objects directly from the same transcript table.
    transcripts_split = ovrlpy.process_coordinates(transcripts)
    transcripts_bottom = transcripts_split.filter(pl.col("z") < pl.col("z_center")).select(["gene", "x", "y", "z"])
    transcripts_top = transcripts_split.filter(pl.col("z") > pl.col("z_center")).select(["gene", "x", "y", "z"])

    print(f"Top subslice transcripts: {transcripts_top.shape[0]:,}")
    print(f"Bottom subslice transcripts: {transcripts_bottom.shape[0]:,}")

    brain_whole = ovrlpy.Ovrlp(transcripts, n_workers=args.n_workers, random_state=args.random_state)
    brain_whole.process_coordinates(shift=False)
    brain_whole.fit_transcripts(min_transcripts=args.min_transcripts, fit_umap=False)
    brain_whole.compute_VSI()
    save_pickle(brain_whole, outdir / "whole_slice.pickle")

    brain_bottom = ovrlpy.Ovrlp(transcripts_bottom, n_workers=args.n_workers, random_state=args.random_state)
    brain_bottom.process_coordinates(shift=False)
    brain_bottom.fit_transcripts(min_transcripts=args.min_transcripts, fit_umap=False)
    brain_bottom.compute_VSI()
    save_pickle(brain_bottom, outdir / "bottom_subslice.pickle")

    brain_top = ovrlpy.Ovrlp(transcripts_top, n_workers=args.n_workers, random_state=args.random_state)
    brain_top.process_coordinates(shift=False)
    brain_top.fit_transcripts(min_transcripts=args.min_transcripts, fit_umap=False)
    brain_top.compute_VSI()
    save_pickle(brain_top, outdir / "top_subslice.pickle")

    # Histogram overview
    t = args.doublet_threshold
    xlim = (0, 1)
    kwargs = dict(bins=100, range=xlim, density=True, histtype="step")

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.hist(brain_whole.integrity_map[brain_whole.signal_map > t], label="whole", **kwargs)
    ax.hist(brain_bottom.integrity_map[brain_bottom.signal_map > t], label="bottom", **kwargs)
    ax.hist(brain_top.integrity_map[brain_top.signal_map > t], label="top", **kwargs)
    ax.set(ylabel="Density", yticks=[], xlabel="vertical signal integrity", xlim=xlim)
    ax.legend()
    fig.savefig(outdir / "integrity_histogram_overview.png", **SAVEFIG_KWARGS)
    plt.close(fig)

    # Downsample whole slice to same size as top slice
    brain_sub = ovrlpy.Ovrlp(
        transcripts.sample(transcripts_top.shape[0], seed=args.random_state),
        n_workers=args.n_workers,
        random_state=args.random_state,
    )
    brain_sub.process_coordinates(shift=False)
    brain_sub.fit_transcripts(min_transcripts=args.min_transcripts, fit_umap=False)
    brain_sub.compute_VSI()
    save_pickle(brain_sub, outdir / "whole_downsampled.pickle")

    # Downsample bottom slice to match the top slice size for a fairer comparison.
    brain_bottom_downsampled = ovrlpy.Ovrlp(
        transcripts_bottom.sample(min(transcripts_top.shape[0], transcripts_bottom.shape[0]), seed=args.random_state),
        n_workers=args.n_workers,
        random_state=args.random_state,
    )
    brain_bottom_downsampled.process_coordinates(shift=False)
    brain_bottom_downsampled.fit_transcripts(min_transcripts=args.min_transcripts, fit_umap=False)
    brain_bottom_downsampled.compute_VSI()
    save_pickle(brain_bottom_downsampled, outdir / "bottom_downsampled.pickle")

    # Detect doublets in whole downsampled object.
    doublets = brain_sub.detect_doublets(
        min_signal=args.doublet_min_signal,
        integrity_sigma=args.integrity_sigma,
    )
    doublets.write_csv(outdir / "detected_doublets.csv")
    print(f"Detected doublets: {doublets.shape[0]:,}")

    scalebar_kwargs = dict(dx=1, units="um", loc="lower left", frameon=False, color="w")

    # Signal integrity panels
    labels = "pqrstuvwxyz"
    n_to_plot = min(args.max_doublets_to_plot, doublets.shape[0])

    with plt.style.context("dark_background"):
        fig, axs = plt.subplots(ncols=4, figsize=(18 * CM, 4.5 * CM), width_ratios=[6, 6, 6, 0.3])

        _ = plot_vsi(brain_sub, axs[0], threshold=t)
        _ = plot_vsi(brain_top, axs[1], threshold=t)
        img = plot_vsi(brain_bottom, axs[2], threshold=t)

        for ax in axs[:3]:
            ax.set_axis_off()
            ax.invert_xaxis()
            ax.invert_yaxis()

        axs[0].add_artist(ScaleBar(**scalebar_kwargs))
        axs[0].set_title("Whole slice")
        axs[1].set_title("Top subslice")
        axs[2].set_title("Bottom subslice")

        _ = fig.colorbar(img, cax=axs[3], label="vertical signal integrity")

        for l, i in zip(labels, range(n_to_plot)):
            x, y = doublets.select(["x", "y"]).row(i)
            for ax in axs[:3]:
                ax.text(x + 100, y, l, color="lime", fontsize=6, va="center")
                ax.scatter(x, y, marker="s", s=5, edgecolors="lime", facecolors="None", lw=1)

        fig.savefig(outdir / "signal_integrity.png", **SAVEFIG_KWARGS)
        plt.close(fig)

    # Integrity histogram (ratio plots)
    from ovrlpy._plotting import BIH_CMAP as cmap

    with plt.style.context("dark_background"):
        fig, axs = plt.subplots(nrows=3, figsize=(9 * CM, 9 * CM), sharex=True)

        density_sub, bins, _ = axs[0].hist(
            brain_sub.integrity_map[brain_sub.signal_map > t],
            label="whole slice",
            color="red",
            **kwargs,
        )
        density_bottom, *_ = axs[0].hist(
            brain_bottom.integrity_map[brain_bottom.signal_map > t],
            label="bottom subslice",
            color="yellow",
            **kwargs,
        )
        density_top, *_ = axs[0].hist(
            brain_top.integrity_map[brain_top.signal_map > t],
            label="top subslice",
            color="cyan",
            **kwargs,
        )
        axs[0].set(ylabel="Density", yticks=[], xlabel=None, xlim=xlim)
        axs[0].legend(frameon=False)

        hist_quotient = np.divide(density_sub, density_top, out=np.full_like(density_sub, np.nan), where=density_top != 0)
        bars = axs[1].bar(bins[:-1], hist_quotient, width=0.01, align="edge", lw=0)
        for bar, color in zip(bars, cmap(bins[:-1])):
            bar.set_color(color)
        axs[1].set(yscale="log", ylim=(0.1, 100))
        axs[1].set_ylabel(r"$\frac{\mathrm{Density(whole)}}{\mathrm{Density(top)}}$", fontsize=9)
        axs[1].hlines(1, 0, 1, linestyles="dashed", color="lime", lw=1)

        hist_quotient = np.divide(density_sub, density_bottom, out=np.full_like(density_sub, np.nan), where=density_bottom != 0)
        bars = axs[2].bar(bins[:-1], hist_quotient, width=0.01, align="edge", lw=0)
        for bar, color in zip(bars, cmap(bins[:-1])):
            bar.set_color(color)
        axs[2].set(yscale="log", xlabel="vertical signal integrity", ylim=(0.1, 100))
        axs[2].set_ylabel(r"$\frac{\mathrm{Density(whole)}}{\mathrm{Density(bottom)}}$", fontsize=9)
        axs[2].hlines(1, 0, 1, linestyles="dashed", color="lime", lw=1)

        fig.savefig(outdir / "integrity_histogram.png", **SAVEFIG_KWARGS)
        plt.close(fig)

    # Zoomed doublet panels
    if n_to_plot > 0:
        with plt.style.context("dark_background"):
            fig, axs = plt.subplots(nrows=3, ncols=n_to_plot, figsize=(3 * n_to_plot * CM, 9 * CM), squeeze=False)
            for i in range(n_to_plot):
                x, y = doublets.select(["x", "y"]).row(i)
                plot_doublet(
                    axs[:, i],
                    [brain_sub, brain_top, brain_bottom],
                    x,
                    y,
                    threshold=t,
                    window_size=args.window_size,
                )
                for ax in axs[:, i]:
                    ax.set_axis_off()
                    ax.invert_xaxis()
                    ax.invert_yaxis()

            axs[0, 0].add_artist(ScaleBar(**scalebar_kwargs))
            fig.savefig(outdir / "doublets.png", **SAVEFIG_KWARGS)
            plt.close(fig)

    print(f"Done. Outputs saved in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
