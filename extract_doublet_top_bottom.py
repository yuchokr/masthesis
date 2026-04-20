#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import ovrlpy
import polars as pl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract local top/bottom transcripts for each detected doublet"
    )
    p.add_argument("--csv", required=True, help="Input transcript CSV")
    p.add_argument("--doublets", required=True, help="detected_doublets.csv from ovrlpy")
    p.add_argument("--outdir", required=True, help="Output directory")

    p.add_argument("--gene-col", default="gene", help="Gene column name")
    p.add_argument("--x-col", default="x", help="X coordinate column name")
    p.add_argument("--y-col", default="y", help="Y coordinate column name")
    p.add_argument("--z-col", default="z", help="Z coordinate column name")

    p.add_argument(
        "--window-size",
        type=float,
        default=40.0,
        help="Half window size around each doublet center, in same units as x/y (default: 40)",
    )
    p.add_argument(
        "--min-local-transcripts",
        type=int,
        default=1,
        help="Minimum number of transcripts required to keep a local top/bottom subset",
    )
    return p.parse_args()


def ensure_required_columns(df: pl.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nAvailable columns: {df.columns}")


def load_transcripts(
    csv_path: str,
    *,
    gene_col: str,
    x_col: str,
    y_col: str,
    z_col: str,
) -> pl.DataFrame:
    df = pl.read_csv(csv_path)
    ensure_required_columns(df, [gene_col, x_col, y_col, z_col])

    df = df.select(
        pl.col(gene_col).cast(pl.String).alias("gene"),
        pl.col(x_col).cast(pl.Float64).alias("x"),
        pl.col(y_col).cast(pl.Float64).alias("y"),
        pl.col(z_col).cast(pl.Float64).alias("z"),
    )

    # ovrlpy examples often shift coordinates so ROI starts at 0,0
    df = df.with_columns(
        (pl.col("x") - pl.col("x").min()).alias("x"),
        (pl.col("y") - pl.col("y").min()).alias("y"),
    )
    return df


def load_doublets(doublet_csv: str) -> pl.DataFrame:
    df = pl.read_csv(doublet_csv)

    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError(
            f"detected_doublets.csv must contain 'x' and 'y' columns. Found: {df.columns}"
        )

    # Add a stable ID if none exists
    if "doublet_id" not in df.columns:
        df = df.with_row_index("doublet_id")

    return df


def extract_local_subslice(
    transcripts_proc: pl.DataFrame,
    x0: float,
    y0: float,
    window_size: float,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    local = transcripts_proc.filter(
        (pl.col("x") >= x0 - window_size)
        & (pl.col("x") <= x0 + window_size)
        & (pl.col("y") >= y0 - window_size)
        & (pl.col("y") <= y0 + window_size)
    )

    top = local.filter(pl.col("z") > pl.col("z_center"))
    bottom = local.filter(pl.col("z") < pl.col("z_center"))

    return local, top, bottom


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading transcripts...")
    transcripts = load_transcripts(
        args.csv,
        gene_col=args.gene_col,
        x_col=args.x_col,
        y_col=args.y_col,
        z_col=args.z_col,
    )
    print(f"  transcripts: {transcripts.shape[0]:,}")

    print("Processing coordinates with ovrlpy...")
    transcripts_proc = ovrlpy.process_coordinates(transcripts)

    required_after_proc = ["gene", "x", "y", "z", "z_center"]
    ensure_required_columns(transcripts_proc, required_after_proc)

    print("Loading detected doublets...")
    doublets = load_doublets(args.doublets)
    print(f"  doublets: {doublets.shape[0]:,}")

    top_rows = []
    bottom_rows = []
    summary_rows = []

    print("Extracting local top/bottom transcripts for each doublet...")

    for row in doublets.iter_rows(named=True):
        did = int(row["doublet_id"])
        x0 = float(row["x"])
        y0 = float(row["y"])

        local, top, bottom = extract_local_subslice(
            transcripts_proc,
            x0=x0,
            y0=y0,
            window_size=args.window_size,
        )

        n_local = local.shape[0]
        n_top = top.shape[0]
        n_bottom = bottom.shape[0]

        summary_rows.append(
            {
                "doublet_id": did,
                "x": x0,
                "y": y0,
                "n_local": n_local,
                "n_top": n_top,
                "n_bottom": n_bottom,
            }
        )

        if n_top >= args.min_local_transcripts:
            top = top.with_columns(
                pl.lit(did).alias("doublet_id"),
                pl.lit(x0).alias("doublet_x"),
                pl.lit(y0).alias("doublet_y"),
            ).select(
                "doublet_id",
                "doublet_x",
                "doublet_y",
                "gene",
                "x",
                "y",
                "z",
                "z_center",
            )
            top_rows.append(top)

        if n_bottom >= args.min_local_transcripts:
            bottom = bottom.with_columns(
                pl.lit(did).alias("doublet_id"),
                pl.lit(x0).alias("doublet_x"),
                pl.lit(y0).alias("doublet_y"),
            ).select(
                "doublet_id",
                "doublet_x",
                "doublet_y",
                "gene",
                "x",
                "y",
                "z",
                "z_center",
            )
            bottom_rows.append(bottom)

    summary_df = pl.DataFrame(summary_rows)

    if top_rows:
        top_df = pl.concat(top_rows, how="vertical")
    else:
        top_df = pl.DataFrame(
            schema={
                "doublet_id": pl.Int64,
                "doublet_x": pl.Float64,
                "doublet_y": pl.Float64,
                "gene": pl.String,
                "x": pl.Float64,
                "y": pl.Float64,
                "z": pl.Float64,
                "z_center": pl.Float64,
            }
        )

    if bottom_rows:
        bottom_df = pl.concat(bottom_rows, how="vertical")
    else:
        bottom_df = pl.DataFrame(
            schema={
                "doublet_id": pl.Int64,
                "doublet_x": pl.Float64,
                "doublet_y": pl.Float64,
                "gene": pl.String,
                "x": pl.Float64,
                "y": pl.Float64,
                "z": pl.Float64,
                "z_center": pl.Float64,
            }
        )

    summary_path = outdir / "doublet_local_summary.csv"
    top_path = outdir / "doublet_top_transcripts.csv"
    bottom_path = outdir / "doublet_bottom_transcripts.csv"

    summary_df.write_csv(summary_path)
    top_df.write_csv(top_path)
    bottom_df.write_csv(bottom_path)

    print("\nDone.")
    print(f"  Summary: {summary_path}")
    print(f"  Top transcripts: {top_path}")
    print(f"  Bottom transcripts: {bottom_path}")


if __name__ == "__main__":
    main()