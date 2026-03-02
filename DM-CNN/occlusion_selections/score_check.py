#!/usr/bin/env python3
"""
Lightweight helpers for browsing score CSVs from the command line.

Examples
--------
# Show top 20 events by score from a single file:
python3 score_check.py /path/to/file_scores.csv --top-score 20

# Find 10 events nearest score 0.085 (with pixel cut):
python3 score_check.py /path/to/file_scores.csv --near-score 0.085 --n 10 --min-pixels 500

# Load a folder of *_scores.csv and show busiest events:
python3 score_check.py /path/to/folder --top-pixels 20 --min-score 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from occlusion_selection_tools import (
    filter_range,
    load_scores_any,
    nearest,
    occlusion_cmd,
    select_top,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "path", type=Path, help="Scores CSV file or folder containing *_scores.csv."
    )
    ap.add_argument("--min-pixels", type=int, default=None)
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument(
        "--top-score", type=int, default=None, help="Show top N by signal_score."
    )
    ap.add_argument(
        "--top-pixels", type=int, default=None, help="Show top N by n_pixels."
    )
    ap.add_argument(
        "--near-score", type=float, default=None, help="Show N nearest to this score."
    )
    ap.add_argument(
        "--near-pixels",
        type=int,
        default=None,
        help="Show N nearest to this pixel count.",
    )
    ap.add_argument("--n", type=int, default=10, help="N for nearest queries.")
    ap.add_argument("--range-score-min", type=float, default=None)
    ap.add_argument("--range-score-max", type=float, default=None)
    ap.add_argument("--range-pix-min", type=int, default=None)
    ap.add_argument("--range-pix-max", type=int, default=None)
    args = ap.parse_args()

    df = load_scores_any(args.path)

    if args.top_score is not None:
        print(
            select_top(
                df,
                by="signal_score",
                n=args.top_score,
                min_pixels=args.min_pixels,
                min_score=args.min_score,
            ).to_string(index=False)
        )

    if args.top_pixels is not None:
        print(
            select_top(
                df,
                by="n_pixels",
                n=args.top_pixels,
                min_pixels=args.min_pixels,
                min_score=args.min_score,
            ).to_string(index=False)
        )

    if args.near_score is not None:
        out = nearest(
            df,
            col="signal_score",
            target=args.near_score,
            n=args.n,
            min_pixels=args.min_pixels,
            min_score=args.min_score,
        )
        print(out.to_string(index=False))
        if not out.empty:
            print(
                "Example occlusion cmd:",
                occlusion_cmd(int(out.iloc[0]["entry_number"])),
            )

    if args.near_pixels is not None:
        out = nearest(
            df,
            col="n_pixels",
            target=float(args.near_pixels),
            n=args.n,
            min_pixels=args.min_pixels,
            min_score=args.min_score,
        )
        print(out.to_string(index=False))
        if not out.empty:
            print(
                "Example occlusion cmd:",
                occlusion_cmd(int(out.iloc[0]["entry_number"])),
            )

    if any(
        v is not None
        for v in [
            args.range_score_min,
            args.range_score_max,
            args.range_pix_min,
            args.range_pix_max,
        ]
    ):
        out = filter_range(
            df,
            score_min=args.range_score_min,
            score_max=args.range_score_max,
            pixels_min=args.range_pix_min,
            pixels_max=args.range_pix_max,
        )
        print(out.head(50).to_string(index=False))


if __name__ == "__main__":
    main()
