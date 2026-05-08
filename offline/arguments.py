"""Unified CLI arguments for offline merge and offline analysis.

offline/main.py is the only CLI entrypoint. It dispatches into:
- offline.merge (multi-rank consolidation)
- offline.analysis (report generation)
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "SparseRL offline tooling. Use --mode to select 'merge' or 'analysis'. "
            "Run from repo root with PYTHONPATH=."
        )
    )
    p.add_argument(
        "--mode",
        required=True,
        choices=["merge", "analysis"],
        help="Which offline pipeline to run.",
    )

    # Common (shared)
    common = p.add_argument_group("Common")
    common.add_argument(
        "--data-dir",
        required=True,
        help="Input directory containing rank*_info.json and rank*_indices_*.pt",
    )
    common.add_argument(
        "--out-dir",
        required=True,
        help="Output directory (for merge: merged dump dir; for analysis: report dir).",
    )

    # Merge args
    mg = p.add_argument_group("Merge (multi-rank -> single logical rank)")
    mg.add_argument(
        "--out-rank",
        type=int,
        default=0,
        help="Synthetic output rank id in filenames (default: 0).",
    )
    mg.add_argument(
        "--intervals",
        type=int,
        nargs="*",
        default=None,
        metavar="N",
        help=(
            "Intervals to merge (space-separated). If omitted, merges all intervals found under "
            "--data-dir. Each interval is one task."
        ),
    )
    mg.add_argument(
        "--skip-info",
        action="store_true",
        help="Do not write rank*_info.json (useful when merging intervals in chunks).",
    )
    mg.add_argument(
        "--merge-device",
        default="cpu",
        help="Merge-time device for tensor ops. Examples: cpu (default), cuda.",
    )
    mg.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help=(
            "GPU worker process count when --merge-device is cuda. "
            "Intervals are distributed across GPUs in round-robin. Default: 1."
        ),
    )
    mg.add_argument(
        "--estimate-interval-size",
        action="store_true",
        help="Print an estimated merged .pt size per interval using only rank*_info.json.",
    )
    mg.add_argument(
        "--assumed-events-per-interval",
        type=int,
        default=1,
        help=(
            "Used with --estimate-interval-size. rank*_info.json does not encode events per interval; "
            "set your expected count (default: 1)."
        ),
    )

    # Analysis args
    an = p.add_argument_group("Analysis (dump -> report)")
    an.add_argument(
        "--rank",
        type=int,
        default=None,
        help="If set, only analyze a single rank (e.g. --rank 0).",
    )
    an.add_argument(
        "--data-kind",
        choices=["auto", "single", "merged"],
        default="auto",
        help="Label input kind for report rendering.",
    )
    an.add_argument(
        "--topk-params",
        type=int,
        default=120,
        help="Top-K params for sparsity heatmap.",
    )
    an.add_argument(
        "--param-internal-topk",
        type=int,
        default=12,
        help="How many params get internal locality plots (0 disables).",
    )
    an.add_argument(
        "--internal-buckets",
        type=int,
        default=128,
        help="Bucket count for param-internal index distribution heatmaps.",
    )
    an.add_argument(
        "--internal-jaccard-thresh",
        type=float,
        default=0.6,
        help="Threshold on mean consecutive-step Jaccard to flag locality_like=1.",
    )

    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)
