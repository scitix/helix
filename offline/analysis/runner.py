from __future__ import annotations

from argparse import Namespace

from .config import OfflineReportConfig
from .pipeline import run_offline_report


def run_analysis_from_args(args: Namespace) -> None:
    config = OfflineReportConfig(
        data_dir=str(args.data_dir),
        out_dir=str(args.out_dir),
        rank=getattr(args, "rank", None),
        data_kind=str(getattr(args, "data_kind", "auto")),
        topk_params_for_heatmap=int(getattr(args, "topk_params", 120)),
        topk_params_for_locality=int(getattr(args, "param_internal_topk", 12)),
        internal_index_buckets=int(getattr(args, "internal_buckets", 128)),
        locality_jaccard_threshold=float(getattr(args, "internal_jaccard_thresh", 0.6)),
    )
    run_offline_report(config)
