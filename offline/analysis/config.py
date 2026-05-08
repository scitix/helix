"""CLI / run configuration for offline sparse-stats reporting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OfflineReportConfig:
    """User-tunable knobs for `pipeline.run_offline_report`."""

    data_dir: str
    out_dir: str
    rank: int | None = None
    # "auto" (infer from rank*_info.json), "single" (raw single-rank dump), "merged" (consolidated)
    data_kind: str = "auto"
    topk_params_for_heatmap: int = 120
    topk_params_for_locality: int = 12
    internal_index_buckets: int = 128
    locality_jaccard_threshold: float = 0.6
