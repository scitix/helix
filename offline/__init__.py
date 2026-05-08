"""Offline tooling: merge multi-rank dumps and analyze dumps into reports.

Prefer running the unified CLI:

    PYTHONPATH=. python -m offline.main --mode merge ...
    PYTHONPATH=. python -m offline.main --mode analysis ...
"""

# Re-export stable APIs (imports resolved after module relocation).
from .analysis.config import OfflineReportConfig
from .analysis.pipeline import run_offline_report
from .merge.merge import merge_sparse_dumps_to_single_rank

__all__ = [
    "OfflineReportConfig",
    "merge_sparse_dumps_to_single_rank",
    "run_offline_report",
]
