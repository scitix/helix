from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from .manager import SparseStatsModelManager

if TYPE_CHECKING:
    from megatron.core.distributed import DistributedDataParallel as DDP


_SPARSE_STATS_MANAGER = None


def get_sparse_stats_manager() -> SparseStatsModelManager:
    global _SPARSE_STATS_MANAGER
    return _SPARSE_STATS_MANAGER


def set_sparse_stats_manager(sparse_stats_manager: SparseStatsModelManager):
    global _SPARSE_STATS_MANAGER
    _SPARSE_STATS_MANAGER = sparse_stats_manager


def _get_sparse_stats_saved_dir(model_tag: str):
    sparserl_root = Path(__file__).resolve().parents[3]
    default_saved_dir = str(sparserl_root / "data")
    sparse_saved_dir = os.getenv("SPARSE_STATS_SAVED_DIR", default_saved_dir)
    return os.path.join(sparse_saved_dir, model_tag)


def save_model_sparse_info(
    models: list[DDP] | DDP, model_tag: str, flush_indices_interval: int = 1
):
    sparse_stats_manager: SparseStatsModelManager | None = get_sparse_stats_manager()
    if sparse_stats_manager is None:
        saved_dir = _get_sparse_stats_saved_dir(model_tag)
        sparse_stats_manager = SparseStatsModelManager(
            models, model_tag, saved_dir, flush_indices_interval
        )
        set_sparse_stats_manager(sparse_stats_manager)

    sparse_stats_manager.save_sparse_info()
