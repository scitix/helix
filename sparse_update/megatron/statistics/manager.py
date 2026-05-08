from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Final

import torch
from torch.nn import Parameter

from ..manager import get_sparse_manager
from ..observer import SparseObserver
from ..sparse_data import SparseData

if TYPE_CHECKING:
    from megatron.core.distributed import DistributedDataParallel as DDP


class _SparseStatsParamManager:
    DEFAULT_TP_ATTRS: Final[dict[str, Any]] = {
        "tensor_model_parallel": False,
        "parallel_mode": None,
        "partition_dim": 0,
        "partition_stride": 1,
    }

    def __init__(self, name: str, param: Parameter):
        self.name = name
        self.param = param

        # Megatron convention: expert-parallel params have param.allreduce == False.
        self._is_moe_param: bool = not getattr(param, "allreduce", True)
        self._init_tp_attrs()

        self.sparse_manager: SparseObserver | None = get_sparse_manager(param)
        self.param_range = None
        if self.sparse_manager is not None:
            self.param_range = self.sparse_manager.param_range

    def _init_tp_attrs(self):
        self.tp_attrs = {}
        for attr, default in self.DEFAULT_TP_ATTRS.items():
            self.tp_attrs[attr] = getattr(self.param, attr, default)

    @property
    def parameter_info(self) -> dict[str, Any]:
        info = {
            "param.shape": list(self.param.shape),
            "param.dtype": str(self.param.dtype),
            "param.numel": self.param.numel(),
            "is_moe_param": self._is_moe_param,
            "tp_attrs": self.tp_attrs,
            "dp_param_range": None,
        }
        if self.param_range is not None:
            info["dp_param_range"] = [self.param_range.start, self.param_range.end]

        return info

    @property
    def sparse_history(self) -> list[SparseData] | None:
        if self.sparse_manager is not None:
            history_sparse_data = self.sparse_manager.history_sparse_data
            self.sparse_manager.clear_history()
            return history_sparse_data
        return None


class SparseStatsModelManager:
    def __init__(
        self,
        models: list[DDP],
        model_tag: str,
        saved_dir: os.PathLike | str,
        flush_interval: int = 1,
    ):
        if not isinstance(models, list):
            models = [models]

        self.models = models
        self.model_tag = model_tag

        self.saved_dir = saved_dir
        self.rank = torch.distributed.get_rank()

        self.flush_interval = flush_interval

        self.saved_interval = -1
        self.param_to_stats: dict[Parameter, _SparseStatsParamManager] = self.init()

        self._save_model_info()

    def init(self):
        param_to_stats: dict[Parameter, _SparseStatsParamManager] = {}
        for model in self.models:
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                param_to_stats[param] = _SparseStatsParamManager(name, param)
        return param_to_stats

    def _save_model_info(self) -> dict[str, Any]:
        """Save baisc model info to json file. Only save once."""
        if getattr(self, "model_info_saved", False):
            return
        self.model_info_saved = True

        os.makedirs(self.saved_dir, exist_ok=True)

        from .distributed_info import DistributedInfo

        distributed_info = DistributedInfo()

        params_info: dict[str, dict[str, Any]] = {}
        for _, stats in self.param_to_stats.items():
            params_info[stats.name] = stats.parameter_info

        model_info: dict[str, dict[str, Any]] = {
            "distributed_info": distributed_info.distributed_info,
            "params_info": params_info,
        }

        file_path = os.path.join(self.saved_dir, f"rank{self.rank}_info.json")
        with open(file_path, "w") as f:
            f.write(json.dumps(model_info, indent=4))

    def save_sparse_info(self):
        self.saved_interval += 1
        # When the first time update_weights() called, Megatron optimizer is not initialized yet.
        if self.saved_interval == 0:
            return

        if self.saved_interval % self.flush_interval != 0:
            return

        os.makedirs(self.saved_dir, exist_ok=True)

        names_to_sparse_history: dict[str, list[SparseData]] = {}
        for _, stats in self.param_to_stats.items():
            sparse_manager: SparseObserver | None = stats.sparse_manager
            if sparse_manager is None:
                continue
            sparse_history: list[SparseData] = sparse_manager.history_sparse_data
            names_to_sparse_history[stats.name] = sparse_history
            sparse_manager.clear_history()

        tensor_path = os.path.join(
            self.saved_dir, f"rank{self.rank}_indices_{self.saved_interval}.pt"
        )

        torch.save(names_to_sparse_history, tensor_path)
