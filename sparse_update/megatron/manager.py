from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from ..common.utils import SparseState, get_sparse_state
from .sparse_data import SparseData

if TYPE_CHECKING:
    from megatron.core.optimizer.distrib_optimizer import Range


class BaseSparseManager(ABC):
    def __init__(self, model_param: torch.nn.Parameter, param_range: "Range") -> None:
        self.param: torch.nn.Parameter = model_param
        self.param_dtype = self.param.dtype
        self.param_shape = self.param.shape
        self.param_device = self.param.device
        self.param_range = param_range

        self.history_sparse_data: list[SparseData] = []

        self.reset()

    def reset(self) -> None:
        self.prev_model_weight = None
        self._indices = torch.tensor([], dtype=torch.int32, device=torch.cuda.current_device())

    @abstractmethod
    def before_copy(self, shard_model_weight: torch.Tensor) -> None: ...

    @abstractmethod
    def after_copy(
        self, shard_model_weight: torch.Tensor, shard_main_weight: torch.Tensor
    ) -> None: ...

    def union_indices(self, indices: torch.Tensor) -> None:
        if self._indices.numel() == 0:
            self._indices = indices.clone()
            return
        self._indices = torch.unique(torch.cat((self._indices, indices), dim=0), sorted=True)

    def fetch_dp_local_indices(self) -> torch.Tensor:
        indices = self._indices
        self._indices = torch.tensor([], dtype=torch.int32, device=torch.cuda.current_device())
        return indices

    def clear_history(self) -> None:
        self.history_sparse_data = []


class NoOpSparseManager(BaseSparseManager):
    def before_copy(self, shard_model_weight: torch.Tensor) -> None:
        return None

    @torch.no_grad()
    def after_copy(self, shard_model_weight: torch.Tensor, shard_main_weight: torch.Tensor) -> None:
        return None


def init_sparse_manager(
    model_param: torch.nn.Parameter,
    shard_model_weight: torch.Tensor,
    shard_main_weight: torch.Tensor,
    param_range: "Range",
):
    """Bind the model_param to the shard_model_weight and shard_main_weight, and creates a BaseSparseManager object.
    Note. This function should be called in DistributedOptimizer.init_param_groups.

    1. In DistributedOptimizer, the shard_model_weight will be created by model_param.view(-1)[param_range.start:param_range.end] for each time it updated.
        Therefore, the shard_model_weight will be different for each time it updated.
        We can find model_param by shard_main_weight.
    2. In HybridDeviceOptimizer, shard_main_weight will be created by HybridDeviceOptimizer() and it holds the shard_model_weight.
        So we can find model_param by shard_main_weight.
    """

    if shard_model_weight is not None:
        shard_model_weight._sparse_owner = model_param
    if shard_main_weight is not None:
        shard_main_weight._sparse_owner = model_param

    sparse_state = get_sparse_state()
    if sparse_state is None:
        sparse_manager = NoOpSparseManager(model_param, param_range)
    elif sparse_state in [SparseState.UPDATE, SparseState.UPDATE_AND_VALIDATE]:
        from .updater import SparseUpdater

        sparse_manager = SparseUpdater(model_param, param_range)
    else:
        assert "observe" in sparse_state.value, f"Invalid sparse state: {sparse_state}"
        from .observer import SparseObserver

        sparse_manager = SparseObserver(model_param, param_range)
    model_param._sparse_manager = sparse_manager
    return sparse_manager


def get_sparse_manager(
    model_param: torch.nn.Parameter | None = None,
    shard_model_weight: torch.Tensor | None = None,
    shard_main_weight: torch.Tensor | None = None,
) -> BaseSparseManager | None:
    if model_param is not None:
        return getattr(model_param, "_sparse_manager", None)

    if shard_model_weight is not None:
        model_param = getattr(shard_model_weight, "_sparse_owner", None)
        if model_param is not None:
            return getattr(model_param, "_sparse_manager", None)

    if shard_main_weight is not None:
        model_param = getattr(shard_main_weight, "_sparse_owner", None)
        if model_param is not None:
            return getattr(model_param, "_sparse_manager", None)

    raise ValueError(
        f"Invalid input: model_param={model_param}, shard_model_weight={shard_model_weight}, shard_main_weight={shard_main_weight}"
    )


@contextmanager
def sparse_diff_context(shard_model_weight: torch.Tensor, shard_main_weight: torch.Tensor):
    """Wrap the shard_model_weight.copy_(shard_main_weight) operation, and update the sparse diff indices."""
    sparse_manager = get_sparse_manager(
        shard_model_weight=shard_model_weight, shard_main_weight=shard_main_weight
    )

    sparse_manager.before_copy(shard_model_weight)
    try:
        yield
    finally:
        sparse_manager.after_copy(shard_model_weight, shard_main_weight)
