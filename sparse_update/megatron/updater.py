import torch

from ..common.compare import get_sparse_diff_indices
from .manager import BaseSparseManager


class SparseUpdater(BaseSparseManager):
    def __init__(self, model_param, param_range) -> None:
        super().__init__(model_param, param_range)
        self.dp_gathered_indices: torch.Tensor | None = None
        self.reset()

    def before_copy(self, shard_model_weight: torch.Tensor) -> None:
        """Save the previous model weight.
        TODO:: This could be optimized by using the param.grad buffer.
        """
        self.prev_model_weight = shard_model_weight.detach().clone()

    @torch.no_grad()
    def after_copy(self, shard_model_weight: torch.Tensor, shard_main_weight: torch.Tensor) -> None:
        assert self.prev_model_weight is not None
        model_weight_indices = get_sparse_diff_indices(
            shard_model_weight, self.prev_model_weight, offset=self.param_range.start
        )

        self.union_indices(model_weight_indices)
        self.prev_model_weight = None
