from __future__ import annotations

import torch

from ..common.convert import indices_and_param_to_dense_nan
from ..megatron.gather_indices import fetch_sparse_indices


def maybe_sparse_param_to_dense_nan(name: str, param: torch.Tensor) -> torch.Tensor:
    """If Megatron sparse indices are attached to this param, build a dense NaN-filled tensor
    that only contains updated entries on those indices. Otherwise return the original tensor.

    Convention: NaN means "unchanged" and will be applied on rollout side via sparse_copy_context().
    """
    # expert_bias is treated as a normal param/buffer and does not participate in sparse update.
    if "expert_bias" in name:
        return param

    indices = fetch_sparse_indices(param)
    if indices is None:
        raise RuntimeError(f"[sparse-update] {name}: sparse indices is None (expected Tensor).")

    dense_nan_tensor = indices_and_param_to_dense_nan(indices, param)
    _copy_tp_attrs(param, dense_nan_tensor)
    return dense_nan_tensor


def _copy_tp_attrs(src: torch.Tensor, dst: torch.Tensor) -> None:
    """Preserve Megatron TP-related attributes required by `all_gather_param()`."""
    # all_gather_param() asserts tensor_model_parallel exists.
    defaults = {
        "tensor_model_parallel": False,
        "parallel_mode": None,
        "partition_dim": 0,
        "partition_stride": 1,
    }
    for attr, default in defaults.items():
        val = getattr(src, attr, default)
        setattr(dst, attr, val)
