"""
This file contains the kernel functions for converting between dense and sparse tensors.

1. indices + param -> dense_nan tensor
2. dense_nan tensor -> indices + values
"""

from collections.abc import Sequence

import torch


def indices_and_param_to_dense_nan(indices: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    dense_nan_tensor = torch.full_like(param, float("nan"))
    if indices.numel() == 0:
        return dense_nan_tensor

    flat = dense_nan_tensor.view(-1)
    src_flat = param.view(-1)
    flat[indices] = src_flat[indices]
    return dense_nan_tensor


def indices_and_values_to_dense_nan(
    indices: torch.Tensor, values: torch.Tensor, origin_shape: torch.Size | Sequence[int]
) -> torch.Tensor:
    origin_shape = (
        origin_shape if isinstance(origin_shape, torch.Size) else torch.Size(origin_shape)
    )
    dense_nan_tensor = torch.full(
        origin_shape, float("nan"), device=values.device, dtype=values.dtype
    )
    flat = dense_nan_tensor.view(-1)
    flat[indices] = values
    return dense_nan_tensor


def dense_nan_to_sparse_tensor(dense_nan_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = dense_nan_tensor.view(-1)
    idx64 = torch.where(~torch.isnan(flat))[0]
    values = flat[idx64]
    return idx64.to(torch.int32), values
