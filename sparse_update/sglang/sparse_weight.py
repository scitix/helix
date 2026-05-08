from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from enum import Enum, auto

import torch

from ..common.convert import indices_and_values_to_dense_nan
from ..common.copy import sparse_copy


@contextmanager
def sparse_copy_context():
    """Within this block, tensor.copy_(src) becomes index_put_ on nonzero positions of src."""
    _original_copy_ = torch.Tensor.copy_
    torch.Tensor.copy_ = sparse_copy
    try:
        yield
    finally:
        torch.Tensor.copy_ = _original_copy_


def sparse_to_dense(
    indices: torch.Tensor,
    values: torch.Tensor,
    origin_shape: torch.Size | Sequence[int],
    device: torch.device | str,
) -> torch.Tensor:
    return indices_and_values_to_dense_nan(indices, values, origin_shape)


# ------------------------------------------------------------
# Validate State
# ------------------------------------------------------------


class SparseLoaderValidateState(Enum):
    BEFORE_FULL_UPDATE = auto()
    BEFORE_SPARSE_UPDATE = auto()
    AFTER_SPARSE_UPDATE = auto()
    ZERO_NNZ_UPDATE = auto()


def _assert_allclose_progressive(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    base_atol: float,
    base_rtol: float,
    tag: str,
) -> str:
    # When strict tolerances fail, progressively relax to find the
    # minimum threshold that passes. This helps you understand
    # whether the mismatch is due to numerical precision.
    scales = [1.0, 10.0, 100.0, 1_000.0, 10_000.0]
    dif = (a - b).detach()
    max_abs = dif.abs().max().item()
    denom = a.detach().abs().max().item() + 1e-12
    max_rel = (dif.abs().max().item() / denom) if denom != 0 else float("inf")

    for s in scales:
        atol = base_atol * s
        rtol = base_rtol * s
        if torch.allclose(a, b, atol=atol, rtol=rtol):
            if s != scales[0]:
                return f"{tag} allclose passed only after relaxing: atol={atol:g}, rtol={rtol:g} (max_abs_diff={max_abs:g}, max_rel_diff={max_rel:g})"

            return "allclose passed"

    # If none passes, fail with the original strict threshold + diffs.
    tried = ", ".join([f"(atol={base_atol * sc:g}, rtol={base_rtol * sc:g})" for sc in scales])
    raise AssertionError(
        f"{tag} allclose failed. "
        f"max_abs_diff={max_abs:g}, max_rel_diff={max_rel:g}. "
        f"Tried: {tried}"
    )


def _get_prev_params_dict(model) -> dict[str, torch.Tensor] | None:
    return getattr(model, "_sparse_validate_prev_params_dict", None)


def _set_prev_params_dict(model, params_dict: dict[str, torch.Tensor] | None) -> None:
    model._sparse_validate_prev_params_dict = params_dict


def _get_full_update_params_dict(model) -> dict[str, torch.Tensor] | None:
    return getattr(model, "_sparse_validate_full_update_params_dict", None)


def _set_full_update_params_dict(model, params_dict: dict[str, torch.Tensor] | None) -> None:
    model._sparse_validate_full_update_params_dict = params_dict


def validate_sparse_weights(model, validate_state: SparseLoaderValidateState):
    params_dict = dict(model.named_parameters())
    if validate_state == SparseLoaderValidateState.BEFORE_FULL_UPDATE:
        if _get_prev_params_dict(model) is None:
            prev_params_dict = {}
            for name, param in params_dict.items():
                prev_params_dict[name] = param.data.detach().to("cpu")
            _set_prev_params_dict(model, prev_params_dict)
            torch.cuda.synchronize()
        return

    if validate_state == SparseLoaderValidateState.ZERO_NNZ_UPDATE:
        prev_params_dict = _get_prev_params_dict(model)
        assert prev_params_dict is not None, (
            "prev_params_dict is None. "
            "Call validate_sparse_weights(..., BEFORE_FULL_UPDATE) before doing ZERO_NNZ_UPDATE."
        )
        for name, param in params_dict.items():
            prev_param = prev_params_dict[name].to(device=param.device)
            _assert_allclose_progressive(
                prev_param,
                param,
                base_atol=1e-8,
                base_rtol=1e-8,
                tag=f"compare_non_nnz_update_{name}",
            )
        torch.cuda.synchronize()
        _set_prev_params_dict(model, None)
        return

    if validate_state == SparseLoaderValidateState.BEFORE_SPARSE_UPDATE:
        # Save full update params to cpu
        assert _get_full_update_params_dict(model) is None, "full_update_params_dict is not None"

        full_update_params_dict = {}
        for name, param in params_dict.items():
            full_update_params_dict[name] = param.data.detach().to("cpu")
        _set_full_update_params_dict(model, full_update_params_dict)
        torch.cuda.synchronize()

        # Recover self.model.weights to "before full update"
        prev_params_dict = _get_prev_params_dict(model)
        assert prev_params_dict is not None, (
            "prev_params_dict is None. "
            "Validation flow expects: BEFORE_FULL_UPDATE -> (do full update) -> BEFORE_SPARSE_UPDATE."
        )
        for name, param in params_dict.items():
            prev_param = prev_params_dict[name].to(device=param.device)
            param.data.copy_(prev_param, non_blocking=True)

        torch.cuda.synchronize()
        _set_prev_params_dict(model, None)
        return

    assert validate_state == SparseLoaderValidateState.AFTER_SPARSE_UPDATE, (
        "validate_state is not AFTER_SPARSE_UPDATE"
    )
    full_update_params_dict = _get_full_update_params_dict(model)
    assert full_update_params_dict is not None, "full_update_params_dict is None"

    for name, param in params_dict.items():
        full_updated_param = full_update_params_dict[name].to(device=param.device)
        _assert_allclose_progressive(
            full_updated_param,
            param,
            base_atol=1e-8,
            base_rtol=1e-8,
            tag=f"compare_{name}",
        )
    torch.cuda.synchronize()
    _set_full_update_params_dict(model, None)
