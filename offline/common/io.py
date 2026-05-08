"""Discover rank files, load metadata, and normalize per-event change indices from checkpoints."""

from __future__ import annotations

import gc
import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

_RANK_INFO_RE = re.compile(r"^rank(?P<rank>\d+)_info\.json$")
_RANK_INDICES_RE = re.compile(r"^rank(?P<rank>\d+)_indices_(?P<interval>\d+)\.pt$")


def _register_sparse_data_pickle_alias() -> None:
    """Map legacy module name ``sparse_update.megatron.data`` to ``sparse_update.megatron.sparse_data``."""
    import sys

    legacy = "sparse_update.megatron.data"
    if legacy in sys.modules:
        return
    try:
        import sparse_update.megatron.sparse_data as sparse_data_mod
    except ImportError:
        return
    sys.modules[legacy] = sparse_data_mod


def load_torch_dict(path: str) -> dict[str, Any]:
    """Load a `torch.save` dict; compatible with PyTorch versions before `weights_only`."""
    _register_sparse_data_pickle_alias()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@dataclass(frozen=True)
class DistributedInfo:
    raw: dict[str, Any]

    @property
    def global_rank(self) -> int:
        return int(self.raw.get("global_rank", -1))

    @property
    def dp_rank(self) -> int:
        return int(self.raw.get("dp_rank", -1))

    @property
    def tp_rank(self) -> int:
        return int(self.raw.get("tp_rank", -1))

    @property
    def ep_rank(self) -> int:
        return int(self.raw.get("ep_rank", -1))


@dataclass(frozen=True)
class ParamInfo:
    name: str
    shape: tuple[int, ...]
    numel: int
    dtype_str: str
    is_moe_param: bool
    tp_attrs: dict[str, Any]
    has_sparse_diff: bool
    dp_param_range: tuple[int, int] | None


def discover_rank_files(
    data_dir: str,
    *,
    allowed_ranks: set[int] | None = None,
) -> tuple[list[int], dict[int, str], dict[int, list[tuple[int, str]]]]:
    """Return sorted ranks, paths to `rank*_info.json`, and per-rank list of (interval, indices_pt_path)."""
    files = sorted(os.listdir(data_dir))
    rank_to_info_path: dict[int, str] = {}
    rank_to_indices_paths: dict[int, list[tuple[int, str]]] = {}
    for fn in files:
        m = _RANK_INFO_RE.match(fn)
        if m:
            r = int(m.group("rank"))
            if allowed_ranks is not None and r not in allowed_ranks:
                continue
            rank_to_info_path[r] = os.path.join(data_dir, fn)
            continue
        m = _RANK_INDICES_RE.match(fn)
        if m:
            r = int(m.group("rank"))
            if allowed_ranks is not None and r not in allowed_ranks:
                continue
            interval = int(m.group("interval"))
            rank_to_indices_paths.setdefault(r, []).append((interval, os.path.join(data_dir, fn)))
            continue
    for r in list(rank_to_indices_paths.keys()):
        rank_to_indices_paths[r] = sorted(rank_to_indices_paths[r], key=lambda x: x[0])
    ranks = sorted(rank_to_info_path.keys())
    return ranks, rank_to_info_path, rank_to_indices_paths


def load_rank_info(info_path: str) -> tuple[DistributedInfo, dict[str, ParamInfo]]:
    with open(info_path) as f:
        obj = json.load(f)
    distributed = DistributedInfo(raw=dict(obj.get("distributed_info", {})))
    # Training may write `model_info` (legacy) or `params_info` (megatron/statistics/manager.py).
    raw_params: dict[str, Any] = {}
    for key in ("model_info", "params_info"):
        block = obj.get(key)
        if isinstance(block, dict):
            raw_params.update(block)
    out: dict[str, ParamInfo] = {}
    for name, info in raw_params.items():
        dp_range = None
        pr = info.get("dp_param_range")
        if pr is not None and isinstance(pr, (list, tuple)) and len(pr) == 2:
            dp_range = (int(pr[0]), int(pr[1]))
        has_sparse = bool(info.get("has_sparse_diff")) or dp_range is not None
        shape_raw = info.get("param.shape")
        if isinstance(shape_raw, (list, tuple)):
            shape = tuple(int(x) for x in shape_raw)
        else:
            shape = tuple()
        tp_attrs = info.get("tp_attrs")
        tp_attrs_dict: dict[str, Any] = dict(tp_attrs) if isinstance(tp_attrs, dict) else {}
        out[name] = ParamInfo(
            name=str(name),
            shape=shape,
            numel=int(info["param.numel"]),
            dtype_str=str(info["param.dtype"]),
            is_moe_param=bool(info.get("is_moe_param", False)),
            tp_attrs=tp_attrs_dict,
            has_sparse_diff=has_sparse,
            dp_param_range=dp_range,
        )
    return distributed, out


def event_to_change_indices(event: Any) -> torch.Tensor:
    """One training sync event -> 1D int tensor of flat indices that changed (shard-local)."""
    if torch.is_tensor(event):
        return event.view(-1)
    model_idx = getattr(event, "model_weight_indices", None)
    if model_idx is not None and torch.is_tensor(model_idx):
        return model_idx.view(-1)
    raise TypeError(f"Expected Tensor or object with model_weight_indices; got {type(event)!r}")


def normalize_event_list(indices_list: Any) -> list[torch.Tensor]:
    """Turn checkpoint `dict[param]` value into a list of per-event index tensors.

    Supported on-disk shapes:
    - ``None`` -> ``[]``
    - ``list[Tensor]`` (legacy / plain indices)
    - ``list[tuple[Tensor, Tensor]]`` -> take the second tensor per event (merged indices)
    - ``list[SparseData]`` (v2) -> ``model_weight_indices`` per event
    """
    if indices_list is None:
        return []
    if not isinstance(indices_list, (list, tuple)):
        raise TypeError(f"Unsupported indices_list type: {type(indices_list)!r}")
    if len(indices_list) == 0:
        return []
    first = indices_list[0]
    if torch.is_tensor(first):
        return [t.view(-1) for t in indices_list]
    if (
        isinstance(first, (list, tuple))
        and len(first) == 2
        and torch.is_tensor(first[0])
        and torch.is_tensor(first[1])
    ):
        return [pair[1].view(-1) for pair in indices_list]
    if getattr(first, "model_weight_indices", None) is not None:
        return [event_to_change_indices(ev) for ev in indices_list]
    raise TypeError(f"Unsupported indices_list element type: {type(first)!r}")


def event_to_analyse_result(event: Any) -> dict[str, Any] | None:
    """Extract per-event analysis dict if present (SparseData.main_weight_delta_analyse_result)."""
    d = getattr(event, "main_weight_delta_analyse_result", None)
    return d if isinstance(d, dict) else None


def normalize_event_analyse_list(indices_list: Any) -> list[dict[str, Any] | None]:
    """Turn checkpoint `dict[param]` value into a list of per-event analyse dicts (or None).

    Mirrors `normalize_event_list()` but returns analysis dicts when available.
    """
    if indices_list is None:
        return []
    if not isinstance(indices_list, (list, tuple)):
        raise TypeError(f"Unsupported indices_list type: {type(indices_list)!r}")
    if len(indices_list) == 0:
        return []
    first = indices_list[0]
    if torch.is_tensor(first):
        return [None for _ in indices_list]
    if isinstance(first, (list, tuple)) and len(first) == 2 and torch.is_tensor(first[0]):
        return [None for _ in indices_list]
    if getattr(first, "model_weight_indices", None) is not None:
        return [event_to_analyse_result(ev) for ev in indices_list]
    return [None for _ in indices_list]


def iter_interval_batches_for_rank(
    rank: int,
    indices_paths: list[tuple[int, str]],
) -> Iterator[tuple[int, int, dict[str, list[torch.Tensor]]]]:
    """Yield ``(interval, num_events, param_name -> list[indices_tensor])`` per ``.pt`` file."""
    for interval, pt_path in indices_paths:
        obj: dict[str, Any] = load_torch_dict(pt_path)
        param_to_events: dict[str, list[torch.Tensor]] = {}
        max_len = 0
        for name, raw_events in obj.items():
            events = normalize_event_list(raw_events)
            param_to_events[str(name)] = events
            if len(events) > max_len:
                max_len = len(events)
        yield interval, max_len, param_to_events


def iter_interval_enriched_batches_for_rank(
    rank: int,
    indices_paths: list[tuple[int, str]],
) -> Iterator[
    tuple[int, int, dict[str, list[torch.Tensor]], dict[str, list[dict[str, Any] | None]]]
]:
    """Yield indices and analysis dicts per ``.pt`` file.

    Returns: (interval, num_events, param->list[indices], param->list[analyse_dict|None])
    """
    for interval, pt_path in indices_paths:
        obj: dict[str, Any] = load_torch_dict(pt_path)
        param_to_events: dict[str, list[torch.Tensor]] = {}
        param_to_analyse: dict[str, list[dict[str, Any] | None]] = {}
        max_len = 0
        for name, raw_events in obj.items():
            events = normalize_event_list(raw_events)
            analyses = normalize_event_analyse_list(raw_events)
            param_to_events[str(name)] = events
            param_to_analyse[str(name)] = analyses
            max_len = max(max_len, len(events), len(analyses))
        yield interval, max_len, param_to_events, param_to_analyse


def build_offline_step_index(
    rank_to_indices_paths: dict[int, list[tuple[int, str]]],
) -> dict[tuple[int, int, int], int]:
    """Map ``(rank, interval, event_idx)`` -> contiguous ``offline_step`` in ``0 .. T-1``.

    Ordering follows the smallest rank's ``(interval, event_idx)`` sequence; there is no on-disk
    ``global_opt_step`` in current dumps.

    Loads each checkpoint **once** per rank (avoids doubling RAM on large merged dumps).
    """
    ranks = sorted(rank_to_indices_paths.keys())
    if not ranks:
        return {}
    base_rank = ranks[0]
    n_events_by_rank_interval: dict[tuple[int, int], int] = {}
    for r in ranks:
        for interval, pt_path in sorted(rank_to_indices_paths[r], key=lambda x: x[0]):
            obj = load_torch_dict(pt_path)
            n_events = 0
            for v in obj.values():
                n_events = max(n_events, len(normalize_event_list(v)))
            n_events_by_rank_interval[int(r), int(interval)] = int(n_events)
            del obj
            gc.collect()

    sequence_keys: list[tuple[int, int]] = []
    for interval, _ in sorted(rank_to_indices_paths[base_rank], key=lambda x: x[0]):
        n_events = n_events_by_rank_interval.get((int(base_rank), int(interval)), 0)
        for e in range(n_events):
            sequence_keys.append((int(interval), e))
    step_of: dict[tuple[int, int], int] = {key: i for i, key in enumerate(sequence_keys)}
    out: dict[tuple[int, int, int], int] = {}
    for r in ranks:
        for interval, _ in sorted(rank_to_indices_paths[r], key=lambda x: x[0]):
            n_events = n_events_by_rank_interval.get((int(r), int(interval)), 0)
            for e in range(n_events):
                out[int(r), int(interval), e] = step_of[int(interval), e]
    return out
