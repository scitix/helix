"""Offline merge of sparse stats across DP/EP/TP into a single-rank compatible dump.

Goal: produce an output directory that looks like a normal training dump:
- rank0_info.json
- rank0_indices_<interval>.pt

But indices are merged into **global (full-model) flat coordinates**:
1) DP indices are assumed already shifted by dp_range.start at dump time.
2) MoE expert ids are corrected across EP ranks by rewriting param names.
3) TP shards are merged by mapping tp-local flat indices into global flat indices.
   For MoE expert params (``is_moe_param=True``), the TP dimension is the
   **expert TP group** (``ep_tp_size``/``ep_tp_rank``), not the dense
   ``tp_size``/``tp_rank``. Non-expert params continue to use the dense TP group.

Replicated TP shards (``dp_param_range`` is ``None`` or covers the full shard ``numel``) appear on
multiple ranks (e.g. DP/EP replicas); we only take **one** rank per (logical param, ``tp_rank``)
when merging, otherwise indices would be duplicated.

**Enriched stats** (``main_weight_delta_analyse_result``): per-step shard dicts are merged by
summing integer counters and weighted-averaging ``*_stats`` dicts where possible.
"""

from __future__ import annotations

import gc
import json
import math
import multiprocessing as mp
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch

from ..common import io as io_mod
from ..common.metrics import torch_unique_int64
from ..common.naming import fix_moe_expert_param_name


@dataclass(frozen=True)
class MergePlan:
    """Derived merge settings from rank0 distributed_info."""

    tp_size: int
    dp_size: int
    ep_size: int
    ep_tp_size: int
    experts_per_ep_rank: int


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _interval_map(
    rank_to_indices_paths: dict[int, list[tuple[int, str]]],
) -> dict[int, dict[int, str]]:
    """interval -> (rank -> pt_path)"""
    out: dict[int, dict[int, str]] = defaultdict(dict)
    for r, items in rank_to_indices_paths.items():
        for interval, p in items:
            out[int(interval)][int(r)] = p
    return {k: dict(v) for k, v in sorted(out.items(), key=lambda x: x[0])}


def _unravel_flat_indices(flat: torch.Tensor, shape: tuple[int, ...]) -> list[torch.Tensor]:
    x = flat.view(-1)
    if x.dtype != torch.int64:
        x = x.to(torch.int64)
    if not shape or len(shape) == 1:
        return [x]
    out: list[torch.Tensor] = []
    rem = x
    for dim in reversed(shape[1:]):
        out.append(rem % int(dim))
        rem = torch.div(rem, int(dim), rounding_mode="floor")
    out.append(rem)
    out.reverse()
    return out


def _ravel_multi_index(multi: list[torch.Tensor], shape: tuple[int, ...]) -> torch.Tensor:
    if not shape or len(shape) == 1:
        x = multi[0].view(-1)
        return x.to(torch.int64) if x.dtype != torch.int64 else x
    stride_vals: list[int] = [1] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        stride_vals[i] = acc
        acc *= int(shape[i])
    base = multi[0].view(-1)
    base = base.to(torch.int64) if base.dtype != torch.int64 else base
    out = torch.zeros_like(base)
    for i, idx in enumerate(multi):
        xi = idx.view(-1)
        xi = xi.to(torch.int64) if xi.dtype != torch.int64 else xi
        out += xi * int(stride_vals[i])
    return out


def tp_local_flat_to_global_flat(
    *,
    flat_tp_local: torch.Tensor,
    tp_local_shape: tuple[int, ...],
    tp_world_size: int,
    tp_rank: int,
    tp_attrs: dict[str, Any],
) -> torch.Tensor:
    """Map tp-local flat indices to global flat indices based on Megatron tp_attrs.

    This follows Megatron's `partition_dim` + `partition_stride` convention.
    For non-TP params, returns indices unchanged.
    """
    is_tp = bool(tp_attrs.get("tensor_model_parallel"))
    if not is_tp or tp_world_size <= 1:
        x = flat_tp_local.view(-1)
        return x.to(torch.int64) if x.dtype != torch.int64 else x

    part_dim = int(tp_attrs.get("partition_dim", 0))
    stride = int(tp_attrs.get("partition_stride", 1) or 1)
    if not tp_local_shape:
        tp_local_shape = (int(flat_tp_local.numel()),)
        part_dim = 0

    global_shape = list(tp_local_shape)
    global_shape[part_dim] = int(global_shape[part_dim]) * int(tp_world_size)
    global_shape_t = tuple(int(x) for x in global_shape)

    local_dim = int(tp_local_shape[part_dim])
    global_dim = int(global_shape_t[part_dim])
    denom = int(tp_world_size) * int(stride)
    if global_dim % denom != 0:
        raise ValueError(
            f"TP global dim {global_dim} not divisible by tp_world_size*stride={denom} "
            f"(shape={tp_local_shape}, part_dim={part_dim}, tp={tp_world_size}, stride={stride})"
        )
    chunk_size = global_dim // denom
    expected_local_dim = int(chunk_size) * int(stride)
    if local_dim != expected_local_dim:
        raise ValueError(
            f"TP local dim mismatch: local_dim={local_dim} != chunk_size*stride={expected_local_dim} "
            f"(global_dim={global_dim}, part_dim={part_dim}, tp={tp_world_size}, stride={stride})"
        )

    multi = _unravel_flat_indices(flat_tp_local, tuple(int(x) for x in tp_local_shape))
    coord = multi[part_dim]
    chunk_id = torch.div(coord, chunk_size, rounding_mode="floor")
    intra = coord - chunk_id * chunk_size
    global_chunk = int(tp_rank) + chunk_id * int(tp_world_size)
    multi[part_dim] = global_chunk * chunk_size + intra

    return _ravel_multi_index(multi, global_shape_t)


def tp_local_shape_to_global(
    tp_local_shape: tuple[int, ...],
    tp_attrs: dict[str, Any],
    tp_world_size: int,
) -> tuple[int, ...]:
    """Megatron TP-local shard shape -> global logical shape (single tensor)."""
    shape = [int(x) for x in tp_local_shape]
    if not tp_attrs.get("tensor_model_parallel") or tp_world_size <= 1:
        return tuple(shape)
    part_dim = int(tp_attrs.get("partition_dim", 0))
    if part_dim < 0 or part_dim >= len(shape):
        return tuple(shape)
    shape[part_dim] = int(shape[part_dim]) * int(tp_world_size)
    return tuple(shape)


def _is_zero1_shard(pi: io_mod.ParamInfo) -> bool:
    """True if ZeRO-1 owns a strict subset of flat elements (needs offset merge across ranks)."""
    dp = pi.dp_param_range
    if dp is None:
        return False
    w = int(dp[1]) - int(dp[0])
    return 0 < w < int(pi.numel)


def _merge_shard_identity(
    local_name: str,
    pi: io_mod.ParamInfo,
    dist: io_mod.DistributedInfo,
    *,
    experts_per_ep_rank: int,
) -> tuple[Any, ...]:
    """Identity for deduping ranks that contribute the same logical shard."""
    ep_rank = int(dist.raw.get("ep_rank", 0) or 0)
    if pi.is_moe_param:
        shard_tp_rank = int(dist.raw.get("ep_tp_rank", 0) or 0)
    else:
        shard_tp_rank = int(dist.raw.get("tp_rank", 0) or 0)
    fixed = fix_moe_expert_param_name(
        local_name, ep_rank=ep_rank, experts_per_ep_rank=experts_per_ep_rank
    )
    if _is_zero1_shard(pi):
        assert pi.dp_param_range is not None
        return ("z1", fixed, shard_tp_rank, int(pi.dp_param_range[0]), int(pi.dp_param_range[1]))
    return ("full", fixed, shard_tp_rank)


def _merge_stats_dicts(parts: list[Any]) -> dict[str, Any]:
    stats = [p for p in parts if isinstance(p, dict) and int(p.get("numel", 0) or 0) > 0]
    if not stats:
        return {"numel": 0, "abs_mean": 0.0, "abs_median": 0.0, "abs_max": 0.0}
    total_n = sum(int(s["numel"]) for s in stats)
    if total_n <= 0:
        return {"numel": 0, "abs_mean": 0.0, "abs_median": 0.0, "abs_max": 0.0}

    def wmean(key: str) -> float:
        return float(
            sum(float(s.get(key, 0.0) or 0.0) * int(s["numel"]) for s in stats) / float(total_n)
        )

    abs_max = max(float(s.get("abs_max", 0.0) or 0.0) for s in stats)
    return {
        "numel": int(total_n),
        "abs_mean": wmean("abs_mean"),
        "abs_median": wmean("abs_median"),
        "abs_max": float(abs_max),
    }


def merge_analyse_dicts(dicts: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    """Merge shard-local ``main_weight_delta_analyse_result`` dicts into one global-ish dict."""
    vals = [d for d in dicts if isinstance(d, dict)]
    if not vals:
        return None
    out: dict[str, Any] = {}
    int_keys = (
        "grad_nnz",
        "main_weight_nnz",
        "updated_indices_count",
        "updated_indices_grad_nnz",
        "updated_indices_grad_zero",
        "acc_loss_count",
        "acc_loss_grad_nonzero",
        "acc_loss_grad_zero",
    )
    for k in int_keys:
        out[k] = int(sum(int(v.get(k, 0) or 0) for v in vals))
    uc = int(out.get("updated_indices_count", 0))
    uz = int(out.get("updated_indices_grad_zero", 0))
    out["updated_indices_grad_zero_ratio"] = (float(uz) / float(uc)) if uc else 0.0

    stat_keys = (
        "updated_indices_grad_stats",
        "updated_indices_fp32_diff_stats",
        "updated_indices_fp32_diff_stats_grad_nonzero",
        "updated_indices_fp32_diff_stats_grad_zero",
        "acc_loss_fp32_diff_stats",
        "acc_loss_fp32_diff_stats_grad_nonzero",
        "nonzero_grad_with_acc_loss_stats",
    )
    for sk in stat_keys:
        out[sk] = _merge_stats_dicts([v.get(sk) for v in vals])

    vr = [str(v.get("validate_result", "")) for v in vals if v.get("validate_result")]
    out["validate_result"] = "; ".join(vr) if vr else vals[0].get("validate_result", "")
    return out


def _infer_output_index_dtype(params_info: dict[str, Any]) -> torch.dtype:
    """If int32 can represent indices for every param (0..numel-1), use int32; else int64."""
    int32_max_index = 2_147_483_647
    max_numel = 0
    for v in params_info.values():
        if not isinstance(v, dict):
            continue
        try:
            n = int(v.get("param.numel", 0) or 0)
        except Exception:
            n = 0
        max_numel = max(max_numel, n)
    return torch.int32 if max_numel <= (int32_max_index + 1) else torch.int64


def _human_bytes(n: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f}{u}"
        x /= 1024.0
    return f"{x:.2f}B"


def _estimate_merged_interval_bytes(
    *,
    merged_params_info: dict[str, Any],
    index_dtype: torch.dtype,
    assumed_param_nnz_ratio: float,
    assumed_events_per_interval: int,
) -> dict[str, float]:
    """Estimate merged rank0_indices_<interval>.pt size from params_info only (indices payload only)."""
    if assumed_events_per_interval <= 0:
        assumed_events_per_interval = 1
    ratio = max(0.0, min(1.0, float(assumed_param_nnz_ratio)))
    index_nbytes = 4 if index_dtype == torch.int32 else 8
    total_numel = 0
    for v in merged_params_info.values():
        if not isinstance(v, dict):
            continue
        try:
            total_numel += int(v.get("param.numel", 0) or 0)
        except Exception:
            continue
    expected_nnz_per_event = float(total_numel) * ratio
    indices_bytes_per_event = expected_nnz_per_event * float(index_nbytes)
    indices_bytes_per_interval = indices_bytes_per_event * float(assumed_events_per_interval)
    return {
        "total_numel": float(total_numel),
        "expected_nnz_per_event": float(expected_nnz_per_event),
        "indices_bytes_per_event": float(indices_bytes_per_event),
        "indices_bytes_per_interval": float(indices_bytes_per_interval),
    }


def _merge_one_interval(
    *,
    interval: int,
    rank_to_path: dict[int, str],
    rank_to_dist: dict[int, io_mod.DistributedInfo],
    rank_to_param_info: dict[int, dict[str, io_mod.ParamInfo]],
    plan: MergePlan,
    out_dir: str,
    out_rank: int,
    out_index_dtype: torch.dtype,
    merge_device: str,
    only_step_event_idx: int | None,
) -> None:
    dev = torch.device(str(merge_device))
    if dev.type == "cuda":
        torch.cuda.set_device(dev)

    per_param_idx_by_event: dict[str, list[list[torch.Tensor]]] = defaultdict(list)
    per_param_an_by_event: dict[str, list[list[dict[str, Any] | None]]] = defaultdict(list)
    seen_shard_by_event: list[set[tuple[Any, ...]]] = []

    def _ensure_slot(lst: list[list[Any]], event_i: int) -> None:
        while len(lst) <= event_i:
            lst.append([])

    for r in sorted(int(x) for x in rank_to_path):
        obj = io_mod.load_torch_dict(rank_to_path[int(r)])
        dist = rank_to_dist[int(r)]
        tp_size = int(dist.raw.get("tp_size", plan.tp_size) or plan.tp_size)
        ep_tp_size = int(dist.raw.get("ep_tp_size", plan.ep_tp_size) or plan.ep_tp_size)
        ep_rank = int(dist.raw.get("ep_rank", 0) or 0)
        tp_rank = int(dist.raw.get("tp_rank", 0) or 0)
        ep_tp_rank = int(dist.raw.get("ep_tp_rank", 0) or 0)
        pinfo = rank_to_param_info[int(r)]

        for raw_name, raw_events in obj.items():
            name = str(raw_name)
            pi = pinfo.get(name)
            if pi is None:
                continue

            events = io_mod.normalize_event_list(raw_events)
            analyses = io_mod.normalize_event_analyse_list(raw_events)
            for event_idx, idx in enumerate(events):
                if only_step_event_idx is not None and int(event_idx) != int(only_step_event_idx):
                    continue
                if not torch.is_tensor(idx):
                    continue
                x = idx.view(-1)
                if x.numel() == 0:
                    continue

                x_max = int(x.max().item())
                if x_max >= int(pi.numel):
                    raise ValueError(
                        f"Index out of bounds for param {name!r} on rank {r}: "
                        f"max={x_max} >= numel={int(pi.numel)}"
                    )

                if x.dtype != torch.int64:
                    x = x.to(torch.int64)
                if x.device != dev:
                    x = x.to(dev, non_blocking=(dev.type == "cuda"))

                shard_id = _merge_shard_identity(
                    name, pi, dist, experts_per_ep_rank=plan.experts_per_ep_rank
                )
                while len(seen_shard_by_event) <= event_idx:
                    seen_shard_by_event.append(set())
                if shard_id in seen_shard_by_event[event_idx]:
                    continue
                seen_shard_by_event[event_idx].add(shard_id)

                fixed_name = fix_moe_expert_param_name(
                    name,
                    ep_rank=ep_rank,
                    experts_per_ep_rank=plan.experts_per_ep_rank,
                )
                if pi.is_moe_param:
                    shard_tp_world_size = ep_tp_size
                    shard_tp_rank = ep_tp_rank
                else:
                    shard_tp_world_size = tp_size
                    shard_tp_rank = tp_rank
                xg = tp_local_flat_to_global_flat(
                    flat_tp_local=x,
                    tp_local_shape=pi.shape,
                    tp_world_size=shard_tp_world_size,
                    tp_rank=shard_tp_rank,
                    tp_attrs=pi.tp_attrs,
                )

                _ensure_slot(per_param_idx_by_event[fixed_name], event_idx)
                per_param_idx_by_event[fixed_name][event_idx].append(xg)
                a = analyses[event_idx] if event_idx < len(analyses) else None
                _ensure_slot(per_param_an_by_event[fixed_name], event_idx)
                per_param_an_by_event[fixed_name][event_idx].append(a)

        del obj
        gc.collect()

    merged_obj: dict[str, list[Any]] = {}
    for pname, ev_lists in per_param_idx_by_event.items():
        for event_idx, tensors in enumerate(ev_lists):
            if only_step_event_idx is not None and int(event_idx) != int(only_step_event_idx):
                continue
            if not tensors:
                continue
            try:
                merged = torch_unique_int64(
                    torch.cat([t.view(-1) for t in tensors], dim=0),
                    device=dev,
                )
            except torch.OutOfMemoryError as e:
                raise RuntimeError(
                    "CUDA OOM during merge. Try fewer intervals in parallel, run with "
                    "--merge-device cpu, or merge interval-by-interval."
                ) from e
            merged = merged.to("cpu")
            if merged.dtype != out_index_dtype:
                merged = merged.to(out_index_dtype)
            an_lists = per_param_an_by_event.get(pname, [])
            merged_an = merge_analyse_dicts(
                an_lists[event_idx] if event_idx < len(an_lists) else []
            )
            merged_obj.setdefault(pname, []).append(
                SimpleNamespace(
                    model_weight_indices=merged,
                    main_weight_delta_analyse_result=merged_an,
                )
            )

    out_pt = os.path.join(out_dir, f"rank{out_rank}_indices_{int(interval)}.pt")
    torch.save(merged_obj, out_pt)
    del merged_obj
    del per_param_idx_by_event
    del per_param_an_by_event
    gc.collect()


def _interval_worker(payload: dict[str, Any]) -> None:
    _merge_one_interval(**payload)


def _build_global_params_info(
    ranks: list[int],
    rank_to_param_info: dict[int, dict[str, io_mod.ParamInfo]],
    rank_to_dist: dict[int, io_mod.DistributedInfo],
    plan: MergePlan,
) -> dict[str, Any]:
    """One entry per logical param (global expert name + TP-global shape + numel)."""
    merged: dict[str, Any] = {}
    seen: set[str] = set()
    tp_sz = int(plan.tp_size)
    ep_tp_sz = int(plan.ep_tp_size)
    for r in sorted(ranks):
        ep_rank = int(rank_to_dist[r].raw.get("ep_rank", 0) or 0)
        for name, pi in rank_to_param_info[r].items():
            fixed = fix_moe_expert_param_name(
                str(name), ep_rank=ep_rank, experts_per_ep_rank=plan.experts_per_ep_rank
            )
            if fixed in seen:
                continue
            seen.add(fixed)
            shard_tp_sz = ep_tp_sz if pi.is_moe_param else tp_sz
            g_shape = tp_local_shape_to_global(pi.shape, pi.tp_attrs, shard_tp_sz)
            g_numel = int(math.prod(g_shape)) if g_shape else int(pi.numel)
            merged[fixed] = {
                "param.shape": list(g_shape),
                "param.dtype": str(pi.dtype_str),
                "param.numel": g_numel,
                "is_moe_param": bool(pi.is_moe_param),
                "tp_attrs": {
                    **dict(pi.tp_attrs),
                    "tensor_model_parallel": False,
                    "partition_dim": -1,
                },
                "dp_param_range": None,
            }
    return merged


def _infer_merge_plan(
    rank_to_dist: dict[int, io_mod.DistributedInfo],
    rank_to_param_info: dict[int, dict[str, io_mod.ParamInfo]],
) -> MergePlan:
    r0 = min(rank_to_dist.keys())
    d = rank_to_dist[r0].raw
    tp_size = int(d.get("tp_size", 1) or 1)
    dp_size = int(d.get("dp_size", 1) or 1)
    ep_size = int(d.get("ep_size", 1) or 1)
    ep_tp_size = int(d.get("ep_tp_size", 1) or 1)

    max_local = -1
    expert_re = re.compile(r"\.mlp\.experts\.linear_fc[12]\.weight(?P<expert>\d+)$")
    for r in rank_to_param_info:
        for n in rank_to_param_info[r]:
            m = expert_re.search(str(n))
            if not m:
                continue
            try:
                max_local = max(max_local, int(m.group("expert")))
            except (TypeError, ValueError):
                continue
    experts_per = int(max_local + 1) if max_local >= 0 else 0
    return MergePlan(
        tp_size=tp_size,
        dp_size=dp_size,
        ep_size=ep_size,
        ep_tp_size=ep_tp_size,
        experts_per_ep_rank=experts_per,
    )


def merge_sparse_dumps_to_single_rank(
    *,
    data_dir: str,
    out_dir: str,
    out_rank: int = 0,
    only_step: int | None = None,
    intervals: list[int] | None = None,
    write_info_json: bool = True,
    merge_device: str = "cpu",
    nproc_per_node: int = 1,
    estimate_interval_size: bool = False,
    assumed_events_per_interval: int = 1,
) -> str:
    """Merge all ranks under `data_dir` and write a single-rank dump under `out_dir`."""
    _ensure_dir(out_dir)
    dev = torch.device(str(merge_device))
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"merge_device={merge_device!r} requested but CUDA is not available in this environment."
        )

    ranks, rank_to_info_path, rank_to_indices_paths = io_mod.discover_rank_files(data_dir)
    if not ranks:
        raise FileNotFoundError(f"No rank*_info.json found under {data_dir!r}")
    if not any(rank_to_indices_paths.get(r) for r in ranks):
        raise FileNotFoundError(f"No rank*_indices_*.pt found under {data_dir!r}")

    rank_to_dist: dict[int, io_mod.DistributedInfo] = {}
    rank_to_param_info: dict[int, dict[str, io_mod.ParamInfo]] = {}
    for r in ranks:
        dist, pinfo = io_mod.load_rank_info(rank_to_info_path[r])
        rank_to_dist[r] = dist
        rank_to_param_info[r] = pinfo

    plan = _infer_merge_plan(rank_to_dist, rank_to_param_info)
    interval_to_rank_path = _interval_map(rank_to_indices_paths)

    only_step_interval: int | None = None
    only_step_event_idx: int | None = None
    if only_step is not None:
        baseline_rank_for_step = ranks[0]
        cur_step = 0
        for interval, pt_path in rank_to_indices_paths.get(baseline_rank_for_step, []):
            obj = io_mod.load_torch_dict(pt_path)
            n_events = 0
            for v in obj.values():
                n_events = max(n_events, len(io_mod.normalize_event_list(v)))
            for e in range(n_events):
                if cur_step == int(only_step):
                    only_step_interval = int(interval)
                    only_step_event_idx = int(e)
                    break
                cur_step += 1
            if only_step_interval is not None:
                break
        if only_step_interval is None or only_step_event_idx is None:
            raise ValueError(
                f"only_step={only_step} out of range for baseline rank {baseline_rank_for_step}"
            )

    merged_params_info = _build_global_params_info(ranks, rank_to_param_info, rank_to_dist, plan)
    out_index_dtype = _infer_output_index_dtype(merged_params_info)
    if estimate_interval_size:
        est = _estimate_merged_interval_bytes(
            merged_params_info=merged_params_info,
            index_dtype=out_index_dtype,
            assumed_param_nnz_ratio=0.01,
            assumed_events_per_interval=int(assumed_events_per_interval),
        )
        print(
            "[sparseRL][merge][estimate] "
            f"assumed_param_nnz_ratio=0.01, assumed_events_per_interval={int(assumed_events_per_interval)}; "
            f"index_dtype={out_index_dtype}; "
            f"indices_per_event≈{_human_bytes(est['indices_bytes_per_event'])}, "
            f"indices_per_interval≈{_human_bytes(est['indices_bytes_per_interval'])} "
            "(indices only; excludes pickle/container overhead)."
        )

    merged_info = {
        "distributed_info": {
            "world_size": 1,
            "pp_size": 1,
            "tp_size": 1,
            "dp_size": 1,
            "ep_size": 1,
            "ep_dp_size": 1,
            "ep_tp_size": 1,
            "global_rank": int(out_rank),
            "tp_rank": 0,
            "dp_rank": 0,
            "ep_rank": 0,
            "ep_dp_rank": 0,
            "ep_tp_rank": 0,
            "orig_tp_size": int(plan.tp_size),
            "orig_dp_size": int(plan.dp_size),
            "orig_ep_size": int(plan.ep_size),
            "orig_ep_tp_size": int(plan.ep_tp_size),
        },
        "params_info": merged_params_info,
    }
    out_info_path = os.path.join(out_dir, f"rank{out_rank}_info.json")
    if write_info_json:
        with open(out_info_path, "w") as f:
            json.dump(merged_info, f, indent=4, sort_keys=True)

    # Decide which intervals to merge.
    if only_step_interval is not None:
        interval_list = [int(only_step_interval)]
    elif intervals is None:
        interval_list = sorted(int(k) for k in interval_to_rank_path)
    else:
        interval_list = sorted({int(x) for x in intervals})

    if dev.type != "cuda":
        for interval in interval_list:
            rank_to_path = interval_to_rank_path.get(int(interval), {})
            if not rank_to_path:
                continue
            _merge_one_interval(
                interval=int(interval),
                rank_to_path=rank_to_path,
                rank_to_dist=rank_to_dist,
                rank_to_param_info=rank_to_param_info,
                plan=plan,
                out_dir=out_dir,
                out_rank=out_rank,
                out_index_dtype=out_index_dtype,
                merge_device=str(merge_device),
                only_step_event_idx=only_step_event_idx,
            )
        return out_dir

    # CUDA merge: distribute intervals across GPUs and run in parallel.
    nproc = max(1, int(nproc_per_node))
    payloads: list[dict[str, Any]] = []
    for i, interval in enumerate(interval_list):
        rank_to_path = interval_to_rank_path.get(int(interval), {})
        if not rank_to_path:
            continue
        gpu_id = int(i % nproc)
        payloads.append(
            {
                "interval": int(interval),
                "rank_to_path": rank_to_path,
                "rank_to_dist": rank_to_dist,
                "rank_to_param_info": rank_to_param_info,
                "plan": plan,
                "out_dir": out_dir,
                "out_rank": out_rank,
                "out_index_dtype": out_index_dtype,
                "merge_device": f"cuda:{gpu_id}",
                "only_step_event_idx": only_step_event_idx,
            }
        )

    ctx = mp.get_context("spawn")
    procs = min(nproc, len(payloads)) if payloads else 0
    if procs <= 1:
        for p in payloads:
            _interval_worker(p)
        return out_dir

    with ctx.Pool(processes=procs) as pool:
        pool.map(_interval_worker, payloads)

    return out_dir
