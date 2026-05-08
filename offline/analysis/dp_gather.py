"""Offline DP gather: merge ZeRO-1 sharded indices within DP groups.

For end-to-end DP + EP expert-id + TP global merge (and replica dedupe), use offline.merge.
This module focuses on the DP dimension only for analysis-time locality computations.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch

from ..common import io as io_mod
from ..common.metrics import torch_unique_cpu_int64


@dataclass(frozen=True)
class DpGatheredEvent:
    """DP-gathered indices for one param at one offline step (tp-local coordinates)."""

    group_key: tuple[Any, ...]
    offline_step: int
    indices_tp_local: torch.Tensor  # CPU int64, unique sorted


def _dp_group_key(dist: io_mod.DistributedInfo, pi: io_mod.ParamInfo) -> tuple[Any, ...]:
    tp_rank = int(dist.raw.get("tp_rank", 0) or 0)
    if not pi.is_moe_param:
        return ("dp", tp_rank)
    ep_rank = int(dist.raw.get("ep_rank", 0) or 0)
    ep_tp_rank = int(dist.raw.get("ep_tp_rank", 0) or 0)
    return ("ep_dp", tp_rank, ep_rank, ep_tp_rank)


def dp_shift_to_tp_local(
    indices_shard_local: torch.Tensor, dp_param_range: tuple[int, int]
) -> torch.Tensor:
    """Map shard-local indices to dp-merged tp-local indices by adding range start."""
    start = int(dp_param_range[0])
    x = indices_shard_local.view(-1).to(device="cpu", dtype=torch.int64)
    if x.numel() == 0:
        return x
    return x + start


def dp_gather_param_step_indices(
    *,
    ranks: list[int],
    rank_to_dist: dict[int, io_mod.DistributedInfo],
    rank_to_param_info: dict[int, dict[str, io_mod.ParamInfo]],
    step_index: dict[tuple[int, int, int], int],
    rank_to_indices_paths: dict[int, list[tuple[int, str]]],
    param_name: str,
) -> list[DpGatheredEvent]:
    """Return DP-gathered per-step indices for `param_name` (tp-local coordinates)."""
    per_group_step: dict[tuple[tuple[Any, ...], int], list[torch.Tensor]] = defaultdict(list)

    for rank in ranks:
        dist = rank_to_dist[rank]
        pinfo = rank_to_param_info[rank]
        pi = pinfo.get(param_name)
        if pi is None:
            continue
        gk = _dp_group_key(dist, pi)

        for interval, n_events, param_to_events in io_mod.iter_interval_batches_for_rank(
            rank, rank_to_indices_paths.get(rank, [])
        ):
            events = param_to_events.get(param_name)
            if not events:
                continue
            for event_idx in range(min(n_events, len(events))):
                offline_step = step_index.get((rank, interval, event_idx))
                if offline_step is None:
                    continue
                idx = events[event_idx]
                if not torch.is_tensor(idx):
                    continue
                x = idx.view(-1).to(device="cpu", dtype=torch.int64)
                if pi.dp_param_range is not None:
                    x = dp_shift_to_tp_local(x, pi.dp_param_range)
                per_group_step[gk, int(offline_step)].append(x)

    out: list[DpGatheredEvent] = []
    for (gk, step), tensors in per_group_step.items():
        if not tensors:
            continue
        merged = torch_unique_cpu_int64(torch.cat([t.view(-1) for t in tensors], dim=0))
        out.append(DpGatheredEvent(group_key=gk, offline_step=int(step), indices_tp_local=merged))
    out.sort(key=lambda e: (e.offline_step, tuple(map(str, e.group_key))))
    return out
