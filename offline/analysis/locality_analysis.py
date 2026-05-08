"""Parameter-internal index distribution and history-similarity Jaccard (locality proxy)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from ..common import io as io_mod
from ..common.metrics import bucketize_indices, jaccard, torch_unique_cpu_int64
from ..common.naming import classify_param_group
from .aggregation import StepAggregation
from .dp_gather import dp_gather_param_step_indices
from .plots import save_heatmap, save_line


def _unravel_flat_indices(flat: torch.Tensor, shape: tuple[int, ...]) -> list[torch.Tensor]:
    x = flat.view(-1).to(device="cpu", dtype=torch.int64)
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
        return multi[0].view(-1).to(device="cpu", dtype=torch.int64)
    stride_vals: list[int] = [1] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        stride_vals[i] = acc
        acc *= int(shape[i])
    out = torch.zeros_like(multi[0].view(-1).to(device="cpu", dtype=torch.int64))
    for i, idx in enumerate(multi):
        out += idx.view(-1).to(device="cpu", dtype=torch.int64) * int(stride_vals[i])
    return out


def _tp_local_flat_to_global_flat(
    *,
    flat_tp_local: torch.Tensor,
    tp_local_shape: tuple[int, ...],
    tp_world_size: int,
    tp_rank: int,
    tp_attrs: dict[str, Any],
) -> torch.Tensor:
    is_tp = bool(tp_attrs.get("tensor_model_parallel"))
    if not is_tp or tp_world_size <= 1:
        return flat_tp_local.view(-1).to(device="cpu", dtype=torch.int64)

    part_dim = int(tp_attrs.get("partition_dim", 0))
    stride = int(tp_attrs.get("partition_stride", 1) or 1)
    if not tp_local_shape:
        tp_local_shape = (int(flat_tp_local.numel()),)
        part_dim = 0

    global_shape = list(tp_local_shape)
    global_shape[part_dim] = int(global_shape[part_dim]) * int(tp_world_size)
    global_shape_t = tuple(int(x) for x in global_shape)

    global_dim = int(global_shape_t[part_dim])
    denom = int(tp_world_size) * int(stride)
    if global_dim % denom != 0:
        raise ValueError(
            f"TP global dim {global_dim} not divisible by tp_world_size*stride={denom} "
            f"(shape={tp_local_shape}, part_dim={part_dim}, tp={tp_world_size}, stride={stride})"
        )
    chunk_size = global_dim // denom

    multi = _unravel_flat_indices(flat_tp_local, tuple(int(x) for x in tp_local_shape))
    coord = multi[part_dim]
    chunk_id = torch.div(coord, chunk_size, rounding_mode="floor")
    intra = coord - chunk_id * chunk_size
    global_chunk = int(tp_rank) + chunk_id * int(tp_world_size)
    multi[part_dim] = global_chunk * chunk_size + intra

    return _ravel_multi_index(multi, global_shape_t)


def _should_count_rank_for_param(dist: io_mod.DistributedInfo, pi: io_mod.ParamInfo) -> bool:
    if not pi.is_moe_param and int(dist.raw.get("ep_rank", 0) or 0) != 0:
        return False
    tp_world = int(dist.raw.get("tp_size", 1) or 1)
    tp_rank = int(dist.raw.get("tp_rank", 0) or 0)
    is_tp = bool(pi.tp_attrs.get("tensor_model_parallel", False)) and tp_world > 1
    if not is_tp and tp_world > 1 and tp_rank != 0:
        return False
    dp_world = int(dist.raw.get("dp_size", 1) or 1)
    dp_rank = int(dist.raw.get("dp_rank", 0) or 0)
    is_dp_sharded = pi.dp_param_range is not None and dp_world > 1
    return is_dp_sharded or dp_world <= 1 or dp_rank == 0


def mean_nnz_ratio_by_param(per_param_step_rows: list[dict[str, Any]]) -> list[tuple[float, str]]:
    by_param: dict[str, list[float]] = defaultdict(list)
    for row in per_param_step_rows:
        by_param[row["param_name"]].append(float(row["nnz_ratio"]))
    scored = [(float(np.mean(v)) if v else 0.0, name) for name, v in by_param.items()]
    scored.sort(reverse=True)
    return scored


def select_locality_candidate_params(
    per_param_step_rows: list[dict[str, Any]],
    heatmap_top_param_names: list[str],
    max_candidates: int,
) -> list[str]:
    ranked = [name for _, name in mean_nnz_ratio_by_param(per_param_step_rows)]
    expert_ranked = [
        name for name in ranked if classify_param_group(name) in ("moe/fc1", "moe/fc2")
    ]
    merged = (expert_ranked[: max(1, max_candidates)] + heatmap_top_param_names)[
        : max(1, max_candidates)
    ]
    seen: set[str] = set()
    out: list[str] = []
    for name in merged:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out[: max(1, max_candidates)]


def run_locality_analysis(
    agg: StepAggregation,
    *,
    candidate_param_names: list[str],
    internal_buckets: int,
    jaccard_threshold: float,
    figures_dir: str,
) -> list[dict[str, Any]]:
    selected = set(candidate_param_names)
    per_param_step_indices: dict[tuple[str, int], list[torch.Tensor]] = defaultdict(list)

    for name in selected:
        pi0 = agg.param_info_baseline_rank.get(name)
        if pi0 is None:
            continue

        dp_events = dp_gather_param_step_indices(
            ranks=agg.ranks,
            rank_to_dist=agg.rank_to_dist,
            rank_to_param_info=agg.rank_to_param_info,
            step_index=agg.step_index,
            rank_to_indices_paths=agg.rank_to_indices_paths,
            param_name=name,
        )

        group_to_tp: dict[tuple[Any, ...], tuple[int, int]] = {}
        for r in agg.ranks:
            dist = agg.rank_to_dist[r]
            pi_r = agg.rank_to_param_info[r].get(name)
            if pi_r is None:
                continue
            if not _should_count_rank_for_param(dist, pi_r):
                continue
            if not pi_r.is_moe_param:
                gk = ("dp", int(dist.raw.get("tp_rank", 0) or 0))
            else:
                gk = (
                    "ep_dp",
                    int(dist.raw.get("tp_rank", 0) or 0),
                    int(dist.raw.get("ep_rank", 0) or 0),
                    int(dist.raw.get("ep_tp_rank", 0) or 0),
                )
            if gk not in group_to_tp:
                group_to_tp[gk] = (
                    int(dist.raw.get("tp_size", 1) or 1),
                    int(dist.raw.get("tp_rank", 0) or 0),
                )

        for ev in dp_events:
            tp_world, tp_rank = group_to_tp.get(ev.group_key, (1, 0))
            idx_global = _tp_local_flat_to_global_flat(
                flat_tp_local=ev.indices_tp_local,
                tp_local_shape=pi0.shape,
                tp_world_size=tp_world,
                tp_rank=tp_rank,
                tp_attrs=pi0.tp_attrs,
            )
            per_param_step_indices[name, ev.offline_step].append(idx_global)

    internal_summaries: list[dict[str, Any]] = []

    for name in candidate_param_names:
        unions: list[torch.Tensor] = []
        nnz_ratios: list[float] = []
        for step in range(agg.num_steps):
            tensors = per_param_step_indices.get((name, step), [])
            if not tensors:
                unions.append(torch.empty((0,), dtype=torch.int64))
                nnz_ratios.append(0.0)
                continue
            concatenated = torch.cat(
                [t.view(-1).to(device="cpu", dtype=torch.int64) for t in tensors], dim=0
            )
            unique_idx = torch_unique_cpu_int64(concatenated)
            unions.append(unique_idx)
            numel = int(agg.numel_sum_by_param_step.get((name, step), 0))
            nnz_ratios.append((float(unique_idx.numel()) / float(numel)) if numel else 0.0)

        hist_jaccard_per_step: list[float] = []
        hist_union = torch.empty((0,), dtype=torch.int64)
        for step in range(agg.num_steps):
            cur = unions[step]
            if step == 0:
                hist_jaccard_per_step.append(float("nan"))
                hist_union = torch_unique_cpu_int64(cur)
                continue
            hist_jaccard_per_step.append(jaccard(hist_union, cur))
            if hist_union.numel() == 0:
                hist_union = torch_unique_cpu_int64(cur)
            elif cur.numel() != 0:
                hist_union = torch_unique_cpu_int64(torch.cat([hist_union, cur], dim=0))

        jaccard_np = np.array(
            [0.0 if np.isnan(x) else float(x) for x in hist_jaccard_per_step], dtype=np.float32
        )
        avg_jaccard = float(np.mean(jaccard_np[1:])) if agg.num_steps > 1 else 0.0
        internal_summaries.append(
            {
                "param_name": name,
                "group": classify_param_group(name),
                "avg_nnz_ratio": float(np.mean(nnz_ratios)) if nnz_ratios else 0.0,
                "avg_history_jaccard": avg_jaccard,
                "locality_like": int(avg_jaccard >= jaccard_threshold),
            }
        )

        max_numel = 0
        for step in range(agg.num_steps):
            max_numel = max(max_numel, int(agg.numel_sum_by_param_step.get((name, step), 0)))
        denom = max(1, max_numel)
        heat = np.zeros((agg.num_steps, internal_buckets), dtype=np.float32)
        for step in range(agg.num_steps):
            u = unions[step]
            if u.numel() == 0:
                continue
            bucket_counts = bucketize_indices(
                u.to(torch.int64), numel=denom, buckets=internal_buckets
            ).astype(np.float32)
            total = float(bucket_counts.sum())
            if total > 0:
                bucket_counts /= total
            heat[step, :] = bucket_counts

        slug = filesystem_safe_param_slug(name)
        save_heatmap(
            f"{figures_dir}/internal_{slug}.png",
            heat,
            title=f"Param internal index distribution (bucketed) — {name}",
            xlabel="bucket",
            ylabel="offline step",
            x_ticks=[str(i) for i in range(internal_buckets)],
            y_ticks=[str(i) for i in range(agg.num_steps)],
            vmax=None,
            cmap="viridis",
        )
        save_line(
            f"{figures_dir}/internal_{slug}_jaccard.png",
            x=list(range(agg.num_steps)),
            y=[0.0 if np.isnan(x) else float(x) for x in hist_jaccard_per_step],
            title=f"History Jaccard vs union(prev) — {name}",
            xlabel="offline step",
            ylabel="Jaccard(union(prev), step)",
        )

    internal_summaries.sort(key=lambda x: x["avg_history_jaccard"], reverse=True)
    return internal_summaries


def filesystem_safe_param_slug(name: str, max_len: int = 160) -> str:
    out: list[str] = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out)
    return slug[-max_len:]
