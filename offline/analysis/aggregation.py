"""Cross-rank aggregation: per-step nnz/numel tables and auxiliary maps for later plots."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch

from ..common import io as io_mod
from ..common.naming import classify_param_group


@dataclass(frozen=True)
class StepAggregation:
    ranks: list[int]
    rank_to_info_path: dict[int, str]
    rank_to_indices_paths: dict[int, list[tuple[int, str]]]
    step_index: dict[tuple[int, int, int], int]
    num_steps: int
    per_param_step_rows: list[dict[str, Any]]
    per_rank_step_rows: list[dict[str, Any]]
    per_step_source_rows: list[dict[str, Any]]
    nnz_sum_by_param_step: dict[tuple[str, int], int]
    numel_sum_by_param_step: dict[tuple[str, int], int]
    rank_to_dist: dict[int, io_mod.DistributedInfo]
    rank_to_param_info: dict[int, dict[str, io_mod.ParamInfo]]
    param_info_baseline_rank: dict[str, io_mod.ParamInfo]


def build_step_aggregation(data_dir: str, *, rank: int | None = None) -> StepAggregation:
    allowed = {int(rank)} if rank is not None else None
    ranks, rank_to_info_path, rank_to_indices_paths = io_mod.discover_rank_files(
        data_dir, allowed_ranks=allowed
    )
    if not ranks:
        raise FileNotFoundError(f"No rank*_info.json found under {data_dir!r}")
    if not any(rank_to_indices_paths.get(rank) for rank in ranks):
        raise FileNotFoundError(
            f"No rank*_indices_*.pt under {data_dir!r}; offline report needs at least one indices checkpoint."
        )

    step_index = io_mod.build_offline_step_index(rank_to_indices_paths)
    num_steps = (max(step_index.values()) + 1) if step_index else 0

    baseline_rank = ranks[0]
    _, param_info_baseline = io_mod.load_rank_info(rank_to_info_path[baseline_rank])

    rank_to_dist: dict[int, io_mod.DistributedInfo] = {}
    rank_to_param_info: dict[int, dict[str, io_mod.ParamInfo]] = {}

    nnz_sum_by_param_step: dict[tuple[str, int], int] = defaultdict(int)
    numel_sum_by_param_step: dict[tuple[str, int], int] = defaultdict(int)
    per_param_step_rows: list[dict[str, Any]] = []
    per_rank_step_rows: list[dict[str, Any]] = []
    per_step_source_rows: list[dict[str, Any]] = []

    # Source analysis accumulators (best-effort; only available when dumps contain analysis dicts).
    source_acc: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    source_float_acc: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for rank in ranks:
        dist, pinfo = io_mod.load_rank_info(rank_to_info_path[rank])
        rank_to_dist[rank] = dist
        rank_to_param_info[rank] = pinfo

        for (
            interval,
            n_events,
            param_to_events,
            param_to_analyse,
        ) in io_mod.iter_interval_enriched_batches_for_rank(
            rank, rank_to_indices_paths.get(rank, [])
        ):
            for event_idx in range(n_events):
                offline_step = step_index.get((rank, interval, event_idx))
                if offline_step is None:
                    continue
                step_nnz = 0
                step_numel = 0
                for param_name, event_tensors in param_to_events.items():
                    pi = rank_to_param_info[rank].get(param_name)
                    if pi is None:
                        continue
                    idx_tensor = (
                        event_tensors[event_idx]
                        if event_idx < len(event_tensors)
                        else torch.empty((0,), dtype=torch.int64)
                    )
                    nnz = int(idx_tensor.numel())
                    step_nnz += nnz
                    step_numel += int(pi.numel)
                    nnz_sum_by_param_step[param_name, offline_step] += nnz
                    numel_sum_by_param_step[param_name, offline_step] += int(pi.numel)

                    # Source analysis: per-param analysis dict (if present) is already per-shard.
                    analyses = param_to_analyse.get(param_name)
                    if analyses is None or event_idx >= len(analyses):
                        continue
                    a = analyses[event_idx]
                    if not isinstance(a, dict):
                        continue
                    sacc = source_acc[int(offline_step)]
                    for k in (
                        "grad_nnz",
                        "main_weight_nnz",
                        "updated_indices_count",
                        "updated_indices_grad_nnz",
                        "acc_loss_count",
                        "acc_loss_grad_nonzero",
                    ):
                        v = a.get(k)
                        if isinstance(v, int):
                            sacc[k] += int(v)

                    # Grad value magnitude features (best-effort aggregation)
                    ug = a.get("updated_indices_grad_stats")
                    if isinstance(ug, dict) and isinstance(ug.get("numel"), int):
                        num = int(ug["numel"])
                        if num > 0:
                            mean = float(ug.get("abs_mean", 0.0) or 0.0)
                            mx = float(ug.get("abs_max", 0.0) or 0.0)
                            source_float_acc[int(offline_step)]["updated_grad_abs_sum"] += (
                                mean * num
                            )
                            source_acc[int(offline_step)]["updated_grad_numel"] += num
                            source_float_acc[int(offline_step)]["updated_grad_abs_max"] = max(
                                source_float_acc[int(offline_step)].get(
                                    "updated_grad_abs_max", 0.0
                                ),
                                mx,
                            )

                    ud = a.get("updated_indices_fp32_diff_stats")
                    if isinstance(ud, dict) and isinstance(ud.get("numel"), int):
                        num = int(ud["numel"])
                        if num > 0:
                            mean = float(ud.get("abs_mean", 0.0) or 0.0)
                            mx = float(ud.get("abs_max", 0.0) or 0.0)
                            source_float_acc[int(offline_step)]["updated_diff_abs_sum"] += (
                                mean * num
                            )
                            source_acc[int(offline_step)]["updated_diff_numel"] += num
                            source_float_acc[int(offline_step)]["updated_diff_abs_max"] = max(
                                source_float_acc[int(offline_step)].get(
                                    "updated_diff_abs_max", 0.0
                                ),
                                mx,
                            )

                    ag = a.get("nonzero_grad_with_acc_loss_stats")
                    if isinstance(ag, dict) and isinstance(ag.get("numel"), int):
                        num = int(ag["numel"])
                        if num > 0:
                            mean = float(ag.get("abs_mean", 0.0) or 0.0)
                            mx = float(ag.get("abs_max", 0.0) or 0.0)
                            source_float_acc[int(offline_step)]["acc_loss_grad_abs_sum"] += (
                                mean * num
                            )
                            source_acc[int(offline_step)]["acc_loss_grad_numel"] += num
                            source_float_acc[int(offline_step)]["acc_loss_grad_abs_max"] = max(
                                source_float_acc[int(offline_step)].get(
                                    "acc_loss_grad_abs_max", 0.0
                                ),
                                mx,
                            )

                    ad = a.get("acc_loss_fp32_diff_stats_grad_nonzero")
                    if isinstance(ad, dict) and isinstance(ad.get("numel"), int):
                        num = int(ad["numel"])
                        if num > 0:
                            mean = float(ad.get("abs_mean", 0.0) or 0.0)
                            mx = float(ad.get("abs_max", 0.0) or 0.0)
                            source_float_acc[int(offline_step)]["acc_loss_diff_abs_sum"] += (
                                mean * num
                            )
                            source_acc[int(offline_step)]["acc_loss_diff_numel"] += num
                            source_float_acc[int(offline_step)]["acc_loss_diff_abs_max"] = max(
                                source_float_acc[int(offline_step)].get(
                                    "acc_loss_diff_abs_max", 0.0
                                ),
                                mx,
                            )

                    sacc["numel"] += int(pi.numel)

                per_rank_step_rows.append(
                    {
                        "rank": rank,
                        "interval": interval,
                        "event_idx": event_idx,
                        "step": offline_step,
                        "nnz": step_nnz,
                        "numel": step_numel,
                        "nnz_ratio": (float(step_nnz) / float(step_numel)) if step_numel else 0.0,
                    }
                )

    if source_acc:
        for step in range(num_steps):
            a = source_acc.get(step, {})
            f = source_float_acc.get(step, {})
            numel = int(a.get("numel", 0))
            grad_nnz = int(a.get("grad_nnz", 0))
            fp32_nnz = int(a.get("main_weight_nnz", 0))
            bf16_upd = int(a.get("updated_indices_count", 0))
            acc_loss = int(a.get("acc_loss_count", 0))

            upd_g_num = int(a.get("updated_grad_numel", 0))
            acc_g_num = int(a.get("acc_loss_grad_numel", 0))
            upd_d_num = int(a.get("updated_diff_numel", 0))
            acc_d_num = int(a.get("acc_loss_diff_numel", 0))
            out = {
                "step": step,
                "numel": numel,
                "grad_nnz": grad_nnz,
                "fp32_changed_nnz": fp32_nnz,
                "bf16_updated_nnz": bf16_upd,
                "acc_loss_count": acc_loss,
                "grad_sparsity": (1.0 - float(grad_nnz) / float(numel)) if numel else 0.0,
                "fp32_change_ratio": (float(fp32_nnz) / float(numel)) if numel else 0.0,
                "bf16_update_ratio": (float(bf16_upd) / float(numel)) if numel else 0.0,
                "acc_loss_ratio": (float(acc_loss) / float(numel)) if numel else 0.0,
                "acc_loss_share_in_fp32_change": (float(acc_loss) / float(fp32_nnz))
                if fp32_nnz
                else 0.0,
                "updated_grad_numel": upd_g_num,
                "updated_grad_abs_mean": (
                    float(f.get("updated_grad_abs_sum", 0.0)) / float(upd_g_num)
                )
                if upd_g_num
                else 0.0,
                "updated_grad_abs_max": float(f.get("updated_grad_abs_max", 0.0) or 0.0),
                "updated_diff_numel": upd_d_num,
                "updated_diff_abs_mean": (
                    float(f.get("updated_diff_abs_sum", 0.0)) / float(upd_d_num)
                )
                if upd_d_num
                else 0.0,
                "updated_diff_abs_max": float(f.get("updated_diff_abs_max", 0.0) or 0.0),
                "acc_loss_grad_numel": acc_g_num,
                "acc_loss_grad_abs_mean": (
                    float(f.get("acc_loss_grad_abs_sum", 0.0)) / float(acc_g_num)
                )
                if acc_g_num
                else 0.0,
                "acc_loss_grad_abs_max": float(f.get("acc_loss_grad_abs_max", 0.0) or 0.0),
                "acc_loss_diff_numel": acc_d_num,
                "acc_loss_diff_abs_mean": (
                    float(f.get("acc_loss_diff_abs_sum", 0.0)) / float(acc_d_num)
                )
                if acc_d_num
                else 0.0,
                "acc_loss_diff_abs_max": float(f.get("acc_loss_diff_abs_max", 0.0) or 0.0),
            }
            per_step_source_rows.append(out)

    for (param_name, offline_step), nnz in sorted(
        nnz_sum_by_param_step.items(), key=lambda x: (x[0][1], x[0][0])
    ):
        numel = numel_sum_by_param_step[param_name, offline_step]
        pi0 = param_info_baseline.get(param_name)
        per_param_step_rows.append(
            {
                "param_name": param_name,
                "group": classify_param_group(param_name),
                "step": offline_step,
                "nnz": int(nnz),
                "numel": int(numel),
                "nnz_ratio": (float(nnz) / float(numel)) if numel else 0.0,
                "dtype": (pi0.dtype_str if pi0 else ""),
            }
        )

    return StepAggregation(
        ranks=ranks,
        rank_to_info_path=rank_to_info_path,
        rank_to_indices_paths=rank_to_indices_paths,
        step_index=step_index,
        num_steps=num_steps,
        per_param_step_rows=per_param_step_rows,
        per_rank_step_rows=per_rank_step_rows,
        per_step_source_rows=per_step_source_rows,
        nnz_sum_by_param_step=dict(nnz_sum_by_param_step),
        numel_sum_by_param_step=dict(numel_sum_by_param_step),
        rank_to_dist=rank_to_dist,
        rank_to_param_info=rank_to_param_info,
        param_info_baseline_rank=param_info_baseline,
    )
