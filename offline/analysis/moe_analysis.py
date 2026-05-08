"""MoE-related parsing of param names, EP-aware expert ids, MoE CSV rows, and heatmaps.

All MoE plots in this module use **sparsity** as the primary metric:
    sparsity = 1 - nnz/numel
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

from ..common import io as io_mod
from ..common.naming import parse_moe_tag
from .aggregation import StepAggregation
from .plots import save_heatmap


def infer_local_experts_per_layer_kind(param_names: Iterable[str]) -> dict[tuple[int, str], int]:
    local_experts_per: dict[tuple[int, str], int] = {}
    for name in param_names:
        tag = parse_moe_tag(name)
        if tag is None or tag.kind not in ("fc1", "fc2") or tag.expert_id is None:
            continue
        key = (tag.layer, tag.kind)
        local_experts_per[key] = max(local_experts_per.get(key, 0), int(tag.expert_id) + 1)
    return local_experts_per


def build_moe_param_step_rows(agg: StepAggregation) -> list[dict[str, Any]]:
    local_experts_per = infer_local_experts_per_layer_kind(agg.param_info_baseline_rank.keys())

    moe_by_key: dict[tuple[int, str, int | None, int, int], tuple[int, int]] = defaultdict(
        lambda: (0, 0)
    )

    for rank in agg.ranks:
        dist = agg.rank_to_dist[rank]
        ep_rank = int(dist.raw.get("ep_rank", 0) or 0)
        pinfo = agg.rank_to_param_info[rank]
        for interval, n_events, param_to_events in io_mod.iter_interval_batches_for_rank(
            rank, agg.rank_to_indices_paths.get(rank, [])
        ):
            for event_idx in range(n_events):
                offline_step = agg.step_index.get((rank, interval, event_idx))
                if offline_step is None:
                    continue
                for param_name, event_tensors in param_to_events.items():
                    tag = parse_moe_tag(param_name)
                    if tag is None:
                        continue
                    pi = pinfo.get(param_name)
                    if pi is None:
                        continue
                    idx_tensor = (
                        event_tensors[event_idx]
                        if event_idx < len(event_tensors)
                        else torch.empty((0,), dtype=torch.int64)
                    )
                    nnz = int(idx_tensor.numel())
                    numel = int(pi.numel)

                    if tag.kind in ("fc1", "fc2") and tag.expert_id is not None:
                        per = int(local_experts_per.get((tag.layer, tag.kind), 0) or 0)
                        global_expert_id = (
                            (ep_rank * per + int(tag.expert_id)) if per > 0 else int(tag.expert_id)
                        )
                    else:
                        global_expert_id = None

                    key = (tag.layer, tag.kind, global_expert_id, int(offline_step), ep_rank)
                    cur = moe_by_key[key]
                    moe_by_key[key] = (cur[0] + nnz, cur[1] + numel)

    moe_rows: list[dict[str, Any]] = []
    for (layer, kind, gid, step, ep_rank), (nnz, numel) in sorted(
        moe_by_key.items(),
        key=lambda x: (
            x[0][0],
            str(x[0][1]),
            -1 if x[0][2] is None else x[0][2],
            x[0][3],
            x[0][4],
        ),
    ):
        moe_rows.append(
            {
                "layer": layer,
                "kind": kind,
                "ep_rank": ep_rank,
                "expert_id_global": "" if gid is None else gid,
                "step": step,
                "nnz": nnz,
                "numel": numel,
                "nnz_ratio": (float(nnz) / float(numel)) if numel else 0.0,
                "sparsity": (1.0 - (float(nnz) / float(numel))) if numel else 0.0,
            }
        )
    return moe_rows


def moe_layer_kind_sparsity_stats(
    moe_rows: list[dict[str, Any]], *, num_steps: int
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for kind in ["router", "fc1", "fc2"]:
        layers = sorted({int(r["layer"]) for r in moe_rows if r["kind"] == kind})
        if not layers:
            continue
        cell_vals: dict[tuple[int, int], list[float]] = defaultdict(list)
        for r in moe_rows:
            if r["kind"] != kind:
                continue
            cell_vals[int(r["layer"]), int(r["step"])].append(float(r.get("sparsity", 0.0)))
        xs: list[float] = []
        for layer in layers:
            for step in range(int(num_steps)):
                vals = cell_vals.get((layer, step), [])
                if not vals:
                    continue
                xs.append(float(np.mean(vals)))
        arr = np.asarray(xs, dtype=np.float64)
        if arr.size == 0:
            continue
        out[kind] = {
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
        }
    return out


def plot_moe_layer_kind_heatmaps(
    moe_rows: list[dict[str, Any]],
    num_steps: int,
    figures_dir: str,
) -> list[str]:
    def layer_kind_matrix(kind: str) -> tuple[np.ndarray, list[str]]:
        layers = sorted({int(r["layer"]) for r in moe_rows if r["kind"] == kind})
        if not layers:
            return np.zeros((0, 0), dtype=np.float32), []
        layer_to_row = {layer: i for i, layer in enumerate(layers)}
        cell_ratios: dict[tuple[int, int], list[float]] = defaultdict(list)
        for r in moe_rows:
            if r["kind"] != kind:
                continue
            cell_ratios[int(r["layer"]), int(r["step"])].append(float(r.get("sparsity", 0.0)))
        mat = np.zeros((len(layers), num_steps), dtype=np.float32)
        for (layer, step), vals in cell_ratios.items():
            mat[layer_to_row[layer], step] = float(np.mean(vals)) if vals else 0.0
        return mat, [f"layer{layer}" for layer in layers]

    written: list[str] = []
    for kind in ["router", "fc1", "fc2"]:
        mat, y_labels = layer_kind_matrix(kind)
        if mat.size == 0:
            continue
        fig_path = f"{figures_dir}/moe_{kind}_layer_heatmap.png"
        save_heatmap(
            fig_path,
            mat,
            title=f"MoE {kind}: per-layer avg sparsity over steps",
            xlabel="offline step",
            ylabel="layer",
            x_ticks=[str(i) for i in range(num_steps)],
            y_ticks=y_labels,
            vmin=0.0,
            vmax=1.0,
            cmap="magma",
        )
        written.append(fig_path)
    return written


def plot_top_moe_expert_heatmaps(
    moe_rows: list[dict[str, Any]],
    num_steps: int,
    figures_dir: str,
    *,
    top_layers_per_kind: int = 2,
) -> list[str]:
    def top_layers_for_kind(kind: str, topn: int) -> list[int]:
        layer_to_ratios: dict[int, list[float]] = defaultdict(list)
        for r in moe_rows:
            if r["kind"] != kind:
                continue
            if r["expert_id_global"] == "":
                continue
            layer_to_ratios[int(r["layer"])].append(float(r.get("sparsity", 0.0)))
        scored = [(float(np.mean(v)) if v else 0.0, layer) for layer, v in layer_to_ratios.items()]
        scored.sort(reverse=True)
        return [layer for _, layer in scored[:topn]]

    expert_figure_paths: list[str] = []
    for kind in ["fc1", "fc2"]:
        for layer in top_layers_for_kind(kind, top_layers_per_kind):
            global_ids = sorted(
                {
                    int(r["expert_id_global"])
                    for r in moe_rows
                    if r["kind"] == kind
                    and int(r["layer"]) == layer
                    and r["expert_id_global"] != ""
                }
            )
            if not global_ids:
                continue
            gid_to_row = {gid: i for i, gid in enumerate(global_ids)}
            mat_ex = np.zeros((len(global_ids), num_steps), dtype=np.float32)
            for r in moe_rows:
                if r["kind"] != kind or int(r["layer"]) != layer or r["expert_id_global"] == "":
                    continue
                gid = int(r["expert_id_global"])
                step = int(r["step"])
                mat_ex[gid_to_row[gid], step] = float(r.get("sparsity", 0.0))
            fig_path = f"{figures_dir}/moe_{kind}_experts_layer{layer}.png"
            save_heatmap(
                fig_path,
                mat_ex,
                title=f"MoE {kind} expert tensors: sparsity (layer={layer}, global experts)",
                xlabel="offline step",
                ylabel="global_expert_id",
                x_ticks=[str(i) for i in range(num_steps)],
                y_ticks=[str(g) for g in global_ids],
                vmin=0.0,
                vmax=1.0,
                cmap="magma",
            )
            expert_figure_paths.append(fig_path)
    return expert_figure_paths
