"""Dense vs MoE and expert-level spread (for merged global dumps and single-rank MoE)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ..common.naming import parse_moe_tag
from .plots import save_multi_line


def param_dense_moe_bucket(param_name: str) -> str:
    tag = parse_moe_tag(param_name)
    if tag is None:
        return "dense"
    if tag.kind == "router":
        return "moe_router"
    return "moe_expert"


def build_dense_vs_moe_step_rows(
    per_param_step_rows: list[dict[str, Any]],
    *,
    num_steps: int,
) -> list[dict[str, Any]]:
    acc: dict[int, dict[str, tuple[int, int]]] = defaultdict(
        lambda: {"dense": (0, 0), "moe_router": (0, 0), "moe_expert": (0, 0)}
    )
    for row in per_param_step_rows:
        b = param_dense_moe_bucket(str(row["param_name"]))
        s = int(row["step"])
        nnz, numel = int(row["nnz"]), int(row["numel"])
        cur_nnz, cur_ne = acc[s][b]
        acc[s][b] = (cur_nnz + nnz, cur_ne + numel)

    out: list[dict[str, Any]] = []
    for step in range(int(num_steps)):
        rows_out: dict[str, Any] = {"step": step}

        d = acc.get(step, {})

        def ratio(dct: dict[str, tuple[int, int]], bucket: str) -> float:
            nnz, n = dct.get(bucket, (0, 0))
            return float(nnz) / float(n) if n else 0.0

        rows_out["dense_element_sparsity"] = 1.0 - ratio(d, "dense")
        rows_out["moe_router_element_sparsity"] = 1.0 - ratio(d, "moe_router")
        rows_out["moe_expert_element_sparsity"] = 1.0 - ratio(d, "moe_expert")
        out.append(rows_out)
    return out


def plot_dense_vs_moe_step(
    rows: list[dict[str, Any]],
    *,
    num_steps: int,
    out_path: str,
) -> None:
    if not rows or num_steps <= 0:
        return
    steps = list(range(int(num_steps)))
    series = {
        "dense": [float(r.get("dense_element_sparsity", 0.0)) for r in rows],
        "moe_router": [float(r.get("moe_router_element_sparsity", 0.0)) for r in rows],
        "moe_expert": [float(r.get("moe_expert_element_sparsity", 0.0)) for r in rows],
    }
    save_multi_line(
        out_path,
        x=steps,
        series={k: [100.0 * v for v in vs] for k, vs in series.items()},
        title="Dense vs MoE: element sparsity (%) over offline steps",
        xlabel="offline step",
        ylabel="sparsity (%)",
    )


def build_moe_expert_spread_rows(moe_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    for r in moe_rows:
        if str(r.get("kind")) not in ("fc1", "fc2"):
            continue
        if r.get("expert_id_global") == "":
            continue
        key = (int(r["layer"]), str(r["kind"]), int(r["step"]))
        groups[key].append(float(r.get("sparsity", 0.0)))

    out: list[dict[str, Any]] = []
    for (layer, kind, step), ratios in sorted(groups.items()):
        arr = np.asarray(ratios, dtype=np.float64)
        if arr.size == 0:
            continue
        out.append(
            {
                "layer": layer,
                "kind": kind,
                "step": step,
                "expert_count": int(arr.size),
                "mean_sparsity": float(np.mean(arr)),
                "std_sparsity": float(np.std(arr)),
                "min_sparsity": float(np.min(arr)),
                "max_sparsity": float(np.max(arr)),
            }
        )
    return out


def plot_moe_expert_spread_top_layers(
    spread_rows: list[dict[str, Any]],
    *,
    num_steps: int,
    figures_dir: str,
    top_layers: int = 4,
) -> list[str]:
    if not spread_rows or num_steps <= 0:
        return []

    def layer_mean_std(layer: int, kind: str) -> float:
        xs = [
            float(r["std_sparsity"])
            for r in spread_rows
            if int(r["layer"]) == layer and str(r["kind"]) == kind
        ]
        return float(np.mean(xs)) if xs else 0.0

    layers_by_kind: dict[str, list[int]] = {"fc1": [], "fc2": []}
    for kind in ("fc1", "fc2"):
        seen = {int(r["layer"]) for r in spread_rows if str(r["kind"]) == kind}
        scored = [(layer_mean_std(L, kind), L) for L in seen]
        scored.sort(reverse=True)
        layers_by_kind[kind] = [L for _, L in scored[: max(1, top_layers)]]

    written: list[str] = []
    for kind in ("fc1", "fc2"):
        layers = layers_by_kind.get(kind, [])
        if not layers:
            continue
        series: dict[str, list[float]] = {}
        for layer in layers:
            by_step = {
                int(r["step"]): float(r["std_sparsity"])
                for r in spread_rows
                if int(r["layer"]) == layer and str(r["kind"]) == kind
            }
            series[f"L{layer}"] = [
                100.0 * float(by_step.get(s, 0.0)) for s in range(int(num_steps))
            ]
        out_path = f"{figures_dir}/moe_expert_nnz_spread_{kind}.png"
        save_multi_line(
            out_path,
            x=list(range(int(num_steps))),
            series=series,
            title=f"MoE {kind}: std(sparsity) across experts (x100 %) vs step (top layers)",
            xlabel="offline step",
            ylabel="std(sparsity) (%)",
        )
        written.append(out_path)
    return written
