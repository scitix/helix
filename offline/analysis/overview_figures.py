"""High-level plots: element sparsity, param-level heatmap, param activity ratio."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from ..common.metrics import dtype_nbytes, summarize_element_sparsity
from ..common.naming import parse_decoder_layer_id
from .plots import save_cdf, save_heatmap, save_hist, save_line, save_multi_line


def element_sparsity_rows(per_param_step_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    elem = summarize_element_sparsity(per_param_step_rows)
    return [
        {
            "step": e.step,
            "nnz": e.nnz,
            "numel": e.numel,
            "nnz_ratio": e.nnz_ratio,
            "sparsity": e.sparsity,
        }
        for e in elem
    ]


def param_inactive_ratio_rows(
    per_param_step_rows: list[dict[str, Any]],
    *,
    num_steps: int,
) -> list[dict[str, Any]]:
    unique_params = {row["param_name"] for row in per_param_step_rows}
    total_params = len(unique_params)
    active_by_step: dict[int, int] = defaultdict(int)
    for row in per_param_step_rows:
        if int(row["nnz"]) > 0:
            active_by_step[int(row["step"])] += 1
    return [
        {
            "step": int(step),
            "inactive_ratio": 1.0 - (active_by_step.get(int(step), 0) / max(1, total_params)),
        }
        for step in range(num_steps)
    ]


def param_sparsity_quantile_rows(
    per_param_step_rows: list[dict[str, Any]],
    *,
    num_steps: int,
    quantiles: list[float] | None = None,
) -> list[dict[str, Any]]:
    if quantiles is None:
        quantiles = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
    by_step: dict[int, list[float]] = defaultdict(list)
    for row in per_param_step_rows:
        step = int(row["step"])
        nnz_ratio = float(row["nnz_ratio"])
        by_step[step].append(1.0 - nnz_ratio)

    out: list[dict[str, Any]] = []
    for step in range(num_steps):
        xs = np.asarray(by_step.get(step, []), dtype=np.float64)
        if xs.size == 0:
            q = [float("nan")] * len(quantiles)
        else:
            q = [float(np.quantile(xs, qq)) for qq in quantiles]
        row_out: dict[str, Any] = {"step": step, "count_params": int(xs.size)}
        for qq, val in zip(quantiles, q, strict=False):
            key = f"q{round(qq * 100):02.0f}" if qq < 1.0 else "q100"
            row_out[key] = val
        out.append(row_out)
    return out


def group_step_rows(per_param_step_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], dict[str, float]] = {}
    for row in per_param_step_rows:
        group = str(row["group"])
        step = int(row["step"])
        key = (group, step)
        acc = by_key.get(key)
        if acc is None:
            acc = {"nnz": 0.0, "numel": 0.0, "params": 0.0, "inactive": 0.0}
            by_key[key] = acc
        nnz = float(row["nnz"])
        numel = float(row["numel"])
        acc["nnz"] += nnz
        acc["numel"] += numel
        acc["params"] += 1.0
        acc["inactive"] += 1.0 if int(row["nnz"]) == 0 else 0.0

    out: list[dict[str, Any]] = []
    for (group, step), acc in sorted(by_key.items(), key=lambda x: (x[0][1], x[0][0])):
        nnz_ratio = (acc["nnz"] / acc["numel"]) if acc["numel"] > 0 else 0.0
        out.append(
            {
                "group": group,
                "step": int(step),
                "element_sparsity": 1.0 - float(nnz_ratio),
                "inactive_ratio": float(acc["inactive"] / max(1.0, acc["params"])),
                "params": int(acc["params"]),
            }
        )
    return out


def layer_bucket_step_rows(
    per_param_step_rows: list[dict[str, Any]],
    *,
    num_steps: int,
    n_tail: int = 4,
) -> list[dict[str, Any]]:
    layer_ids: list[int] = []
    for row in per_param_step_rows:
        lid = parse_decoder_layer_id(str(row["param_name"]))
        if lid is not None:
            layer_ids.append(int(lid))
    if not layer_ids:
        return []
    max_layer = max(layer_ids)
    total_layers = max_layer + 1
    tail = min(n_tail, max(1, total_layers // 3))
    front_max = tail - 1
    back_min = max(0, total_layers - tail)

    def bucket(lid: int) -> str:
        if lid <= front_max:
            return "front"
        if lid >= back_min:
            return "back"
        return "mid"

    by_key: dict[tuple[str, int], dict[str, float]] = {}
    for row in per_param_step_rows:
        lid = parse_decoder_layer_id(str(row["param_name"]))
        if lid is None:
            continue
        b = bucket(int(lid))
        step = int(row["step"])
        key = (b, step)
        acc = by_key.get(key)
        if acc is None:
            acc = {"nnz": 0.0, "numel": 0.0, "params": 0.0}
            by_key[key] = acc
        acc["nnz"] += float(row["nnz"])
        acc["numel"] += float(row["numel"])
        acc["params"] += 1.0

    out: list[dict[str, Any]] = []
    for step in range(num_steps):
        for b in ["front", "mid", "back"]:
            acc = by_key.get((b, step), {"nnz": 0.0, "numel": 0.0, "params": 0.0})
            nnz_ratio = (acc["nnz"] / acc["numel"]) if acc["numel"] > 0 else 0.0
            out.append(
                {
                    "bucket": b,
                    "step": step,
                    "element_sparsity": 1.0 - float(nnz_ratio),
                    "params": int(acc["params"]),
                    "bucket_def": f"layers: front<= {front_max}, back>= {back_min} (total={total_layers})",
                }
            )
    return out


def layer_sparsity_last_step_rows(
    per_param_step_rows: list[dict[str, Any]],
    *,
    step: int,
) -> list[dict[str, Any]]:
    acc: dict[int, dict[str, float]] = {}
    for row in per_param_step_rows:
        if int(row.get("step", -1)) != int(step):
            continue
        lid = parse_decoder_layer_id(str(row.get("param_name", "")))
        if lid is None:
            continue
        nnz = float(row.get("nnz", 0.0))
        numel = float(row.get("numel", 0.0))
        if lid not in acc:
            acc[lid] = {"nnz": 0.0, "numel": 0.0, "params": 0.0}
        acc[lid]["nnz"] += nnz
        acc[lid]["numel"] += numel
        acc[lid]["params"] += 1.0
    out: list[dict[str, Any]] = []
    for lid in sorted(acc.keys()):
        a = acc[lid]
        nnz_ratio = (a["nnz"] / a["numel"]) if a["numel"] > 0 else 0.0
        out.append(
            {
                "step": int(step),
                "layer": int(lid),
                "element_sparsity": 1.0 - float(nnz_ratio),
                "params": int(a["params"]),
            }
        )
    return out


def plot_layer_sparsity_cdf(
    layer_rows: list[dict[str, Any]],
    *,
    out_path: str,
    step: int,
) -> None:
    if not layer_rows:
        return
    vals = np.asarray([float(r.get("element_sparsity", 0.0)) for r in layer_rows], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    vals = np.clip(vals, 0.0, 1.0)
    save_cdf(
        out_path,
        series={f"step{int(step)}": vals},
        title=f"Per-layer element sparsity CDF (step={int(step)})",
        xlabel="layer element sparsity (1 - nnz/numel)",
    )


def plot_group_sparsity_lines(
    group_rows: list[dict[str, Any]],
    *,
    out_path: str,
    topk_groups: int = 8,
) -> None:
    if not group_rows:
        return
    max_step = max(int(r["step"]) for r in group_rows)
    steps = list(range(max_step + 1))
    groups = sorted({str(r["group"]) for r in group_rows})
    group_to_params: dict[str, int] = defaultdict(int)
    for r in group_rows:
        group_to_params[str(r["group"])] = max(group_to_params[str(r["group"])], int(r["params"]))
    groups.sort(key=lambda g: group_to_params.get(g, 0), reverse=True)
    chosen = groups[: max(1, topk_groups)]

    series: dict[str, list[float]] = {}
    for g in chosen:
        ys = [float("nan")] * len(steps)
        for r in group_rows:
            if str(r["group"]) != g:
                continue
            s = int(r["step"])
            ys[s] = 100.0 * float(r["element_sparsity"])
        last = 0.0
        for i, v in enumerate(ys):
            if np.isnan(v):
                ys[i] = last
            else:
                last = float(v)
        series[g] = ys

    save_multi_line(
        out_path,
        x=steps,
        series=series,
        title=f"Group element sparsity (top-{len(series)} by param count)",
        xlabel="offline step",
        ylabel="element sparsity (1 - nnz/numel, %)",
        y_lim=(0.0, 100.0),
    )


def plot_layer_bucket_lines(
    layer_bucket_rows: list[dict[str, Any]],
    *,
    out_path: str,
) -> None:
    if not layer_bucket_rows:
        return
    max_step = max(int(r["step"]) for r in layer_bucket_rows)
    steps = list(range(max_step + 1))
    buckets = ["front", "mid", "back"]
    series: dict[str, list[float]] = {b: [0.0] * len(steps) for b in buckets}
    for r in layer_bucket_rows:
        b = str(r["bucket"])
        if b not in series:
            continue
        s = int(r["step"])
        series[b][s] = 100.0 * float(r["element_sparsity"])
    save_multi_line(
        out_path,
        x=steps,
        series=series,
        title="Layer position element sparsity (front/mid/back)",
        xlabel="offline step",
        ylabel="element sparsity (1 - nnz/numel, %)",
        y_lim=(0.0, 100.0),
    )


def plot_overall_and_inactive(
    *,
    element_rows: list[dict[str, Any]],
    inactive_rows: list[dict[str, Any]],
    out_path: str,
) -> None:
    if not element_rows or not inactive_rows:
        return
    steps = [int(r["step"]) for r in element_rows]
    y_spars = [100.0 * float(r.get("sparsity", 0.0)) for r in element_rows]
    y_inact = [100.0 * float(r.get("inactive_ratio", 0.0)) for r in inactive_rows]
    n = min(len(steps), len(y_inact))
    steps, y_spars, y_inact = steps[:n], y_spars[:n], y_inact[:n]

    def zoom(vals: list[float]) -> tuple[float, float]:
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        pad = max(0.05, 0.2 * (vmax - vmin))
        return (max(0.0, vmin - pad), min(100.0, vmax + pad))

    s_lim = zoom(y_spars) if y_spars else (0.0, 100.0)
    i_lim = zoom(y_inact) if y_inact else (0.0, 100.0)

    plt.figure(figsize=(10.2, 4.6))
    ax1 = plt.gca()
    ax1.plot(steps, y_spars, linewidth=1.8, color="#111827", label="element sparsity (%)")
    ax1.set_xlabel("offline step")
    ax1.set_ylabel("element sparsity (%)", color="#111827")
    ax1.tick_params(axis="y", labelcolor="#111827")
    ax1.set_ylim(s_lim)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(steps, y_inact, linewidth=1.8, color="#2563eb", label="param inactive ratio (%)")
    ax2.set_ylabel("param inactive ratio (%)", color="#2563eb")
    ax2.tick_params(axis="y", labelcolor="#2563eb")
    ax2.set_ylim(i_lim)

    plt.title("Overall element sparsity vs param inactivity (zoomed axes)")
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, fontsize=8, ncol=2, loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def rank_params_by_mean_nnz_ratio(per_param_step_rows: list[dict[str, Any]]) -> list[str]:
    by_param: dict[str, list[float]] = defaultdict(list)
    for row in per_param_step_rows:
        by_param[row["param_name"]].append(float(row["nnz_ratio"]))
    scored = [(float(np.mean(v)) if v else 0.0, name) for name, v in by_param.items()]
    scored.sort(reverse=True)
    return [name for _, name in scored]


def plot_param_nnz_heatmap(
    per_param_step_rows: list[dict[str, Any]],
    *,
    top_param_names: list[str],
    num_steps: int,
    out_path: str,
) -> None:
    name_to_row = {name: i for i, name in enumerate(top_param_names)}
    top_set = set(top_param_names)
    mat = np.zeros((len(top_param_names), num_steps), dtype=np.float32)
    for row in per_param_step_rows:
        name = row["param_name"]
        if name not in top_set:
            continue
        step = int(row["step"])
        mat[name_to_row[name], step] = float(row["nnz_ratio"])

    y_labels = [_short_param_label(name) for name in top_param_names]
    nonzero = mat[mat > 0]
    if nonzero.size:
        vmax = float(np.quantile(nonzero, 0.995))
        vmax = max(vmax, 1e-8)
    else:
        vmax = 1.0
    save_heatmap(
        out_path,
        mat,
        title=f"Param-level nnz_ratio heatmap (top-{len(top_param_names)} by avg nnz_ratio)",
        xlabel="offline step",
        ylabel="param",
        x_ticks=[str(i) for i in range(num_steps)],
        y_ticks=y_labels,
        vmin=0.0,
        vmax=vmax,
        cmap="magma",
    )


def plot_param_sparsity_histogram(
    per_param_step_rows: list[dict[str, Any]],
    *,
    step: int,
    out_path: str,
    bins: int = 50,
) -> None:
    xs: list[float] = []
    for row in per_param_step_rows:
        if int(row["step"]) != int(step):
            continue
        xs.append(100.0 * (1.0 - float(row["nnz_ratio"])))
    save_hist(
        out_path,
        np.asarray(xs, dtype=np.float64),
        bins=bins,
        range_=(0.0, 100.0),
        title=f"Param sparsity distribution at step {int(step)} (unweighted)",
        xlabel="param sparsity (1 - nnz/numel, %)",
        ylabel="num params",
    )


def _short_param_label(full_name: str, max_len: int = 60) -> str:
    if len(full_name) <= max_len:
        return full_name
    return "..." + full_name[-(max_len - 3) :]


def plot_param_activity_ratio(
    per_param_step_rows: list[dict[str, Any]],
    *,
    num_steps: int,
    out_path: str,
) -> None:
    rows = param_inactive_ratio_rows(per_param_step_rows, num_steps=num_steps)
    ys = [100.0 * float(r["inactive_ratio"]) for r in rows]
    if ys:
        y_min = float(np.min(ys))
        y_max = float(np.max(ys))
        pad = max(0.05, 0.2 * (y_max - y_min))
        y0 = max(0.0, y_min - pad)
        y1 = min(100.0, y_max + pad)
        y_lim = (y0, y1)
    else:
        y_lim = (0.0, 100.0)
    save_line(
        out_path,
        x=[int(r["step"]) for r in rows],
        y=ys,
        title="Param-level sparsity (no update) ratio over steps",
        xlabel="offline step",
        ylabel="inactive_params / total_params (%)",
        y_lim=y_lim,
    )


def build_communication_estimate_rows(
    per_param_step_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in per_param_step_rows:
        dtype_str = str(row.get("dtype", ""))
        if not dtype_str:
            continue
        nnz = int(row["nnz"])
        numel = int(row["numel"])
        step = int(row["step"])
        dense_bytes = numel * dtype_nbytes(dtype_str)
        sparse_bytes = nnz * (4 + dtype_nbytes(dtype_str))
        rows.append(
            {
                "param_name": row["param_name"],
                "step": step,
                "dtype": dtype_str,
                "nnz": nnz,
                "numel": numel,
                "dense_bytes": dense_bytes,
                "sparse_bytes_est": sparse_bytes,
                "saving_ratio": (1.0 - (float(sparse_bytes) / float(dense_bytes)))
                if dense_bytes
                else 0.0,
            }
        )
    return rows
