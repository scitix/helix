"""Global locality analysis across *all* parameters (merged single-rank dumps).

Definition (per param, per step t>0):
    locality_ratio(t) = |I_t ∩ U_{<t}| / |I_t|
"""

from __future__ import annotations

import gc
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from ..common import io as io_mod
from ..common.metrics import torch_unique_cpu_int64
from .plots import save_cdf


def _overlap_count(cur: torch.Tensor, hist: torch.Tensor) -> int:
    if cur.numel() == 0 or hist.numel() == 0:
        return 0
    cur = cur.view(-1).to(device="cpu", dtype=torch.int64)
    hist = hist.view(-1).to(device="cpu", dtype=torch.int64)
    return int(torch.isin(cur, hist).sum().item())


def run_global_param_locality(
    *,
    data_dir: str,
    rank: int = 0,
    step_index: dict[tuple[int, int, int], int],
    rank_to_indices_paths: dict[int, list[tuple[int, str]]],
    out_fig_path: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indices_paths = rank_to_indices_paths.get(int(rank), [])
    if not indices_paths:
        return [], []

    traversal: list[tuple[int, int, int]] = []
    for interval, pt_path in sorted(indices_paths, key=lambda x: x[0]):
        obj = io_mod.load_torch_dict(pt_path)
        n_events = 0
        for v in obj.values():
            n_events = max(n_events, len(io_mod.normalize_event_list(v)))
        del obj
        gc.collect()
        for e in range(int(n_events)):
            s = step_index.get((int(rank), int(interval), int(e)))
            if s is None:
                continue
            traversal.append((int(s), int(interval), int(e)))
    traversal.sort(key=lambda x: x[0])
    if not traversal:
        return [], []

    all_param_names: set[str] = set()
    interval_to_path = {int(i): p for i, p in indices_paths}
    for _, interval, _ in traversal:
        pt_path = interval_to_path.get(int(interval))
        if pt_path is None:
            continue
        obj = io_mod.load_torch_dict(pt_path)
        all_param_names.update(str(k) for k in obj)
        del obj
        gc.collect()

    hist: dict[str, torch.Tensor] = {
        name: torch.empty((0,), dtype=torch.int64) for name in all_param_names
    }

    step_param_rows: list[dict[str, Any]] = []
    step_to_vals: dict[int, list[float]] = defaultdict(list)
    step_to_active: dict[int, int] = defaultdict(int)

    for step, interval, event_idx in traversal:
        pt_path = interval_to_path.get(int(interval))
        if pt_path is None:
            continue
        obj = io_mod.load_torch_dict(pt_path)
        for raw_name, raw_events in obj.items():
            name = str(raw_name)
            events = io_mod.normalize_event_list(raw_events)
            if event_idx >= len(events):
                continue
            cur = events[event_idx].view(-1).to(device="cpu", dtype=torch.int64)
            if cur.numel() == 0:
                continue
            cur_u = torch_unique_cpu_int64(cur)
            h = hist.get(name, torch.empty((0,), dtype=torch.int64))
            seen = _overlap_count(cur_u, h)
            ratio = float(seen) / float(cur_u.numel()) if cur_u.numel() else 0.0
            step_to_vals[int(step)].append(float(ratio))
            step_to_active[int(step)] += 1
            step_param_rows.append(
                {
                    "step": int(step),
                    "param_name": name,
                    "nnz": int(cur_u.numel()),
                    "seen_count": int(seen),
                    "locality_ratio": float(ratio),
                }
            )
            hist[name] = (
                cur_u if h.numel() == 0 else torch_unique_cpu_int64(torch.cat([h, cur_u], dim=0))
            )
        del obj
        gc.collect()

    steps = sorted(step_to_vals.keys())
    cdf_series: dict[str, np.ndarray] = {}
    step_summary_rows: list[dict[str, Any]] = []
    for s in steps:
        vals = np.asarray(step_to_vals.get(int(s), []), dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        vals = np.clip(vals, 0.0, 1.0)
        cdf_series[f"step{s}"] = vals
        if vals.size == 0:
            step_summary_rows.append(
                {
                    "step": int(s),
                    "active_params": int(step_to_active.get(int(s), 0)),
                    "mean": 0.0,
                    "p50": float("nan"),
                    "p90": float("nan"),
                    "p99": float("nan"),
                }
            )
        else:
            step_summary_rows.append(
                {
                    "step": int(s),
                    "active_params": int(step_to_active.get(int(s), 0)),
                    "mean": float(np.mean(vals)),
                    "p50": float(np.quantile(vals, 0.50)),
                    "p90": float(np.quantile(vals, 0.90)),
                    "p99": float(np.quantile(vals, 0.99)),
                }
            )

    save_cdf(
        out_fig_path,
        series=cdf_series,
        title="Global locality across params: CDF of |I_t ∩ U_{<t}| / |I_t|",
        xlabel="locality ratio per param at step t (higher = more repeated indices)",
    )
    return step_summary_rows, step_param_rows
