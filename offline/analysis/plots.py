from __future__ import annotations

import os
from collections.abc import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# NOTE: Keep this module pure-matplotlib (Agg backend) for headless environments.


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_line(
    out_path: str,
    x: Iterable[float],
    y: Iterable[float],
    title: str,
    xlabel: str,
    ylabel: str,
    *,
    y_lim: tuple[float, float] | None = None,
) -> None:
    xs = list(x)
    ys = list(y)
    plt.figure(figsize=(10, 4))
    plt.plot(xs, ys, linewidth=1.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_heatmap(
    out_path: str,
    mat: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    x_ticks: list[str] | None = None,
    y_ticks: list[str] | None = None,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
) -> None:
    if mat.ndim != 2:
        raise ValueError("mat must be 2D")
    h, w = mat.shape
    fig_w = min(18.0, max(8.0, w * 0.12))
    fig_h = min(18.0, max(6.0, h * 0.16))
    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(im, fraction=0.03, pad=0.02)

    if x_ticks is not None:
        step = max(1, int(len(x_ticks) / 16))
        xs = list(range(0, len(x_ticks), step))
        plt.xticks(xs, [x_ticks[i] for i in xs], rotation=45, ha="right", fontsize=8)
    if y_ticks is not None:
        step = max(1, int(len(y_ticks) / 40))
        ys = list(range(0, len(y_ticks), step))
        plt.yticks(ys, [y_ticks[i] for i in ys], fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_hist(
    out_path: str,
    values: np.ndarray,
    *,
    bins: int,
    title: str,
    xlabel: str,
    ylabel: str = "count",
    range_: tuple[float, float] | None = None,
) -> None:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    plt.figure(figsize=(8.5, 4.2))
    plt.hist(x, bins=bins, range=range_, color="#3b82f6", alpha=0.85, edgecolor="#1f2937")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_multi_line(
    out_path: str,
    *,
    x: list[int],
    series: dict[str, list[float]],
    title: str,
    xlabel: str,
    ylabel: str,
    y_lim: tuple[float, float] | None = None,
) -> None:
    plt.figure(figsize=(10, 4.4))
    for name, ys in series.items():
        plt.plot(x, ys, linewidth=1.6, label=name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2, loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_cdf(
    out_path: str,
    *,
    series: dict[str, np.ndarray],
    title: str,
    xlabel: str,
    ylabel: str = "CDF",
) -> None:
    """Plot empirical CDF curves."""
    plt.figure(figsize=(9.5, 4.6))
    for name, vals in series.items():
        x = np.asarray(vals, dtype=np.float64).reshape(-1)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        x.sort()
        y = np.linspace(0.0, 1.0, num=x.size, endpoint=True)
        plt.plot(x, y, linewidth=1.6, label=name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2, loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
