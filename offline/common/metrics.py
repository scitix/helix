from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch


def dtype_nbytes(dtype_str: str) -> int:
    # info.json stores dtype like "torch.bfloat16"
    if dtype_str.endswith("bfloat16") or dtype_str.endswith("float16"):
        return 2
    if dtype_str.endswith("float32"):
        return 4
    if dtype_str.endswith("float64"):
        return 8
    if dtype_str.endswith("int8") or dtype_str.endswith("uint8") or dtype_str.endswith("bool"):
        return 1
    if dtype_str.endswith("int16") or dtype_str.endswith("uint16"):
        return 2
    if dtype_str.endswith("int32") or dtype_str.endswith("uint32"):
        return 4
    if dtype_str.endswith("int64") or dtype_str.endswith("uint64"):
        return 8
    raise ValueError(f"Unknown dtype string: {dtype_str!r}")


def torch_unique_cpu_int64(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.to(device="cpu", dtype=torch.int64).view(-1)
    x = x.view(-1)
    if x.device.type != "cpu":
        x = x.to("cpu")
    if x.dtype != torch.int64:
        x = x.to(torch.int64)
    return torch.unique(x, sorted=True)


def torch_unique_int64(
    x: torch.Tensor, *, device: torch.device | str | None = None
) -> torch.Tensor:
    """Return sorted unique int64 indices on the requested device.

    Intended for merge-time acceleration. Callers can move the result back to CPU for saving.
    """
    if x.numel() == 0:
        if device is None:
            return x.to(dtype=torch.int64).view(-1)
        return x.to(device=device, dtype=torch.int64).view(-1)
    x = x.view(-1)
    if device is not None:
        x = x.to(device=device)
    if x.dtype != torch.int64:
        x = x.to(torch.int64)
    return torch.unique(x, sorted=True)


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    a_u = torch_unique_cpu_int64(a)
    b_u = torch_unique_cpu_int64(b)
    if a_u.numel() == 0 and b_u.numel() == 0:
        return 1.0
    inter = int(torch.isin(a_u, b_u).sum().item())
    uni = int(a_u.numel() + b_u.numel() - inter)
    return float(inter) / float(uni) if uni else 1.0


@dataclass(frozen=True)
class ElementSparsity:
    step: int
    nnz: int
    numel: int

    @property
    def nnz_ratio(self) -> float:
        return float(self.nnz) / float(self.numel) if self.numel else 0.0

    @property
    def sparsity(self) -> float:
        return 1.0 - self.nnz_ratio


def summarize_element_sparsity(per_param_step_rows: Iterable[dict]) -> list[ElementSparsity]:
    # rows must include: step, nnz, numel
    accum: dict[int, tuple[int, int]] = {}
    for r in per_param_step_rows:
        s = int(r["step"])
        nnz = int(r["nnz"])
        numel = int(r["numel"])
        cur = accum.get(s, (0, 0))
        accum[s] = (cur[0] + nnz, cur[1] + numel)
    out = [ElementSparsity(step=s, nnz=v[0], numel=v[1]) for s, v in sorted(accum.items())]
    return out


def bucketize_indices(indices: torch.Tensor, numel: int, buckets: int) -> np.ndarray:
    """Return bucket counts (length=buckets) for indices in [0,numel).

    If indices are in global coordinates, caller should first shift to local [0,numel) if desired.
    """
    if buckets <= 0:
        raise ValueError("buckets must be > 0")
    if numel <= 0:
        return np.zeros((buckets,), dtype=np.int64)
    x = indices
    if x.device.type != "cpu":
        x = x.to("cpu")
    x = x.view(-1)
    if x.numel() == 0:
        return np.zeros((buckets,), dtype=np.int64)
    if x.dtype != torch.int64:
        x = x.to(torch.int64)
    # clamp to valid range to avoid rare outliers breaking plots
    x = torch.clamp(x, 0, numel - 1)
    # bucket id = floor(x / numel * buckets)
    bid = torch.div(x * buckets, max(numel, 1), rounding_mode="floor")
    bid = torch.clamp(bid, 0, buckets - 1)
    bc = torch.bincount(bid, minlength=buckets)
    return bc.to(torch.int64).cpu().numpy()
