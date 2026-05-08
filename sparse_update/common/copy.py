import torch


def sparse_copy(self: torch.Tensor, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    assert self.shape == src.shape, f"self.shape: {self.shape}, src.shape: {src.shape}"
    mask = ~torch.isnan(src)
    self[mask] = src[mask]
    return self
