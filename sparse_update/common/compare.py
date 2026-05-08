import torch


@torch.no_grad()
def get_sparse_diff_indices(
    curr_param: torch.Tensor, prev_param: torch.Tensor, offset: int = 0
) -> torch.Tensor:
    """Given two tensors, return the flattened indices of the elements that are different."""
    # TODO: Use triton to implement this.
    mask = curr_param.view(-1) != prev_param.view(-1)
    return mask.nonzero(as_tuple=False).view(-1).to(torch.int32) + offset


@torch.no_grad()
def get_sparse_diff(
    curr_param: torch.Tensor, prev_param: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Given two tensors, return the flattened indices of the elements that are different."""
    # TODO: Use triton to implement this.
    indices = get_sparse_diff_indices(curr_param, prev_param)

    curr_sparse_values = curr_param.index_select(0, indices)
    prev_sparse_values = prev_param.index_select(0, indices)

    return indices, curr_sparse_values, prev_sparse_values
