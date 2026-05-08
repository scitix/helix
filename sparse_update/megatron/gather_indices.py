import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

import torch
import torch.distributed as dist
from torch.nn import Parameter

if TYPE_CHECKING:
    from megatron.core.distributed import DistributedDataParallel as DDP


class SparseDPGatherHelper(ABC):
    @staticmethod
    def _collect_params(
        models: list["DDP"],
    ) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
        """Collect dense and expert-parallel params from models.
        Copy from megatron.core.distributed.distributed_data_parallel.py : DistributedDataParallel.__init__() method.

        Returns:
            dense_params: list of dense params
            expert_parallel_params: list of expert-parallel params
        """

        dense_params = []
        expert_parallel_params = []

        for model in models:
            for _, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if getattr(param, "allreduce", True):
                    dense_params.append(param)
                else:
                    expert_parallel_params.append(param)

        return dense_params, expert_parallel_params

    def __init__(self, models: Sequence["DDP"]) -> None:
        from megatron.core import mpu
        from megatron.core.distributed import DistributedDataParallel as DDP

        if not isinstance(models, list):
            models = [models]
        assert len(models) > 0, "models must be a non-empty list"
        assert all(isinstance(model, DDP) for model in models), "models must be a list of DDP"

        self.models = models
        self.dense_params, self.expert_parallel_params = self._collect_params(models)

        self._dp_group = mpu.get_data_parallel_group(with_context_parallel=True)
        self._dp_ep_group = mpu.get_expert_data_parallel_group()

    def sync_sparse_indices(self) -> None:
        """Get dp-local data and gather them as global item."""
        # Init dp-local indices for each param
        dense_param_to_local_item = self._init_dp_local_item(self.dense_params)
        expert_parallel_param_to_local_item = self._init_dp_local_item(self.expert_parallel_params)

        # Gather dp-local data as global item
        dense_param_to_global_item = self._gather_dp_items(
            dense_param_to_local_item, self._dp_group
        )
        moe_param_to_global_item = self._gather_dp_items(
            expert_parallel_param_to_local_item, self._dp_ep_group
        )

        # Set dp-global data to params
        self._set_dp_global_item(dense_param_to_global_item)
        self._set_dp_global_item(moe_param_to_global_item)

    @abstractmethod
    def _init_dp_local_item(
        self, params: list[torch.nn.Parameter]
    ) -> dict[torch.nn.Parameter, Any]: ...

    @abstractmethod
    def _set_dp_global_item(self, param_to_global_item: dict[torch.nn.Parameter, Any]) -> None: ...

    def _gather_dp_items(
        self, param_to_local_item: dict[torch.nn.Parameter, Any], group: dist.ProcessGroup
    ) -> dict[torch.nn.Parameter, Any]:
        if not param_to_local_item:
            return param_to_local_item
        if torch.distributed.get_world_size(group) == 1:
            return param_to_local_item
        return self._gather_dp_items_impl(param_to_local_item, group)

    @abstractmethod
    def _gather_dp_items_impl(
        self, param_to_local_item: dict[torch.nn.Parameter, Any], group: dist.ProcessGroup
    ) -> dict[torch.nn.Parameter, Any]:
        raise NotImplementedError


class SparseIndicesDPGatherHelper(SparseDPGatherHelper):
    _GLOBAL_INDICES_ATTR_NAME: Final[str] = "_sparse_dp_global_indices"

    def __init__(self, models: list[torch.nn.Module]) -> None:
        super().__init__(models)

    def _init_dp_local_item(
        self, params: list[torch.nn.Parameter]
    ) -> dict[torch.nn.Parameter, Any]:
        param_to_local_item: dict[torch.nn.Parameter, Any] = {}
        for param in params:
            sparse_manager = getattr(param, "_sparse_manager", None)

            if sparse_manager is None:
                dp_local_indices = None
            else:
                dp_local_indices = sparse_manager.fetch_dp_local_indices()

            param_to_local_item[param] = dp_local_indices

        return param_to_local_item

    def _set_dp_global_item(self, param_to_global_item: dict[torch.nn.Parameter, Any]) -> None:
        for param, global_item in param_to_global_item.items():
            assert global_item is not None
            setattr(param, self._GLOBAL_INDICES_ATTR_NAME, global_item)

    @staticmethod
    def _get_padded_flat_tensor(
        unpadded_object: list[int] | int,
        padding_value: int,
        device: torch.device,
        dtype: torch.dtype = torch.int32,
        alignment: int = 256,
    ) -> torch.Tensor:
        """Get a padded flat tensor from an unpadded object. Only recieve list[int] or int as unpadded_object."""
        if isinstance(unpadded_object, list):
            origin_len = len(unpadded_object)
        elif isinstance(unpadded_object, int):
            origin_len = unpadded_object
        else:
            raise ValueError(
                f"unpadded_object must be a list[int] or int, got {type(unpadded_object)}"
            )

        aligned_len = math.ceil(origin_len / alignment) * alignment
        padding_len = aligned_len - origin_len

        if isinstance(unpadded_object, list):
            return torch.tensor(
                unpadded_object + [padding_value] * padding_len, dtype=dtype, device=device
            )
        else:
            return torch.full((aligned_len,), padding_value, dtype=dtype, device=device)

    @staticmethod
    def _get_global_nnz_list(
        local_indices_list: list[torch.Tensor | None], group: dist.ProcessGroup
    ) -> torch.Tensor:
        """Get global nnz list for each param.
        Assumption: for each param, at most one dp-rank owns/maintains indices (others nnz=0).
        Under this assumption, SUM and MAX should produce identical tensors. We validate this  to catch protocol violations early.
        """
        # params_num = len(local_indices_list)
        device = torch.cuda.current_device()
        local_nnz_list: list[int] = [
            int(indices.numel()) if indices is not None else 0 for indices in local_indices_list
        ]  # [params_num]
        local_nnz_tensor: torch.Tensor = SparseIndicesDPGatherHelper._get_padded_flat_tensor(
            unpadded_object=local_nnz_list, padding_value=0, device=device
        )  # [params_num_padded]

        global_nnz_tensor_gathered = [
            torch.empty_like(local_nnz_tensor) for _ in range(dist.get_world_size(group))
        ]  # [world_size, params_num_padded]
        dist.all_gather(global_nnz_tensor_gathered, local_nnz_tensor, group=group)

        return torch.stack(global_nnz_tensor_gathered)

    def _gather_dp_items_impl(
        self, param_to_local_indices: dict[Parameter, torch.Tensor | None], group: dist.ProcessGroup
    ) -> dict[Parameter, torch.Tensor | None]:
        """Gather sparse indices between dp-group."""
        # Step-1 : Init && Check
        params: list[Parameter] = list(param_to_local_indices.keys())
        local_indices_list: list[torch.Tensor | None] = list(param_to_local_indices.values())
        params_num = len(params)
        group_rank = dist.get_rank(group)

        # Step-2 : All-reduce nnz for each param
        global_nnz_tensor_gathered: torch.Tensor = self._get_global_nnz_list(
            local_indices_list, group
        )  # [world_size, params_num_padded]

        # Step-3 : Calculate global offset and nnz for each param
        global_nnz_tensor_gathered_sum: torch.Tensor = global_nnz_tensor_gathered.sum(dim=0)[
            :params_num
        ]  # [params_num]
        global_offset_list_tensor: torch.Tensor = (
            torch.cumsum(global_nnz_tensor_gathered_sum, dim=0) - global_nnz_tensor_gathered_sum
        )  # [params_num]

        param_idx_to_nnz_list: torch.Tensor = global_nnz_tensor_gathered.transpose(
            0, 1
        )  # [params_num_padded, world_size]
        # Per-param prefix sum over ranks (dim=1), not over params.
        param_idx_to_local_offset: torch.Tensor = (
            torch.cumsum(param_idx_to_nnz_list, dim=1) - param_idx_to_nnz_list
        )  # [params_num_padded, world_size]

        # To CPU List for easy access
        global_nnz_list: list[int] = global_nnz_tensor_gathered_sum.tolist()  # [params_num]
        global_offset_list: list[int] = global_offset_list_tensor.tolist()  # [params_num]
        param_idx_to_local_offset: list[list[int]] = param_idx_to_local_offset.tolist()[
            :params_num
        ]  # [params_num, world_size]

        # Step-4 : Create global merged indices and init by local indices
        total_nnz_num = int(global_nnz_tensor_gathered_sum.sum().item())
        global_merged_indices: torch.Tensor = self._get_padded_flat_tensor(
            unpadded_object=total_nnz_num, padding_value=-1, device=torch.cuda.current_device()
        )  # [total_nnz_num_padded]

        # Step-5 : Copy local indices to global merged indices
        for param_idx in range(params_num):
            # Find indices for this param
            global_param_nnz: int = global_nnz_list[param_idx]
            global_param_offset: int = global_offset_list[param_idx]

            local_indices: torch.Tensor | None = local_indices_list[param_idx]
            local_param_nnz: int = local_indices.numel() if local_indices is not None else 0
            local_param_offset: int = param_idx_to_local_offset[param_idx][group_rank]

            assert local_param_nnz <= global_param_nnz
            if local_param_nnz == 0:
                continue

            final_offset = global_param_offset + local_param_offset
            global_merged_indices[final_offset : final_offset + local_param_nnz].copy_(
                local_indices,
                non_blocking=True,
            )

        # Step-6 : All-reduce global merged indices
        torch.distributed.all_reduce(global_merged_indices, op=dist.ReduceOp.MAX, group=group)

        # Step-7 : Set sparse indices to params
        param_to_global_indices: dict[Parameter, torch.Tensor] = {}
        for param, offset, nnz in zip(params, global_offset_list, global_nnz_list, strict=True):
            param_to_global_indices[param] = global_merged_indices[offset : offset + nnz]

        return param_to_global_indices


class SparseDataDPGatherHelper(SparseDPGatherHelper):
    def __init__(self, models: list[torch.nn.Module]) -> None:
        raise NotImplementedError

    def _init_dp_local_item(
        self, params: list[torch.nn.Parameter]
    ) -> dict[torch.nn.Parameter, Any]: ...

    def _set_dp_global_item(self, param_to_global_item: dict[torch.nn.Parameter, Any]) -> None: ...

    def _gather_dp_items_impl(
        self, param_to_local_item: dict[torch.nn.Parameter, Any], group: dist.ProcessGroup
    ) -> dict[torch.nn.Parameter, Any]: ...


_SPARSE_DP_GATHER_HELPER = None


def _get_sparse_dp_gather_helper(check_initialized: bool = False) -> SparseDPGatherHelper:
    global _SPARSE_DP_GATHER_HELPER
    if not check_initialized:
        assert _SPARSE_DP_GATHER_HELPER is not None, (
            "SparseDataDPGatherHelper has not been initialized"
        )
    return _SPARSE_DP_GATHER_HELPER


def _init_sparse_dp_gather_helper(models: list["DDP"], gathered_data_type: str = "indices") -> None:
    global _SPARSE_DP_GATHER_HELPER
    assert _SPARSE_DP_GATHER_HELPER is None, "SparseDataDPGatherHelper has already been initialized"

    assert gathered_data_type == "indices", (
        f"Only support gathered data type 'indices', got {gathered_data_type}"
    )

    _SPARSE_DP_GATHER_HELPER = SparseIndicesDPGatherHelper(models)


def fetch_sparse_indices(param: torch.nn.Parameter) -> torch.Tensor | None:
    return getattr(param, SparseIndicesDPGatherHelper._GLOBAL_INDICES_ATTR_NAME, None)


def gather_sparse_indices_in_dp_group(models: list["DDP"]) -> None:
    sparse_dp_gather_helper = _get_sparse_dp_gather_helper(check_initialized=True)
    if sparse_dp_gather_helper is None:
        _init_sparse_dp_gather_helper(models)
        sparse_dp_gather_helper = _get_sparse_dp_gather_helper()

    sparse_dp_gather_helper.sync_sparse_indices()
