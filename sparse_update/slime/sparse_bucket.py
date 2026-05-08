from typing import Any

import torch

from ..common.convert import dense_nan_to_sparse_tensor


def _ceil_to_multiple(x: int, multiple: int = 256) -> int:
    if x == 0:
        return 0
    return (x + multiple - 1) // multiple * multiple


class _SparseWeight:
    indices_dtype = torch.int32

    def __init__(self, name: str, dense_nan_tensor: torch.Tensor):
        self.name = name
        self.dtype = dense_nan_tensor.dtype
        self.shape = dense_nan_tensor.shape

        self._to_sparse(dense_nan_tensor)

    def _to_sparse(self, dense_nan_tensor: torch.Tensor):
        self.indices, self.values = dense_nan_to_sparse_tensor(dense_nan_tensor)

    @property
    def nnz(self) -> int:
        return int(self.values.numel())


class _SparseBucketWithDtype:
    indices_dtype = torch.int32

    def __init__(self, dtype: torch.dtype, device: torch.device):
        self.dtype = dtype
        self.device = device
        self.sparse_weights = []

        self.names = []
        self.dtypes = []
        self.shapes = []
        self.nnz_list = []
        self.merged_numel = 0
        self.merged_indices = None
        self.merged_values = None

    def append(self, sparse_weight: _SparseWeight):
        self.sparse_weights.append(sparse_weight)

    def empty(self) -> bool:
        return self.merged_numel == 0

    def stats(self) -> dict[str, Any]:
        pass

    def finalize(self, filter_zero_nnz: bool = True):
        # This object is reused across multiple flushes. Clear previous exported metadata
        # to keep merged buffers aligned with (names/shapes/dtypes/nnz_list).
        self.names.clear()
        self.dtypes.clear()
        self.shapes.clear()
        self.nnz_list.clear()
        self.merged_numel = 0
        self.merged_indices = None
        self.merged_values = None

        nnz_list = [sparse_weight.nnz for sparse_weight in self.sparse_weights]
        self.merged_numel = _ceil_to_multiple(sum(nnz_list))

        if self.merged_numel == 0 and filter_zero_nnz:
            return

        if self.merged_numel > 0:
            self.merged_indices = torch.empty(
                self.merged_numel, dtype=self.indices_dtype, device=self.device
            )
            self.merged_values = torch.empty(
                self.merged_numel, dtype=self.dtype, device=self.device
            )

        offset = 0
        for nnz, sparse_weight in zip(nnz_list, self.sparse_weights, strict=True):
            # Keep metadata for nnz>0, and optionally keep nnz==0 entries when not filtering.
            if nnz > 0 or (not filter_zero_nnz):
                self.names.append(sparse_weight.name)
                self.dtypes.append(sparse_weight.dtype)
                self.shapes.append(sparse_weight.shape)
                self.nnz_list.append(nnz)

            if nnz == 0:
                continue

            self.merged_indices[offset : offset + nnz].copy_(sparse_weight.indices.view(-1))
            self.merged_values[offset : offset + nnz].copy_(sparse_weight.values.view(-1))

            offset += nnz

        self.sparse_weights.clear()


class SparseBucket:
    def __init__(self):
        self.clear()
        self.sparse_cuda_stream = torch.cuda.Stream(device=torch.cuda.current_device())

    def clear(self):
        self.dtype_to_sparse_bucket = {}

        # HTTP Server (Ray.remote) Args
        self.names: list[str] = []
        self.dtypes: list[torch.dtype] = []
        self.shapes: list[torch.Size] = []

        # Extra Args for update_weights_from_distributed()
        self.nnz_list: list[int] = []
        self.merged_dtype_list: list[torch.dtype] = []
        self.merged_numel_list: list[int] = []
        self.merged_indices_list: list[torch.Tensor] = []
        self.merged_values_list: list[torch.Tensor] = []

        self.records_stats = {
            "total_param_nums": 0,
            "updated_param_nums": 0,
            "total_numel": 0,
            "updated_numel": 0,
            "full_update_nbytes": 0,
            "sparse_update_nbytes": 0,
            "compression_ratio": 0,
        }

    def empty(self) -> bool:
        return len(self.dtype_to_sparse_bucket) == 0

    def append(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], force_sync: bool = False
    ):
        if force_sync:
            torch.cuda.current_stream().wait_stream(self.sparse_cuda_stream)
        if not converted_named_tensors:
            return

        for name, dense_nan_tensor in converted_named_tensors:
            dtype = dense_nan_tensor.dtype

            sparse_bucket = self.dtype_to_sparse_bucket.setdefault(
                dtype, _SparseBucketWithDtype(dtype, dense_nan_tensor.device)
            )

            with torch.cuda.stream(self.sparse_cuda_stream):
                sparse_bucket.append(_SparseWeight(name, dense_nan_tensor))

            self.records_stats["total_param_nums"] += 1
            self.records_stats["total_numel"] += dense_nan_tensor.numel()
            self.records_stats["full_update_nbytes"] += dense_nan_tensor.nbytes

    def finalize(self, filter_zero_nnz: bool = True) -> dict[str, Any]:
        assert not self.empty()

        self.names.clear()
        self.dtypes.clear()
        self.shapes.clear()
        self.nnz_list.clear()
        self.merged_dtype_list.clear()
        self.merged_numel_list.clear()
        self.merged_indices_list.clear()
        self.merged_values_list.clear()

        for sparse_bucket in self.dtype_to_sparse_bucket.values():
            sparse_bucket.finalize(filter_zero_nnz)
            sparse_bucket.stats()

            # When filtering, skip dtype-buckets that contain no nnz>0 entries.
            if sparse_bucket.empty() and filter_zero_nnz:
                continue

            self.names.extend(sparse_bucket.names)
            self.dtypes.extend(sparse_bucket.dtypes)
            self.shapes.extend(sparse_bucket.shapes)

            self.nnz_list.extend(sparse_bucket.nnz_list)

            self.merged_dtype_list.append(sparse_bucket.dtype)
            self.merged_numel_list.append(sparse_bucket.merged_numel)
            self.merged_indices_list.append(sparse_bucket.merged_indices)
            self.merged_values_list.append(sparse_bucket.merged_values)

            self.records_stats["updated_param_nums"] += len(sparse_bucket.nnz_list)
            self.records_stats["updated_numel"] += sum(sparse_bucket.nnz_list)
            self.records_stats["sparse_update_nbytes"] += (
                sparse_bucket.merged_indices.nbytes + sparse_bucket.merged_values.nbytes
            )

        return self.stats()

    def stats(self) -> dict[str, Any]:
        full_update_nbytes_GB = self.records_stats["full_update_nbytes"] / 1024**3
        sparse_update_nbytes_GB = self.records_stats["sparse_update_nbytes"] / 1024**3
        compression_ratio = full_update_nbytes_GB / (sparse_update_nbytes_GB + 1e-6)

        stats_result = ""
        stats_result += f"total_param_nums: {self.records_stats['total_param_nums']}, "
        stats_result += f"updated_param_nums: {self.records_stats['updated_param_nums']}, "
        stats_result += f"updated_param_ratio: {self.records_stats['updated_param_nums'] / self.records_stats['total_param_nums']:.2%}; "
        stats_result += f"total_numel: {self.records_stats['total_numel']}, "
        stats_result += f"updated_numel: {self.records_stats['updated_numel']}, "
        stats_result += f"updated_numel_ratio: {self.records_stats['updated_numel'] / self.records_stats['total_numel']:.2%}; "
        stats_result += f"full_update_nbytes: {full_update_nbytes_GB:.2f} GB, "
        stats_result += f"sparse_update_nbytes: {sparse_update_nbytes_GB:.2f} GB, "
        stats_result += f"compression_ratio: {compression_ratio:.2f} x; "

        self.records_stats = {
            "total_param_nums": 0,
            "updated_param_nums": 0,
            "total_numel": 0,
            "updated_numel": 0,
            "full_update_nbytes": 0,
            "sparse_update_nbytes": 0,
        }

        return stats_result
