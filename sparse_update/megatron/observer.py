import os

import torch

from ..common.compare import get_sparse_diff_indices
from .manager import BaseSparseManager
from .sparse_data import SparseData


class SparseObserver(BaseSparseManager):
    def __init__(self, model_param, param_range) -> None:
        super().__init__(model_param, param_range)

        self.prev_main_weight = None  # FP32 snapshot initialized after the first time it is copied

    def before_copy(self, shard_model_weight: torch.Tensor) -> None:
        self.prev_model_weight = shard_model_weight.detach().clone()

    @torch.no_grad()
    def after_copy(self, shard_model_weight: torch.Tensor, shard_main_weight: torch.Tensor) -> None:
        assert self.prev_model_weight is not None
        model_weight_indices = get_sparse_diff_indices(shard_model_weight, self.prev_model_weight)

        save_values = os.environ.get("SPARSE_SAVE_VALUES", "0") == "1"
        prev_model_weight_values = (
            self.prev_model_weight.index_select(0, model_weight_indices) if save_values else None
        )
        curr_model_weight_values = (
            shard_model_weight.index_select(0, model_weight_indices) if save_values else None
        )

        # FP32 Delta (optional, can be expensive; gate by env)
        main_weight_delta_analyse_result: dict | None = self._analyse_main_weight_and_grad_delta(
            shard_main_weight, model_weight_indices
        )

        self.prev_main_weight = shard_main_weight.detach().clone().to("cpu")
        self.prev_model_weight = None

        # Add the param range start to make the indices global
        fixed_model_weight_indices = model_weight_indices + self.param_range.start
        self.union_indices(fixed_model_weight_indices)

        sparse_data = SparseData(
            model_weight_indices=fixed_model_weight_indices,
            prev_model_weight_values=prev_model_weight_values,
            curr_model_weight_values=curr_model_weight_values,
            main_weight_delta_analyse_result=main_weight_delta_analyse_result,
        )

        sparse_data.to_cpu()

        self.history_sparse_data.append(sparse_data)

    @torch.no_grad()
    def _analyse_main_weight_and_grad_delta(
        self, shard_main_weight: torch.Tensor, model_weight_indices: torch.Tensor
    ) -> dict | None:
        # Off by default: this analysis can be expensive for near-dense fp32 diffs.
        if os.environ.get("SPARSE_ENRICH_STATS", "1") != "1":
            return None
        if self.prev_main_weight is None:
            return None
        if shard_main_weight.dtype == self.param_dtype:
            return None

        grad = getattr(self.param, "main_grad", self.param.grad)
        assert grad is not None
        assert grad.shape == self.param_shape
        self.grad = grad.detach().view(-1)[self.param_range.start : self.param_range.end]

        # To CUDA
        prev_main_weight: torch.Tensor = self.prev_main_weight.to(
            device=self.param_device, non_blocking=True
        )
        shard_main_weight = shard_main_weight.to(device=self.param_device, non_blocking=True)

        # Keep a copy: we need both the fp32 delta and a bf16-delta consistent with model_weight_indices.
        prev_main_weight32: torch.Tensor = prev_main_weight

        # main_weight_diff32(fp32) = shard_main_weight(fp32) - prev_main_weight(fp32)
        main_weight_diff32: torch.Tensor = shard_main_weight - prev_main_weight32  # FP32

        # A bf16 delta consistent with the model copy path:
        # model (bf16) updates match: bf16(main_curr) - bf16(main_prev)
        prev_main_weight16: torch.Tensor = prev_main_weight32.to(self.param_dtype)  # BF16/FP16
        curr_main_weight16: torch.Tensor = shard_main_weight.to(self.param_dtype)  # BF16/FP16
        main_weight_diff16_consistent: torch.Tensor = curr_main_weight16 - prev_main_weight16

        # nnz counts (avoid nonzero().numel() ambiguity)
        main_weight_nnz: int = int(torch.count_nonzero(main_weight_diff32).item())

        # Validate (best-effort): bf16 updated positions should match model_weight_indices.
        # Important: compare against bf16(main_curr) - bf16(main_prev), not bf16(main_curr - main_prev).
        validate_result = "success"
        try:
            nz16 = (main_weight_diff16_consistent != 0).nonzero(as_tuple=False).view(-1)
            nz16 = nz16.to(
                device=model_weight_indices.device,
                dtype=model_weight_indices.dtype,
                non_blocking=True,
            )
            assert nz16.numel() == model_weight_indices.numel(), (
                "nz16.numel() != model_weight_indices.numel()"
            )
            assert torch.equal(nz16, model_weight_indices), "nz16 != model_weight_indices"
        except AssertionError as e:
            validate_result = "failed: " + str(e)

        # Grad must be aligned to this optimizer shard (same flat slice as shard_main_weight).
        if self.grad.numel() != main_weight_diff32.numel():
            raise ValueError(
                f"grad/shard_main_weight numel mismatch: grad.numel()={self.grad.numel()}, main_weight_diff32.numel()={main_weight_diff32.numel()}"
            )
        grad_nnz: int = int(torch.count_nonzero(self.grad).item())

        # Stats restricted to *bf16-updated indices* (model_weight_indices).
        # This answers: "do bf16 updates mostly come from grad==0 or grad!=0?"
        updated_count = int(model_weight_indices.numel())
        updated_grad_nnz = 0
        updated_grad_zero = 0
        updated_fp32_diff_stats: dict = {
            "numel": 0,
            "abs_mean": 0.0,
            "abs_median": 0.0,
            "abs_max": 0.0,
        }
        updated_fp32_diff_stats_grad_nonzero: dict = {
            "numel": 0,
            "abs_mean": 0.0,
            "abs_median": 0.0,
            "abs_max": 0.0,
        }
        updated_fp32_diff_stats_grad_zero: dict = {
            "numel": 0,
            "abs_mean": 0.0,
            "abs_median": 0.0,
            "abs_max": 0.0,
        }
        updated_grad_stats: dict = {
            "numel": 0,
            "abs_mean": 0.0,
            "abs_median": 0.0,
            "abs_max": 0.0,
        }
        if updated_count > 0:
            idx64 = model_weight_indices.to(
                device=self.grad.device, dtype=torch.int64, non_blocking=True
            )
            updated_grad = self.grad.index_select(0, idx64)
            updated_diff32 = main_weight_diff32.index_select(0, idx64)

            updated_grad_nnz = int(torch.count_nonzero(updated_grad).item())
            updated_grad_zero = int(updated_count - updated_grad_nnz)
            updated_fp32_diff_stats = self._stats_tensor(updated_diff32)
            updated_grad_stats = self._stats_tensor(updated_grad)

            mask_gnz = updated_grad != 0
            updated_fp32_diff_stats_grad_nonzero = self._stats_tensor(updated_diff32[mask_gnz])
            updated_fp32_diff_stats_grad_zero = self._stats_tensor(updated_diff32[~mask_gnz])

        # Mask of cast acc loss.
        mask_of_acc_loss: torch.Tensor = (main_weight_diff32 != 0) & (
            main_weight_diff16_consistent == 0
        )
        # Mask of cast acc loss and grad is not 0.
        mask_of_acc_loss_and_non_zero_grad: torch.Tensor = mask_of_acc_loss & (self.grad != 0)

        # A) fp32 changed && bf16 unchanged (cast accumulation loss)
        diff_acc_loss_tensor: torch.Tensor = main_weight_diff32[mask_of_acc_loss]
        diff_acc_loss_tensor_result: dict = self._stats_tensor(diff_acc_loss_tensor)

        # B) fp32 changed b&& bf16 unchanged  && grad != 0 (grad is tiny)
        # B.1) Statistics of fp32 diff values
        diff_acc_loss_with_nonzero_grad_tensor: torch.Tensor = main_weight_diff32[
            mask_of_acc_loss_and_non_zero_grad
        ]
        diff_acc_loss_with_nonzero_grad_tensor_result: dict = self._stats_tensor(
            diff_acc_loss_with_nonzero_grad_tensor
        )

        # B.2) Statistics of grad values
        nonzero_grad_with_acc_loss_tensor: torch.Tensor = self.grad[
            mask_of_acc_loss_and_non_zero_grad
        ]
        nonzero_grad_with_acc_loss_tensor_result: dict = self._stats_tensor(
            nonzero_grad_with_acc_loss_tensor
        )

        # Also track how many of acc-loss positions have grad==0
        acc_cnt = int(mask_of_acc_loss.sum().item())
        gnz_cnt = int(mask_of_acc_loss_and_non_zero_grad.sum().item())
        g0_cnt = acc_cnt - gnz_cnt

        return {
            "grad_nnz": grad_nnz,
            "main_weight_nnz": main_weight_nnz,
            "updated_indices_count": updated_count,
            "updated_indices_grad_nnz": updated_grad_nnz,
            "updated_indices_grad_zero": updated_grad_zero,
            "updated_indices_grad_zero_ratio": (float(updated_grad_zero) / float(updated_count))
            if updated_count
            else 0.0,
            "updated_indices_grad_stats": updated_grad_stats,
            "updated_indices_fp32_diff_stats": updated_fp32_diff_stats,
            "updated_indices_fp32_diff_stats_grad_nonzero": updated_fp32_diff_stats_grad_nonzero,
            "updated_indices_fp32_diff_stats_grad_zero": updated_fp32_diff_stats_grad_zero,
            "acc_loss_count": acc_cnt,
            "acc_loss_grad_nonzero": gnz_cnt,
            "acc_loss_grad_zero": g0_cnt,
            "acc_loss_fp32_diff_stats": diff_acc_loss_tensor_result,
            "acc_loss_fp32_diff_stats_grad_nonzero": diff_acc_loss_with_nonzero_grad_tensor_result,
            "nonzero_grad_with_acc_loss_stats": nonzero_grad_with_acc_loss_tensor_result,
            "validate_result": validate_result,
        }

    @staticmethod
    def _stats_tensor(tensor: torch.Tensor) -> dict:
        """Statistics of the tensor, including numel, abs_mean, abs_median, abs_max."""
        if tensor.numel() == 0:
            return {"numel": 0, "abs_mean": 0.0, "abs_median": 0.0, "abs_max": 0.0}
        tensor = tensor.abs().to(torch.float32)
        return {
            "numel": int(tensor.numel()),
            "abs_mean": float(tensor.mean().item()),
            "abs_median": float(tensor.median().item()),
            "abs_max": float(tensor.max().item()),
        }
