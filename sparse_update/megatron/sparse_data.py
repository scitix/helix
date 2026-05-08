from dataclasses import dataclass

import torch


@dataclass
class SparseData:
    model_weight_indices: torch.Tensor
    # Optional (large): can be derived online if needed.
    prev_model_weight_values: torch.Tensor | None = None
    curr_model_weight_values: torch.Tensor | None = None

    # # Optional: summary stats about fp32-delta vs bf16-delta mismatch (cast accumulation loss)
    main_weight_delta_analyse_result: dict | None = None

    def dict_to_cpu(self, data: dict | None) -> dict | None:
        if not data:
            return None
        keys = list(data.keys())
        for key in keys:
            value = data[key]
            if isinstance(value, torch.Tensor):
                data[key] = value.to("cpu")
            elif isinstance(value, dict):
                data[key] = self.dict_to_cpu(value)
        return data

    def to_cpu(self) -> None:
        self.model_weight_indices = self.model_weight_indices.to("cpu")
        if self.prev_model_weight_values is not None:
            self.prev_model_weight_values = self.prev_model_weight_values.to("cpu")
        if self.curr_model_weight_values is not None:
            self.curr_model_weight_values = self.curr_model_weight_values.to("cpu")
        self.main_weight_delta_analyse_result = self.dict_to_cpu(
            self.main_weight_delta_analyse_result
        )
