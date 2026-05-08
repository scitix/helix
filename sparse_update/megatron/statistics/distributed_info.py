from __future__ import annotations

import torch.distributed as dist


class DistributedInfo:
    def __init__(self):
        from megatron.core import mpu

        self.world_size_info = {
            "world_size": dist.get_world_size(),
            "pp_size": mpu.get_pipeline_model_parallel_world_size(),
            "tp_size": mpu.get_tensor_model_parallel_world_size(),
            "dp_size": mpu.get_data_parallel_world_size(with_context_parallel=True),
            "ep_size": mpu.get_expert_model_parallel_world_size(),
            "ep_dp_size": mpu.get_expert_data_parallel_world_size(),
            "ep_tp_size": mpu.get_expert_tensor_parallel_world_size(),
        }

        self.rank_info = {
            "global_rank": dist.get_rank(),
            "tp_rank": mpu.get_tensor_model_parallel_rank(),
            "dp_rank": mpu.get_data_parallel_rank(with_context_parallel=True),
            "ep_rank": mpu.get_expert_model_parallel_rank(),
            "ep_dp_rank": mpu.get_expert_data_parallel_rank(),
            "ep_tp_rank": mpu.get_expert_tensor_parallel_rank(),
        }

    @property
    def distributed_info(self):
        return {
            **self.world_size_info,
            **self.rank_info,
        }
