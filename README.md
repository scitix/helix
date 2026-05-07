# Helix
high-performance, large-scale post-training framework including SFT and RL.

Codes will be released soon.

## Features
- 🔥 **SparseRL-sync**: reduce parameter sync size by **30x ~ 100x with 100% accuracy**, valided on models 8B ~ 671B, GRPO, DAPO, GSPO etc, generic reasoning and agentic. Check our [tech report](https://github.com/scitix/helix/blob/main/paper/sparseRL-sync.pdf)

<p align="center">
  <img src="assets/model_payload_size.png" alt="Per-synchronization payload across model scales" width="49%" />
  <img src="assets/origin_reward_validation.png" alt="Reward curves: full-update baseline vs. SparseRL-Sync" width="49%" />
</p>

**Core result at a glance.** SparseRL-Sync reduces the Trainer-to-Rollout weight-synchronization payload by 32×–54× raw and up to ~100× after lossless compression across model scales (left) while preserving training dynamics bit-exactly (right).

- **(Left) Per-synchronization payload across model scales:** full update (BF16) vs. sparse (I, V) uncompressed vs. sparse (I, V) compressed. Sparse synchronization reduces the transfer by 32×–54× raw and ≈60×–101× after lossless compression.
- **(Right) Reward curves** of the full-update baseline vs. SparseRL-Sync on Qwen3-30B-A3B-Instruct-2507 over 500 training steps. The two curves are nearly indistinguishable, confirming lossless fidelity.



## License
This project is licensed under the Apache License 2.0.
