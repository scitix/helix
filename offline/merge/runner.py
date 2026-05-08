from __future__ import annotations

from argparse import Namespace

from .merge import merge_sparse_dumps_to_single_rank


def run_merge_from_args(args: Namespace) -> None:
    merge_sparse_dumps_to_single_rank(
        data_dir=str(args.data_dir),
        out_dir=str(args.out_dir),
        out_rank=int(getattr(args, "out_rank", 0)),
        intervals=getattr(args, "intervals", None),
        write_info_json=not bool(getattr(args, "skip_info", False)),
        merge_device=str(getattr(args, "merge_device", "cpu")),
        nproc_per_node=int(getattr(args, "nproc_per_node", 1)),
        estimate_interval_size=bool(getattr(args, "estimate_interval_size", False)),
        assumed_events_per_interval=int(getattr(args, "assumed_events_per_interval", 1)),
    )
