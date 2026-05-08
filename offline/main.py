"""Unified CLI entrypoint for offline merge and offline analysis."""

from __future__ import annotations

from .arguments import parse_args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.mode == "merge":
        from .merge.runner import run_merge_from_args

        run_merge_from_args(args)
        return

    if args.mode == "analysis":
        from .analysis.runner import run_analysis_from_args

        run_analysis_from_args(args)
        return

    raise ValueError(f"Unknown mode: {args.mode!r}")


if __name__ == "__main__":
    main()
