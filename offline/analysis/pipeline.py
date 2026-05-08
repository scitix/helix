"""End-to-end offline report: load checkpoints → tables → figures → report.md/html."""

from __future__ import annotations

import json
import os

from .aggregation import build_step_aggregation
from .config import OfflineReportConfig
from .global_locality import run_global_param_locality
from .global_moe_analysis import (
    build_dense_vs_moe_step_rows,
    build_moe_expert_spread_rows,
    plot_dense_vs_moe_step,
    plot_moe_expert_spread_top_layers,
)
from .locality_analysis import run_locality_analysis, select_locality_candidate_params
from .moe_analysis import (
    build_moe_param_step_rows,
    moe_layer_kind_sparsity_stats,
    plot_moe_layer_kind_heatmaps,
    plot_top_moe_expert_heatmaps,
)
from .overview_figures import (
    build_communication_estimate_rows,
    element_sparsity_rows,
    group_step_rows,
    layer_bucket_step_rows,
    layer_sparsity_last_step_rows,
    param_inactive_ratio_rows,
    param_sparsity_quantile_rows,
    plot_group_sparsity_lines,
    plot_layer_bucket_lines,
    plot_layer_sparsity_cdf,
    plot_overall_and_inactive,
    plot_param_activity_ratio,
    plot_param_nnz_heatmap,
    plot_param_sparsity_histogram,
    rank_params_by_mean_nnz_ratio,
)
from .plots import ensure_dir, save_multi_line
from .report import make_report_paths
from .report_document import build_report_sections, write_reports
from .tables import write_csv_dict_rows


def _infer_data_kind(data_dir: str, default: str) -> str:
    if default in ("single", "merged"):
        return default
    try:
        p = os.path.join(data_dir, "rank0_info.json")
        if not os.path.exists(p):
            return "auto"
        with open(p) as f:
            obj = json.load(f)
        di = obj.get("distributed_info", {}) if isinstance(obj, dict) else {}
        if int(di.get("world_size", 0) or 0) == 1 and (
            di.get("orig_tp_size") is not None or di.get("orig_dp_size") is not None
        ):
            return "merged"
        return "single"
    except Exception:
        return "auto"


def run_offline_report(config: OfflineReportConfig) -> None:
    paths = make_report_paths(config.out_dir)
    ensure_dir(paths.out_dir)
    ensure_dir(paths.figures_dir)
    ensure_dir(paths.tables_dir)

    data_kind = _infer_data_kind(config.data_dir, str(config.data_kind))
    agg = build_step_aggregation(config.data_dir, rank=config.rank)

    write_csv_dict_rows(
        os.path.join(paths.tables_dir, "per_param_step.csv"), agg.per_param_step_rows
    )
    write_csv_dict_rows(os.path.join(paths.tables_dir, "per_rank_step.csv"), agg.per_rank_step_rows)

    if agg.per_step_source_rows:
        write_csv_dict_rows(
            os.path.join(paths.tables_dir, "source_step.csv"), agg.per_step_source_rows
        )
        steps = [int(r["step"]) for r in agg.per_step_source_rows]
        save_multi_line(
            os.path.join(paths.figures_dir, "source_updated_grad.png"),
            x=steps,
            series={
                "abs_mean": [
                    float(r.get("updated_grad_abs_mean", 0.0)) for r in agg.per_step_source_rows
                ],
                "abs_max": [
                    float(r.get("updated_grad_abs_max", 0.0)) for r in agg.per_step_source_rows
                ],
            },
            title="bf16-updated positions: |grad| stats over steps",
            xlabel="offline step",
            ylabel="abs(value)",
        )
        save_multi_line(
            os.path.join(paths.figures_dir, "source_acc_loss_grad.png"),
            x=steps,
            series={
                "abs_mean": [
                    float(r.get("acc_loss_grad_abs_mean", 0.0)) for r in agg.per_step_source_rows
                ],
                "abs_max": [
                    float(r.get("acc_loss_grad_abs_max", 0.0)) for r in agg.per_step_source_rows
                ],
            },
            title="acc-loss positions (fp32 changed, bf16 unchanged): |grad| stats over steps",
            xlabel="offline step",
            ylabel="abs(value)",
        )
        save_multi_line(
            os.path.join(paths.figures_dir, "source_updated_fp32_diff.png"),
            x=steps,
            series={
                "abs_mean": [
                    float(r.get("updated_diff_abs_mean", 0.0)) for r in agg.per_step_source_rows
                ],
                "abs_max": [
                    float(r.get("updated_diff_abs_max", 0.0)) for r in agg.per_step_source_rows
                ],
            },
            title="bf16-updated positions: |fp32 diff| stats over steps",
            xlabel="offline step",
            ylabel="abs(value)",
        )
        save_multi_line(
            os.path.join(paths.figures_dir, "source_acc_loss_fp32_diff.png"),
            x=steps,
            series={
                "abs_mean": [
                    float(r.get("acc_loss_diff_abs_mean", 0.0)) for r in agg.per_step_source_rows
                ],
                "abs_max": [
                    float(r.get("acc_loss_diff_abs_max", 0.0)) for r in agg.per_step_source_rows
                ],
            },
            title="acc-loss positions (fp32 changed, bf16 unchanged): |fp32 diff| stats over steps",
            xlabel="offline step",
            ylabel="abs(value)",
        )

    element_rows = element_sparsity_rows(agg.per_param_step_rows)
    write_csv_dict_rows(os.path.join(paths.tables_dir, "element_sparsity.csv"), element_rows)

    inactive_rows = param_inactive_ratio_rows(agg.per_param_step_rows, num_steps=agg.num_steps)
    write_csv_dict_rows(os.path.join(paths.tables_dir, "param_inactive_ratio.csv"), inactive_rows)

    fig_overall_inactive = os.path.join(paths.figures_dir, "overall_sparsity_and_inactive.png")
    plot_overall_and_inactive(
        element_rows=element_rows, inactive_rows=inactive_rows, out_path=fig_overall_inactive
    )

    fig_active = os.path.join(paths.figures_dir, "param_active_ratio.png")
    plot_param_activity_ratio(agg.per_param_step_rows, num_steps=agg.num_steps, out_path=fig_active)

    ranked_param_names = rank_params_by_mean_nnz_ratio(agg.per_param_step_rows)
    last_step = max(0, agg.num_steps - 1)
    fig_param_sparsity_hist = os.path.join(
        paths.figures_dir, f"param_sparsity_hist_step{last_step}.png"
    )
    plot_param_sparsity_histogram(
        agg.per_param_step_rows, step=last_step, out_path=fig_param_sparsity_hist, bins=60
    )
    quantile_rows = param_sparsity_quantile_rows(agg.per_param_step_rows, num_steps=agg.num_steps)
    write_csv_dict_rows(
        os.path.join(paths.tables_dir, "param_sparsity_quantiles.csv"), quantile_rows
    )

    fig_param_heatmap = ""
    top_for_heatmap: list[str] = []
    try:
        last_inactive = float(inactive_rows[-1]["inactive_ratio"]) if inactive_rows else 1.0
        q_last = quantile_rows[-1] if quantile_rows else {}
        q25 = float(q_last.get("q25", 0.0))
        q50 = float(q_last.get("q50", 0.0))
        very_sparse = (q25 >= 0.995) and (q50 >= 0.995)
        mostly_active = last_inactive <= 0.01
        if not (very_sparse and mostly_active):
            top_for_heatmap = ranked_param_names[: max(1, config.topk_params_for_heatmap)]
            fig_param_heatmap = os.path.join(paths.figures_dir, "param_heatmap_topk.png")
            plot_param_nnz_heatmap(
                agg.per_param_step_rows,
                top_param_names=top_for_heatmap,
                num_steps=agg.num_steps,
                out_path=fig_param_heatmap,
            )
    except Exception:
        top_for_heatmap = ranked_param_names[: max(1, config.topk_params_for_heatmap)]
        fig_param_heatmap = os.path.join(paths.figures_dir, "param_heatmap_topk.png")
        plot_param_nnz_heatmap(
            agg.per_param_step_rows,
            top_param_names=top_for_heatmap,
            num_steps=agg.num_steps,
            out_path=fig_param_heatmap,
        )

    param_extrema: dict[str, object] = {}
    try:
        last_rows = [r for r in agg.per_param_step_rows if int(r.get("step", -1)) == int(last_step)]
        if last_rows:

            def sparsity(rr: dict) -> float:
                return 1.0 - float(rr.get("nnz_ratio", 0.0) or 0.0)

            max_r = max(last_rows, key=sparsity)
            min_r = min(last_rows, key=sparsity)
            param_extrema = {
                "step": int(last_step),
                "max_param_name": str(max_r.get("param_name", "")),
                "max_param_sparsity": float(sparsity(max_r)),
                "min_param_name": str(min_r.get("param_name", "")),
                "min_param_sparsity": float(sparsity(min_r)),
            }
    except Exception:
        param_extrema = {}

    fig_global_locality_cdf = ""
    global_locality_step_rows: list[dict] = []
    if len(agg.ranks) == 1:
        fig_global_locality_cdf = os.path.join(paths.figures_dir, "param_locality_cdf.png")
        global_locality_step_rows, _param_rows = run_global_param_locality(
            data_dir=config.data_dir,
            rank=int(agg.ranks[0]),
            step_index=agg.step_index,
            rank_to_indices_paths=agg.rank_to_indices_paths,
            out_fig_path=fig_global_locality_cdf,
        )
        write_csv_dict_rows(
            os.path.join(paths.tables_dir, "param_locality_step.csv"), global_locality_step_rows
        )

    moe_rows = build_moe_param_step_rows(agg)
    write_csv_dict_rows(os.path.join(paths.tables_dir, "moe_param_step.csv"), moe_rows)

    moe_layer_stats = moe_layer_kind_sparsity_stats(moe_rows, num_steps=agg.num_steps)
    moe_heatmap_std_thresh = 5e-4

    dense_vs_moe_rows = build_dense_vs_moe_step_rows(
        agg.per_param_step_rows, num_steps=agg.num_steps
    )
    write_csv_dict_rows(os.path.join(paths.tables_dir, "dense_vs_moe_step.csv"), dense_vs_moe_rows)
    fig_dense_vs_moe = os.path.join(paths.figures_dir, "dense_vs_moe_element_nnz.png")
    plot_dense_vs_moe_step(dense_vs_moe_rows, num_steps=agg.num_steps, out_path=fig_dense_vs_moe)

    moe_spread_rows = build_moe_expert_spread_rows(moe_rows)
    write_csv_dict_rows(
        os.path.join(paths.tables_dir, "moe_expert_nnz_spread.csv"), moe_spread_rows
    )
    moe_spread_paths = plot_moe_expert_spread_top_layers(
        moe_spread_rows, num_steps=agg.num_steps, figures_dir=paths.figures_dir, top_layers=4
    )

    moe_layer_paths: list[str] = []
    if any(v.get("std", 0.0) >= moe_heatmap_std_thresh for v in moe_layer_stats.values()):
        moe_layer_paths = plot_moe_layer_kind_heatmaps(moe_rows, agg.num_steps, paths.figures_dir)

    expert_paths: list[str] = []
    if (
        moe_layer_stats.get("fc1", {}).get("std", 0.0) >= moe_heatmap_std_thresh
        or moe_layer_stats.get("fc2", {}).get("std", 0.0) >= moe_heatmap_std_thresh
    ):
        expert_paths = plot_top_moe_expert_heatmaps(moe_rows, agg.num_steps, paths.figures_dir)
    expert_relpaths = [os.path.relpath(p, paths.out_dir) for p in expert_paths]

    if config.topk_params_for_locality > 0:
        locality_candidates = select_locality_candidate_params(
            agg.per_param_step_rows,
            heatmap_top_param_names=top_for_heatmap,
            max_candidates=config.topk_params_for_locality,
        )
        internal_summaries = run_locality_analysis(
            agg,
            candidate_param_names=locality_candidates,
            internal_buckets=config.internal_index_buckets,
            jaccard_threshold=config.locality_jaccard_threshold,
            figures_dir=paths.figures_dir,
        )
        write_csv_dict_rows(
            os.path.join(paths.tables_dir, "param_internal_locality.csv"), internal_summaries
        )
    else:
        internal_summaries = []

    group_rows = group_step_rows(agg.per_param_step_rows)
    write_csv_dict_rows(os.path.join(paths.tables_dir, "group_step.csv"), group_rows)
    fig_group_lines = os.path.join(paths.figures_dir, "group_element_sparsity.png")
    plot_group_sparsity_lines(group_rows, out_path=fig_group_lines, topk_groups=10)

    layer_bucket_rows = layer_bucket_step_rows(agg.per_param_step_rows, num_steps=agg.num_steps)
    write_csv_dict_rows(os.path.join(paths.tables_dir, "layer_bucket_step.csv"), layer_bucket_rows)
    fig_layer_bucket_lines = os.path.join(paths.figures_dir, "layer_bucket_element_sparsity.png")
    plot_layer_bucket_lines(layer_bucket_rows, out_path=fig_layer_bucket_lines)

    layer_last_rows = layer_sparsity_last_step_rows(agg.per_param_step_rows, step=last_step)
    write_csv_dict_rows(
        os.path.join(paths.tables_dir, "layer_sparsity_last_step.csv"), layer_last_rows
    )
    fig_layer_cdf = os.path.join(paths.figures_dir, f"layer_sparsity_cdf_step{last_step}.png")
    plot_layer_sparsity_cdf(layer_last_rows, out_path=fig_layer_cdf, step=last_step)

    comm_rows = build_communication_estimate_rows(agg.per_param_step_rows)
    write_csv_dict_rows(os.path.join(paths.tables_dir, "comm_estimate_param_step.csv"), comm_rows)

    sections = build_report_sections(
        paths=paths,
        data_dir=config.data_dir,
        data_kind=data_kind,
        ranks=agg.ranks,
        num_steps=agg.num_steps,
        fig_element_sparsity=fig_overall_inactive,
        fig_param_activity=fig_active,
        fig_param_heatmap=fig_param_heatmap,
        fig_param_sparsity_hist=fig_param_sparsity_hist,
        fig_group_lines=fig_group_lines,
        fig_layer_bucket_lines=fig_layer_bucket_lines,
        fig_layer_sparsity_cdf=fig_layer_cdf,
        moe_layer_figure_paths=moe_layer_paths,
        moe_expert_figure_relpaths=expert_relpaths,
        fig_dense_vs_moe=fig_dense_vs_moe,
        moe_spread_figure_paths=moe_spread_paths,
        moe_spread_rows=moe_spread_rows,
        fig_global_locality_cdf=fig_global_locality_cdf,
        global_locality_step_rows=global_locality_step_rows,
        param_extrema=param_extrema,
        moe_layer_stats=moe_layer_stats,
        internal_summaries=internal_summaries,
        element_rows=element_rows,
        inactive_rows=inactive_rows,
        quantile_rows=quantile_rows,
        group_rows=group_rows,
        layer_bucket_rows=layer_bucket_rows,
        source_rows=agg.per_step_source_rows,
        moe_rows=moe_rows,
    )
    write_reports(paths, sections)
