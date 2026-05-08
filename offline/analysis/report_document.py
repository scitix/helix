"""Assemble Markdown/HTML report sections from figure paths and summary tables."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .report import (
    ReportPaths,
    img_section,
    make_summary_table_md,
    write_html_from_markdownish,
    write_markdown,
)


def markdown_image_or_missing(fig_abs_path: str, report_root: str, alt: str) -> str:
    if not os.path.exists(fig_abs_path):
        return "_(missing figure)_"
    rel = os.path.relpath(fig_abs_path, report_root)
    return f"![{alt}]({rel})"


def build_summary_lines(
    *,
    data_dir: str,
    data_kind: str,
    num_ranks: int,
    num_steps: int,
    element_rows: list[dict[str, Any]],
    inactive_rows: list[dict[str, Any]],
    quantile_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    layer_bucket_rows: list[dict[str, Any]],
    internal_summaries: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    moe_rows: list[dict[str, Any]],
    moe_spread_rows: list[dict[str, Any]] | None = None,
    global_locality_step_rows: list[dict[str, Any]] | None = None,
    param_extrema: dict[str, Any] | None = None,
    moe_layer_stats: dict[str, dict[str, float]] | None = None,
) -> list[str]:
    def pct(x: float) -> str:
        return f"{100.0 * float(x):.2f}%"

    def safe_last(xs: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
        if not xs:
            return default
        try:
            return float(xs[-1].get(key, default))
        except (TypeError, ValueError):
            return default

    last_sparsity = safe_last(element_rows, "sparsity", 0.0)
    mean_sparsity = (
        float(np.mean([float(x["sparsity"]) for x in element_rows])) if element_rows else 0.0
    )
    last_inactive = safe_last(inactive_rows, "inactive_ratio", 0.0)
    mean_inactive = (
        float(np.mean([float(x["inactive_ratio"]) for x in inactive_rows]))
        if inactive_rows
        else 0.0
    )
    moe_fc1_layers = len({int(r["layer"]) for r in moe_rows if r["kind"] == "fc1"})
    moe_fc2_layers = len({int(r["layer"]) for r in moe_rows if r["kind"] == "fc2"})

    q_text = ""
    if quantile_rows:
        q = quantile_rows[-1]
        q50 = float(q.get("q50", 0.0))
        q90 = float(q.get("q90", 0.0))
        q99 = float(q.get("q99", 0.0))
        q_text = f"(last step: P50={pct(q50)}, P90={pct(q90)}, P99={pct(q99)})"

    param_text = ""
    if param_extrema:
        try:
            mxn = str(param_extrema.get("max_param_name", ""))
            mnn = str(param_extrema.get("min_param_name", ""))
            mxs = float(param_extrema.get("max_param_sparsity", 0.0))
            mns = float(param_extrema.get("min_param_sparsity", 0.0))
            param_text = f"(last step: max `{mxn}`={pct(mxs)}, min `{mnn}`={pct(mns)})"
        except (TypeError, ValueError):
            param_text = ""

    layer_text = ""
    if layer_bucket_rows:
        last_step = max(int(r.get("step", 0)) for r in layer_bucket_rows)
        last = [r for r in layer_bucket_rows if int(r.get("step", 0)) == last_step]
        if last:
            defs = ""
            for r in last:
                defs = str(r.get("bucket_def", "")) or defs
            by_b = {str(r["bucket"]): float(r["element_sparsity"]) for r in last}
            if by_b:
                layer_text = (
                    f"({defs}; last step: front={pct(by_b.get('front', 0.0))}, "
                    f"mid={pct(by_b.get('mid', 0.0))}, back={pct(by_b.get('back', 0.0))})"
                )

    jac_text = ""
    if internal_summaries:
        vals = [float(r.get("avg_history_jaccard", 0.0)) for r in internal_summaries]
        jac_text = f"(top candidates' mean history similarity: Jaccard≈{float(np.mean(vals)):.3f})"

    gl_text = ""
    if global_locality_step_rows:
        last = global_locality_step_rows[-1]
        try:
            gl_text = (
                f"(last step: mean≈{float(last.get('mean', 0.0)):.3f}, "
                f"P50≈{float(last.get('p50', 0.0)):.3f}, "
                f"P90≈{float(last.get('p90', 0.0)):.3f})"
            )
        except (TypeError, ValueError):
            gl_text = ""

    source_lines: list[str] = []
    one_liner = ""
    if source_rows:
        last = source_rows[-1]
        one_liner = (
            "TL;DR: bf16 updates are very sparse, fp32 changes are close to dense, and most fp32 changes "
            "are attributable to casting/accumulation loss (acc_loss); grad is near-dense at the element level."
        )
        source_lines = [
            "- **Source analysis (single rank, element-level)**:",
            f"  - grad sparsity≈{pct(float(last.get('grad_sparsity', 0.0)))} (1 - grad_nnz/numel)",
            f"  - fp32 changed ratio≈{pct(float(last.get('fp32_change_ratio', 0.0)))}",
            f"  - bf16 updated ratio≈{pct(float(last.get('bf16_update_ratio', 0.0)))}",
            f"  - acc_loss share within fp32 changes≈{pct(float(last.get('acc_loss_share_in_fp32_change', 0.0)))}",
        ]

    lines: list[str] = [
        f"- This report is generated on CPU from `{data_dir}` (data_kind=`{data_kind}`, offline steps={num_steps}).",
        "- **Sparsity overview**:",
        f"  - element sparsity (1-nnz/numel) mean≈{pct(mean_sparsity)}, last step≈{pct(last_sparsity)}.",
        f"  - param inactive ratio (#params with nnz==0) mean≈{pct(mean_inactive)}, last step≈{pct(last_inactive)}.",
        f"  - param element-sparsity distribution {q_text}".rstrip(),
        f"  - param extrema {param_text}".rstrip(),
        f"  - layer front/mid/back comparison {layer_text}".rstrip(),
    ]
    if one_liner:
        lines.insert(1, f"- **{one_liner}**")
    if source_lines:
        lines.extend(source_lines)
    lines.append(
        f"- **Locality**: global |I_t ∩ U_{{<t}}|/|I_t| CDF/quantiles {gl_text}; "
        f"top-candidate internal plots (if enabled) {jac_text}".rstrip()
    )
    if moe_rows:
        lines.append(
            f"- **MoE (if present)**: fc1 covers≈{moe_fc1_layers} layer slices; fc2 covers≈{moe_fc2_layers}."
        )
        if moe_layer_stats:
            try:
                router = moe_layer_stats.get("router", {})
                fc1 = moe_layer_stats.get("fc1", {})
                fc2 = moe_layer_stats.get("fc2", {})
                lines.append(
                    "- **MoE variance (used to decide whether heatmaps are informative)**: "
                    f"router(std≈{float(router.get('std', 0.0)):.4g}, "
                    f"min≈{float(router.get('min', 0.0)):.3f}, max≈{float(router.get('max', 0.0)):.3f}); "
                    f"fc1(std≈{float(fc1.get('std', 0.0)):.4g}); "
                    f"fc2(std≈{float(fc2.get('std', 0.0)):.4g})."
                )
            except Exception:
                pass
    return [ln for ln in lines if ln.strip() and ln.strip() != "-"]


def build_report_sections(
    *,
    paths: ReportPaths,
    data_dir: str,
    data_kind: str,
    ranks: list[int],
    num_steps: int,
    fig_element_sparsity: str,
    fig_param_activity: str,
    fig_param_heatmap: str,
    fig_param_sparsity_hist: str,
    fig_group_lines: str,
    fig_layer_bucket_lines: str,
    fig_layer_sparsity_cdf: str = "",
    moe_layer_figure_paths: list[str],
    moe_expert_figure_relpaths: list[str],
    fig_dense_vs_moe: str = "",
    moe_spread_figure_paths: list[str] | None = None,
    moe_spread_rows: list[dict[str, Any]] | None = None,
    fig_global_locality_cdf: str = "",
    global_locality_step_rows: list[dict[str, Any]] | None = None,
    param_extrema: dict[str, Any] | None = None,
    moe_layer_stats: dict[str, dict[str, float]] | None = None,
    internal_summaries: list[dict[str, Any]],
    element_rows: list[dict[str, Any]],
    inactive_rows: list[dict[str, Any]],
    quantile_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    layer_bucket_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    moe_rows: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []

    sections.append(
        (
            "Summary (TL;DR)",
            "\n".join(
                build_summary_lines(
                    data_dir=data_dir,
                    data_kind=data_kind,
                    num_ranks=len(ranks),
                    num_steps=num_steps,
                    element_rows=element_rows,
                    inactive_rows=inactive_rows,
                    quantile_rows=quantile_rows,
                    group_rows=group_rows,
                    layer_bucket_rows=layer_bucket_rows,
                    internal_summaries=internal_summaries,
                    source_rows=source_rows,
                    moe_rows=moe_rows,
                    moe_spread_rows=moe_spread_rows,
                    global_locality_step_rows=global_locality_step_rows,
                    param_extrema=param_extrema,
                    moe_layer_stats=moe_layer_stats,
                )
            ),
        )
    )

    sections.append(
        (
            "Inputs & offline step definition",
            "\n".join(
                [
                    f"- data_dir: `{data_dir}`",
                    f"- data_kind: `{data_kind}`",
                    f"- ranks: `{ranks}`",
                    f"- offline steps: `{num_steps}` (constructed by sorting `(interval, event_idx)` on baseline rank)",
                    "- Note: current dumps do not include `global_opt_step`; offline step is a best-effort proxy.",
                ]
            ),
        )
    )

    sections.append(
        (
            "Overall sparsity & param inactivity (combined)",
            img_section(
                os.path.relpath(fig_element_sparsity, paths.out_dir),
                "overall sparsity and inactivity",
                extra_lines=[
                    "- left y-axis: element sparsity = (1 - nnz/numel) in percent (%)",
                    "- right y-axis: param inactive ratio = (#params with nnz==0) / (#params) in percent (%)",
                    "- both axes are zoomed to show small variations",
                ],
            ),
        )
    )
    sections.append(
        (
            "Param inactivity over steps",
            img_section(
                os.path.relpath(fig_param_activity, paths.out_dir),
                "param inactive ratio",
                extra_lines=[
                    "- y-axis is inactive ratio = (#params with nnz==0) / (#params) in percent (%)",
                    "- y-axis is zoomed to reveal small changes when values are near 0",
                ],
            ),
        )
    )
    if fig_param_heatmap and os.path.exists(fig_param_heatmap):
        sections.append(
            (
                "Param-level heatmap (top-K by avg nnz_ratio)",
                img_section(os.path.relpath(fig_param_heatmap, paths.out_dir), "param heatmap"),
            )
        )

    sections.append(
        (
            "Param sparsity distribution (unweighted, last step)",
            img_section(
                os.path.relpath(fig_param_sparsity_hist, paths.out_dir),
                "param sparsity histogram",
                extra_lines=["- x-axis is param sparsity = (1 - nnz/numel) in percent (%)"],
            ),
        )
    )

    if os.path.exists(fig_group_lines):
        sections.append(
            (
                "Group/module element sparsity over time (single rank)",
                img_section(
                    os.path.relpath(fig_group_lines, paths.out_dir),
                    "group element sparsity",
                    extra_lines=["- y-axis is element sparsity = (1 - nnz/numel) in percent (%)"],
                ),
            )
        )

    if fig_layer_sparsity_cdf and os.path.exists(fig_layer_sparsity_cdf):
        sections.append(
            (
                "Layer element sparsity distribution (CDF, last step)",
                img_section(
                    os.path.relpath(fig_layer_sparsity_cdf, paths.out_dir),
                    "layer sparsity CDF",
                    extra_lines=[
                        "- Each point is one decoder layer's pooled element sparsity at the last step",
                        "- This is a CDF over layers (not over parameters)",
                        "- Table: `tables/layer_sparsity_last_step.csv`",
                    ],
                ),
            )
        )

    if os.path.exists(fig_layer_bucket_lines):
        sections.append(
            (
                "Layer position element sparsity over time (single rank)",
                img_section(
                    os.path.relpath(fig_layer_bucket_lines, paths.out_dir),
                    "layer bucket element sparsity",
                    extra_lines=["- y-axis is element sparsity = (1 - nnz/numel) in percent (%)"],
                ),
            )
        )

    moe_md_parts: list[str] = []
    for fig_path in moe_layer_figure_paths:
        if os.path.exists(fig_path):
            moe_md_parts.append(f"![moe layer heatmap]({os.path.relpath(fig_path, paths.out_dir)})")
    if moe_md_parts:
        sections.append(
            ("MoE (per-layer over steps, sparsity heatmaps)", "\n\n".join(moe_md_parts))
        )
    elif moe_layer_stats:
        lines = ["- Heatmaps skipped (variance too small / uninformative).", "- Stats (sparsity):"]
        for kind, st in moe_layer_stats.items():
            lines.append(
                f"  - {kind}: mean={float(st.get('mean', 0.0)):.4f}, std={float(st.get('std', 0.0)):.4g}, "
                f"min={float(st.get('min', 0.0)):.4f}, max={float(st.get('max', 0.0)):.4f}"
            )
        sections.append(("MoE (per-layer stats, heatmaps skipped)", "\n".join(lines)))

    if fig_dense_vs_moe and os.path.exists(fig_dense_vs_moe):
        sections.append(
            (
                "Dense vs MoE (element sparsity over steps)",
                img_section(
                    os.path.relpath(fig_dense_vs_moe, paths.out_dir),
                    "dense vs moe router vs moe expert",
                ),
            )
        )

    if moe_spread_figure_paths:
        parts = [
            markdown_image_or_missing(p, paths.out_dir, "moe expert spread")
            for p in moe_spread_figure_paths
            if os.path.exists(p)
        ]
        if parts:
            sections.append(
                (
                    "MoE: spread of expert sparsity (std across experts)",
                    "\n\n".join(parts)
                    + "\n\n- Table: `tables/moe_expert_nnz_spread.csv` (mean/std/min/max per layer/kind/step).",
                )
            )

    if moe_expert_figure_relpaths:
        sections.append(
            (
                "MoE expert tensors (expert-level heatmaps)",
                "\n\n".join([f"![moe expert heatmap]({p})" for p in moe_expert_figure_relpaths]),
            )
        )

    if fig_global_locality_cdf and os.path.exists(fig_global_locality_cdf):
        body = img_section(
            os.path.relpath(fig_global_locality_cdf, paths.out_dir),
            "param locality CDF",
            extra_lines=[
                "- locality_ratio(t) = |I_t ∩ U_{<t}| / |I_t|, per-param distribution shown as CDF",
                "- Table: `tables/param_locality_step.csv` (mean/P50/P90/P99 per step, over active params).",
            ],
        )
        sections.append(("Global locality across ALL params (CDF)", body))

    table_keys = ["param_name", "group", "avg_nnz_ratio", "avg_history_jaccard", "locality_like"]
    sections.append(
        (
            "Param-internal locality summary",
            make_summary_table_md(internal_summaries, keys=table_keys, max_rows=30),
        )
    )

    from .locality_analysis import filesystem_safe_param_slug

    for row in internal_summaries[: min(6, len(internal_summaries))]:
        name = row["param_name"]
        slug = filesystem_safe_param_slug(name)
        fig_internal = os.path.join(paths.figures_dir, f"internal_{slug}.png")
        fig_jac = os.path.join(paths.figures_dir, f"internal_{slug}_jaccard.png")
        body = "\n\n".join(
            [
                f"- param: `{name}`",
                markdown_image_or_missing(fig_internal, paths.out_dir, "internal heatmap"),
                markdown_image_or_missing(fig_jac, paths.out_dir, "jaccard"),
            ]
        )
        sections.append((f"Internal heatmap & Jaccard: {name[-80:]}", body))

    return sections


def write_reports(paths: ReportPaths, sections: list[tuple[str, str]]) -> None:
    write_markdown(paths.report_md, title="Sparse Update Offline Report", sections=sections)
    write_html_from_markdownish(
        paths.report_html, title="Sparse Update Offline Report", sections=sections
    )
