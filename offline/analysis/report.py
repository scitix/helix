from __future__ import annotations

import html
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReportPaths:
    out_dir: str
    figures_dir: str
    tables_dir: str
    report_md: str
    report_html: str


def make_report_paths(out_dir: str) -> ReportPaths:
    figures_dir = os.path.join(out_dir, "figures")
    tables_dir = os.path.join(out_dir, "tables")
    return ReportPaths(
        out_dir=out_dir,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        report_md=os.path.join(out_dir, "report.md"),
        report_html=os.path.join(out_dir, "report.html"),
    )


def _md_img(rel_path: str, alt: str) -> str:
    return f"![{alt}]({rel_path})"


def write_markdown(
    path: str,
    *,
    title: str,
    sections: list[tuple[str, str]],
) -> None:
    lines: list[str] = []
    lines.append(f"## {title}")
    lines.append("")
    for h, body in sections:
        lines.append(f"### {h}")
        lines.append("")
        lines.append(body.rstrip())
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_html_from_markdownish(
    path: str,
    *,
    title: str,
    sections: list[tuple[str, str]],
) -> None:
    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append("<html><head>")
    parts.append('<meta charset="utf-8"/>')
    parts.append(f"<title>{html.escape(title)}</title>")
    parts.append(
        "<style>"
        "body{font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial;max-width:1100px;margin:28px auto;padding:0 16px;}"
        "h1,h2,h3{margin:18px 0 10px;}"
        "code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;}"
        "pre{background:#0b1020;color:#e6e6e6;padding:12px 14px;border-radius:8px;overflow:auto;}"
        "table{border-collapse:collapse;margin:10px 0;width:100%;}"
        "th,td{border:1px solid #ddd;padding:6px 8px;font-size:13px;}"
        "th{background:#f6f6f6;text-align:left;}"
        "img{max-width:100%;border:1px solid #eee;border-radius:6px;}"
        ".muted{color:#666;}"
        "</style>"
    )
    parts.append("</head><body>")
    parts.append(f"<h1>{html.escape(title)}</h1>")
    for h, body in sections:
        parts.append(f"<h2>{html.escape(h)}</h2>")
        for line in body.splitlines():
            if line.startswith("![") and "](" in line and line.endswith(")"):
                try:
                    alt = line[2 : line.index("]")]
                    src = line[line.index("](") + 2 : -1]
                    parts.append(f'<p><img src="{html.escape(src)}" alt="{html.escape(alt)}"/></p>')
                    continue
                except Exception:
                    pass
            if line.startswith("- "):
                parts.append(f"<p>• {html.escape(line[2:])}</p>")
            elif line.strip() == "":
                parts.append("<br/>")
            else:
                parts.append(f"<p>{html.escape(line)}</p>")
    parts.append("</body></html>")
    with open(path, "w") as f:
        f.write("\n".join(parts))


def make_summary_table_md(rows: list[dict[str, Any]], keys: list[str], max_rows: int = 30) -> str:
    if not rows:
        return "_(empty)_"
    head = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join(["---"] * len(keys)) + " |"
    body_lines = []
    for r in rows[:max_rows]:
        body_lines.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    extra = ""
    if len(rows) > max_rows:
        extra = f"\n\n_(showing {max_rows}/{len(rows)} rows)_"
    return "\n".join([head, sep, *body_lines]) + extra


def img_section(fig_rel_path: str, alt: str, extra_lines: list[str] | None = None) -> str:
    lines = []
    lines.append(_md_img(fig_rel_path, alt))
    if extra_lines:
        lines.append("")
        lines.extend(extra_lines)
    return "\n".join(lines)
