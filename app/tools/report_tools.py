"""
Report tools — Phase 8 full implementation.

generate_risk_matrix_chart  — matplotlib 5×5 heatmap PNG
compile_executive_summary   — structured plain-text executive summary
render_pdf_report           — weasyprint (primary) / reportlab (fallback) PDF
save_report_locally         — resolves pdf_handle, writes to ./reports/{deal_id}/,
                               registers session artifact; wrapped with
                               require_confirmation=True in reporter agent
"""

from __future__ import annotations

import html as _html_mod
import io
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Module-level cache: render_pdf_report stores bytes here keyed by a UUID handle.
# save_report_locally resolves the handle and then removes it.
_PDF_CACHE: dict[str, bytes] = {}


# ---------------------------------------------------------------------------
# 1. Risk matrix chart
# ---------------------------------------------------------------------------

def generate_risk_matrix_chart(risk_matrix: list, deal_id: str = "default") -> dict:
    """
    Render a 5×5 risk heatmap PNG and save to ./reports/{deal_id}/risk_matrix.png.

    Each entry in risk_matrix must be a dict with at minimum:
      likelihood (int 1-5), impact (int 1-5), risk_name (str), workstream (str).

    Args:
        risk_matrix: List of risk items from RiskAssessment.risk_matrix.
        deal_id: Unique deal identifier; used as the output sub-directory name.

    Returns:
        Dict with 'chart_path' (absolute path) and 'status'.
    """
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive; safe in server/CI environments
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    import numpy as np

    # ----- background: 5×5 score grid ----------------------------------------
    grid = np.array([[i * j for j in range(1, 6)] for i in range(1, 6)], dtype=float)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "risk", ["#27ae60", "#f39c12", "#e74c3c"], N=256
    )

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(grid, cmap=cmap, vmin=1, vmax=25, origin="lower", aspect="auto",
              extent=[-0.5, 4.5, -0.5, 4.5])

    # Cell score labels
    for row in range(5):
        for col in range(5):
            score = (row + 1) * (col + 1)
            text_color = "white" if score >= 15 else "#1a1a2e"
            ax.text(col, row, str(score), ha="center", va="center",
                    fontsize=11, color=text_color, fontweight="bold", zorder=4)

    # Axes
    tick_labels = ["1\n(Very Low)", "2\n(Low)", "3\n(Medium)", "4\n(High)", "5\n(Very High)"]
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("Impact", fontsize=12, labelpad=10, fontweight="bold")
    ax.set_ylabel("Likelihood", fontsize=12, labelpad=10, fontweight="bold")
    ax.set_title(f"Risk Matrix — {deal_id}", fontsize=14, fontweight="bold", pad=14)

    # Grid lines
    for i in range(6):
        ax.axhline(i - 0.5, color="white", linewidth=0.8, zorder=3)
        ax.axvline(i - 0.5, color="white", linewidth=0.8, zorder=3)

    # ----- risk item markers -------------------------------------------------
    workstream_colors = {
        "financial":      "#2c3e50",
        "legal":          "#8e44ad",
        "market":         "#1a5276",
        "news_sentiment": "#117a65",
        "people_culture": "#784212",
    }

    cell_counts: dict[tuple[int, int], int] = {}
    for item in risk_matrix:
        try:
            lh = max(0, min(4, int(item.get("likelihood", 1)) - 1))
            imp = max(0, min(4, int(item.get("impact", 1)) - 1))
            ws = str(item.get("workstream", "financial"))
            name = str(item.get("risk_name", "Risk"))
            color = workstream_colors.get(ws, "#2c3e50")

            key = (lh, imp)
            offset = cell_counts.get(key, 0) * 0.15
            cell_counts[key] = cell_counts.get(key, 0) + 1

            ax.plot(imp + offset, lh + offset, "o",
                    color=color, markersize=9,
                    markeredgecolor="white", markeredgewidth=1.5, zorder=5)
            ax.annotate(
                name[:22],
                xy=(imp + offset, lh + offset),
                xytext=(6, 4), textcoords="offset points",
                fontsize=6, color=color, zorder=6,
            )
        except (ValueError, TypeError, KeyError):
            continue

    # Workstream legend
    patches = [
        mpatches.Patch(color=c, label=ws.replace("_", " ").title())
        for ws, c in workstream_colors.items()
    ]
    ax.legend(handles=patches, loc="upper left",
              bbox_to_anchor=(1.02, 1), borderaxespad=0,
              fontsize=8, title="Workstream", title_fontsize=9)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=25))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.18)
    cbar.set_label("Risk Score (L × I)", fontsize=9)
    cbar.set_ticks([1, 6, 12, 18, 25])
    cbar.set_ticklabels(["Low\n(1)", "6", "12", "18", "Critical\n(25)"])

    plt.tight_layout()

    out_dir = Path(f"./reports/{deal_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_path = out_dir / "risk_matrix.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"status": "success", "chart_path": str(chart_path.resolve())}


# ---------------------------------------------------------------------------
# 2. Executive summary text
# ---------------------------------------------------------------------------

def compile_executive_summary(risk_assessment: dict, deal_metadata: dict) -> dict:
    """
    Produce a structured plain-text executive summary (≤ 2 pages when printed).

    Args:
        risk_assessment: Dict with top_10_risks, all_dealbreakers, deal_recommendation,
                         rationale, price_adjustment_recommendation_usd, recommended_escrow_pct.
        deal_metadata:   Dict with target_company, deal_value, deal_type, analysis_date.

    Returns:
        Dict with 'summary' (str) and 'status'.
    """
    target = deal_metadata.get("target_company", "Unknown Target")
    deal_value = deal_metadata.get("deal_value", 0)
    deal_type = str(deal_metadata.get("deal_type", "acquisition")).title()
    date_str = deal_metadata.get(
        "analysis_date",
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )

    recommendation = risk_assessment.get("deal_recommendation", "PROCEED_WITH_CONDITIONS")
    rationale = str(risk_assessment.get("rationale", ""))
    top_10 = list(risk_assessment.get("top_10_risks", []))
    dealbreakers = list(risk_assessment.get("all_dealbreakers", []))
    price_adj = float(risk_assessment.get("price_adjustment_recommendation_usd", 0) or 0)
    escrow_pct = float(risk_assessment.get("recommended_escrow_pct", 0) or 0)

    w = 72
    sep = "=" * w
    thin = "-" * w

    def section(title: str) -> str:
        return f"\n{thin}\n{title.upper()}\n{thin}"

    lines: list[str] = [
        sep,
        "DEALBREAKER — M&A DUE DILIGENCE EXECUTIVE SUMMARY".center(w),
        sep,
        f"  Target Company   : {target}",
        f"  Transaction Type : {deal_type}",
        f"  Deal Value       : ${deal_value:,.0f}" if isinstance(deal_value, (int, float))
        else f"  Deal Value       : {deal_value}",
        f"  Analysis Date    : {date_str}",
        f"  Recommendation   : *** {recommendation} ***",
        sep,
    ]

    lines.append(section("Recommendation Rationale"))
    lines.append(rationale)

    lines.append(section(
        f"Dealbreakers Identified ({len(dealbreakers)})" if dealbreakers
        else "Dealbreakers: None Identified"
    ))
    if dealbreakers:
        for i, db in enumerate(dealbreakers, 1):
            lines.append(f"  {i:>2}. {db}")
    else:
        lines.append("  No dealbreakers were identified across any workstream.")

    lines.append(section("Top 5 Priority Risks"))
    if top_10:
        for i, risk in enumerate(top_10[:5], 1):
            lines.append(f"  {i}. {risk}")
    else:
        lines.append("  Risk data not available.")

    if price_adj > 0 or escrow_pct > 0:
        lines.append(section("Deal Structure Recommendations"))
        if price_adj > 0:
            lines.append(f"  Price Adjustment   : -${price_adj:,.0f}")
        if escrow_pct > 0:
            lines.append(f"  Escrow Hold-back   : {escrow_pct:.1f}% of deal value")

    if len(top_10) > 5:
        lines.append(section("Additional Risks (6–10)"))
        for i, risk in enumerate(top_10[5:], 6):
            lines.append(f"  {i:>2}. {risk}")

    lines += [
        "",
        sep,
        "See the full report PDF for detailed workstream findings and source citations.",
        sep,
    ]

    return {"status": "success", "summary": "\n".join(lines)}


# ---------------------------------------------------------------------------
# 3. PDF rendering
# ---------------------------------------------------------------------------

def render_pdf_report(sections: dict, chart_paths: list) -> dict:
    """
    Assemble a full due diligence PDF from sections text and embedded chart images.

    Attempts weasyprint first (higher quality); falls back to reportlab if
    weasyprint's native GTK/Pango dependencies are unavailable (common on Windows).

    The rendered bytes are stored in an in-memory cache under a UUID handle.
    Pass the returned 'pdf_handle' to save_report_locally to write to disk.

    Args:
        sections:    Dict keyed by workstream / section name containing the text for
                     each section. Recognised keys: executive_summary, financial, legal,
                     market, news_sentiment, people_culture, risk_assessment, appendix.
        chart_paths: List of absolute PNG file paths to embed in the Charts section.

    Returns:
        Dict with 'pdf_handle', 'renderer' ('weasyprint' or 'reportlab'),
        'size_bytes', and 'status'.
    """
    renderer = "reportlab"
    pdf_bytes: bytes = b""

    # Try weasyprint (primary)
    try:
        pdf_bytes = _render_weasyprint(sections, chart_paths)
        renderer = "weasyprint"
    except Exception:
        pass

    # Fallback to reportlab
    if not pdf_bytes:
        try:
            pdf_bytes = _render_reportlab(sections, chart_paths)
            renderer = "reportlab"
        except Exception as exc:
            return {"status": "error", "message": f"PDF rendering failed: {exc}"}

    handle = str(uuid.uuid4())
    _PDF_CACHE[handle] = pdf_bytes
    return {
        "status": "success",
        "pdf_handle": handle,
        "renderer": renderer,
        "size_bytes": len(pdf_bytes),
    }


# ---------------------------------------------------------------------------
# 4. Save to disk + register artifact
# ---------------------------------------------------------------------------

async def save_report_locally(
    pdf_handle: str,
    deal_id: str,
    filename: str,
    tool_context=None,
) -> dict:
    """
    Resolve a pdf_handle from render_pdf_report, write it to
    ./reports/{deal_id}/{filename}, and register it as a session artifact.

    This tool is wrapped with require_confirmation=True in the reporter agent
    so the analyst must approve before the file is written.

    Args:
        pdf_handle: The handle string returned by render_pdf_report.
        deal_id:    Unique deal identifier; determines the output sub-directory.
        filename:   Output file name, e.g. 'due_diligence_report.pdf'.
    """
    pdf_bytes = _PDF_CACHE.get(pdf_handle)
    if not pdf_bytes:
        return {
            "status": "error",
            "message": (
                f"PDF handle '{pdf_handle}' not found in cache. "
                "Ensure render_pdf_report was called first and succeeded."
            ),
        }

    out_path = Path(f"./reports/{deal_id}/{filename}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(pdf_bytes)

    # Register as session artifact (best-effort — requires InMemoryArtifactService)
    if tool_context is not None:
        try:
            from google.genai import types as _gtypes
            part = _gtypes.Part(
                inline_data=_gtypes.Blob(mime_type="application/pdf", data=pdf_bytes)
            )
            await tool_context.save_artifact(f"reports/{deal_id}/{filename}", part)
        except Exception:
            pass  # Artifact registration is best-effort; don't fail the save

    _PDF_CACHE.pop(pdf_handle, None)

    return {"status": "success", "local_path": str(out_path.resolve())}


# ---------------------------------------------------------------------------
# Private rendering helpers
# ---------------------------------------------------------------------------

_SECTION_ORDER = [
    "executive_summary",
    "financial",
    "legal",
    "market",
    "news_sentiment",
    "people_culture",
    "risk_assessment",
    "appendix",
]

_SECTION_LABELS = {
    "executive_summary": "Executive Summary",
    "financial":         "Financial Due Diligence",
    "legal":             "Legal Review",
    "market":            "Market Research",
    "news_sentiment":    "News & Sentiment Analysis",
    "people_culture":    "People & Culture",
    "risk_assessment":   "Risk Assessment",
    "appendix":          "Appendix: Source Citations",
}


def _render_reportlab(sections: dict, chart_paths: list) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer,
        HRFlowable, PageBreak,
    )
    # Image import: use a local alias to avoid shadowing builtins
    from reportlab.platypus import Image as _RLImage

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
        title="DealBreaker Due Diligence Report",
        author="DealBreaker AI",
    )

    styles = getSampleStyleSheet()

    cover_style = ParagraphStyle(
        "Cover", parent=styles["Heading1"],
        fontSize=30, textColor=colors.HexColor("#c0392b"),
        spaceAfter=6,
    )
    sub_cover = ParagraphStyle(
        "SubCover", parent=styles["Heading2"],
        fontSize=16, textColor=colors.HexColor("#16213e"),
        spaceAfter=20,
    )
    h1_style = ParagraphStyle(
        "H1", parent=styles["Heading1"],
        fontSize=16, textColor=colors.HexColor("#1a1a2e"),
        spaceBefore=18, spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=9.5, leading=14, spaceAfter=5,
    )
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    story = []

    # Cover page
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("DEALBREAKER", cover_style))
    story.append(Paragraph("M&amp;A Due Diligence Report", sub_cover))
    story.append(HRFlowable(width="100%", thickness=3, color=colors.HexColor("#c0392b")))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(f"Generated: {date_str}", body_style))
    story.append(PageBreak())

    # Content sections
    for key in _SECTION_ORDER:
        content = sections.get(key, "")
        if not content:
            continue
        label = _SECTION_LABELS.get(key, key.replace("_", " ").title())
        story.append(Paragraph(label, h1_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
        story.append(Spacer(1, 0.25 * cm))

        for block in str(content).split("\n\n"):
            block = block.strip()
            if not block:
                continue
            # Escape XML special chars for reportlab
            safe = (block
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace("\n", "<br/>"))
            story.append(Paragraph(safe, body_style))

        story.append(Spacer(1, 0.4 * cm))

    # Charts section
    valid_charts = [p for p in chart_paths if Path(p).exists()]
    if valid_charts:
        story.append(PageBreak())
        story.append(Paragraph("Charts", h1_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
        story.append(Spacer(1, 0.25 * cm))
        for cp in valid_charts:
            story.append(_RLImage(str(cp), width=14 * cm, height=9.5 * cm, kind="proportional"))
            story.append(Spacer(1, 0.5 * cm))

    doc.build(story)
    return buf.getvalue()


def _render_weasyprint(sections: dict, chart_paths: list) -> bytes:
    import base64
    import weasyprint  # Will raise ImportError/OSError on Windows without Pango

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<style>",
        "  @page { margin: 2cm; size: A4; }",
        "  body { font-family: Arial, sans-serif; font-size: 10pt; color: #1a1a2e; }",
        "  h1.cover { font-size: 30pt; color: #c0392b; margin-bottom: 4px; }",
        "  h2.cover { font-size: 16pt; color: #16213e; }",
        "  hr.thick { border: none; border-top: 3px solid #c0392b; }",
        "  hr.thin  { border: none; border-top: 1px solid #ccc; }",
        "  h1 { font-size: 14pt; color: #1a1a2e; border-bottom: 1px solid #ccc; ",
        "       padding-bottom: 4px; margin-top: 20px; }",
        "  pre { white-space: pre-wrap; word-break: break-word; font-family: inherit; ",
        "        font-size: 9pt; line-height: 1.4; }",
        "  img { max-width: 100%; page-break-inside: avoid; }",
        "  .page-break { page-break-after: always; }",
        "</style></head><body>",
        "<div class='page-break'>",
        "<br><br><br>",
        "<h1 class='cover'>DEALBREAKER</h1>",
        "<h2 class='cover'>M&amp;A Due Diligence Report</h2>",
        "<hr class='thick'>",
        f"<p>Generated: {_html_mod.escape(date_str)}</p>",
        "</div>",
    ]

    for key in _SECTION_ORDER:
        content = sections.get(key, "")
        if not content:
            continue
        label = _SECTION_LABELS.get(key, key.replace("_", " ").title())
        parts.append(f"<h1>{_html_mod.escape(label)}</h1>")
        parts.append(f"<pre>{_html_mod.escape(str(content))}</pre>")

    valid_charts = [p for p in chart_paths if Path(p).exists()]
    if valid_charts:
        parts.append("<h1>Charts</h1>")
        for cp in valid_charts:
            raw = Path(cp).read_bytes()
            b64 = base64.b64encode(raw).decode()
            parts.append(f'<img src="data:image/png;base64,{b64}" alt="Risk Matrix Chart"/>')

    parts.append("</body></html>")
    return weasyprint.HTML(string="".join(parts)).write_pdf()
