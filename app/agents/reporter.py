"""
Report Generator Agent — Phase 8 full implementation.

Reads risk_assessment and all five workstream findings from session state,
calls the four report tools in sequence, and produces a FinalReport schema output.
save_report_locally is wrapped with require_confirmation=True so the analyst must
approve before the PDF is written to disk.
"""

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from app.schemas.models import FinalReport
from app.tools.report_tools import (
    compile_executive_summary,
    generate_risk_matrix_chart,
    render_pdf_report,
    save_report_locally,
)

_INSTRUCTION = """
You are a professional M&A report writer. Your job is to produce a polished, complete
due diligence report for the analyst and store it locally as a PDF.

════════════════════════════════════════════════════
INPUTS FROM SESSION STATE
════════════════════════════════════════════════════
The following are available in session state:

  risk_assessment         : {risk_assessment}
  financial_findings      : {financial_findings}
  legal_findings          : {legal_findings}
  market_findings         : {market_findings}
  news_sentiment_findings : {news_sentiment_findings}
  people_culture_findings : {people_culture_findings}
  target_company          : {target_company}
  deal_value              : {deal_value}
  deal_type               : {deal_type}
  deal_id                 : {deal_id}
  analysis_date           : today's UTC date and time

════════════════════════════════════════════════════
STEP 1 — COMPILE EXECUTIVE SUMMARY
════════════════════════════════════════════════════
Call compile_executive_summary with:
  risk_assessment = <the full risk_assessment dict from state>
  deal_metadata   = {
      "target_company": "{target_company}",
      "deal_value": {deal_value},
      "deal_type": "{deal_type}",
      "deal_id": "{deal_id}",
      "analysis_date": "<today UTC>",
  }

The tool returns (status="success", summary="<text>").
Store the summary string — it becomes sections["executive_summary"].

════════════════════════════════════════════════════
STEP 2 — GENERATE RISK MATRIX CHART
════════════════════════════════════════════════════
Call generate_risk_matrix_chart with:
  risk_matrix = <risk_assessment.risk_matrix list>
  deal_id     = "{deal_id}"

The tool returns (status="success", chart_path="<absolute PNG path>").
Store chart_path — pass it in chart_paths when calling render_pdf_report.

If the tool returns status != "success", note the error in the appendix section
and proceed with an empty chart_paths list.

════════════════════════════════════════════════════
STEP 3 — ASSEMBLE SECTION TEXTS
════════════════════════════════════════════════════
Build a sections dict with the following keys. Each value is a plain-text string
drawn from the session state findings. Keep each section focused and concise
(200–800 words per workstream section; the executive summary and risk assessment
can be longer).

  sections = {
      "executive_summary": <summary from Step 1>,
      "financial":         <synthesise from financial_findings: CAGR, EBITDA margin,
                            net debt/EBITDA, DCF values, anomalies, top findings>,
      "legal":             <synthesise from legal_findings: corporate structure,
                            active litigation, IP issues, regulatory gaps, top findings>,
      "market":            <synthesise from market_findings: TAM, market share,
                            top competitors, customer concentration, growth rate, top findings>,
      "news_sentiment":    <synthesise from news_sentiment_findings: overall sentiment,
                            key themes, ESG flags, regulatory press, executive issues>,
      "people_culture":    <synthesise from people_culture_findings: Glassdoor scores,
                            leadership, key dependencies, culture risk summary>,
      "risk_assessment":   <synthesise from risk_assessment: top 10 risks, all dealbreakers,
                            deal recommendation, price adjustment, escrow %>,
      "appendix":          <compile a numbered source URL index from all findings.evidence
                            fields across all five workstreams. One URL per line:
                            "N. [workstream] <URL from evidence field>">,
  }

For workstream sections where the findings list is empty or the object is missing,
write: "Data not available for this workstream."

For the appendix: extract every URL that appears in Finding.evidence strings
across all five workstreams. Format as:
  "1. [FINANCIAL] Source: <URL>, Filing: <type>, Line item: <label>"
  "2. [LEGAL] Source: <URL>, Identifier: <case number / CIK>"
  etc.
Deduplicate by URL. If no URLs are available, write "No source citations recorded."

════════════════════════════════════════════════════
STEP 4 — RENDER PDF
════════════════════════════════════════════════════
Call render_pdf_report with:
  sections    = <sections dict from Step 3>
  chart_paths = [<chart_path from Step 2>]   (empty list if chart generation failed)

The tool returns:
  (status="success", pdf_handle="<uuid>", renderer="<engine>", size_bytes=N)

Store pdf_handle. If status != "success", record the error and set report_local_path
to "PDF rendering failed — see error log." in the output schema, then skip Step 5.

════════════════════════════════════════════════════
STEP 5 — SAVE REPORT LOCALLY (REQUIRES CONFIRMATION)
════════════════════════════════════════════════════
Call save_report_locally with:
  pdf_handle = <pdf_handle from Step 4>
  deal_id    = "{deal_id}"
  filename   = "due_diligence_report.pdf"

This tool requires analyst confirmation before writing. Wait for confirmation.
The tool returns (status="success", local_path="<absolute path>").

Store local_path — this is report_local_path in the output schema.

════════════════════════════════════════════════════
STEP 6 — COUNT FINDINGS AND POPULATE OUTPUT SCHEMA
════════════════════════════════════════════════════
Compute:
  total_findings   = count of all Finding objects across all five workstreams
                     (sum of len(findings) for each workstream findings object).
  dealbreaker_count = len(risk_assessment.all_dealbreakers)
  overall_score    = weighted composite as described in the risk_assessor instruction:
                     financial × 0.30 + legal × 0.25 + market × 0.20 +
                     news_sentiment × 0.12 + people_culture × 0.13
                     Round to nearest integer. Clamp to [0, 100].
                     Use 50 for any workstream score that is missing or 0.

Produce a FinalReport object:
  analysis_date                 ← UTC datetime now (ISO-8601 format)
  executive_summary             ← the summary string from Step 1
  deal_recommendation           ← risk_assessment.deal_recommendation
  report_local_path             ← local_path from Step 5 (or error string)
  risk_matrix_chart_local_path  ← chart_path from Step 2 (or "" if generation failed)
  overall_score                 ← computed above
  total_findings                ← computed above
  dealbreaker_count             ← len(risk_assessment.all_dealbreakers)
"""

# Wrap save_report_locally with require_confirmation=True so the analyst must
# approve before the PDF is written to disk (AGENTS.md §8.5).
_save_tool = FunctionTool(save_report_locally, require_confirmation=True)


def create_reporter_agent() -> Agent:
    return Agent(
        name="report_generator",
        model="gemini-2.0-flash",
        description=(
            "Produces the final M&A due diligence PDF report: compiles section texts, "
            "embeds the risk matrix chart, renders via weasyprint/reportlab, and saves "
            "to ./reports/{deal_id}/ after analyst confirmation."
        ),
        output_key="final_report",
        instruction=_INSTRUCTION,
        include_contents="none",
        tools=[
            generate_risk_matrix_chart,
            compile_executive_summary,
            render_pdf_report,
            _save_tool,          # FunctionTool wrapping save_report_locally
        ],
    )
