"""
Financial guardrails plugin — Phase 9 full implementation.

Intercepts model text responses (after_model_callback) and rejects any response
that contains a specific financial figure (revenue, EBITDA, DCF values, ratios,
percentages in financial context) without an accompanying public source URL
citation in the same paragraph.

When a violation is detected the plugin returns a replacement LlmResponse that
instructs the agent to re-generate the answer with proper source citations,
preventing hallucinated numbers from entering the risk assessment.

Agents excluded from guardrail checks:
  - dealbreaker_coordinator  (repeats deal parameters supplied by the user)

All specialist and synthesis agents are subject to the guardrail.
"""

from __future__ import annotations

import re
import structlog
from typing import Any, Optional

from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types as _gtypes

_logger = structlog.get_logger("dealbreaker.guardrails")

# ---------------------------------------------------------------------------
# Agents exempt from the financial citation guardrail
# ---------------------------------------------------------------------------
_EXEMPT_AGENTS = frozenset({"dealbreaker_coordinator"})

# ---------------------------------------------------------------------------
# Pattern: specific financial figures that require a public source citation.
#
# Matches claims about a company's actual financial performance — revenue,
# EBITDA, FCF, profit/loss, debt, and financial ratios. Intentionally does NOT
# match generic score numbers (risk score: 4/25) or deal parameters.
# ---------------------------------------------------------------------------
_FIN_FIGURE_RE = re.compile(
    r"""
    (?:
        # ── Financial metric keyword followed by a currency amount ──
        # Dollar sign is optional — agents may write "revenue was 1.2 billion"
        (?:
            (?:revenue|EBITDA|FCF|free\s+cash\s+flow|gross\s+profit|
               net\s+(?:income|loss|profit)|operating\s+(?:income|loss)|
               enterprise\s+value|market\s+cap(?:italiz\w+)?|
               total\s+(?:assets|debt|liabilities))
            \s{0,10}
            (?:of|was|were|is|are|totaled?|reached?|grew\s+to|
               declined?\s+to|amounted?\s+to)?
            \s{0,5}
            \$?\s*[\d,]+(?:\.\d+)?\s*(?:million|billion|thousand|[MBKT])?
        )
    |
        # ── Amount (with or without $) followed by "in/of <financial metric>" ──
        (?:
            \$?\s*[\d,]+(?:\.\d+)?\s*(?:million|billion|[MB])
            \s{0,10}(?:in|of)\s{0,5}
            (?:revenue|EBITDA|FCF|free\s+cash\s+flow|profit|income|
               loss|debt|sales|bookings|ARR|MRR)
        )
    |
        # ── Percentage paired with a financial metric keyword ──
        (?:
            [\d]+(?:\.\d+)?\s*%
            \s{0,10}
            (?:EBITDA\s+)?
            (?:margin|CAGR|revenue\s+growth|gross\s+margin|
               operating\s+margin|growth\s+rate|of\s+revenue|
               year[- ]over[- ]year)
        |
            (?:EBITDA\s+margin|revenue\s+CAGR|gross\s+margin|
               operating\s+margin|revenue\s+growth(?:\s+rate)?)
            \s{0,10}(?:of|was|is|at|:)?\s{0,5}
            [\d]+(?:\.\d+)?\s*%
        )
    |
        # ── Financial leverage / coverage ratios with a numeric value ──
        (?:
            (?:net\s+debt\s*[/÷]\s*EBITDA|leverage\s+ratio|
               interest\s+coverage|current\s+ratio|quick\s+ratio|
               debt[- ]to[- ]equity|D[/\\]E\s+ratio)
            \s{0,10}(?:of|was|is|at|:)?\s{0,5}
            [\d]+(?:\.\d+)?\s*[xX]?
        )
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Pattern: valid source citation evidence.
#
# A paragraph "has a citation" if it contains at least one of:
#   • An https:// URL (minimum 15 chars total to exclude bare domains)
#   • A "Source: https://..." inline citation
#   • An explicit SEC filing reference with a year (10-K 2023, 10-Q FY2022)
#   • "SEC EDGAR" reference
#   • An annual/quarterly report reference
# ---------------------------------------------------------------------------
_CITATION_RE = re.compile(
    r"""
    (?:
        https?://[\w\-./?=&#%+@]{10,}    # https:// URL (≥10 chars of path/query)
    |
        Source:\s*https?://              # inline "Source: <url>"
    |
        (?:10-K|10-Q|8-K|DEF\s*14A|S-1|20-F)   # SEC form type …
        [\s,]+(?:FY|fiscal\s+year\s+)?\d{4}      # … with a year
    |
        SEC\s+EDGAR
    |
        (?:annual|quarterly)\s+report\s+\(?(?:FY|fiscal)?\s*\d{4}\)?
    |
        (?:filed|retrieved)\s+(?:from|via|at)\s+https?://
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Rejection message template
# ---------------------------------------------------------------------------
_REJECTION_TEMPLATE = (
    "GUARDRAIL VIOLATION — Your response was blocked because it contains "
    "specific financial figures without public source URL citations. "
    "The following paragraph(s) triggered the guardrail:\n\n"
    "{violations}\n\n"
    "Re-generate your response and ensure that every financial figure "
    "(revenue amount, EBITDA value, margin percentage, ratio, DCF value, etc.) "
    "is immediately followed by a citation in the format:\n"
    '  "Source: <full https:// URL>, Filing: <form type> <fiscal year>, '
    'Line item: <exact label>"\n\n'
    "If the figure cannot be sourced from a public URL, omit it entirely. "
    "Do NOT fabricate or estimate financial figures."
)


class FinancialGuardrailsPlugin(BasePlugin):
    """
    Rejects model responses that assert specific financial figures without a
    public source URL citation in the same paragraph.
    """

    def __init__(self) -> None:
        super().__init__(name="financial_guardrails")

    # ------------------------------------------------------------------
    # Core check
    # ------------------------------------------------------------------

    def _check_text(self, text: str) -> list[str]:
        """
        Return a list of offending paragraph excerpts (≤ 120 chars each).
        Empty list means the text passes the guardrail.
        """
        violations: list[str] = []

        # Split on blank lines (paragraph boundaries).
        # Also handle single newline-separated lists/bullets.
        paragraphs = re.split(r"\n{2,}", text.strip())
        if len(paragraphs) == 1:
            # Fallback: treat each non-empty line as its own block
            paragraphs = [ln for ln in text.split("\n") if ln.strip()]

        for para in paragraphs:
            if not para.strip():
                continue
            if _FIN_FIGURE_RE.search(para) and not _CITATION_RE.search(para):
                # Truncate for the violation report
                snippet = para.strip().replace("\n", " ")[:120]
                if len(para.strip()) > 120:
                    snippet += "…"
                violations.append(snippet)

        return violations

    # ------------------------------------------------------------------
    # after_model_callback
    # ------------------------------------------------------------------

    async def after_model_callback(
        self,
        *,
        callback_context: Any,
        llm_response: Any,
    ) -> Optional[LlmResponse]:
        """
        Scan prose text parts of the model response.
        If any paragraph contains a specific financial figure without a
        public source URL citation, replace the entire response with a
        rejection message that instructs the agent to re-cite and retry.
        """
        agent_name = callback_context.agent_name

        # Coordinator repeats user-supplied deal parameters — exempt
        if agent_name in _EXEMPT_AGENTS:
            return None

        if llm_response.content is None or not llm_response.content.parts:
            return None

        all_violations: list[str] = []

        for part in llm_response.content.parts:
            # Only scan prose text; skip function_call / function_response parts
            if part.text is None:
                continue
            all_violations.extend(self._check_text(part.text))

        if not all_violations:
            return None

        # Build a formatted list of violation excerpts
        numbered = "\n".join(
            f"  {i}. {v}" for i, v in enumerate(all_violations, 1)
        )
        rejection_text = _REJECTION_TEMPLATE.format(violations=numbered)

        _logger.warning(
            "guardrail_triggered",
            agent=agent_name,
            deal_id=callback_context.state.get("deal_id"),
            violation_count=len(all_violations),
            first_violation=all_violations[0],
        )

        rejection_response = LlmResponse(
            content=_gtypes.Content(
                role="model",
                parts=[_gtypes.Part(text=rejection_text)],
            )
        )
        return rejection_response

    # ------------------------------------------------------------------
    # Tool callbacks — guardrail does not apply to tool I/O
    # ------------------------------------------------------------------

    async def before_tool_callback(
        self,
        *,
        tool: Any,
        tool_args: dict[str, Any],
        tool_context: Any,
    ) -> Optional[dict]:
        return None

    async def after_tool_callback(
        self,
        *,
        tool: Any,
        tool_args: dict[str, Any],
        tool_context: Any,
        result: dict,
    ) -> Optional[dict]:
        return None
