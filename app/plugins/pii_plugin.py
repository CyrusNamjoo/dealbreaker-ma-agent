"""
PII redaction plugin — Phase 9 full implementation.

Intercepts model text responses (after_model_callback) and scrubs PERSON names,
EMAIL_ADDRESS, and PHONE_NUMBER entities from the prose text parts using
presidio-analyzer + presidio-anonymizer.

Design decisions:
- Only prose text parts (part.text is not None) are scanned and scrubbed.
  Function call / function response parts pass through unmodified so that the
  agent's tool invocations and structured JSON outputs are never altered.
- AnalyzerEngine and AnonymizerEngine are lazy-loaded on first use — they load
  a spaCy NLP model (~400 MB) so cold init is slow; subsequent calls are fast.
- The modified LlmResponse is returned only when at least one entity is found,
  avoiding unnecessary object allocation on clean responses.
- Returns None from tool callbacks so tool inputs/outputs are never modified.
"""

from __future__ import annotations

import structlog
from typing import Any, Optional

from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types as _gtypes

_logger = structlog.get_logger("dealbreaker.pii")

# Entities scrubbed from prose model output.
# LOCATION is intentionally excluded — the agents legitimately discuss
# geographic jurisdictions and court locations in their analysis.
_SCRUB_ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]


class PIIRedactionPlugin(BasePlugin):
    """
    Scrubs PERSON, EMAIL_ADDRESS, and PHONE_NUMBER from model text output
    using presidio-analyzer before responses propagate through the pipeline.
    """

    def __init__(self) -> None:
        super().__init__(name="pii_redaction")
        # Engines are None until first use — see _engines property.
        self._analyzer = None
        self._anonymizer = None

    # ------------------------------------------------------------------
    # Lazy engine initialisation
    # ------------------------------------------------------------------

    @property
    def _engines(self):
        """Return (AnalyzerEngine, AnonymizerEngine), loading on first access."""
        if self._analyzer is None:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
        return self._analyzer, self._anonymizer

    # ------------------------------------------------------------------
    # Core scrubbing helper
    # ------------------------------------------------------------------

    def _scrub(self, text: str) -> tuple[str, int]:
        """
        Analyze text for PII and return (anonymized_text, entity_count).

        Returns the original text unchanged if no PII is detected.
        """
        if not text or not text.strip():
            return text, 0

        analyzer, anonymizer = self._engines
        results = analyzer.analyze(
            text=text,
            language="en",
            entities=_SCRUB_ENTITIES,
            # 0.4 catches phone numbers (presidio scores them 0.40);
            # false-positive risk is acceptable for these three specific types.
            score_threshold=0.4,
        )

        if not results:
            return text, 0

        anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text, len(results)

    # ------------------------------------------------------------------
    # after_model_callback — scrub prose text parts
    # ------------------------------------------------------------------

    async def after_model_callback(
        self,
        *,
        callback_context: Any,
        llm_response: Any,
    ) -> Optional[Any]:
        """
        Scan each text part of the model response for PII.
        Replace detected entities with typed placeholders (<PERSON>, etc.).
        Return a modified LlmResponse if any part was changed; else None.
        """
        if llm_response.content is None or not llm_response.content.parts:
            return None

        new_parts = []
        total_entities = 0
        any_modified = False

        for part in llm_response.content.parts:
            # Only process prose text — leave function_call / function_response intact
            if part.text is None:
                new_parts.append(part)
                continue

            scrubbed, entity_count = self._scrub(part.text)
            total_entities += entity_count

            if entity_count > 0:
                new_parts.append(_gtypes.Part(text=scrubbed))
                any_modified = True
            else:
                new_parts.append(part)

        if not any_modified:
            return None

        _logger.info(
            "pii_redacted",
            agent=callback_context.agent_name,
            deal_id=callback_context.state.get("deal_id"),
            entity_count=total_entities,
        )

        new_content = _gtypes.Content(
            role=llm_response.content.role,
            parts=new_parts,
        )
        return llm_response.model_copy(update={"content": new_content})

    # ------------------------------------------------------------------
    # Tool callbacks — pass through without modification
    # ------------------------------------------------------------------

    async def before_tool_callback(
        self,
        *,
        tool: Any,
        tool_args: dict[str, Any],
        tool_context: Any,
    ) -> Optional[dict]:
        """Tool inputs are intentionally not scrubbed — agents need real values to query."""
        return None

    async def after_tool_callback(
        self,
        *,
        tool: Any,
        tool_args: dict[str, Any],
        tool_context: Any,
        result: dict,
    ) -> Optional[dict]:
        """Structured tool results are not scrubbed — they form typed schema outputs."""
        return None
