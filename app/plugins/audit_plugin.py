"""
Audit plugin — Phase 9 full implementation.

Logs every tool call and model interaction to a structured JSON audit trail using
structlog. Logs only metadata (keys, not values; status, not content) to avoid
PII leaking into the audit log.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Optional

import structlog
from structlog.processors import JSONRenderer, TimeStamper, add_log_level

from google.adk.plugins.base_plugin import BasePlugin

# ---------------------------------------------------------------------------
# Configure structlog once at module import time.
# Uses JSON output → stdout so log aggregators (CloudWatch, Datadog, etc.) can
# ingest it without further parsing. Re-configuration is idempotent.
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        add_log_level,
        TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
    cache_logger_on_first_use=True,
)

_logger = structlog.get_logger("dealbreaker.audit")


class AuditPlugin(BasePlugin):
    """Structured audit trail for every tool call and model response in the pipeline."""

    def __init__(self) -> None:
        super().__init__(name="audit")

    # ------------------------------------------------------------------
    # Tool callbacks
    # ------------------------------------------------------------------

    async def before_tool_callback(
        self,
        *,
        tool: Any,
        tool_args: dict[str, Any],
        tool_context: Any,
    ) -> Optional[dict]:
        """Log tool invocation metadata — arg keys only, never values."""
        _logger.info(
            "tool_call",
            tool=tool.name,
            deal_id=tool_context.state.get("deal_id"),
            analyst_id=tool_context.state.get("user:analyst_id"),
            args_keys=sorted(tool_args.keys()),
        )
        return None  # proceed normally

    async def after_tool_callback(
        self,
        *,
        tool: Any,
        tool_args: dict[str, Any],
        tool_context: Any,
        result: dict,
    ) -> Optional[dict]:
        """Log tool completion status — result status only, never payload values."""
        status = result.get("status") if isinstance(result, dict) else "non-dict-result"
        _logger.info(
            "tool_result",
            tool=tool.name,
            deal_id=tool_context.state.get("deal_id"),
            status=status,
        )
        return None  # do not modify the result

    # ------------------------------------------------------------------
    # Model callbacks
    # ------------------------------------------------------------------

    async def before_model_callback(
        self,
        *,
        callback_context: Any,
        llm_request: Any,
    ) -> None:
        """Log that a model request is about to be sent."""
        _logger.info(
            "model_request",
            agent=callback_context.agent_name,
            deal_id=callback_context.state.get("deal_id"),
        )
        return None

    async def after_model_callback(
        self,
        *,
        callback_context: Any,
        llm_response: Any,
    ) -> None:
        """Log model response metadata — finish_reason and token counts, no text."""
        finish_reason = None
        total_tokens: Optional[int] = None

        if llm_response.finish_reason is not None:
            finish_reason = str(llm_response.finish_reason)

        if getattr(llm_response, "usage_metadata", None) is not None:
            um = llm_response.usage_metadata
            total_tokens = getattr(um, "total_token_count", None)

        _logger.info(
            "model_response",
            agent=callback_context.agent_name,
            deal_id=callback_context.state.get("deal_id"),
            finish_reason=finish_reason,
            total_tokens=total_tokens,
        )
        return None

    # ------------------------------------------------------------------
    # Error callbacks
    # ------------------------------------------------------------------

    async def on_tool_error_callback(
        self,
        *,
        tool: Any,
        tool_args: dict[str, Any],
        tool_context: Any,
        error: Exception,
    ) -> Optional[dict]:
        """Log tool errors with type and message — no stack trace to avoid PII."""
        _logger.error(
            "tool_error",
            tool=tool.name,
            deal_id=tool_context.state.get("deal_id"),
            error_type=type(error).__name__,
            error_message=str(error)[:200],
        )
        return None  # re-raise by returning None

    async def on_model_error_callback(
        self,
        *,
        callback_context: Any,
        llm_request: Any,
        error: Exception,
    ) -> None:
        """Log model errors."""
        _logger.error(
            "model_error",
            agent=callback_context.agent_name,
            deal_id=callback_context.state.get("deal_id"),
            error_type=type(error).__name__,
            error_message=str(error)[:200],
        )
        return None
