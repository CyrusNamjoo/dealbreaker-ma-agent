import os
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

"""
DealBreaker root coordinator — Phase 7 full pipeline.

Replaces the Phase 3 isolation shim. Wires investigation_phase (ParallelAgent)
and synthesis_phase (SequentialAgent) into a single due_diligence_pipeline, then
wraps it in the dealbreaker_coordinator LlmAgent with the full AGENTS.md §2.2 instruction.
"""

from pathlib import Path

from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps import App
from google.adk.artifacts import InMemoryArtifactService
from google.adk.sessions import InMemorySessionService

Path("./reports").mkdir(parents=True, exist_ok=True)

from app.agents.financial import create_financial_agent
from app.agents.legal import create_legal_agent
from app.agents.market import create_market_agent
from app.agents.news_sentiment import create_news_sentiment_agent
from app.agents.people_culture import create_people_culture_agent
from app.agents.risk import create_risk_agent
from app.agents.reporter import create_reporter_agent
from app.plugins.audit_plugin import AuditPlugin
from app.plugins.pii_plugin import PIIRedactionPlugin
from app.plugins.guardrails_plugin import FinancialGuardrailsPlugin

# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------

session_service = InMemorySessionService()
artifact_service = InMemoryArtifactService()

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

investigation_phase = ParallelAgent(
    name="investigation_phase",
    sub_agents=[
        create_financial_agent(),
        create_legal_agent(),
        create_market_agent(),
        create_news_sentiment_agent(),
        create_people_culture_agent(),
    ],
)

synthesis_phase = SequentialAgent(
    name="synthesis_phase",
    sub_agents=[
        create_risk_agent(),
        create_reporter_agent(),
    ],
)

pipeline = SequentialAgent(
    name="due_diligence_pipeline",
    sub_agents=[investigation_phase, synthesis_phase],
)

# ---------------------------------------------------------------------------
# Default session state
# ---------------------------------------------------------------------------
# adk web creates a fresh, empty session with no state. Seed it with a demo
# deal so the pipeline has the variables it needs (target_company, deal_value,
# etc.) without requiring replay.json. Only fills in keys that are missing, so
# a replay-supplied or user-supplied session's state is left untouched.

DEFAULT_SESSION_STATE = {
    "target_company": "Salesforce",
    "industry": "SaaS",
    "deal_value": 50000000,
    "deal_id": "deal_demo_001",
    "deal_type": "acquisition",
    "scope": "full",
    "acquirer_description": "A mid-size enterprise software company focused on productivity and CRM tools",
    "jurisdictions": ["US"],
}


async def seed_default_state(callback_context: CallbackContext) -> None:
    for key, value in DEFAULT_SESSION_STATE.items():
        if key not in callback_context.state:
            callback_context.state[key] = value


# ---------------------------------------------------------------------------
# Root coordinator
# ---------------------------------------------------------------------------

_COORDINATOR_INSTRUCTION = """
You are DealBreaker, an expert M&A due diligence coordinator.

Deal context:
- Target company: {target_company}
- Deal type: {deal_type}
- Deal value: {deal_value}
- Investigation scope: {scope}

Your job:
1. Confirm the deal parameters with the user before launching the investigation.
2. Delegate the full investigation to the due_diligence_pipeline.
3. Present the final risk-rated report and highlight any DEALBREAKERS.

A DEALBREAKER is any finding that should cause reconsideration of the deal:
material misstatement, undisclosed litigation above {deal_value} * 0.05,
severe reputational crisis, hostile leadership culture, or regulatory non-compliance
that cannot be remediated.

All research uses publicly available data only. Do not request or process
any confidential documents.
"""

root_agent = Agent(
    name="dealbreaker_coordinator",
    model="gemini-2.0-flash",
    description="Orchestrates a full M&A due diligence investigation.",
    instruction=_COORDINATOR_INSTRUCTION,
    sub_agents=[pipeline],
    before_agent_callback=seed_default_state,
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = App(
    name="app",
    root_agent=root_agent,
    plugins=[AuditPlugin(), PIIRedactionPlugin(), FinancialGuardrailsPlugin()],
)
