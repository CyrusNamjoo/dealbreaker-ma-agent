# DealBreaker — M&A Due Diligence AI Agent System

**Stack:** Google ADK (Python) · Gemini 2.0 Flash / Pro · AI Studio · InMemory Services · MCP Servers

---

## 1. Project Structure

```
dealbreaker-ai-agent/
├── app/                            # ADK app directory (name must match App(name="app"))
│   ├── __init__.py
│   ├── agent.py                    # Root coordinator agent (entry point)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── financial.py            # FinancialAnalystAgent
│   │   ├── legal.py                # LegalReviewAgent
│   │   ├── market.py               # MarketResearchAgent
│   │   ├── news_sentiment.py       # NewsSentimentAgent
│   │   ├── people_culture.py       # PeopleCultureAgent
│   │   ├── risk.py                 # RiskAssessmentAgent
│   │   └── reporter.py             # ReportGeneratorAgent
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── financial_tools.py      # Ratio analysis, DCF, anomaly detection
│   │   ├── legal_tools.py          # Litigation search, corporate records, compliance
│   │   ├── market_tools.py         # Market data, competitor analysis
│   │   ├── news_sentiment_tools.py # News search, sentiment scoring, ESG
│   │   ├── people_culture_tools.py # Executive research, Glassdoor, culture fit
│   │   └── report_tools.py         # Report generation, risk matrix, local file save
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── models.py               # All Pydantic output schemas
│   ├── plugins/
│   │   ├── __init__.py
│   │   ├── audit_plugin.py         # Audit trail — every tool call logged
│   │   ├── pii_plugin.py           # PII scrub before logging
│   │   └── guardrails_plugin.py    # Block hallucinated financial figures
│   └── .env                        # Environment variables (never commit)
├── mcp_servers/
│   ├── sec_edgar_server/           # SEC EDGAR public filings MCP
│   │   ├── server.py
│   │   └── requirements.txt
│   └── legal_db_server/            # Court records / regulatory filings MCP
│       ├── server.py
│       └── requirements.txt
├── reports/                        # Local output directory for generated reports
├── tests/
│   ├── eval/
│   │   ├── eval_config.yaml        # Criteria and thresholds per agent
│   │   └── datasets/               # JSON eval datasets per specialist agent
│   ├── integration/                # End-to-end pipeline tests
│   └── unit/                       # Per-tool unit tests
├── pyproject.toml
└── AGENTS.md                       # This file
```

---

## 2. Agent Architecture

### 2.1 Orchestration Overview

```
DealBreakerCoordinator (LlmAgent)
│
├── [ParallelAgent] InvestigationPhase
│   ├── FinancialAnalystAgent     (mode="task", output_key="financial_findings")
│   ├── LegalReviewAgent          (mode="task", output_key="legal_findings")
│   ├── MarketResearchAgent       (mode="task", output_key="market_findings")
│   ├── NewsSentimentAgent        (mode="task", output_key="news_sentiment_findings")
│   └── PeopleCultureAgent        (mode="task", output_key="people_culture_findings")
│
├── [SequentialAgent] SynthesisPhase
│   ├── RiskAssessmentAgent       (reads all five output_keys from state)
│   └── ReportGeneratorAgent      (produces final PDF saved to ./reports/)
│
└── [LoopAgent] RefinementLoop    (optional HITL: analyst can request deeper dives)
    ├── AnalystReviewAgent        (presents findings, requests confirmation)
    └── EscalationChecker         (escalate=True when analyst approves)
```

### 2.2 Agent Definitions

#### Root Coordinator
```python
# app/agent.py
from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.apps import App
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
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

root_agent = Agent(
    name="dealbreaker_coordinator",
    model="gemini-2.0-flash-exp",
    description="Orchestrates a full M&A due diligence investigation.",
    instruction="""
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
""",
    sub_agents=[pipeline],
)

app = App(
    name="app",
    root_agent=root_agent,
    plugins=[AuditPlugin(), PIIRedactionPlugin(), FinancialGuardrailsPlugin()],
)
```

#### Financial Analyst Agent
```python
# app/agents/financial.py
from google.adk.agents import Agent
from app.schemas.models import FinancialFindings
from app.tools.financial_tools import (
    calculate_financial_ratios,
    build_dcf_model,
    analyze_cash_flow_quality,
    detect_accounting_anomalies,
    compare_industry_benchmarks,
)
from google.adk.tools.load_web_page import load_web_page

def create_financial_agent() -> Agent:
    return Agent(
        name="financial_analyst",
        model="gemini-2.0-pro-exp",
        mode="task",
        description="Performs deep financial due diligence using public filings and market data.",
        output_schema=FinancialFindings,
        output_key="financial_findings",
        instruction="""
You are a senior M&A financial analyst. All data comes from public sources only.

Target: {target_company}, Industry: {industry}, Deal value: {deal_value}

Tasks (complete ALL before calling finish_task):
1. Retrieve 3 years of P&L, balance sheet, and cash flow data from SEC EDGAR filings
   (10-K, 10-Q) or equivalent public filings for non-US companies.
2. Calculate: revenue CAGR, EBITDA margins, working capital cycle, leverage ratios.
3. Build a DCF model with bear/base/bull scenarios using publicly disclosed financials.
4. Detect anomalies: channel stuffing signals, aggressive revenue recognition patterns,
   off-balance-sheet item disclosures in footnotes.
5. Compare all metrics to industry benchmarks via public data.
6. Flag any DEALBREAKER findings (material restatements, SEC enforcement actions,
   negative working capital trend, debt covenant breaches).

Always cite the public source URL, filing type, and line item for every finding.
""",
        tools=[
            load_web_page,
            calculate_financial_ratios,
            build_dcf_model,
            analyze_cash_flow_quality,
            detect_accounting_anomalies,
            compare_industry_benchmarks,
        ],
    )
```

#### Legal Review Agent
```python
# app/agents/legal.py
from google.adk.agents import Agent
from app.schemas.models import LegalFindings
from app.tools.legal_tools import (
    search_litigation_records,
    check_ip_ownership,
    verify_corporate_structure,
    screen_regulatory_compliance,
)
from google.adk.tools.load_web_page import load_web_page

def create_legal_agent() -> Agent:
    return Agent(
        name="legal_reviewer",
        model="gemini-2.0-pro-exp",
        mode="task",
        description="Reviews public legal records, regulatory filings, and compliance status.",
        output_schema=LegalFindings,
        output_key="legal_findings",
        instruction="""
You are a senior M&A attorney performing legal due diligence using public sources only.

Target: {target_company}, Jurisdictions: {deal_jurisdictions}

Tasks:
1. Map the corporate structure via public registry data: subsidiaries, ownership, jurisdictions.
2. Search for all public litigation (active, settled) from the past 5 years via court records.
3. Check IP ownership: public patent databases (USPTO, EPO), trademark registries.
4. Screen for regulatory compliance: check sector-specific licenses via public regulatory websites,
   GDPR/CCPA enforcement actions, export control violations.
5. Review publicly disclosed material contracts from SEC filings (Exhibit 10 attachments).
6. Flag DEALBREAKERS: unresolved material litigation, regulatory disqualification,
   OFAC sanctions, corrupt-practice findings (FCPA, UK Bribery Act enforcement).

Cite the public source URL and case/filing number for every finding.
""",
        tools=[
            load_web_page,
            search_litigation_records,
            check_ip_ownership,
            verify_corporate_structure,
            screen_regulatory_compliance,
        ],
    )
```

#### Market Research Agent
```python
# app/agents/market.py
from google.adk.agents import Agent
from app.schemas.models import MarketFindings
from app.tools.market_tools import (
    fetch_market_size_data,
    analyze_competitive_landscape,
    score_customer_concentration,
    evaluate_growth_drivers,
)
from google.adk.tools.load_web_page import load_web_page

def create_market_agent() -> Agent:
    return Agent(
        name="market_researcher",
        model="gemini-2.0-flash-exp",
        mode="task",
        description="Evaluates the target's market position, growth, and competitive dynamics.",
        output_schema=MarketFindings,
        output_key="market_findings",
        instruction="""
You are a strategy consultant performing commercial due diligence using public data only.

Target: {target_company}, Industry: {industry}

Tasks:
1. Size the TAM, SAM, and the target's estimated market share using public industry reports,
   analyst research, and government data.
2. Map the top 10 competitors: relative size, growth rate, pricing strategy, moat.
3. Evaluate customer concentration from public disclosures (SEC filings list major customers
   >10% of revenue). Flag if top 3 customers exceed 40% of revenue.
4. Assess growth drivers and headwinds for the next 5 years.
5. Verify the target's stated growth narrative against independent market data.
6. Flag DEALBREAKERS: structurally shrinking market, customer churn above industry norm,
   no defensible moat, key customer publicly signalling vendor switch.

Cite public source URL for every data point.
""",
        tools=[
            load_web_page,
            fetch_market_size_data,
            analyze_competitive_landscape,
            score_customer_concentration,
            evaluate_growth_drivers,
        ],
    )
```

#### News & Sentiment Agent
```python
# app/agents/news_sentiment.py
from google.adk.agents import Agent
from app.schemas.models import NewsSentimentFindings
from app.tools.news_sentiment_tools import (
    search_news_coverage,
    analyze_media_sentiment,
    check_esg_ratings,
    search_regulatory_press,
    scan_social_media_signals,
)
from google.adk.tools.load_web_page import load_web_page

def create_news_sentiment_agent() -> Agent:
    return Agent(
        name="news_sentiment_analyst",
        model="gemini-2.0-flash-exp",
        mode="task",
        description="Assesses the target's public reputation, media sentiment, and ESG standing.",
        output_schema=NewsSentimentFindings,
        output_key="news_sentiment_findings",
        instruction="""
You are a reputation and sentiment analyst. All research uses public sources only.

Target: {target_company}, Industry: {industry}

Tasks:
1. Search news coverage from the past 24 months across major outlets. Classify sentiment
   (positive / neutral / negative / mixed) and compute an aggregate sentiment score (-1 to 1).
2. Identify the top recurring themes in media coverage (product issues, growth milestones,
   leadership changes, controversy, regulatory actions).
3. Check ESG ratings and scores from public sources (sustainalytics summaries, MSCI snippets,
   B Corp status, published sustainability reports).
4. Search for regulatory press: fines, consent orders, enforcement actions, investigations
   reported in the press.
5. Assess executive reputation: search for CEO/C-suite controversies, criminal records,
   prior company failures, public misconduct allegations.
6. Scan public social media signals (LinkedIn, Twitter/X, Reddit) for brand sentiment,
   employee sentiment, and customer complaints at scale.
7. Flag DEALBREAKERS: active reputational crisis, executive criminal investigation,
   ESG controversy that would prevent investor participation, viral customer backlash.

Cite the URL and publication date for every significant finding.
""",
        tools=[
            load_web_page,
            search_news_coverage,
            analyze_media_sentiment,
            check_esg_ratings,
            search_regulatory_press,
            scan_social_media_signals,
        ],
    )
```

#### People & Culture Agent
```python
# app/agents/people_culture.py
from google.adk.agents import Agent
from app.schemas.models import PeopleCultureFindings
from app.tools.people_culture_tools import (
    fetch_glassdoor_data,
    research_executive_backgrounds,
    check_executive_controversies,
    assess_culture_fit,
    analyze_employee_review_themes,
)
from google.adk.tools.load_web_page import load_web_page

def create_people_culture_agent() -> Agent:
    return Agent(
        name="people_culture_analyst",
        model="gemini-2.0-flash-exp",
        mode="task",
        description="Evaluates leadership quality, workforce culture, and people integration risks.",
        output_schema=PeopleCultureFindings,
        output_key="people_culture_findings",
        instruction="""
You are a people and culture due diligence specialist. All data is from public sources only.

Target: {target_company}, Acquirer context: {acquirer_description}

Tasks:
1. Fetch Glassdoor data: overall rating (out of 5), CEO approval %, recommend-to-friend %,
   trend over the past 2 years. Flag if rating < 3.0 or declining sharply.
2. Research the leadership team (CEO, CFO, CTO, key SVPs): prior company track records,
   exit outcomes, public bios, LinkedIn tenure patterns.
3. Search for executive controversies: litigation, harassment allegations, prior
   company failures, regulatory sanctions against individuals.
4. Identify key-person dependencies from public sources: frequent mentions of specific
   individuals as critical to the business in press releases or analyst reports.
5. Assess culture fit risk: compare publicly stated values, work-from-home policy,
   diversity reports, and employee review themes between target and acquirer.
6. Analyze employee review themes on Glassdoor / Indeed / Blind for recurring issues
   (toxic management, high attrition, poor compensation, unclear strategy).
7. Flag DEALBREAKERS: CEO under active investigation, Glassdoor rating < 2.5,
   documented mass exodus of senior leadership, culture that is fundamentally
   incompatible with acquirer's integration model.

Cite public source URLs and review platform names for every finding.
""",
        tools=[
            load_web_page,
            fetch_glassdoor_data,
            research_executive_backgrounds,
            check_executive_controversies,
            assess_culture_fit,
            analyze_employee_review_themes,
        ],
    )
```

#### Risk Assessment Agent
```python
# app/agents/risk.py
from google.adk.agents import Agent
from app.schemas.models import RiskAssessment

def create_risk_agent() -> Agent:
    return Agent(
        name="risk_assessor",
        model="gemini-2.0-pro-exp",
        mode="task",
        description="Synthesizes all workstream findings into a structured risk matrix.",
        output_schema=RiskAssessment,
        output_key="risk_assessment",
        instruction="""
You are a senior M&A risk officer. Synthesize findings from all five workstreams.

Inputs from session state:
- Financial: {financial_findings}
- Legal: {legal_findings}
- Market: {market_findings}
- News & Sentiment: {news_sentiment_findings}
- People & Culture: {people_culture_findings}

Produce:
1. A risk matrix: each risk rated on Likelihood (1-5) x Impact (1-5).
2. A prioritized list of the top 10 risks across all workstreams.
3. A list of all DEALBREAKER findings across workstreams (empty list = no dealbreakers).
4. Recommended deal adjustments: price reductions, escrow holdbacks, reps & warranties,
   retention packages for key people.
5. An overall deal recommendation: PROCEED / PROCEED WITH CONDITIONS / DO NOT PROCEED.
""",
    )
```

#### Report Generator Agent
```python
# app/agents/reporter.py
from google.adk.agents import Agent
from app.schemas.models import FinalReport
from app.tools.report_tools import (
    generate_risk_matrix_chart,
    compile_executive_summary,
    render_pdf_report,
    save_report_locally,
)

def create_reporter_agent() -> Agent:
    return Agent(
        name="report_generator",
        model="gemini-2.0-flash-exp",
        mode="task",
        description="Produces the final due diligence report as a local PDF file.",
        output_schema=FinalReport,
        output_key="final_report",
        instruction="""
You are a professional M&A report writer.

Using:
- Risk assessment: {risk_assessment}
- All workstream findings (financial, legal, market, news_sentiment, people_culture in state)
- Deal metadata: target={target_company}, value={deal_value}

Produce:
1. An executive summary (max 2 pages): deal overview, top 5 risks, recommendation.
2. A detailed findings section per workstream.
3. A visual risk matrix (call generate_risk_matrix_chart).
4. Appendix: full source URL index for all citations.

Then render to PDF (call render_pdf_report) and save locally (call save_report_locally).
Return the local file path and a one-paragraph verdict for the coordinator.
""",
        tools=[
            generate_risk_matrix_chart,
            compile_executive_summary,
            render_pdf_report,
            save_report_locally,
        ],
    )
```

---

## 3. Pydantic Output Schemas

```python
# app/schemas/models.py
from pydantic import BaseModel, Field
from typing import Literal, Optional
from enum import Enum

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Finding(BaseModel):
    category: str
    description: str
    evidence: str           # "Source: <URL>, Published: <date>"
    risk_level: RiskLevel
    is_dealbreaker: bool
    recommended_action: Optional[str] = None

class FinancialFindings(BaseModel):
    revenue_cagr_3yr: float
    ebitda_margin: float
    net_debt_to_ebitda: float
    dcf_value_bear: float
    dcf_value_base: float
    dcf_value_bull: float
    anomalies_detected: list[str]
    findings: list[Finding]
    dealbreakers: list[str]

class LegalFindings(BaseModel):
    corporate_structure_summary: str
    material_contracts_count: int
    change_of_control_clauses: list[str]
    active_litigation: list[str]
    ip_issues: list[str]
    regulatory_gaps: list[str]
    findings: list[Finding]
    dealbreakers: list[str]

class MarketFindings(BaseModel):
    tam_usd_millions: float
    market_share_pct: float
    top_competitors: list[str]
    customer_concentration_top3_pct: float
    market_growth_rate_5yr: float
    findings: list[Finding]
    dealbreakers: list[str]

class NewsSentimentFindings(BaseModel):
    overall_sentiment: Literal["positive", "neutral", "negative", "mixed"]
    sentiment_score: float              # -1.0 (very negative) to 1.0 (very positive)
    key_news_themes: list[str]
    esg_red_flags: list[str]
    regulatory_press: list[str]
    executive_reputation_issues: list[str]
    findings: list[Finding]
    dealbreakers: list[str]

class PeopleCultureFindings(BaseModel):
    glassdoor_overall_score: float      # out of 5.0
    glassdoor_ceo_approval_pct: float
    glassdoor_recommend_pct: float
    key_executives: list[str]
    leadership_red_flags: list[str]
    key_person_dependencies: list[str]
    recurring_culture_complaints: list[str]
    culture_integration_risk_summary: str
    findings: list[Finding]
    dealbreakers: list[str]

class RiskMatrixItem(BaseModel):
    risk_name: str
    likelihood: int                     # 1-5
    impact: int                         # 1-5
    risk_score: int                     # likelihood * impact
    workstream: str
    mitigation: str

class RiskAssessment(BaseModel):
    risk_matrix: list[RiskMatrixItem]
    top_10_risks: list[str]
    all_dealbreakers: list[str]
    price_adjustment_recommendation_usd: float
    recommended_escrow_pct: float
    deal_recommendation: Literal["PROCEED", "PROCEED_WITH_CONDITIONS", "DO_NOT_PROCEED"]
    rationale: str

class FinalReport(BaseModel):
    executive_summary: str
    deal_recommendation: Literal["PROCEED", "PROCEED_WITH_CONDITIONS", "DO_NOT_PROCEED"]
    report_local_path: str
    risk_matrix_chart_local_path: str
    total_findings: int
    dealbreaker_count: int
```

---

## 4. Tools Specification

### 4.1 Financial Tools (`app/tools/financial_tools.py`)

All financial data sourced from public filings via SEC EDGAR MCP or `load_web_page`.

| Tool | Args | Returns | Description |
|------|------|---------|-------------|
| `calculate_financial_ratios` | `financials: dict` | `dict` | Compute 20+ ratios (liquidity, leverage, profitability, efficiency) |
| `build_dcf_model` | `fcf_history: list, wacc: float, terminal_growth: float` | `dict` | Bear/base/bull DCF with sensitivity table |
| `analyze_cash_flow_quality` | `cash_flow_statements: list` | `dict` | Detect non-recurring items, working capital manipulation signals |
| `detect_accounting_anomalies` | `financials_3yr: list` | `dict` | Beneish M-Score, channel stuffing indicators, accruals analysis |
| `compare_industry_benchmarks` | `metrics: dict, industry_code: str` | `dict` | Compare against industry medians from public data sources |

### 4.2 Legal Tools (`app/tools/legal_tools.py`)

All legal data sourced from public court records via legal DB MCP, SEC filings, and `load_web_page`.

| Tool | Args | Returns | Description |
|------|------|---------|-------------|
| `search_litigation_records` | `company_name: str, jurisdiction: str` | `dict` | Query CourtListener and PACER-equivalent public APIs |
| `check_ip_ownership` | `company_name: str` | `dict` | USPTO, EPO, trademark registries — public patent and IP search |
| `verify_corporate_structure` | `company_name: str, jurisdiction: str` | `dict` | Public corporate registry lookup |
| `screen_regulatory_compliance` | `business_description: str, jurisdictions: list` | `dict` | Map required licenses and flag gaps from public regulatory sources |

### 4.3 Market Tools (`app/tools/market_tools.py`)

| Tool | Args | Returns | Description |
|------|------|---------|-------------|
| `fetch_market_size_data` | `industry: str, geography: str` | `dict` | TAM/SAM from public industry reports and government data |
| `analyze_competitive_landscape` | `target_company: str, industry: str` | `dict` | Identify and profile top competitors from public sources |
| `score_customer_concentration` | `major_customers: list` | `dict` | Concentration analysis from SEC 10-K customer disclosures |
| `evaluate_growth_drivers` | `industry: str, company_description: str` | `dict` | PESTLE and Porter's Five Forces from public research |

### 4.4 News & Sentiment Tools (`app/tools/news_sentiment_tools.py`)

| Tool | Args | Returns | Description |
|------|------|---------|-------------|
| `search_news_coverage` | `company_name: str, days_back: int` | `dict` | Aggregate recent news from public news APIs and web search |
| `analyze_media_sentiment` | `articles: list` | `dict` | Score sentiment -1.0 to 1.0 with theme extraction |
| `check_esg_ratings` | `company_name: str` | `dict` | ESG scores from publicly accessible ratings summaries and sustainability reports |
| `search_regulatory_press` | `company_name: str` | `dict` | Search press releases from regulators (SEC, FTC, DOJ, EU) |
| `scan_social_media_signals` | `company_name: str` | `dict` | Aggregate public social media sentiment from LinkedIn, Reddit, Twitter/X |

### 4.5 People & Culture Tools (`app/tools/people_culture_tools.py`)

| Tool | Args | Returns | Description |
|------|------|---------|-------------|
| `fetch_glassdoor_data` | `company_name: str` | `dict` | Glassdoor overall score, CEO approval %, trend over 24 months |
| `research_executive_backgrounds` | `company_name: str` | `dict` | Public bios, LinkedIn tenure, prior company outcomes |
| `check_executive_controversies` | `executive_names: list` | `dict` | News search for individual exec misconduct, litigation, prior failures |
| `assess_culture_fit` | `target_company: str, acquirer_description: str` | `dict` | Compare values, remote policy, diversity reports, review themes |
| `analyze_employee_review_themes` | `company_name: str` | `dict` | Recurring themes from Glassdoor, Indeed, Blind public reviews |

### 4.6 Report Tools (`app/tools/report_tools.py`)

| Tool | Args | Returns | Description |
|------|------|---------|-------------|
| `generate_risk_matrix_chart` | `risk_matrix: list` | `dict` | Render a 5×5 risk heatmap as PNG, save to `./reports/{deal_id}/` |
| `compile_executive_summary` | `risk_assessment: dict, deal_metadata: dict` | `dict` | Structured 2-page executive summary text |
| `render_pdf_report` | `sections: dict, chart_paths: list` | `dict` | Assemble full PDF report (cover + sections + appendix) |
| `save_report_locally` | `report_bytes: bytes, deal_id: str, filename: str` | `dict` | Save to `./reports/{deal_id}/` and return absolute file path |

---

## 5. MCP Server Design

### 5.1 SEC EDGAR Server (`mcp_servers/sec_edgar_server/`)

Custom Python MCP server exposing:
- `search_company_filings(company_name, form_types)` — 10-K, 10-Q, 8-K, proxy statements
- `get_filing_document(accession_number)` — fetch a specific filing's text
- `search_enforcement_actions(company_name)` — SEC enforcement history

Uses the public SEC EDGAR REST API (no auth required, rate-limited to 10 req/s).

```python
# server.py skeleton
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("sec-edgar")

@mcp.tool()
def search_company_filings(company_name: str, form_types: list[str]) -> dict:
    """Search SEC EDGAR for filings by a company name."""
    ...
```

### 5.2 Legal Database Server (`mcp_servers/legal_db_server/`)

Custom Python MCP server exposing:
- `search_federal_court_records(company_name)` — PACER-style case search (via CourtListener API)
- `check_ofac_sanctions(entity_name)` — OFAC SDN list check (public API)
- `search_state_ucc_filings(company_name, state)` — UCC lien filings (public)

---

## 6. State Management

Session state flows down through agents. Use these keys consistently.
All state is held in `InMemorySessionService` — no external persistence.

```python
# app/agent.py — session service setup
from google.adk.sessions import InMemorySessionService
session_service = InMemorySessionService()

# Set by root_agent at start of session
state["target_company"]            = "Acme Corp"
state["deal_type"]                 = "acquisition"    # acquisition | merger | minority_stake
state["deal_value"]                = 50_000_000       # USD
state["industry"]                  = "SaaS"
state["deal_id"]                   = "deal_2024_acme"
state["deal_jurisdictions"]        = ["US", "UK"]
state["acquirer_description"]      = "Large enterprise SaaS acquirer, remote-first culture"
state["scope"]                     = ["financial", "legal", "market",
                                      "news_sentiment", "people_culture"]

# Written by specialist agents (via output_key)
state["financial_findings"]        = FinancialFindings(...)
state["legal_findings"]            = LegalFindings(...)
state["market_findings"]           = MarketFindings(...)
state["news_sentiment_findings"]   = NewsSentimentFindings(...)
state["people_culture_findings"]   = PeopleCultureFindings(...)

# Written by synthesis agents
state["risk_assessment"]           = RiskAssessment(...)
state["final_report"]              = FinalReport(...)
```

---

## 7. Artifact Service

Reports and charts are saved to the local filesystem and tracked as in-memory artifacts
within the session. No cloud storage dependency.

```python
# app/agent.py
from google.adk.artifacts import InMemoryArtifactService
from google.genai import types

artifact_service = InMemoryArtifactService()

# In report tools — save chart/PDF to disk, then register as artifact for session tracking:
async def save_report_locally(
    report_bytes: bytes, deal_id: str, filename: str, tool_context: ToolContext
) -> dict:
    """Save report PDF to ./reports/{deal_id}/ and register as session artifact."""
    path = Path(f"./reports/{deal_id}/{filename}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(report_bytes)

    part = types.Part(inline_data=types.Blob(mime_type="application/pdf", data=report_bytes))
    await tool_context.save_artifact(f"reports/{deal_id}/{filename}", part)

    return {"status": "success", "local_path": str(path.resolve())}
```

**Local output directory:** `./reports/{deal_id}/`
- `final_report.pdf` — full due diligence report
- `risk_matrix.png` — 5×5 risk heatmap chart
- `executive_summary.md` — plain-text summary for quick review

---

## 8. Security Requirements

### 8.1 Deal Isolation
- Each deal runs in a separate ADK session with a unique `deal_id`.
- No cross-deal data sharing — each session is independent in `InMemorySessionService`.
- `deal_id` is always generated server-side (UUID), never accepted from user input.
- Public data only: agents are instructed and guardrailed to use public sources. No
  confidential document upload pathway exists in this system.

### 8.2 Audit Plugin
```python
# app/plugins/audit_plugin.py
from google.adk.plugins.base_plugin import BasePlugin
import structlog

logger = structlog.get_logger()

class AuditPlugin(BasePlugin):
    async def before_tool_callback(self, *, tool, args, tool_context):
        logger.info("tool_call",
            tool=tool.name,
            deal_id=tool_context.state.get("deal_id"),
            user_id=tool_context.state.get("user:analyst_id"),
            args_keys=list(args.keys()),  # log keys, not values (may contain PII)
        )
        return None

    async def after_tool_callback(self, *, tool, args, tool_context, tool_response):
        logger.info("tool_result",
            tool=tool.name,
            deal_id=tool_context.state.get("deal_id"),
            status=tool_response.get("status"),
        )
        return None
```

### 8.3 PII Redaction Plugin
```python
# app/plugins/pii_plugin.py
# Scrub names, email addresses, phone numbers from log payloads before they leave the agent.
# Uses presidio-analyzer on all model outputs before structured logging.
```

### 8.4 Financial Guardrails Plugin
```python
# app/plugins/guardrails_plugin.py
# Intercepts model output (after_model_callback).
# Rejects any response that cites a financial figure without a public source URL citation.
# Prevents hallucinated numbers from entering the risk assessment.
```

### 8.5 Tool Confirmation for High-Stakes Actions
```python
from google.adk.tools import FunctionTool

# Require explicit analyst approval before writing the final report to disk
save_report_locally_tool = FunctionTool(
    save_report_locally,
    require_confirmation=True,
)
```

### 8.6 Secrets Management
- All API keys in `.env` (never committed to source control).
- `.env` template:
```
GOOGLE_API_KEY=...              # AI Studio key for Gemini (dev)
GOOGLE_GENAI_USE_VERTEXAI=False
COURTLISTENER_API_KEY=...
```

---

## 9. Step-by-Step Build Order

Work through these phases sequentially. Do not skip to a later phase before the prior one
runs cleanly in `adk web` or `adk run`.

### Phase 0 — Scaffold
```bash
agents-cli scaffold create dealbreaker-ai-agent
cd dealbreaker-ai-agent
agents-cli info   # confirm project config
```

### Phase 1 — Schemas and Core Infrastructure
1. Write all Pydantic models in `app/schemas/models.py`.
2. Write stub implementations for all tools (return `{"status": "stub", "data": {}}` for now).
3. Wire up `InMemoryArtifactService` and `InMemorySessionService` in `app/agent.py`.
4. Create the `./reports/` output directory.
5. Verify `adk web app/` starts without import errors.

### Phase 2 — MCP Servers
1. Build `mcp_servers/sec_edgar_server/` using `FastMCP` and the public EDGAR REST API.
2. Build `mcp_servers/legal_db_server/` (CourtListener + OFAC public APIs).
3. Wire `McpToolset` into `financial_analyst` and `legal_reviewer` agents.
4. Test: query a real company name (e.g., "Salesforce") and verify filings are returned.

### Phase 3 — Financial Analyst Agent (Full)
1. Implement all financial tools with real logic using SEC EDGAR data and `load_web_page`.
2. Test in isolation: `adk run app/ --agent financial_analyst`.
3. Write 5 eval cases in `tests/eval/datasets/financial_eval.json` using public companies.
4. Run `agents-cli eval app/ tests/eval/datasets/financial_eval.json`.

### Phase 4 — Legal Review Agent (Full)
1. Implement all legal tools using CourtListener, USPTO, and `load_web_page`.
2. Test in isolation, write 5 eval cases, run eval.

### Phase 5 — Market Research Agent (Full)
1. Implement market tools using `load_web_page` and public market reports.
2. Test in isolation, write 5 eval cases, run eval.

### Phase 6 — News & Sentiment and People & Culture Agents
1. Implement `news_sentiment_tools.py` — integrate a public news API (e.g., NewsAPI free tier)
   and web scraping of public social signals.
2. Implement `people_culture_tools.py` — scrape public Glassdoor pages, LinkedIn bios,
   and employee review platforms.
3. Write a specific eval case for each: one with a positive target, one with a reputational crisis,
   one with a toxic culture Glassdoor profile.
4. Test in isolation, write eval cases, run eval.

### Phase 7 — Parallel Investigation Phase
1. Assemble `ParallelAgent("investigation_phase", sub_agents=[...])` with all five agents.
2. Run a full parallel session against a real public company (e.g., a mid-cap with SEC filings).
3. Verify all five `output_key`s land in session state without race conditions.

### Phase 8 — Risk Assessment + Report Generator
1. Implement `RiskAssessmentAgent` — verify it reads all five output keys from state correctly.
2. Implement `ReportGeneratorAgent` with local PDF rendering (`weasyprint` or `reportlab`).
3. Implement `generate_risk_matrix_chart` with `matplotlib` saving to `./reports/{deal_id}/`.
4. Implement `save_report_locally` with `require_confirmation=True`.
5. Wire the `SequentialAgent("synthesis_phase", ...)`.

### Phase 9 — Security Plugins
1. Implement `AuditPlugin` with structured logging via `structlog`.
2. Implement `PIIRedactionPlugin` using `presidio-analyzer`.
3. Implement `FinancialGuardrailsPlugin` requiring source URLs on all financial figures.
4. Register all three in `App(plugins=[...])`.
5. Test that audit logs appear and PII is scrubbed from log output.

### Phase 10 — Root Coordinator + Full Pipeline
1. Wire `SequentialAgent` wrapping `investigation_phase` → `synthesis_phase`.
2. Build the root `DealBreakerCoordinator` agent with full instruction template.
3. Add the optional `LoopAgent` HITL refinement layer.
4. Run an end-to-end test against a real public company and verify PDF output in `./reports/`.

### Phase 11 — Evaluation Suite
1. Write integration eval datasets: `tests/eval/datasets/full_pipeline_eval.json` (3+ cases).
2. Each case: a public company name + expected `deal_recommendation` and `dealbreaker_count`.
3. Write `tests/eval/eval_config.yaml` with criteria: `trajectory_match`, `safety`,
   custom `dealbreaker_recall`, `source_citation_rate`.
4. Run: `agents-cli eval app/ tests/eval/`.

### Phase 12 — Local Dev Hardening
1. Add rate-limiting to all web tools (respect `robots.txt`, stay within fair-use limits).
2. Add caching layer for web requests within a session (avoid re-fetching the same URL).
3. Add timeout handling for all external HTTP calls.
4. Document the local run procedure in `README.md`.

---

## 10. Dependencies

```toml
# pyproject.toml
[project]
name = "dealbreaker-ai-agent"
requires-python = ">=3.11"
dependencies = [
    "google-adk[a2a]>=1.0",
    "matplotlib>=3.9",
    "weasyprint>=62",               # PDF rendering (primary)
    "reportlab>=4.2",               # PDF rendering (fallback)
    "pydantic>=2.0",
    "structlog>=24.0",
    "mcp>=1.0",
    "httpx>=0.27",                  # async HTTP for EDGAR / CourtListener / news APIs
    "presidio-analyzer>=2.2",       # PII detection in guardrails plugin
    "presidio-anonymizer>=2.2",
    "textblob>=0.18",               # lightweight sentiment baseline
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.23", "ruff", "pyright"]
```

---

## 11. Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Orchestration pattern | `ParallelAgent` → `SequentialAgent` | Five workstreams are independent; synthesis requires all outputs |
| Specialist agent mode | `mode="task"` with `output_schema` | Typed, structured outputs prevent hallucinated data entering the risk matrix |
| Model assignment | Pro for financial/legal/risk, Flash for market/news/people/reporter | Pro for complex analytical reasoning; Flash for speed and cost on web-search-heavy tasks |
| State transport | `output_key` → session state | Native ADK pattern; avoids custom message-passing plumbing |
| Session persistence | `InMemorySessionService` | No cloud dependency for local dev; swap to `VertexAiSessionService` for production |
| Artifact storage | `InMemoryArtifactService` + local disk | Reports written to `./reports/` for immediate access; no GCS dependency |
| Data sources | Public web + SEC EDGAR MCP + legal DB MCP | Public-data-only constraint; no confidential document upload pathway |
| PDF report rendering | `weasyprint` (primary) + `reportlab` (fallback) | `weasyprint` produces higher-quality output from HTML/CSS templates |
| Security guardrails | Plugin layer (not per-agent callbacks) | Cross-cutting; applies to every agent without per-agent repetition |
| HITL integration | `LoopAgent` + `EscalationChecker` | Analyst can request deeper investigation without restarting the pipeline |
| Source citation enforcement | `FinancialGuardrailsPlugin` | Blocks any financial figure without a public URL — prevents hallucinated numbers |
