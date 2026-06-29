# DealBreaker — M&A Due Diligence AI Agent

DealBreaker is a multi-agent AI system that performs comprehensive M&A due diligence
on a publicly traded acquisition target. It runs five specialist research workstreams
in parallel (financial, legal, market, news/sentiment, people & culture), synthesises
the findings into a ranked risk matrix, and generates a PDF report with a deal
recommendation.

**Stack:** Google ADK 2.3.0 · Gemini 2.0 Flash · Python 3.11+  
**Data sources:** SEC EDGAR, CourtListener, USPTO PatentsView, GDELT, Reddit (all public, no paywalls)

---

## Architecture

```
dealbreaker_coordinator (root LlmAgent)
└── due_diligence_pipeline (SequentialAgent)
    ├── investigation_phase (ParallelAgent)
    │   ├── financial_analyst   → FinancialFindings
    │   ├── legal_reviewer      → LegalFindings  (+ Legal DB MCP server)
    │   ├── market_researcher   → MarketFindings
    │   ├── news_sentiment_analyst → NewsSentimentFindings
    │   └── people_culture_analyst → PeopleCultureFindings
    └── synthesis_phase (SequentialAgent)
        ├── risk_assessor       → RiskAssessment
        └── report_generator    → FinalReport + PDF
```

Three security plugins run on every agent turn:
- **AuditPlugin** — structured JSON audit trail via structlog (logs metadata only)
- **PIIRedactionPlugin** — scrubs names/emails/phones from model text with presidio
- **FinancialGuardrailsPlugin** — rejects financial figures without a public source citation

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.11+ | 3.12–3.14 tested |
| `pip` or `uv` | For dependency installation |
| Google AI Studio API key | [aistudio.google.com](https://aistudio.google.com) — free tier works for testing (quota limits apply) |
| CourtListener API key | Free at [courtlistener.com/sign-in](https://www.courtlistener.com/sign-in/) |

---

## Installation

### 1. Clone and enter the project

```bash
git clone <repo-url>
cd dealbreaker-ai-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install main dependencies

```bash
pip install -e .
```

This installs the `app` package and all runtime dependencies from `pyproject.toml`.

### 4. Install spaCy model (required by PIIRedactionPlugin)

```bash
python -m spacy download en_core_web_lg
```

### 5. Install MCP server dependencies

The two MCP servers each have their own lightweight requirements:

```bash
pip install -r mcp_servers/sec_edgar_server/requirements.txt
pip install -r mcp_servers/legal_db_server/requirements.txt
```

---

## Configuration

### 1. Create `app/.env`

Copy the example and fill in your keys:

```bash
cp .env.example app/.env
```

Edit `app/.env`:

```env
# Google AI Studio API key — aistudio.google.com
GOOGLE_API_KEY=your_key_here

# Leave False for AI Studio; set True only for Vertex AI
GOOGLE_GENAI_USE_VERTEXAI=False

# CourtListener — free token at courtlistener.com/sign-in
COURTLISTENER_API_KEY=your_token_here

# Your email — included in SEC EDGAR User-Agent as required by SEC Fair Access policy
CONTACT_EMAIL=your_email@example.com
```

> **Important:** `app/.env` is git-ignored. Never commit real credentials.

### 2. Enable billing (for production runs)

The free tier of Google AI Studio has low quota limits. For a full pipeline run
(5 parallel agents + 2 synthesis agents), enable billing at
[aistudio.google.com](https://aistudio.google.com) or switch to a Vertex AI project
by setting `GOOGLE_GENAI_USE_VERTEXAI=True` and configuring `GOOGLE_CLOUD_PROJECT`.

---

## Running the Pipeline

### Start the MCP servers (background processes)

The Legal reviewer agent connects to two stdio MCP servers. Start them before
invoking `adk run`:

**Terminal 1:**
```bash
python mcp_servers/sec_edgar_server/server.py
```

**Terminal 2:**
```bash
python mcp_servers/legal_db_server/server.py
```

> The servers use stdio transport and will block waiting for MCP protocol input.
> ADK spawns its own subprocess connections; these manual instances are for
> verification only. In production, ADK manages server lifecycle automatically.

### Prepare the initial session state

`adk run` requires deal parameters to be in session state before the pipeline
starts. Create a `replay.json` file:

```json
{
  "state": {
    "target_company": "Salesforce",
    "industry": "SaaS",
    "deal_value": 50000000,
    "deal_type": "acquisition",
    "deal_id": "deal_test_001",
    "deal_jurisdictions": ["US"],
    "scope": "Full acquisition due diligence — financial, legal, market, news/sentiment, and people/culture workstreams",
    "acquirer_description": "Strategic acquirer in the enterprise software industry seeking to expand SaaS capabilities"
  },
  "queries": [
    "Analyze Salesforce as an acquisition target. Industry is SaaS. Deal value is 50000000 USD. Deal ID is deal_test_001. Jurisdictions are US."
  ]
}
```

### Run the agent

```bash
adk run app/ --replay replay.json --in_memory
```

Or use `--state` with an inline JSON string (Linux/macOS):

```bash
adk run app/ \
  --state '{"target_company":"Salesforce","industry":"SaaS","deal_value":50000000,"deal_type":"acquisition","deal_id":"deal_001","deal_jurisdictions":["US"],"scope":"Full acquisition due diligence","acquirer_description":"Strategic SaaS acquirer"}' \
  --in_memory \
  "Analyze Salesforce as an acquisition target."
```

### Output

- **Console** — structured JSON audit logs from AuditPlugin, then the coordinator's final report
- **`./reports/{deal_id}/risk_matrix.png`** — 5×5 risk heatmap
- **`./reports/{deal_id}/due_diligence_report.pdf`** — full PDF report (requires analyst confirmation)

---

## Project Structure

```
dealbreaker-ai-agent/
├── app/
│   ├── agent.py              # Root agent, pipeline, App object, plugins
│   ├── agents/
│   │   ├── financial.py      # FinancialAnalystAgent
│   │   ├── legal.py          # LegalReviewerAgent (+ MCP)
│   │   ├── market.py         # MarketResearcherAgent
│   │   ├── news_sentiment.py # NewsSentimentAnalystAgent
│   │   ├── people_culture.py # PeopleCultureAnalystAgent
│   │   ├── risk.py           # RiskAssessorAgent
│   │   ├── reporter.py       # ReportGeneratorAgent
│   │   └── investigation.py  # ParallelAgent wrapper (5 specialists)
│   ├── tools/
│   │   ├── financial_tools.py     # DCF, ratios, Beneish M-Score
│   │   ├── legal_tools.py         # CourtListener, EDGAR, patents
│   │   ├── market_tools.py        # TAM search, competitive landscape
│   │   ├── news_sentiment_tools.py# GDELT, Reddit, ESG
│   │   ├── people_culture_tools.py# Glassdoor proxy, exec background check
│   │   └── report_tools.py        # matplotlib chart, PDF rendering
│   ├── plugins/
│   │   ├── audit_plugin.py        # structlog JSON audit trail
│   │   ├── pii_plugin.py          # presidio PII redaction
│   │   └── guardrails_plugin.py   # financial citation guardrail
│   └── schemas/
│       └── models.py              # Pydantic output schemas
├── mcp_servers/
│   ├── sec_edgar_server/
│   │   └── server.py   # FastMCP: search_company_filings, get_filing_document,
│   │                   #          search_enforcement_actions
│   └── legal_db_server/
│       └── server.py   # FastMCP: search_federal_court_records,
│                       #          check_ofac_sanctions, search_state_ucc_filings
├── .env.example
├── pyproject.toml
└── AGENTS.md           # Full system specification (all 12 phases)
```

---

## Development

### Run all syntax checks

```bash
python -m compileall app/ -q
python -m compileall mcp_servers/ -q
```

### Lint and type-check

```bash
pip install -e ".[dev]"
ruff check app/
pyright app/
```

### Run tests

```bash
pytest tests/ -v
```

---

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Google AI Studio or Vertex AI key |
| `GOOGLE_GENAI_USE_VERTEXAI` | No | `True` to use Vertex AI instead of AI Studio |
| `GOOGLE_CLOUD_PROJECT` | If Vertex | GCP project ID |
| `COURTLISTENER_API_KEY` | Yes* | CourtListener PACER search token |
| `CONTACT_EMAIL` | Recommended | Included in EDGAR User-Agent (SEC requirement) |

\* Without `COURTLISTENER_API_KEY`, the legal agent still runs but federal court
case search returns an error; all other legal tools (OFAC, UCC, IP) continue to work.

---

## Known Limitations

- **Windows PDF rendering** — `weasyprint` requires GTK/Pango native libraries
  (not available on Windows without extra setup). The pipeline automatically falls
  back to `reportlab` for PDF generation.
- **Quota** — On the Google AI Studio free tier, a full 7-agent pipeline run may
  exhaust the daily `gemini-2.0-flash` quota. Enable billing for uninterrupted runs.
- **Private companies** — EDGAR-based tools return no data for companies without
  public SEC filings. The agents handle this gracefully and note the gap.
- **Glassdoor/LinkedIn scores** — No public unauthenticated API exists. The people
  & culture agent returns platform URLs for manual review via `load_web_page`.
