from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Finding(BaseModel):
    category: str
    description: str
    evidence: str  # "Source: <URL>, Published: <date>"
    risk_level: RiskLevel
    is_dealbreaker: bool
    recommended_action: Optional[str] = None


# ---------------------------------------------------------------------------
# Specialist agent output schemas
# ---------------------------------------------------------------------------

class FinancialFindings(BaseModel):
    revenue_cagr_3yr: float = Field(description="3-year revenue compound annual growth rate as a decimal (e.g. 0.15 = 15%)")
    ebitda_margin: float = Field(description="EBITDA as a fraction of revenue (e.g. 0.22 = 22%)")
    net_debt_to_ebitda: float = Field(description="Net debt divided by EBITDA; negative means net cash position")
    dcf_value_bear: float = Field(description="Bear-case DCF enterprise value in USD")
    dcf_value_base: float = Field(description="Base-case DCF enterprise value in USD")
    dcf_value_bull: float = Field(description="Bull-case DCF enterprise value in USD")
    anomalies_detected: list[str] = Field(description="List of accounting anomaly signals identified")
    overall_score: int = Field(ge=0, le=100, description="Financial health score: 0 = critical risk, 100 = no concerns")
    findings: list[Finding]
    dealbreakers: list[str] = Field(description="Plain-language dealbreaker descriptions; empty list if none")


class LegalFindings(BaseModel):
    corporate_structure_summary: str = Field(description="Narrative summary of subsidiaries, jurisdictions, and ownership chain")
    material_contracts_count: int = Field(ge=0, description="Number of material contracts identified in public filings")
    change_of_control_clauses: list[str] = Field(description="Contracts with change-of-control provisions that may trigger on acquisition")
    active_litigation: list[str] = Field(description="Active or recent legal proceedings with case identifiers and status")
    ip_issues: list[str] = Field(description="Patent, trademark, or open-source IP concerns identified")
    regulatory_gaps: list[str] = Field(description="Missing or lapsed licenses and compliance gaps")
    overall_score: int = Field(ge=0, le=100, description="Legal risk score: 0 = critical risk, 100 = no concerns")
    findings: list[Finding]
    dealbreakers: list[str] = Field(description="Plain-language dealbreaker descriptions; empty list if none")


class MarketFindings(BaseModel):
    tam_usd_millions: float = Field(ge=0, description="Total addressable market in USD millions")
    market_share_pct: float = Field(ge=0, le=100, description="Target company's estimated market share as a percentage")
    top_competitors: list[str] = Field(description="Ordered list of top competitors by market presence")
    customer_concentration_top3_pct: float = Field(ge=0, le=100, description="Revenue percentage attributable to top 3 customers")
    market_growth_rate_5yr: float = Field(description="Expected market CAGR over 5 years as a decimal (e.g. 0.12 = 12%)")
    overall_score: int = Field(ge=0, le=100, description="Market position score: 0 = critical risk, 100 = no concerns")
    findings: list[Finding]
    dealbreakers: list[str] = Field(description="Plain-language dealbreaker descriptions; empty list if none")


class NewsSentimentFindings(BaseModel):
    overall_sentiment: Literal["positive", "neutral", "negative", "mixed"]
    sentiment_score: float = Field(ge=-1.0, le=1.0, description="Aggregate sentiment score: -1.0 very negative, 0 neutral, 1.0 very positive")
    key_news_themes: list[str] = Field(description="Top recurring themes in media coverage over the past 24 months")
    esg_red_flags: list[str] = Field(description="ESG controversies or low scores from publicly available sources")
    regulatory_press: list[str] = Field(description="Reported fines, consent orders, or enforcement actions in the press")
    executive_reputation_issues: list[str] = Field(description="Public controversies, misconduct allegations, or prior failures by named executives")
    overall_score: int = Field(ge=0, le=100, description="News/sentiment score: 0 = critical risk, 100 = no concerns")
    findings: list[Finding]
    dealbreakers: list[str] = Field(description="Plain-language dealbreaker descriptions; empty list if none")


class PeopleCultureFindings(BaseModel):
    glassdoor_overall_score: float = Field(ge=0.0, le=5.0, description="Glassdoor overall company rating out of 5.0; use 0.0 if not found")
    glassdoor_ceo_approval_pct: float = Field(ge=0.0, le=100.0, description="Glassdoor CEO approval percentage; use 0.0 if not found")
    glassdoor_recommend_pct: float = Field(ge=0.0, le=100.0, description="Percentage of Glassdoor reviewers who recommend the company to a friend")
    key_executives: list[str] = Field(description="Named C-suite and key VP roles with incumbents identified from public sources")
    leadership_red_flags: list[str] = Field(description="Public controversies, legal issues, or track-record concerns for named executives")
    key_person_dependencies: list[str] = Field(description="Individuals publicly identified as critical to business continuity")
    recurring_culture_complaints: list[str] = Field(description="Top employee grievances from Glassdoor, Indeed, and Blind public reviews")
    culture_integration_risk_summary: str = Field(description="Narrative assessment of culture-fit risk between target and acquirer")
    overall_score: int = Field(ge=0, le=100, description="People & culture score: 0 = critical risk, 100 = no concerns")
    findings: list[Finding]
    dealbreakers: list[str] = Field(description="Plain-language dealbreaker descriptions; empty list if none")


# ---------------------------------------------------------------------------
# Synthesis agent output schemas
# ---------------------------------------------------------------------------

class RiskMatrixItem(BaseModel):
    risk_name: str
    likelihood: int = Field(ge=1, le=5, description="Probability of the risk materialising: 1 = very unlikely, 5 = near-certain")
    impact: int = Field(ge=1, le=5, description="Severity if the risk materialises: 1 = negligible, 5 = deal-threatening")
    risk_score: int = Field(ge=1, le=25, description="likelihood × impact")
    workstream: Literal["financial", "legal", "market", "news_sentiment", "people_culture"]
    mitigation: str = Field(description="Recommended mitigation action or deal term adjustment")


class RiskAssessment(BaseModel):
    risk_matrix: list[RiskMatrixItem]
    top_10_risks: list[str] = Field(description="Ordered list of the ten highest-priority risks across all workstreams")
    all_dealbreakers: list[str] = Field(description="Consolidated dealbreakers from all workstreams; empty list if none")
    price_adjustment_recommendation_usd: float = Field(description="Recommended reduction to purchase price in USD to account for identified risks; 0 if none")
    recommended_escrow_pct: float = Field(ge=0.0, le=100.0, description="Percentage of deal value recommended to be held in escrow")
    deal_recommendation: Literal["PROCEED", "PROCEED_WITH_CONDITIONS", "DO_NOT_PROCEED"]
    rationale: str = Field(description="2-3 sentence explanation of the deal recommendation")


class FinalReport(BaseModel):
    analysis_date: datetime = Field(description="UTC timestamp when the due diligence analysis was completed")
    executive_summary: str = Field(description="2-page maximum narrative: deal overview, top 5 risks, and recommendation")
    deal_recommendation: Literal["PROCEED", "PROCEED_WITH_CONDITIONS", "DO_NOT_PROCEED"]
    report_local_path: str = Field(description="Absolute path to the generated PDF report on the local filesystem")
    risk_matrix_chart_local_path: str = Field(description="Absolute path to the risk matrix PNG chart on the local filesystem")
    overall_score: int = Field(ge=0, le=100, description="Aggregate due diligence score: 0 = do not proceed, 100 = proceed with confidence")
    total_findings: int = Field(ge=0, description="Total number of Finding objects across all five workstreams")
    dealbreaker_count: int = Field(ge=0, description="Number of dealbreaker items across all workstreams")
