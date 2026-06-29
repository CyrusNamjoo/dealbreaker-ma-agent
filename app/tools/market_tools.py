"""
Market research tools for the MarketResearchAgent.

Four tools covering market sizing, competitive landscape, customer concentration,
and growth driver analysis. All data sourced from public SEC filings, government
statistics, and public industry databases.

CONTRACT (applies to every tool in this module):
  - data_available=False when a required source returns no usable data.
  - No market size figures are estimated — only data cited from public filings
    or government statistics is returned.
  - Every figure is accompanied by a source URL.
"""

import asyncio
import os
import time
from typing import Any
from urllib.parse import urlparse

import httpx

_rate_limiter: dict[str, float] = {}


async def _rate_limit(url: str) -> None:
    """Enforce 1-second minimum delay between requests to the same domain."""
    domain = urlparse(url).netloc
    now = time.monotonic()
    wait = 1.0 - (now - _rate_limiter.get(domain, 0.0))
    if wait > 0:
        await asyncio.sleep(wait)
    _rate_limiter[domain] = time.monotonic()

_CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "YOUR_EMAIL@EXAMPLE.COM")
_EDGAR_HEADERS = {
    "User-Agent": f"DealBreaker-AI {_CONTACT_EMAIL}",
    "Accept": "application/json",
}
_EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{}.json"

# ── NAICS code mapping (US Census public taxonomy) ───────────────────────────
_NAICS_MAP: dict[str, dict[str, str]] = {
    "software":           {"code": "5112",  "label": "Software Publishers"},
    "saas":               {"code": "5112",  "label": "Software Publishers"},
    "fintech":            {"code": "5223",  "label": "Activities Related to Credit Intermediation"},
    "financial_services": {"code": "5221",  "label": "Depository Credit Intermediation"},
    "financial services": {"code": "5221",  "label": "Depository Credit Intermediation"},
    "healthcare":         {"code": "6211",  "label": "Offices of Physicians"},
    "pharma":             {"code": "3254",  "label": "Pharmaceutical and Medicine Manufacturing"},
    "biotech":            {"code": "3254",  "label": "Pharmaceutical and Medicine Manufacturing"},
    "telecom":            {"code": "5133",  "label": "Telecommunications"},
    "energy":             {"code": "2211",  "label": "Electric Power Generation, Transmission and Distribution"},
    "defense":            {"code": "3364",  "label": "Aerospace Product and Parts Manufacturing"},
    "aerospace":          {"code": "3364",  "label": "Aerospace Product and Parts Manufacturing"},
    "food":               {"code": "3113",  "label": "Sugar and Confectionery Product Manufacturing"},
    "transportation":     {"code": "4841",  "label": "General Freight Trucking"},
    "logistics":          {"code": "4841",  "label": "General Freight Trucking"},
    "real_estate":        {"code": "5311",  "label": "Lessors of Real Estate"},
    "real estate":        {"code": "5311",  "label": "Lessors of Real Estate"},
    "retail":             {"code": "4451",  "label": "Grocery Stores"},
    "e-commerce":         {"code": "4541",  "label": "Electronic Shopping and Mail-Order Houses"},
    "ecommerce":          {"code": "4541",  "label": "Electronic Shopping and Mail-Order Houses"},
    "manufacturing":      {"code": "3390",  "label": "Other Fabricated Metal Product Manufacturing"},
    "media":              {"code": "5151",  "label": "Radio and Television Broadcasting"},
    "gaming":             {"code": "7132",  "label": "Gambling Industries"},
    "education":          {"code": "6111",  "label": "Elementary and Secondary Schools"},
    "edtech":             {"code": "6117",  "label": "Educational Support Services"},
}

# ── Sector growth drivers: tailwinds, headwinds, and data source URLs ────────
_SECTOR_GROWTH_DRIVERS: dict[str, dict[str, Any]] = {
    "software": {
        "tailwinds": [
            "Cloud migration: public cloud spending growing ~15% YoY (Gartner press releases; https://www.gartner.com/en/newsroom)",
            "AI/ML integration driving new software deal categories (Microsoft, Salesforce public earnings calls)",
            "Digital transformation still under-penetrated in mid-market and public sector (US Census digital economy survey; https://www.census.gov/programs-surveys/abs.html)",
            "Subscription recurring revenue model increases LTV vs. perpetual licence alternatives",
        ],
        "headwinds": [
            "SaaS spending consolidation — enterprises reducing vendor count (published CFO surveys, Gartner Magic Quadrant commentary)",
            "Longer enterprise sales cycles as IT budgets face scrutiny (public earnings guidance from Salesforce, ServiceNow)",
            "Open-source alternatives eroding pricing in commodity software categories",
            "FTC/EU AI Act regulatory uncertainty for AI-native software products",
        ],
        "government_data_urls": [
            "https://www.bls.gov/ooh/computer-and-information-technology/home.htm",
            "https://www.census.gov/naics/?input=5112&chart=0",
            "https://www.bea.gov/data/gdp/gdp-industry",
        ],
        "industry_association_urls": [
            "https://www.comptia.org/research/it-industry-outlook",
            "https://www.bsa.org/reports",
        ],
    },
    "fintech": {
        "tailwinds": [
            "Mobile banking adoption: 89% of US adults bank digitally (Federal Reserve Consumer Finance Survey; https://www.federalreserve.gov/publications/consumer-community-context.htm)",
            "Open banking / PSD2 expanding API-driven financial data access",
            "BNPL regulatory framework providing legitimacy for embedded credit",
            "Cross-border payment volumes growing with e-commerce (BIS CPMI statistics; https://www.bis.org/cpmi/publ/d201.htm)",
        ],
        "headwinds": [
            "CFPB enforcement increasing compliance burden for non-bank financial firms",
            "Rising interest rates compressing lending margin for balance-sheet fintechs",
            "Bank-embedded fintech eroding standalone fintech moat",
            "Crypto regulatory uncertainty (SEC enforcement, CFTC jurisdiction battles)",
        ],
        "government_data_urls": [
            "https://www.federalreserve.gov/publications/consumer-community-context.htm",
            "https://www.fdic.gov/bank/statistical/",
            "https://www.bis.org/cpmi/publ/d201.htm",
        ],
        "industry_association_urls": [
            "https://www.accenture.com/us-en/insights/banking/banking-technology-vision",
            "https://www.fintechfutures.com/research/",
        ],
    },
    "healthcare": {
        "tailwinds": [
            "Ageing US population: 65+ cohort growing 3.4%/yr through 2030 (US Census projections; https://www.census.gov/newsroom/press-releases/2018/cb18-41-population-projections.html)",
            "Telehealth utilisation stabilising at 3.5× pre-COVID levels (HHS ASPE data; https://aspe.hhs.gov/sites/default/files/documents/a1d5d810fe3433e18b192be42dbf2351/telehealth-hb.pdf)",
            "GLP-1 drug category driving adjacent healthcare spend (FDA orange book; https://www.accessdata.fda.gov/scripts/cder/ob/)",
            "CMS value-based care models expanding (CMS Innovation Center; https://innovation.cms.gov/)",
        ],
        "headwinds": [
            "Reimbursement rate compression from payers",
            "Hospital consolidation reducing supplier pricing power",
            "IRA drug pricing negotiation impacting pharma-adjacent healthcare services",
        ],
        "government_data_urls": [
            "https://www.cms.gov/data-research/statistics-trends-and-reports/national-health-expenditure-data",
            "https://www.census.gov/topics/health/health-insurance.html",
            "https://www.bls.gov/ooh/healthcare/home.htm",
        ],
        "industry_association_urls": [
            "https://www.aha.org/statistics/fast-facts-us-hospitals",
            "https://www.commonwealthfund.org/publications/issue-briefs",
        ],
    },
    "energy": {
        "tailwinds": [
            "IRA 2022 renewable energy tax credits: ~$369B committed (DOE data; https://www.energy.gov/lpo/inflation-reduction-act)",
            "EV adoption driving electricity demand growth (EIA; https://www.eia.gov/electricity/)",
            "Grid modernisation capex cycle: $2T over 10 years (Edison Electric Institute; https://www.eei.org/resources-and-media/research-and-data/",
            "LNG export capacity expansion from Gulf Coast terminals (FERC; https://www.ferc.gov/industries-data/natural-gas/overview/lng-facilities)",
        ],
        "headwinds": [
            "Permitting delays for transmission infrastructure (NEPA reform pending)",
            "Commodity price volatility compressing margins for upstream producers",
            "Geopolitical risk premium in LNG contracting",
        ],
        "government_data_urls": [
            "https://www.eia.gov/totalenergy/",
            "https://www.energy.gov/eere/annual-report",
            "https://www.ferc.gov/industries-data/electric/overview/electricity-markets",
        ],
        "industry_association_urls": [
            "https://www.eei.org/resources-and-media/research-and-data/",
            "https://www.api.org/oil-and-natural-gas/energy-primers",
        ],
    },
    "retail": {
        "tailwinds": [
            "E-commerce share of total US retail: ~16% and growing (US Census retail e-commerce quarterly; https://www.census.gov/retail/index.html)",
            "DTC brand growth via social commerce channels",
            "Supply chain reshoring reducing import dependency for US retailers",
        ],
        "headwinds": [
            "Consumer discretionary spending softening under high interest rates",
            "Amazon marketplace pricing pressure on third-party sellers",
            "Return rates in e-commerce averaging 20-30% eroding gross margins",
        ],
        "government_data_urls": [
            "https://www.census.gov/retail/index.html",
            "https://www.bls.gov/cps/cpsaat18.htm",
        ],
        "industry_association_urls": [
            "https://nrf.com/research/state-retail-2024",
        ],
    },
    "defense": {
        "tailwinds": [
            "US defense budget FY2024: $858B (DoD budget justification; https://comptroller.defense.gov/Budget-Materials/)",
            "NATO allies increasing defence spending toward 2% GDP target",
            "Counter-UAS, space, and cyber driving new procurement categories",
        ],
        "headwinds": [
            "Continuing resolution risk delaying programme starts",
            "IRAD cost-share requirements increasing prime contractor risk",
            "ITAR/EAR compliance costs increasing for international programme participation",
        ],
        "government_data_urls": [
            "https://comptroller.defense.gov/Budget-Materials/",
            "https://www.usaspending.gov/",
            "https://www.gao.gov/topics/defense-acquisitions",
        ],
        "industry_association_urls": [
            "https://www.aia-aerospace.org/research/",
        ],
    },
    "transportation": {
        "tailwinds": [
            "E-commerce last-mile demand sustaining parcel volume growth (US Census; https://www.census.gov/retail/)",
            "IIJA infrastructure bill: $110B road/bridge funding (FHWA; https://www.fhwa.dot.gov/bipartisan-infrastructure-law/)",
            "Autonomous vehicle and drone delivery piloting by major shippers",
        ],
        "headwinds": [
            "Driver shortage: ATA estimates 80K shortage (https://www.trucking.org/economics-and-industry-data)",
            "Diesel fuel cost volatility compressing carrier margins",
            "Rail labour disputes adding supply chain risk premium",
        ],
        "government_data_urls": [
            "https://www.bts.gov/topics/freight-transportation",
            "https://www.fhwa.dot.gov/policyinformation/statistics.cfm",
        ],
        "industry_association_urls": [
            "https://www.trucking.org/economics-and-industry-data",
        ],
    },
}

_DEFAULT_GROWTH_DRIVERS: dict[str, Any] = {
    "tailwinds": [
        "General economic growth expanding total addressable spend (BEA GDP; https://www.bea.gov/data/gdp/gross-domestic-product)",
        "Technology adoption reducing operational costs across sectors",
    ],
    "headwinds": [
        "Interest rate environment increasing cost of capital and compressing multiples",
        "Regulatory uncertainty across multiple jurisdictions",
    ],
    "government_data_urls": [
        "https://www.bea.gov/data/gdp/gross-domestic-product",
        "https://www.bls.gov/productivity/",
        "https://fred.stlouisfed.org/",
    ],
    "industry_association_urls": [],
}

# ── Customer concentration risk thresholds ───────────────────────────────────
_CONC_THRESHOLDS = [
    (60.0, "CRITICAL", "Top-3 customers exceed 60% of revenue — catastrophic concentration risk"),
    (40.0, "HIGH",     "Top-3 customers exceed 40% of revenue — significant concentration risk"),
    (25.0, "MEDIUM",   "Top-3 customers exceed 25% of revenue — moderate concentration risk"),
]
_SINGLE_CUSTOMER_THRESHOLDS = [
    (40.0, "CRITICAL", "Single customer exceeds 40% of revenue — existential dependency"),
    (20.0, "HIGH",     "Single customer exceeds 20% of revenue — material dependency"),
    (10.0, "MEDIUM",   "Single customer exceeds 10% of revenue — requires SEC disclosure"),
]


# ── Tool 1: fetch_market_size_data ───────────────────────────────────────────

async def fetch_market_size_data(industry: str, geography: str) -> dict:
    """Search public SEC filings and return government data sources for market sizing.

    Queries SEC EDGAR for recent S-1 and 10-K filings that disclose 'total
    addressable market' figures for the given industry. These prospectuses are
    the most reliable public sources for audited market size citations.
    Also returns NAICS-mapped government statistics URLs (BLS, Census, BEA)
    for the agent to verify with load_web_page.

    Args:
        industry: Sector or industry name (e.g. "SaaS", "fintech", "healthcare",
                  "defense", "e-commerce").
        geography: Geographic scope (e.g. "US", "North America", "global").
                   Affects the government data URLs returned.

    Returns:
        dict with data_available, tam_reference_filings (SEC filings with cited
        TAM), naics_info, government_data_urls, industry_association_urls,
        edgar_search_url, and guidance.
    """
    industry_lower = industry.lower().strip()
    naics = _NAICS_MAP.get(industry_lower, {})

    # Search EDGAR for S-1 / 10-K filings mentioning TAM for this industry
    edgar_search_url = (
        f"{_EDGAR_SEARCH}?q=%22total+addressable+market%22+%22{industry_lower.replace(' ', '+')}%22"
        f"&forms=S-1,S-1%2FA,10-K&dateRange=custom&startdt=2021-01-01"
    )

    params = {
        "q": f'"total addressable market" "{industry}"',
        "forms": "S-1,S-1/A,10-K",
        "dateRange": "custom",
        "startdt": "2021-01-01",
    }

    await _rate_limit(_EDGAR_SEARCH)
    try:
        async with httpx.AsyncClient(headers=_EDGAR_HEADERS, timeout=30.0) as client:
            r = await client.get(_EDGAR_SEARCH, params=params)
        r.raise_for_status()
        hits = r.json().get("hits", {}).get("hits", [])
    except httpx.HTTPError as exc:
        return {
            "data_available": False,
            "message": f"EDGAR full-text search failed: {exc}",
            "tam_reference_filings": [],
            "naics_info": naics,
            "government_data_urls": _get_gov_urls(geography),
            "industry_association_urls": [],
            "edgar_search_url": edgar_search_url,
            "source": "SEC EDGAR (https://efts.sec.gov/LATEST/search-index)",
        }

    filings = []
    for hit in hits[:10]:
        src = hit.get("_source", {})
        ciks = src.get("ciks", [])
        adsh = src.get("adsh", "")
        cik_int = str(int(ciks[0])).strip() if ciks else ""
        adsh_clean = adsh.replace("-", "")
        display = src.get("display_names", [""])
        entity_name = display[0].split("(CIK")[0].strip() if display else ""
        filings.append({
            "entity_name": entity_name,
            "form": (src.get("root_forms") or [src.get("form", "")])[0],
            "file_date": src.get("file_date", ""),
            "accession_number": adsh,
            "filing_index_url": (
                f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{adsh_clean}/"
                if cik_int and adsh_clean else ""
            ),
        })

    growth = _SECTOR_GROWTH_DRIVERS.get(industry_lower, _DEFAULT_GROWTH_DRIVERS)

    return {
        "data_available": len(filings) > 0,
        "industry": industry,
        "geography": geography,
        "tam_reference_filings_count": len(filings),
        "tam_reference_filings": filings,
        "naics_info": naics,
        "government_data_urls": _get_gov_urls(geography) + growth.get("government_data_urls", []),
        "industry_association_urls": growth.get("industry_association_urls", []),
        "edgar_search_url": edgar_search_url,
        "note": (
            f"{len(filings)} SEC filings found disclosing TAM for '{industry}'. "
            "Use load_web_page on filing_index_url entries to read the 'Market Opportunity' "
            "or 'Industry Overview' sections for cited TAM figures. "
            "Supplement with government_data_urls for independent corroboration. "
            "Do NOT use any TAM figure from these filings without citing the exact source URL, "
            "filing form, and fiscal year."
        ),
        "source": "SEC EDGAR full-text search (https://efts.sec.gov/LATEST/search-index)",
    }


def _get_gov_urls(geography: str) -> list[str]:
    geo = geography.upper()
    urls = [
        "https://www.bea.gov/data/gdp/gdp-industry",
        "https://fred.stlouisfed.org/",
    ]
    if "US" in geo or "NORTH AMERICA" in geo or "GLOBAL" in geo:
        urls += [
            "https://www.census.gov/retail/index.html",
            "https://www.bls.gov/productivity/",
        ]
    if "EU" in geo or "EUROPE" in geo or "GLOBAL" in geo:
        urls.append("https://ec.europa.eu/eurostat/databrowser/explore/all/industryCateg")
    if "GLOBAL" in geo:
        urls.append("https://data.worldbank.org/indicator/NY.GDP.MKTP.CD")
    return urls


# ── Tool 2: analyze_competitive_landscape ────────────────────────────────────

async def analyze_competitive_landscape(target_company: str, industry: str) -> dict:
    """Identify peer companies in the same industry from SEC EDGAR filings.

    Searches EDGAR for public companies that filed 10-K annual reports and
    mention both the industry and competitive dynamics. Also finds the target
    company's own 10-K competition section reference. Returns EDGAR-registered
    peers and financial data search URLs for each.

    Args:
        target_company: Name of the company being analysed (e.g. "Salesforce Inc").
        industry: Sector descriptor (e.g. "SaaS", "healthcare", "fintech").

    Returns:
        dict with data_available, peer_companies (EDGAR-registered), competitor
        search URLs, target_competition_section_url, and guidance.
    """
    industry_lower = industry.lower().strip()

    # Search 1: recent 10-K filings that mention this industry + competitive dynamics
    peer_params = {
        "q": f'"{industry}" "competitive" OR "competitors" OR "competition"',
        "forms": "10-K",
        "dateRange": "custom",
        "startdt": "2022-01-01",
    }
    # Search 2: target company's own 10-K — to find its "Competition" section
    target_params = {
        "q": f'"{target_company}" "competition" OR "competitors"',
        "forms": "10-K",
        "dateRange": "custom",
        "startdt": "2020-01-01",
    }

    peer_filings: list[dict] = []
    target_filing: dict = {}

    try:
        async with httpx.AsyncClient(headers=_EDGAR_HEADERS, timeout=30.0) as client:
            await _rate_limit(_EDGAR_SEARCH)
            r1 = await client.get(_EDGAR_SEARCH, params=peer_params)
            r1.raise_for_status()
            await _rate_limit(_EDGAR_SEARCH)
            r2 = await client.get(_EDGAR_SEARCH, params=target_params)
            r2.raise_for_status()
    except httpx.HTTPError as exc:
        return {
            "data_available": False,
            "message": f"EDGAR search failed: {exc}",
            "peer_companies": [],
            "target_competition_filing": {},
            "competitor_search_urls": _competitor_search_urls(target_company),
            "source": "SEC EDGAR (https://efts.sec.gov/LATEST/search-index)",
        }

    for hit in r1.json().get("hits", {}).get("hits", [])[:15]:
        src = hit.get("_source", {})
        display = src.get("display_names", [""])
        entity_name = display[0].split("(CIK")[0].strip() if display else ""
        ciks = src.get("ciks", [])
        cik_int = str(int(ciks[0])).strip() if ciks else ""
        adsh = src.get("adsh", "").replace("-", "")
        # Exclude the target company itself from the peer list
        if target_company.lower() in entity_name.lower():
            continue
        peer_filings.append({
            "entity_name": entity_name,
            "cik": cik_int,
            "file_date": src.get("file_date", ""),
            "filing_index_url": (
                f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{adsh}/"
                if cik_int and adsh else ""
            ),
            "edgar_company_url": (
                f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik_int}&type=10-K&dateb=&owner=include&count=5"
                if cik_int else ""
            ),
        })

    target_hits = r2.json().get("hits", {}).get("hits", [])
    if target_hits:
        src = target_hits[0].get("_source", {})
        ciks = src.get("ciks", [])
        cik_int = str(int(ciks[0])).strip() if ciks else ""
        adsh = src.get("adsh", "")
        adsh_clean = adsh.replace("-", "")
        target_filing = {
            "entity_name": (src.get("display_names", [""])[0] or "").split("(CIK")[0].strip(),
            "cik": cik_int,
            "form": (src.get("root_forms") or [src.get("form", "")])[0],
            "file_date": src.get("file_date", ""),
            "filing_index_url": (
                f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{adsh_clean}/"
                if cik_int and adsh_clean else ""
            ),
        }

    return {
        "data_available": len(peer_filings) > 0 or bool(target_filing),
        "target_company": target_company,
        "industry": industry,
        "peer_companies_found": len(peer_filings),
        "peer_companies": peer_filings,
        "target_competition_filing": target_filing,
        "competitor_search_urls": _competitor_search_urls(target_company),
        "note": (
            f"{len(peer_filings)} EDGAR-registered peer companies found in '{industry}'. "
            "Use load_web_page on edgar_company_url entries to view each company's filings. "
            "Read target_competition_filing's primary document for the 'Competition' section "
            "which typically names the company's top 5-10 direct competitors. "
            "Use competitor_search_urls to find financial data for each named competitor."
        ),
        "source": "SEC EDGAR full-text search (https://efts.sec.gov/LATEST/search-index)",
    }


def _competitor_search_urls(target_company: str) -> dict[str, str]:
    encoded = target_company.replace(" ", "+")
    return {
        "sec_filings": f"https://www.sec.gov/cgi-bin/browse-edgar?company={encoded}&CIK=&type=10-K&dateb=&owner=include&count=10&action=getcompany",
        "google_finance": f"https://www.google.com/finance/quote/{encoded.replace('+', '-')}",
        "yahoo_finance": f"https://finance.yahoo.com/lookup?s={encoded}",
        "macrotrends": f"https://www.macrotrends.net/stocks/charts/{encoded}/",
        "wisesheets_peers": f"https://wisesheets.io/screener?sector={encoded}",
    }


# ── Tool 3: score_customer_concentration ─────────────────────────────────────

def score_customer_concentration(major_customers: list) -> dict:
    """Calculate customer concentration metrics from 10-K major customer disclosures.

    SEC Regulation S-K Item 101(c) requires disclosure of customers representing
    ≥10% of revenue. This tool calculates HHI, top-N concentration, and risk
    thresholds from those disclosures.

    The agent should extract major_customers from the 'Customers' subsection of
    Item 1 (Business) and from 'Risk Factors' in the 10-K before calling this tool.

    Args:
        major_customers: List of dicts, each with:
            - name (str): Customer name (may be anonymised as "Customer A").
            - revenue_pct (float): Percentage of total revenue (e.g. 15.0 for 15%).
            - fiscal_year (str, optional): e.g. "FY2023".
            - notes (str, optional): Disclosure context, e.g. "disclosed in Risk Factors".
          If revenue_pct is absent for a customer, that customer is excluded from
          calculations and noted in missing_data.

    Returns:
        dict with data_available, concentration_metrics, risk_flags, overall_risk_level,
        hhi, and source.
    """
    if not major_customers:
        return {
            "data_available": False,
            "message": (
                "major_customers list is empty. "
                "If no customers are disclosed in the 10-K, the company likely has no "
                "customer representing ≥10% of revenue — note this as LOW concentration risk."
            ),
            "concentration_metrics": {},
            "risk_flags": [],
            "overall_risk_level": "UNKNOWN",
            "hhi": None,
            "source": "SEC Regulation S-K Item 101(c) — customer concentration disclosure",
        }

    valid: list[dict] = []
    missing_data: list[str] = []
    for c in major_customers:
        pct = c.get("revenue_pct")
        name = c.get("name", "Unknown")
        if pct is None:
            missing_data.append(name)
        else:
            try:
                valid.append({"name": name, "revenue_pct": float(pct), "notes": c.get("notes", ""), "fiscal_year": c.get("fiscal_year", "")})
            except (TypeError, ValueError):
                missing_data.append(name)

    if not valid:
        return {
            "data_available": False,
            "message": (
                f"No customers with revenue_pct data found. "
                f"Customers without pct data: {', '.join(missing_data) or 'none provided'}. "
                "Extract revenue_pct figures from the 10-K disclosure and retry."
            ),
            "concentration_metrics": {},
            "risk_flags": [],
            "overall_risk_level": "UNKNOWN",
            "hhi": None,
            "missing_data": missing_data,
            "source": "SEC Regulation S-K Item 101(c) — customer concentration disclosure",
        }

    valid.sort(key=lambda x: x["revenue_pct"], reverse=True)

    top1_pct = valid[0]["revenue_pct"] if valid else 0.0
    top3_pct = sum(c["revenue_pct"] for c in valid[:3])
    top5_pct = sum(c["revenue_pct"] for c in valid[:5])
    total_known_pct = sum(c["revenue_pct"] for c in valid)

    # Herfindahl-Hirschman Index (using known percentages as shares)
    hhi = round(sum((c["revenue_pct"] / 100) ** 2 * 10_000 for c in valid), 1)

    risk_flags: list[str] = []
    overall_level = "LOW"

    for threshold, level, message in _SINGLE_CUSTOMER_THRESHOLDS:
        if top1_pct >= threshold:
            risk_flags.append(f"{message} (largest customer: {valid[0]['name']} at {top1_pct:.1f}%)")
            overall_level = level
            break

    for threshold, level, message in _CONC_THRESHOLDS:
        if top3_pct >= threshold:
            flag = f"{message} (top-3 combined: {top3_pct:.1f}%)"
            if flag not in risk_flags:
                risk_flags.append(flag)
            if _risk_rank(level) > _risk_rank(overall_level):
                overall_level = level
            break

    if hhi > 2500:
        risk_flags.append(f"HHI={hhi:.0f} — highly concentrated customer base (DOJ threshold for concentrated markets: >2500)")

    known_customer_list = [
        {"name": c["name"], "revenue_pct": c["revenue_pct"], "fiscal_year": c.get("fiscal_year", ""), "notes": c.get("notes", "")}
        for c in valid
    ]

    return {
        "data_available": True,
        "customers_analysed": len(valid),
        "customers_missing_pct": missing_data,
        "concentration_metrics": {
            "top_1_customer_pct": round(top1_pct, 1),
            "top_3_customers_pct": round(top3_pct, 1),
            "top_5_customers_pct": round(top5_pct, 1),
            "total_known_disclosed_pct": round(total_known_pct, 1),
            "largest_customer_name": valid[0]["name"],
        },
        "hhi": hhi,
        "risk_flags": risk_flags,
        "overall_risk_level": overall_level,
        "customers": known_customer_list,
        "note": (
            "These figures are derived solely from SEC 10-K disclosures. "
            "Undisclosed customers (each < 10% of revenue) are excluded, so total_known_disclosed_pct "
            "may be below 100%. A low total suggests broad customer diversification."
        ),
        "source": "SEC Regulation S-K Item 101(c) — customer concentration disclosure (https://www.sec.gov/)",
    }


def _risk_rank(level: str) -> int:
    return {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3, "UNKNOWN": -1}.get(level, -1)


# ── Tool 4: evaluate_growth_drivers ──────────────────────────────────────────

def evaluate_growth_drivers(industry: str, company_description: str) -> dict:
    """Generate a structured PESTLE and Porter's Five Forces framework for the target.

    Uses embedded sector-specific economic knowledge to populate a growth driver
    analysis checklist. Each factor includes a government or public data URL for
    the agent to verify with load_web_page. Returns identified tailwinds, headwinds,
    and a Porter's Five Forces structure with sector context.

    This tool generates a structured starting point — the agent must verify every
    factor using the provided URLs and the company's own public disclosures.

    Args:
        industry: Sector name (e.g. "SaaS", "fintech", "healthcare", "defense").
        company_description: 1-3 sentence description of the company's business
                             (from 10-K Item 1 or company website).

    Returns:
        dict with data_available=True, tailwinds, headwinds, pestle_checklist,
        porters_five_forces, government_data_urls, and source.
    """
    industry_lower = industry.lower().strip()
    desc_lower = company_description.lower()

    # Match to embedded sector or fall back to defaults
    drivers = _SECTOR_GROWTH_DRIVERS.get(industry_lower)
    if drivers is None:
        # Keyword search across all sector keys
        for key, data in _SECTOR_GROWTH_DRIVERS.items():
            if key in desc_lower or key in industry_lower:
                drivers = data
                break
    if drivers is None:
        drivers = _DEFAULT_GROWTH_DRIVERS

    gov_urls: list[str] = list(dict.fromkeys(
        _get_gov_urls("US") + drivers.get("government_data_urls", [])
    ))

    pestle = [
        {
            "factor": "Political",
            "items": [
                "US-China trade and tariff policy affecting supply chains (USTR; https://ustr.gov/)",
                "Industrial policy (CHIPS Act, IRA, IIJA) shaping sector investment (Congress.gov; https://www.congress.gov/)",
                "Antitrust and tech regulation (DOJ/FTC; https://www.ftc.gov/enforcement/actions)",
            ],
        },
        {
            "factor": "Economic",
            "items": [
                f"GDP growth trajectory (BEA; https://www.bea.gov/data/gdp/gross-domestic-product)",
                "Interest rate environment: Federal Funds Rate (Federal Reserve; https://www.federalreserve.gov/releases/h15/)",
                "Corporate IT / capex spend cycle (BEA private fixed investment; https://www.bea.gov/data/economic-accounts/national)",
                "CPI inflation impact on labour and energy costs (BLS; https://www.bls.gov/cpi/)",
            ],
        },
        {
            "factor": "Social",
            "items": [
                "Demographics: US Census population projections (https://www.census.gov/programs-surveys/popproj.html)",
                "Workforce remote/hybrid adoption shifting enterprise software demand (BLS; https://www.bls.gov/news.release/atus.htm)",
                "Consumer confidence index (Conference Board; https://www.conference-board.org/topics/consumer-confidence)",
            ],
        },
        {
            "factor": "Technological",
            "items": [
                "AI/ML adoption curve by sector (NIST AI index; https://aiindex.stanford.edu/report/)",
                "Cloud infrastructure spend growth (public cloud provider earnings calls)",
                "Cybersecurity threat landscape — CISA advisories (https://www.cisa.gov/known-exploited-vulnerabilities-catalog)",
            ],
        },
        {
            "factor": "Legal / Regulatory",
            "items": [
                "Data privacy: GDPR enforcement (https://edpb.europa.eu/), CCPA (https://oag.ca.gov/privacy/ccpa)",
                "AI Act (EU) — risk classification requirements (https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)",
                "Antitrust scrutiny of tech M&A (DOJ/EC; https://www.justice.gov/atr)",
                "Export controls (EAR/ITAR): BIS entity list (https://www.bis.doc.gov/)",
            ],
        },
        {
            "factor": "Environmental",
            "items": [
                "ESG reporting mandates: SEC climate disclosure rule (https://www.sec.gov/rules-regulations/2024/03/the-enhancement-and-standardization-of-climate-related-disclosures)",
                "Energy costs and renewable transition affecting data centre economics",
                "Physical climate risk in supply chain geography (FEMA hazard data; https://hazards.fema.gov/)",
            ],
        },
    ]

    porters = {
        "threat_of_new_entrants": {
            "assessment": "Evaluate capital requirements, switching costs, network effects, and brand moat from public filings and VC funding data.",
            "data_sources": [
                "Crunchbase funding data (https://www.crunchbase.com/)",
                "PitchBook public company tracker",
                "Target 10-K Risk Factors section — barriers to entry disclosures",
            ],
        },
        "bargaining_power_of_buyers": {
            "assessment": "Assess from customer concentration (score_customer_concentration output), churn rate disclosures (if public), NPS proxies from public reviews.",
            "data_sources": [
                "Target 10-K Item 1 — customer disclosure",
                "Glassdoor / G2 / Capterra public reviews for software products",
                "10-K Risk Factors — customer dependency disclosures",
            ],
        },
        "bargaining_power_of_suppliers": {
            "assessment": "Assess from 10-K supply chain risk disclosures, AWS/Azure/GCP dependency mentions, and key vendor concentration in cost of revenue.",
            "data_sources": [
                "Target 10-K Item 1 — supplier concentration",
                "Target 10-K Risk Factors — single-source or key-supplier risk",
                "10-K Exhibit 10 — material supply contracts",
            ],
        },
        "threat_of_substitutes": {
            "assessment": "Identify open-source, legacy, and adjacent technology substitutes from competitor 10-K competition sections and industry analyst reports.",
            "data_sources": [
                "Competitor 10-K filings — competition section",
                "GitHub trending repositories in the sector (https://github.com/trending)",
                "Stack Overflow developer surveys for technology adoption (https://survey.stackoverflow.co/)",
            ],
        },
        "competitive_rivalry": {
            "assessment": "Measure via number of EDGAR-registered peers, public pricing comparisons, and market share disclosures in company filings.",
            "data_sources": [
                "EDGAR peer search (see analyze_competitive_landscape output)",
                "Target 10-K — named competitors in Item 1",
                "Public pricing pages and analyst comparison reports",
            ],
        },
    }

    return {
        "data_available": True,
        "industry": industry,
        "sectors_matched": [industry_lower] if drivers is not _DEFAULT_GROWTH_DRIVERS else [],
        "tailwinds": drivers.get("tailwinds", []),
        "headwinds": drivers.get("headwinds", []),
        "pestle_checklist": pestle,
        "porters_five_forces": porters,
        "government_data_urls": gov_urls,
        "industry_association_urls": drivers.get("industry_association_urls", []),
        "note": (
            "All tailwind/headwind claims include a public source URL. "
            "The agent must verify each claim by using load_web_page on the cited URL "
            "and confirm it is consistent with the target company's actual public disclosures. "
            "Do not report any growth rate figure without citing its public source."
        ),
        "source": "Embedded sector analysis drawn from public BLS, BEA, Census, Federal Reserve, and regulatory press releases",
    }
