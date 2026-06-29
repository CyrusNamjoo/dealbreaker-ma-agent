"""
Legal analysis tools for the LegalReviewAgent.

Four tools that query public legal, patent, and corporate databases.

CONTRACT (applies to every tool in this module):
  - data_available=False when an API key is missing or the source returns
    no usable data. A clear explanation is always included.
  - No legal conclusion is estimated — only facts drawn from public records.
  - Every case, patent, and filing is cited with its source URL and identifier.
  - Confidential documents are never requested or processed.
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
_UA = f"DealBreaker-AI {_CONTACT_EMAIL}"
_EDGAR_HEADERS = {"User-Agent": _UA, "Accept": "application/json"}
_EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{}.json"
_CL_BASE = "https://www.courtlistener.com/api/rest/v4"
_PATENTSVIEW = "https://api.patentsview.org/patents/query"

# ── Secretary of State registry portals ─────────────────────────────────────
_SOS_PORTALS: dict[str, dict[str, str]] = {
    "DE": {"url": "https://icis.corp.delaware.gov/Ecorp/EntitySearch/NameSearch.aspx", "notes": "Delaware Division of Corporations entity search."},
    "CA": {"url": "https://bizfileonline.sos.ca.gov/search/business", "notes": "California SOS business entity search."},
    "NY": {"url": "https://www.dos.ny.gov/corps/bus_entity_search.html", "notes": "New York DOS business entity search."},
    "TX": {"url": "https://mycpa.cpa.state.tx.us/coa/Index.html", "notes": "Texas SOS entity search via Comptroller."},
    "FL": {"url": "https://dos.myflorida.com/sunbiz/search/", "notes": "Florida SOS Sunbiz entity search."},
    "WA": {"url": "https://www.sos.wa.gov/corps/search.aspx", "notes": "Washington SOS entity search."},
    "MA": {"url": "https://corp.sec.state.ma.us/CorpWeb/CorpSearch/CorpSearch.aspx", "notes": "Massachusetts SOS corporate search."},
    "NV": {"url": "https://esos.nv.gov/EntitySearch/OnlineEntitySearch", "notes": "Nevada SOS entity search."},
    "IL": {"url": "https://www.ilsos.gov/corporatellc/", "notes": "Illinois SOS corporate/LLC search."},
    "GA": {"url": "https://ecorp.sos.ga.gov/BusinessSearch", "notes": "Georgia SOS business entity search."},
}

# ── State court public search portals ────────────────────────────────────────
_STATE_COURT_PORTALS: dict[str, dict[str, str]] = {
    "DE": {"url": "https://courts.delaware.gov/forms/search.aspx", "notes": "Delaware Court of Chancery — primary corporate litigation venue."},
    "CA": {"url": "https://www.courts.ca.gov/selfhelp-courtfindermap.htm", "notes": "California Courts case search portal."},
    "NY": {"url": "https://iapps.courts.state.ny.us/nyscef/HomePage", "notes": "New York eCourts / NYSCEF case filing search."},
    "TX": {"url": "https://search.txcourts.gov/", "notes": "Texas court case search."},
    "FL": {"url": "https://www.myflcourtaccess.com/", "notes": "Florida Courts e-Filing Portal public case search."},
    "IL": {"url": "https://www.illinoiscourts.gov/courts/electronic-case-filings/", "notes": "Illinois Courts e-filing access."},
    "NJ": {"url": "https://portal.njcourts.gov/webe10/CivilCaseJacketWeb/", "notes": "New Jersey Courts civil case jacket search."},
}

# ── Regulatory knowledge base for screen_regulatory_compliance ───────────────

# Maps sector slug → list of keywords that identify that sector
_SECTOR_KEYWORDS: dict[str, list[str]] = {
    "software": ["software", "saas", "cloud", "platform", "api", "tech", "digital", "ai ", "machine learning", "data analytics"],
    "fintech": ["fintech", "payment", "neobank", "digital bank", "lending", "insurtech", "robo-advis", "buy now pay later", "bnpl", "crypto", "blockchain", "defi"],
    "financial_services": ["bank", "broker", "dealer", "investment", "asset manag", "hedge fund", "securities", "trading", "wealth manag", "private equity"],
    "healthcare": ["healthcare", "health care", "medical", "hospital", "clinical", "patient", "telehealth", "ehr", "emr", "health information"],
    "pharma": ["pharma", "drug", "biologic", "biotech", "therapeutics", "gene therapy", "vaccine", "clinical trial", "pharmaceutical"],
    "telecom": ["telecom", "wireless", "spectrum", "broadcast", "cable", "5g", "satellite", "internet service provider", "isp"],
    "energy": ["energy", "utility", "power", "solar", "wind", "oil", "gas", "pipeline", "renewable", "electric grid"],
    "defense": ["defense", "defence", "military", "aerospace", "government contractor", "dod", "national security", "intelligence"],
    "food": ["food", "beverage", "restaurant", "cpg", "packaged food", "alcohol", "spirits", "beer", "wine", "dietary supplement"],
    "transportation": ["transport", "logistics", "shipping", "freight", "trucking", "rail", "aviation", "airline", "fleet management"],
    "real_estate": ["real estate", "property", "mortgage", "reit", "housing", "construction", "proptech"],
    "retail": ["retail", "e-commerce", "ecommerce", "marketplace", "consumer goods", "d2c", "direct-to-consumer", "brand"],
}

_SECTOR_REGULATORS: dict[str, list[dict[str, str]]] = {
    "software": [
        {"body": "FTC", "area": "Privacy / data security / unfair practices", "search_url": "https://www.ftc.gov/enforcement/actions", "notes": "Review for FTC enforcement actions; data broker registration if applicable."},
        {"body": "BIS/Commerce", "area": "Export controls (EAR) — dual-use technology", "search_url": "https://www.bis.doc.gov/index.php/policy-guidance/lists-of-parties-of-concern", "notes": "Verify export classification; check Entity List."},
        {"body": "FCC", "area": "TCPA (if product uses SMS/auto-dialers)", "search_url": "https://www.fcc.gov/enforcement/orders", "notes": "Relevant if product involves outbound communications."},
    ],
    "fintech": [
        {"body": "FinCEN", "area": "BSA / AML / money transmitter", "search_url": "https://www.fincen.gov/resources/enforcement", "notes": "MSB registration; MTL required in most US states."},
        {"body": "CFPB", "area": "Consumer financial protection", "search_url": "https://www.consumerfinance.gov/enforcement/actions/", "notes": "Applies to consumer lending, payments, credit reporting."},
        {"body": "NMLS", "area": "State money transmitter / lender licenses", "search_url": "https://www.nmlsconsumeraccess.org/", "notes": "Check NMLS for license status across all states."},
        {"body": "SEC", "area": "Investment adviser / broker-dealer (if applicable)", "search_url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany", "notes": "Check if registered; look for enforcement actions."},
    ],
    "financial_services": [
        {"body": "SEC", "area": "Securities registration / enforcement", "search_url": "https://efts.sec.gov/LATEST/search-index?q=%22AP%22&forms=AP", "notes": "Check for AP orders and 34-Act violations."},
        {"body": "FINRA", "area": "Broker-dealer / registered rep compliance", "search_url": "https://brokercheck.finra.org/", "notes": "BrokerCheck — disciplinary history, disclosures, registrations."},
        {"body": "CFTC", "area": "Derivatives / commodity futures", "search_url": "https://www.cftc.gov/LawRegulation/Enforcement/enforcementactions.html", "notes": "Required for futures commission merchants and swap dealers."},
        {"body": "OCC / FDIC / Federal Reserve", "area": "Banking charter / prudential supervision", "search_url": "https://www.ffiec.gov/nicpubweb/content/NICWELCOME.aspx", "notes": "NIC — charter status and enforcement orders."},
        {"body": "CFPB", "area": "Consumer financial protection", "search_url": "https://www.consumerfinance.gov/enforcement/actions/", "notes": "Applies to retail banking, mortgage, auto, student loans."},
    ],
    "healthcare": [
        {"body": "HHS / OCR", "area": "HIPAA privacy and security", "search_url": "https://ocrportal.hhs.gov/ocr/breach/breach_report.jsf", "notes": "Breach Portal — data incidents; review right-of-access cases."},
        {"body": "CMS", "area": "Medicare / Medicaid provider enrollment", "search_url": "https://data.cms.gov/provider-data/", "notes": "Verify enrollment; check exclusion from federal programs."},
        {"body": "OIG / HHS", "area": "Anti-kickback / Stark Law / exclusions", "search_url": "https://oig.hhs.gov/exclusions/exclusions_list.asp", "notes": "Search OIG exclusion list for all entities and principals."},
        {"body": "State licensing boards", "area": "Provider licensure by state", "search_url": "https://www.fsmb.org/", "notes": "FSMB directory of state medical boards."},
    ],
    "pharma": [
        {"body": "FDA", "area": "Drug / device / biologic approval and compliance", "search_url": "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters", "notes": "Warning letters, consent decrees, 483 observations."},
        {"body": "DEA", "area": "Controlled substance scheduling / registration", "search_url": "https://www.deadiversion.usdoj.gov/drug_chem_info/index.html", "notes": "Required registration for Schedule I–V substances."},
        {"body": "CMS", "area": "Drug pricing / 340B program", "search_url": "https://data.cms.gov/", "notes": "Check rebate agreements and 340B participation."},
    ],
    "telecom": [
        {"body": "FCC", "area": "Spectrum / broadcast licenses / TCPA", "search_url": "https://www.fcc.gov/uls/index.php?job=license&page=license_search", "notes": "ULS — check license status and renewal dates."},
        {"body": "FTC", "area": "Consumer protection / privacy (ISPs)", "search_url": "https://www.ftc.gov/enforcement/actions", "notes": "Applies to ISPs and data-intensive telecom companies."},
    ],
    "energy": [
        {"body": "FERC", "area": "Electricity / gas / pipeline regulation", "search_url": "https://www.ferc.gov/industries-data/enforcement/enforcement-actions", "notes": "FERC registration, tariffs, and enforcement actions."},
        {"body": "EPA", "area": "Environmental compliance", "search_url": "https://echo.epa.gov/", "notes": "ECHO — facility-level compliance and enforcement history."},
        {"body": "NRC", "area": "Nuclear materials / reactors (if applicable)", "search_url": "https://www.nrc.gov/reactors/operating/list-power-reactor-units.html", "notes": "License status for nuclear operations."},
    ],
    "defense": [
        {"body": "DDTC / State", "area": "ITAR — defense articles and services", "search_url": "https://www.pmddtc.state.gov/ddtc_public?id=ddtc_kb_article_page&sys_id=24d528fddbfc930044f9ff621f961987", "notes": "ITAR registration required; check debarred party list."},
        {"body": "BIS / Commerce", "area": "EAR — dual-use export controls", "search_url": "https://www.bis.doc.gov/index.php/policy-guidance/lists-of-parties-of-concern", "notes": "Denied Persons and Entity Lists."},
        {"body": "SAM.gov", "area": "Federal contractor registration / debarment", "search_url": "https://sam.gov/content/exclusions", "notes": "Active exclusions and suspension."},
    ],
    "food": [
        {"body": "FDA", "area": "Food safety / FSMA / labeling", "search_url": "https://www.fda.gov/food/compliance-enforcement/warning-letters", "notes": "Warning letters and recall database."},
        {"body": "USDA / FSIS", "area": "Meat, poultry, egg products inspection", "search_url": "https://www.fsis.usda.gov/establishments/inspection-data", "notes": "Establishment listing and inspection results."},
        {"body": "TTB", "area": "Alcohol / tobacco permit", "search_url": "https://www.ttb.gov/wine/winery-permit-applications", "notes": "Federal basic permit required for alcohol production/distribution."},
    ],
    "transportation": [
        {"body": "DOT / FMCSA", "area": "Motor carrier / trucking safety rating", "search_url": "https://safer.fmcsa.dot.gov/CompanySnapshot.aspx", "notes": "SAFER system — carrier registration, safety rating, inspections."},
        {"body": "FAA", "area": "Aviation certificates / drone operations", "search_url": "https://amsrvs.registry.faa.gov/airmeninquiry/", "notes": "Airman certificates, aircraft registration, repair station certificates."},
        {"body": "FRA", "area": "Rail safety (if applicable)", "search_url": "https://www.fra.dot.gov/View/ViewFRA_SafetyDB.aspx", "notes": "FRA enforcement orders and accident data."},
    ],
    "real_estate": [
        {"body": "CFPB", "area": "Mortgage / RESPA / TILA", "search_url": "https://www.consumerfinance.gov/enforcement/actions/", "notes": "Applies to mortgage origination, servicing, and brokerage."},
        {"body": "HUD", "area": "Fair Housing / FHA programs", "search_url": "https://www.hud.gov/program_offices/fair_housing_equal_opp/online-complaint", "notes": "Fair housing complaint search."},
        {"body": "State real estate commissions", "area": "Broker / agent licensing", "search_url": "https://www.arello.org/member-jurisdictions/", "notes": "ARELLO directory of all state real estate regulators."},
    ],
    "retail": [
        {"body": "FTC", "area": "Advertising / CAN-SPAM / subscription disclosure", "search_url": "https://www.ftc.gov/enforcement/actions", "notes": "Truth in advertising, endorsement guides."},
        {"body": "CPSC", "area": "Product safety / recalls", "search_url": "https://www.cpsc.gov/Recalls", "notes": "CPSC recall database — product safety history."},
        {"body": "CBP / Customs", "area": "Import compliance / UFLPA (forced labor)", "search_url": "https://www.cbp.gov/trade/forced-labor/UFLPA", "notes": "Uyghur Forced Labor Prevention Act entity list."},
    ],
}

_INTL_REGULATORS: dict[str, list[dict[str, str]]] = {
    "EU": [
        {"body": "DPA (EU)", "area": "GDPR compliance", "search_url": "https://edpb.europa.eu/our-work-tools/consistency-findings_en", "notes": "EDPB decisions and relevant DPA enforcement by member state."},
        {"body": "EC / DG COMP", "area": "EU competition law / merger control", "search_url": "https://ec.europa.eu/competition/mergers/cases/", "notes": "EU merger notifications above EC thresholds."},
        {"body": "EBA / ECB", "area": "Banking / financial services authorisation", "search_url": "https://www.eba.europa.eu/regulation-and-policy/consumer-protection", "notes": "EBA register for authorised credit institutions."},
        {"body": "ECHA", "area": "REACH / RoHS — chemical / product safety", "search_url": "https://echa.europa.eu/information-on-chemicals", "notes": "Substance restrictions for products sold in EU."},
    ],
    "UK": [
        {"body": "FCA", "area": "Financial services authorisation", "search_url": "https://register.fca.org.uk/s/", "notes": "FCA Register — authorisation status and disciplinary history."},
        {"body": "ICO", "area": "UK GDPR / data protection", "search_url": "https://ico.org.uk/action-weve-taken/enforcement/", "notes": "ICO enforcement notices and monetary penalty notices."},
        {"body": "CMA", "area": "UK competition / merger control", "search_url": "https://www.gov.uk/cma-cases", "notes": "CMA mergers register — UK filing obligations."},
    ],
    "CA": [
        {"body": "OPC", "area": "PIPEDA / Bill C-27 (CPPA) privacy", "search_url": "https://www.priv.gc.ca/en/opc-actions-and-decisions/", "notes": "Office of the Privacy Commissioner enforcement findings."},
        {"body": "OSFI", "area": "Federal financial institutions", "search_url": "https://www.osfi-bsif.gc.ca/en/guidance/guidance-library", "notes": "Office of the Superintendent of Financial Institutions."},
        {"body": "Competition Bureau", "area": "Merger notification / antitrust", "search_url": "https://www.canada.ca/en/competition-bureau.html", "notes": "Pre-merger notification required above Canadian thresholds."},
    ],
    "AU": [
        {"body": "ASIC", "area": "Financial services / corporate regulation", "search_url": "https://asic.gov.au/regulatory-resources/find-a-licensee/", "notes": "ASIC registers — financial services licensees and credit providers."},
        {"body": "ACCC", "area": "Competition / merger review", "search_url": "https://www.accc.gov.au/regulated-infrastructure/mergers/merger-assessments", "notes": "ACCC merger review database."},
        {"body": "OAIC", "area": "Australian Privacy Act", "search_url": "https://www.oaic.gov.au/privacy/privacy-decisions", "notes": "OAIC privacy decisions and determinations."},
    ],
}

# Universal — always included regardless of sector
_UNIVERSAL_CHECKS: list[dict[str, str]] = [
    {"body": "OFAC / Treasury", "area": "Sanctions compliance", "search_url": "https://sanctionssearch.ofac.treas.gov/", "notes": "Check all principals, subsidiaries, and major customers against OFAC SDN list."},
    {"body": "DOJ / FBI", "area": "FCPA — Foreign Corrupt Practices Act", "search_url": "https://www.justice.gov/criminal-fraud/fcpa/fcpa-digest", "notes": "FCPA digest of enforcement actions — mandatory for targets with non-US operations."},
    {"body": "SAM.gov / EPLS", "area": "Federal debarment / exclusion", "search_url": "https://sam.gov/content/exclusions", "notes": "System for Award Management — debarment, suspension, ineligibility."},
]


# ── Tool 1: search_litigation_records ────────────────────────────────────────

async def search_litigation_records(company_name: str, jurisdiction: str) -> dict:
    """Search federal and state court records for litigation involving a company.

    Queries the CourtListener PACER index for federal cases. Returns direct
    links to state court public search portals for the specified jurisdiction.
    Also provides SEC enforcement action search URL for securities-related cases.

    COURTLISTENER_API_KEY environment variable must be set for federal case data.
    Register for a free token at https://www.courtlistener.com/sign-in/

    The legal DB MCP server tool `search_federal_court_records` provides
    complementary coverage — use both for thorough federal case research.

    Args:
        company_name: Company name to search (e.g. "Theranos Inc").
        jurisdiction: "federal" for federal PACER courts only; a US state code
                      (e.g. "DE", "CA", "NY") to also receive that state's court
                      portal; or "all" for federal + all configured state portals.

    Returns:
        dict with data_available, federal_cases, total_federal_available,
        state_court_portals, sec_enforcement_search_url, source, and warnings.
    """
    token = os.getenv("COURTLISTENER_API_KEY", "").strip()
    if not token:
        return {
            "data_available": False,
            "message": (
                "COURTLISTENER_API_KEY is not set. "
                "Register for a free API token at https://www.courtlistener.com/sign-in/ "
                "and add COURTLISTENER_API_KEY=<token> to app/.env."
            ),
            "federal_cases": [],
            "total_federal_available": 0,
            "state_court_portals": _build_state_court_portals(jurisdiction),
            "sec_enforcement_search_url": _sec_enforcement_url(company_name),
            "source": "CourtListener (https://www.courtlistener.com/api/)",
        }

    headers = {
        "Authorization": f"Token {token}",
        "User-Agent": _UA,
        "Accept": "application/json",
    }
    params: dict[str, Any] = {
        "q": company_name,
        "type": "r",
        "order_by": "score desc",
        "page_size": 20,
    }

    await _rate_limit(f"{_CL_BASE}/dockets/")
    try:
        async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
            r = await client.get(f"{_CL_BASE}/dockets/", params=params)
        if r.status_code == 401:
            return {
                "data_available": False,
                "message": "CourtListener API token is invalid or expired. Check COURTLISTENER_API_KEY in app/.env.",
                "federal_cases": [],
                "total_federal_available": 0,
                "state_court_portals": _build_state_court_portals(jurisdiction),
                "sec_enforcement_search_url": _sec_enforcement_url(company_name),
                "source": "CourtListener (https://www.courtlistener.com/api/)",
            }
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPError as exc:
        return {
            "data_available": False,
            "message": f"CourtListener API request failed: {exc}",
            "federal_cases": [],
            "total_federal_available": 0,
            "state_court_portals": _build_state_court_portals(jurisdiction),
            "sec_enforcement_search_url": _sec_enforcement_url(company_name),
            "source": "CourtListener (https://www.courtlistener.com/api/)",
        }

    cases = [
        {
            "case_name": item.get("case_name", ""),
            "court": item.get("court_id", ""),
            "date_filed": item.get("date_filed", ""),
            "date_terminated": item.get("date_terminated", ""),
            "docket_number": item.get("docket_number", ""),
            "nature_of_suit": item.get("nature_of_suit", ""),
            "cause": item.get("cause", ""),
            "pacer_case_id": item.get("pacer_case_id", ""),
            "case_url": f"https://www.courtlistener.com{item.get('absolute_url', '')}",
        }
        for item in data.get("results", [])
    ]
    total = data.get("count", 0)

    return {
        "data_available": True,
        "company_name": company_name,
        "total_federal_available": total,
        "federal_cases_returned": len(cases),
        "federal_cases": cases,
        "state_court_portals": _build_state_court_portals(jurisdiction),
        "sec_enforcement_search_url": _sec_enforcement_url(company_name),
        "note": (
            f"{total} federal PACER cases found. "
            "Supplement with the MCP tool `search_federal_court_records` for additional coverage. "
            "Use `load_web_page` on case_url entries to read docket sheets."
        ),
        "source": "CourtListener PACER federal index (https://www.courtlistener.com/api/)",
    }


def _build_state_court_portals(jurisdiction: str) -> list[dict[str, str]]:
    j = jurisdiction.strip().upper()
    if j == "ALL":
        return [{"state": k, **v} for k, v in _STATE_COURT_PORTALS.items()]
    if j in _STATE_COURT_PORTALS:
        return [{"state": j, **_STATE_COURT_PORTALS[j]}]
    return [{"state": j, "url": f"https://www.google.com/search?q={j}+court+case+search+public+records", "notes": f"No configured portal for {j}. Use this search to locate the court's public case search."}]


def _sec_enforcement_url(company_name: str) -> str:
    encoded = company_name.replace(" ", "+")
    return f"https://efts.sec.gov/LATEST/search-index?q=%22{encoded}%22&forms=AP,34-12G4&dateRange=custom&startdt=2019-01-01"


# ── Tool 2: check_ip_ownership ───────────────────────────────────────────────

async def check_ip_ownership(company_name: str) -> dict:
    """Search USPTO PatentsView for US patents assigned to a company.

    Queries the public PatentsView API (no authentication required) for granted
    US patents with the company as assignee. Returns direct search URLs for
    USPTO trademark, EPO, and UK IPO databases for the agent to review with
    load_web_page.

    Args:
        company_name: Company name or assignee organisation (e.g. "Salesforce Inc").

    Returns:
        dict with data_available, us_patents (list), patent_count,
        search_urls (USPTO patent, USPTO trademark, EPO, UK IPO),
        and source.
    """
    payload = {
        "q": {"_text_phrase": {"assignee_organization": company_name}},
        "f": ["patent_number", "patent_title", "patent_date", "patent_type", "assignee_organization"],
        "o": {"matched_subentities_only": True, "per_page": 25, "sort_by": "patent_date", "sort_order": "desc"},
    }
    encoded_name = company_name.replace(" ", "+")

    search_urls = {
        "uspto_patents": f"https://ppubs.uspto.gov/pubwebapp/static/pages/ppubsbasic.html",
        "uspto_trademark": f"https://tmsearch.uspto.gov/search/search-information",
        "epo_espacenet": f"https://worldwide.espacenet.com/patent/search?q=pa%3D%22{encoded_name}%22",
        "uk_ipo": f"https://www.ipo.gov.uk/p-ipsum/Case/PublicationNumber",
        "google_patents": f"https://patents.google.com/?assignee={encoded_name}&sort=new",
    }

    await _rate_limit(_PATENTSVIEW)
    try:
        async with httpx.AsyncClient(headers={"User-Agent": _UA}, timeout=30.0) as client:
            r = await client.post(_PATENTSVIEW, json=payload)
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPError as exc:
        return {
            "data_available": False,
            "message": f"PatentsView API request failed: {exc}. Use search_urls with load_web_page for manual patent research.",
            "us_patents": [],
            "patent_count": 0,
            "search_urls": search_urls,
            "source": "USPTO PatentsView (https://patentsview.org/)",
        }

    patents_raw = data.get("patents") or []
    patents = []
    for p in patents_raw:
        assignees = p.get("assignees") or [{}]
        patents.append({
            "patent_number": p.get("patent_number", ""),
            "title": p.get("patent_title", ""),
            "date": p.get("patent_date", ""),
            "type": p.get("patent_type", ""),
            "assignee": (assignees[0] or {}).get("assignee_organization", company_name),
            "patent_url": f"https://patents.google.com/patent/US{p.get('patent_number', '')}",
        })

    total = int(data.get("total_patent_count") or 0)

    return {
        "data_available": True,
        "company_name": company_name,
        "patent_count": total,
        "patents_returned": len(patents),
        "us_patents": patents,
        "search_urls": search_urls,
        "note": (
            f"{total} US patents found via PatentsView. "
            "Use search_urls with load_web_page to search USPTO Trademark and EPO databases. "
            "Also check Google Patents (search_urls.google_patents) for cross-jurisdiction portfolio view."
        ),
        "source": "USPTO PatentsView API (https://patentsview.org/download/data-download-tables)",
    }


# ── Tool 3: verify_corporate_structure ──────────────────────────────────────

async def verify_corporate_structure(company_name: str, jurisdiction: str) -> dict:
    """Look up SEC EDGAR registrant data and return corporate registry search URLs.

    Queries the EDGAR full-text search API for the most recent 10-K filing,
    extracts the CIK, then fetches the EDGAR submissions JSON to return the
    registered company name, SIC code, state of incorporation, fiscal year end,
    and filing history. For the specified jurisdiction, returns the relevant
    Secretary of State portal URL.

    Args:
        company_name: Company name as registered with SEC or SOS
                      (e.g. "Salesforce Inc", "Apple Inc").
        jurisdiction: "US" or "SEC" to search EDGAR; a US state code (e.g. "DE")
                      to also receive that state's SOS portal; or "all" to receive
                      all configured SOS portals.

    Returns:
        dict with data_available, registrant (SEC-registered entity details),
        recent_annual_filings, sec_filings_url, sos_search_urls, and source.
    """
    # Step 1: Search EDGAR for the company's most recent 10-K to get CIK
    search_params = {
        "q": f'"{company_name}"',
        "forms": "10-K",
        "dateRange": "custom",
        "startdt": "2018-01-01",
    }
    await _rate_limit(_EDGAR_SEARCH)
    try:
        async with httpx.AsyncClient(headers=_EDGAR_HEADERS, timeout=30.0) as client:
            r = await client.get(_EDGAR_SEARCH, params=search_params)
        r.raise_for_status()
        hits = r.json().get("hits", {}).get("hits", [])
    except httpx.HTTPError as exc:
        return {
            "data_available": False,
            "message": f"EDGAR search failed: {exc}",
            "registrant": {},
            "recent_annual_filings": [],
            "sec_filings_url": _edgar_company_url(company_name),
            "sos_search_urls": _build_sos_urls(jurisdiction),
            "source": "SEC EDGAR (https://www.sec.gov/cgi-bin/browse-edgar)",
        }

    if not hits:
        return {
            "data_available": False,
            "message": (
                f"No SEC EDGAR 10-K filings found for '{company_name}'. "
                "The company may not be a US public registrant. "
                f"Check the SOS portals in sos_search_urls for private entity registration."
            ),
            "registrant": {},
            "recent_annual_filings": [],
            "sec_filings_url": _edgar_company_url(company_name),
            "sos_search_urls": _build_sos_urls(jurisdiction),
            "source": "SEC EDGAR (https://www.sec.gov/cgi-bin/browse-edgar)",
        }

    src = hits[0].get("_source", {})
    ciks = src.get("ciks", [])
    if not ciks:
        return {
            "data_available": False,
            "message": "EDGAR search returned results but no CIK was found.",
            "registrant": {},
            "recent_annual_filings": [],
            "sec_filings_url": _edgar_company_url(company_name),
            "sos_search_urls": _build_sos_urls(jurisdiction),
            "source": "SEC EDGAR (https://www.sec.gov/cgi-bin/browse-edgar)",
        }

    cik_raw = str(ciks[0]).lstrip("0")
    cik_padded = cik_raw.zfill(10)

    # Step 2: Fetch submissions JSON for full company profile
    await _rate_limit("https://data.sec.gov/")
    try:
        async with httpx.AsyncClient(headers=_EDGAR_HEADERS, timeout=30.0) as client:
            r2 = await client.get(_EDGAR_SUBMISSIONS.format(cik_padded))
        r2.raise_for_status()
        sub = r2.json()
    except httpx.HTTPError as exc:
        return {
            "data_available": False,
            "message": f"EDGAR submissions fetch failed for CIK {cik_padded}: {exc}",
            "registrant": {},
            "recent_annual_filings": [],
            "sec_filings_url": _edgar_company_url(company_name),
            "sos_search_urls": _build_sos_urls(jurisdiction),
            "source": "SEC EDGAR (https://www.sec.gov/cgi-bin/browse-edgar)",
        }

    tickers = sub.get("tickers", [])
    exchanges = sub.get("exchanges", [])
    former_names = [fn.get("name", "") for fn in sub.get("formerNames", [])]

    # Extract last 5 annual filings (10-K, 20-F, 40-F)
    filings_data = sub.get("filings", {}).get("recent", {})
    forms = filings_data.get("form", [])
    dates = filings_data.get("filingDate", [])
    accessions = filings_data.get("accessionNumber", [])
    primary_docs = filings_data.get("primaryDocument", [])

    annual_filings: list[dict[str, str]] = []
    annual_forms = {"10-K", "20-F", "40-F"}
    for i, form in enumerate(forms):
        if form in annual_forms and len(annual_filings) < 5:
            adsh = accessions[i] if i < len(accessions) else ""
            doc = primary_docs[i] if i < len(primary_docs) else ""
            adsh_clean = adsh.replace("-", "")
            annual_filings.append({
                "form": form,
                "date": dates[i] if i < len(dates) else "",
                "accession_number": adsh,
                "filing_index_url": (
                    f"https://www.sec.gov/Archives/edgar/data/{cik_raw}/{adsh_clean}/"
                    if cik_raw and adsh_clean else ""
                ),
                "primary_doc_url": (
                    f"https://www.sec.gov/Archives/edgar/data/{cik_raw}/{adsh_clean}/{doc}"
                    if cik_raw and adsh_clean and doc else ""
                ),
            })

    registrant = {
        "name": sub.get("name", ""),
        "cik": cik_raw,
        "sic": sub.get("sic", ""),
        "sic_description": sub.get("sicDescription", ""),
        "state_of_incorporation": sub.get("stateOfIncorporation", ""),
        "fiscal_year_end": sub.get("fiscalYearEnd", ""),
        "entity_type": sub.get("entityType", ""),
        "category": sub.get("category", ""),
        "tickers": tickers,
        "exchanges": exchanges,
        "ein": sub.get("ein", ""),
        "former_names": former_names,
        "edgar_filings_url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik_raw}&type=10-K&dateb=&owner=include&count=10",
    }

    return {
        "data_available": True,
        "company_name": company_name,
        "registrant": registrant,
        "annual_filings_found": len(annual_filings),
        "recent_annual_filings": annual_filings,
        "sec_filings_url": registrant["edgar_filings_url"],
        "sos_search_urls": _build_sos_urls(jurisdiction),
        "note": (
            "Use load_web_page on primary_doc_url entries to read the annual report. "
            "For subsidiary structure, review the 10-K Exhibit 21 (Subsidiaries list)."
        ),
        "source": f"SEC EDGAR submissions API (https://data.sec.gov/submissions/CIK{cik_padded}.json)",
    }


def _edgar_company_url(company_name: str) -> str:
    encoded = company_name.replace(" ", "+")
    return f"https://www.sec.gov/cgi-bin/browse-edgar?company={encoded}&CIK=&type=10-K&dateb=&owner=include&count=10&action=getcompany"


def _build_sos_urls(jurisdiction: str) -> list[dict[str, str]]:
    j = jurisdiction.strip().upper()
    if j in ("US", "SEC", "FEDERAL"):
        return []
    if j == "ALL":
        return [{"state": k, **v} for k, v in _SOS_PORTALS.items()]
    if j in _SOS_PORTALS:
        return [{"state": j, **_SOS_PORTALS[j]}]
    return [{"state": j, "url": f"https://www.google.com/search?q={j}+Secretary+of+State+business+entity+search", "notes": f"No configured portal for {j}. Use this search to locate the SOS registry."}]


# ── Tool 4: screen_regulatory_compliance ────────────────────────────────────

def screen_regulatory_compliance(business_description: str, jurisdictions: list) -> dict:
    """Generate a structured regulatory compliance checklist for the given business.

    Uses embedded regulatory framework knowledge to identify applicable regulatory
    bodies based on the business description and requested jurisdictions. Returns
    a checklist of agencies, compliance areas, and public search URLs for the
    agent to verify current compliance status using load_web_page.

    This tool identifies what to check — it does NOT confirm compliance.
    The agent must verify each item by reviewing the linked public databases.

    Args:
        business_description: Free-text description of the company's business
                              (e.g. "SaaS financial data platform for hedge funds",
                              "pharmaceutical company developing mRNA therapies").
        jurisdictions: List of jurisdiction codes to include
                      (e.g. ["US", "EU", "UK", "CA", "AU"]).

    Returns:
        dict with data_available=True, sectors_identified, compliance_checklist,
        jurisdictions_covered, unrecognized_jurisdictions, and source.
    """
    desc_lower = business_description.lower()

    # Identify applicable sectors via keyword match
    sectors_identified: list[str] = []
    for sector, keywords in _SECTOR_KEYWORDS.items():
        if any(kw in desc_lower for kw in keywords):
            sectors_identified.append(sector)

    # Build checklist: universal + sector-specific (US) + international
    checklist: list[dict[str, str]] = list(_UNIVERSAL_CHECKS)

    seen_bodies: set[str] = {item["body"] for item in _UNIVERSAL_CHECKS}
    for sector in sectors_identified:
        for item in _SECTOR_REGULATORS.get(sector, []):
            if item["body"] not in seen_bodies:
                checklist.append(item)
                seen_bodies.add(item["body"])

    # Add international regulators for requested jurisdictions
    recognized_jurisdictions: list[str] = []
    unrecognized_jurisdictions: list[str] = []
    for jur in jurisdictions:
        jur_upper = str(jur).upper()
        if jur_upper in ("US", "USA", "UNITED STATES"):
            recognized_jurisdictions.append("US")
            continue  # US regulators already added from sector lists above
        if jur_upper in _INTL_REGULATORS:
            recognized_jurisdictions.append(jur_upper)
            for item in _INTL_REGULATORS[jur_upper]:
                if item["body"] not in seen_bodies:
                    checklist.append({**item, "jurisdiction": jur_upper})
                    seen_bodies.add(item["body"])
        else:
            unrecognized_jurisdictions.append(jur_upper)

    if not sectors_identified:
        sector_note = (
            "No sectors were identified from the business description. "
            "Only universal compliance checks are included. "
            "Add sector-specific terms (e.g. 'SaaS', 'fintech', 'healthcare') for targeted results."
        )
    else:
        sector_note = f"Sectors identified: {', '.join(sectors_identified)}."

    return {
        "data_available": True,
        "sectors_identified": sectors_identified,
        "compliance_checklist": checklist,
        "checklist_item_count": len(checklist),
        "jurisdictions_covered": recognized_jurisdictions,
        "unrecognized_jurisdictions": unrecognized_jurisdictions,
        "note": (
            f"{sector_note} "
            "Use load_web_page on each search_url to verify current compliance status. "
            "This checklist is a starting point — engage qualified legal counsel for final compliance determination."
        ),
        "source": "Embedded regulatory framework (US CFR, EU regulations, UK statutory instruments — public law)",
    }
