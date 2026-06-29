"""
Legal database MCP server.

Exposes three tools to ADK agents:
  - search_federal_court_records : search CourtListener for federal PACER cases
  - check_ofac_sanctions         : check the OFAC Consolidated Sanctions List
  - search_state_ucc_filings     : search EDGAR UCC disclosures + state SOS links
"""

import asyncio
import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

_log = logging.getLogger(__name__)

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("legal-db")

_CL_BASE = "https://www.courtlistener.com/api/rest/v4"
_OFAC_URL = "https://www.treasury.gov/ofac/downloads/consolidated/consolidated.xml"
_EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"

_EDGAR_HEADERS = {
    "User-Agent": "DealBreaker-AI research@dealbreaker.ai",
    "Accept": "application/json",
}


def _cl_headers() -> dict:
    token = os.getenv("COURTLISTENER_API_KEY", "")
    if not token:
        raise RuntimeError(
            "COURTLISTENER_API_KEY environment variable is not set. "
            "Register for a free token at https://www.courtlistener.com/sign-in/ "
            "and add it to your .env file."
        )
    return {
        "Authorization": f"Token {token}",
        "User-Agent": "DealBreaker-AI research@dealbreaker.ai",
        "Accept": "application/json",
    }


# ---------------------------------------------------------------------------
# OFAC XML — module-level cache (file is ~1 MB; OFAC updates ~weekly)
# ---------------------------------------------------------------------------

_ofac_root: ET.Element | None = None
_ofac_loaded_at: float = 0.0
_OFAC_TTL = 86_400.0  # 24 hours


def _strip_ns(element: ET.Element) -> ET.Element:
    """Strip all XML namespace prefixes in-place (modifies the tree)."""
    for el in element.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return element


async def _load_ofac() -> ET.Element:
    global _ofac_root, _ofac_loaded_at
    now = time.monotonic()
    if _ofac_root is not None and (now - _ofac_loaded_at) <= _OFAC_TTL:
        return _ofac_root
    headers = {
        "User-Agent": "DealBreaker-AI research@dealbreaker.ai",
        "Accept": "application/xml",
    }
    try:
        async with httpx.AsyncClient(
            headers=headers, timeout=120.0, follow_redirects=True
        ) as client:
            r = await client.get(_OFAC_URL)
            r.raise_for_status()
        _ofac_root = _strip_ns(ET.fromstring(r.content))
        _ofac_loaded_at = now
    except Exception as exc:
        _log.error("Failed to fetch OFAC Consolidated Sanctions List: %s", exc)
        if _ofac_root is not None:
            return _ofac_root  # return stale cache rather than crashing
        raise RuntimeError(
            f"OFAC sanctions list unavailable and no cached data: {exc}"
        ) from exc
    return _ofac_root


def _ofac_name_score(candidate: str, query: str) -> float:
    """Return overlap fraction of query tokens found in candidate (0.0–1.0)."""
    q_tokens = set(query.upper().split())
    c_tokens = set(candidate.upper().split())
    if not q_tokens:
        return 0.0
    return len(q_tokens & c_tokens) / len(q_tokens)


# ---------------------------------------------------------------------------
# State SOS UCC portal URLs (public, no auth)
# ---------------------------------------------------------------------------

_SOS_PORTALS: dict[str, dict] = {
    "DE": {
        "url": "https://icis.corp.delaware.gov/Ecorp/EntitySearch/NameSearch.aspx",
        "notes": "Delaware Division of Corporations entity and lien search.",
    },
    "CA": {
        "url": "https://bizfileonline.sos.ca.gov/search/ucc",
        "notes": "California SOS UCC filings search.",
    },
    "NY": {
        "url": "https://www.dos.ny.gov/corps/bus_entity_search.html",
        "notes": "New York DOS business entity and UCC search.",
    },
    "TX": {
        "url": "https://www.sos.state.tx.us/ucc/index.shtml",
        "notes": "Texas SOS UCC online search.",
    },
    "FL": {
        "url": "https://dos.myflorida.com/sunbiz/search/",
        "notes": "Florida SOS Sunbiz — includes secured transaction filings.",
    },
    "IL": {
        "url": "https://www.cyberdriveillinois.com/departments/business_services/ucc/home.html",
        "notes": "Illinois SOS UCC search portal.",
    },
    "WA": {
        "url": "https://www.sos.wa.gov/corps/search.aspx",
        "notes": "Washington SOS entity and UCC search.",
    },
    "MA": {
        "url": "https://corp.sec.state.ma.us/CorpWeb/CorpSearch/CorpSearch.aspx",
        "notes": "Massachusetts Secretary of State corporate and UCC search.",
    },
}


# ---------------------------------------------------------------------------
# Tool 1 — search_federal_court_records
# ---------------------------------------------------------------------------

@mcp.tool()
async def search_federal_court_records(company_name: str) -> dict:
    """Search CourtListener for federal court cases involving a company.

    Queries the CourtListener PACER index across all federal courts.
    Requires the COURTLISTENER_API_KEY environment variable.

    Args:
        company_name: Company name to search across federal courts
                      (e.g. "Theranos Inc").

    Returns:
        dict with status, company_name, total_available, cases_returned, and
        cases list. Each case has case_name, court, date_filed, date_terminated,
        docket_number, nature_of_suit, cause, and case_url.
    """
    try:
        headers = _cl_headers()
    except RuntimeError as exc:
        return {"status": "error", "message": str(exc), "cases": []}

    params = {
        "q": company_name,
        "type": "r",            # PACER/federal records
        "order_by": "score desc",
        "page_size": 20,
    }

    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        r = await client.get(f"{_CL_BASE}/dockets/", params=params)
        if r.status_code == 401:
            return {
                "status": "error",
                "message": (
                    "CourtListener API token is invalid or expired. "
                    "Check COURTLISTENER_API_KEY in your .env file."
                ),
                "cases": [],
            }
        r.raise_for_status()
        data = r.json()

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

    return {
        "status": "success",
        "company_name": company_name,
        "total_available": data.get("count", 0),
        "cases_returned": len(cases),
        "cases": cases,
        "source": "CourtListener PACER federal court index",
    }


# ---------------------------------------------------------------------------
# Tool 2 — check_ofac_sanctions
# ---------------------------------------------------------------------------

@mcp.tool()
async def check_ofac_sanctions(entity_name: str) -> dict:
    """Check whether an entity appears on the OFAC Consolidated Sanctions List.

    Downloads the official US Treasury OFAC XML and caches it in memory for
    24 hours. Matches by primary name and all registered aliases.

    Args:
        entity_name: Company or individual name to check (e.g. "Rosneft Oil").

    Returns:
        dict with status, is_sanctioned (bool), match_count, and matches list.
        Each match has uid, name, entity_type, aliases, sanction_programs,
        and remarks.
    """
    try:
        root = await _load_ofac()
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Could not fetch OFAC Consolidated Sanctions List: {exc}",
            "is_sanctioned": False,
            "matches": [],
        }

    matches: list[dict] = []
    min_score = 0.6  # require ≥60% of query tokens to match

    for entry in root.findall("sdnEntry"):
        last = entry.findtext("lastName") or ""
        first = entry.findtext("firstName") or ""
        primary_name = f"{first} {last}".strip() if first else last

        # Collect all aliases from akaList
        aliases: list[str] = []
        aka_list = entry.find("akaList")
        if aka_list is not None:
            for aka in aka_list.findall("aka"):
                aka_last = aka.findtext("lastName") or ""
                aka_first = aka.findtext("firstName") or ""
                alias = f"{aka_first} {aka_last}".strip() if aka_first else aka_last
                if alias:
                    aliases.append(alias)

        all_names = [primary_name] + aliases
        best_score = max(_ofac_name_score(n, entity_name) for n in all_names)
        if best_score < min_score:
            continue

        programs = [
            p.text
            for p in (entry.find("programList") or [])
            if hasattr(p, "text") and p.text
        ]

        matches.append({
            "uid": entry.findtext("uid") or "",
            "name": primary_name,
            "entity_type": entry.findtext("sdnType") or "",
            "aliases": aliases,
            "sanction_programs": programs,
            "remarks": (entry.findtext("remarks") or "")[:500],
            "match_score": round(best_score, 2),
        })

    matches.sort(key=lambda m: m["match_score"], reverse=True)

    return {
        "status": "success",
        "entity_name": entity_name,
        "is_sanctioned": len(matches) > 0,
        "match_count": len(matches),
        "matches": matches[:10],
        "source": "US Treasury OFAC Consolidated Sanctions List",
        "list_url": _OFAC_URL,
    }


# ---------------------------------------------------------------------------
# Tool 3 — search_state_ucc_filings
# ---------------------------------------------------------------------------

@mcp.tool()
async def search_state_ucc_filings(company_name: str, state: str) -> dict:
    """Search for UCC lien filings against a company.

    Searches EDGAR full-text disclosures for publicly disclosed UCC liens and
    returns direct links to the relevant state Secretary of State UCC search
    portal for manual verification.

    Note: Comprehensive lien searches require state-specific paid services
    (e.g. CT Corp, CSC). This tool returns EDGAR-disclosed UCC information
    plus the appropriate SOS portal link.

    Args:
        company_name: Debtor company name (e.g. "Acme Corp").
        state: Two-letter US state code (e.g. "DE", "CA", "NY").

    Returns:
        dict with status, edgar_ucc_disclosures list, sos_search_url,
        sos_notes, and instructions for completing the state UCC search.
    """
    state_code = state.strip().upper()
    _configured_states = set(_SOS_PORTALS.keys())
    _state_not_configured = state_code not in _configured_states
    portal = _SOS_PORTALS.get(
        state_code,
        {
            "url": (
                f"https://www.google.com/search?q="
                f"{state_code}+Secretary+of+State+UCC+lien+search"
            ),
            "notes": (
                f"No configured portal for {state_code}. "
                "Use the link to locate the state SOS UCC search."
            ),
        },
    )

    # Search EDGAR for public disclosures of UCC/lien filings mentioning the company.
    five_years_ago = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")
    params = {
        "q": f'"{company_name}" "UCC" OR "lien" OR "security interest" OR "pledge"',
        "dateRange": "custom",
        "startdt": five_years_ago,
        "enddt": today,
    }

    edgar_refs: list[dict] = []
    try:
        await asyncio.sleep(0.11)
        async with httpx.AsyncClient(headers=_EDGAR_HEADERS, timeout=30.0) as client:
            r = await client.get(_EDGAR_SEARCH, params=params)
            r.raise_for_status()
        for hit in r.json().get("hits", {}).get("hits", [])[:10]:
            src = hit.get("_source", {})
            ciks = src.get("ciks", [])
            adsh = src.get("adsh", "")
            cik_int = str(int(ciks[0])) if ciks else ""
            display = src.get("display_names", [""])
            entity_name = display[0].split("(CIK")[0].strip() if display else ""
            edgar_refs.append({
                "form_type": (src.get("root_forms") or [src.get("form", "")])[0],
                "entity_name": entity_name,
                "file_date": src.get("file_date", ""),
                "accession_number": adsh,
                "filing_index_url": (
                    f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{adsh.replace('-', '')}/"
                    if cik_int and adsh
                    else ""
                ),
            })
    except Exception:
        pass  # EDGAR search is best-effort; SOS link is always returned

    configured_states_list = sorted(_configured_states)
    unconfigured_warning = (
        f"WARNING: '{state_code}' does not have a pre-configured SOS portal. "
        f"Only these states have direct portal links: {', '.join(configured_states_list)}. "
        "A Google search URL has been substituted — verify the correct SOS portal manually."
        if _state_not_configured
        else None
    )

    return {
        "status": "success",
        "company_name": company_name,
        "state": state_code,
        "sos_search_url": portal["url"],
        "sos_notes": portal["notes"],
        "unconfigured_state_warning": unconfigured_warning,
        "edgar_ucc_disclosures": edgar_refs,
        "instructions": (
            f"1. Visit {portal['url']} and search for '{company_name}' to find "
            f"UCC-1 financing statements filed in {state_code}. "
            "2. Review edgar_ucc_disclosures for publicly reported liens in SEC filings. "
            "3. For a full 50-state lien search, use a national UCC search firm "
            "(CT Corp, CSC, or equivalent)."
        ),
    }


if __name__ == "__main__":
    mcp.run()
