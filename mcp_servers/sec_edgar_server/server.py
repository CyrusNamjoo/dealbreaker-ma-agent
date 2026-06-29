"""
SEC EDGAR MCP server.

Exposes three tools to ADK agents:
  - search_company_filings   : find filings by company name via submissions API
  - get_filing_document      : fetch and strip the text of a filing document URL
  - search_enforcement_actions: search for SEC enforcement actions against a company
"""

import asyncio
import re
import time
from datetime import datetime, timedelta

import httpx
from mcp.server.fastmcp import FastMCP
from rapidfuzz import fuzz, process as fuzz_process

mcp = FastMCP("sec-edgar")

# EDGAR requires an identifiable User-Agent on every request.
# THIS MUST BE A REAL EMAIL — EDGAR blocks requests with uncontactable addresses.
# Replace YOUR_EMAIL@EXAMPLE.COM before running in production.
_HEADERS = {
    "User-Agent": "DealBreaker-AI sirousnamjoo@gmail.com",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json",
}

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_BASE = "https://data.sec.gov/submissions"
_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
_SEARCH_BASE = "https://efts.sec.gov/LATEST/search-index"

# Module-level ticker cache — refreshed every hour.
_tickers: dict | None = None
_tickers_loaded_at: float = 0.0
_TICKER_TTL = 3600.0
_tickers_lock = asyncio.Lock()


async def _load_tickers() -> dict:
    global _tickers, _tickers_loaded_at
    now = time.monotonic()
    if _tickers is not None and (now - _tickers_loaded_at) <= _TICKER_TTL:
        return _tickers
    async with _tickers_lock:
        # Re-check inside the lock — another coroutine may have populated it.
        now = time.monotonic()
        if _tickers is not None and (now - _tickers_loaded_at) <= _TICKER_TTL:
            return _tickers
        try:
            async with httpx.AsyncClient(headers=_HEADERS, timeout=30.0) as client:
                r = await client.get(_TICKERS_URL)
                r.raise_for_status()
                _tickers = r.json()
                _tickers_loaded_at = now
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error(
                "Failed to fetch EDGAR ticker registry: %s", exc
            )
            if _tickers is not None:
                return _tickers  # return stale cache rather than crashing
            raise RuntimeError(
                f"EDGAR ticker registry unavailable and no cached data: {exc}"
            ) from exc
    return _tickers


async def _resolve_cik(company_name: str) -> tuple[str, str] | None:
    """
    Return (cik_padded_10, matched_title) for the best matching company,
    or None if not found. Uses fuzzy matching to handle partial names and
    rebranded companies (e.g. Google/Alphabet, Facebook/Meta).
    """
    tickers = await _load_tickers()
    needle = company_name.strip().upper()

    # Build upper-title -> (cik_padded, original_title) map; short-circuit on exact hit.
    title_map: dict[str, tuple[str, str]] = {}
    for entry in tickers.values():
        title: str = entry["title"]
        cik_padded = str(entry["cik_str"]).zfill(10)
        key = title.upper()
        if key == needle:
            return (cik_padded, title)
        title_map[key] = (cik_padded, title)

    # Fall back to fuzzy match (handles partial names, rebrands, abbreviations).
    result = fuzz_process.extractOne(
        needle,
        list(title_map.keys()),
        scorer=fuzz.WRatio,
        score_cutoff=80,
    )
    if result is None:
        return None
    matched_key = result[0]
    return title_map[matched_key]


def _strip_html(raw: str) -> str:
    """Remove HTML/XBRL tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Tool 1 — search_company_filings
# ---------------------------------------------------------------------------

@mcp.tool()
async def search_company_filings(company_name: str, form_types: list[str]) -> dict:
    """Search SEC EDGAR for filings by a company name.

    Looks up the company in the EDGAR ticker registry to obtain its CIK, then
    retrieves recent filings from the EDGAR submissions API.

    Args:
        company_name: Full or partial company name (e.g. "Salesforce Inc").
        form_types: SEC form types to filter by (e.g. ["10-K", "10-Q", "8-K"]).

    Returns:
        dict with status, cik, matched_company, total_found, and filings list.
        Each filing contains accession_number, form, filing_date, report_date,
        primary_document, and document_url (pass this to get_filing_document).
    """
    resolved = await _resolve_cik(company_name)
    if resolved is None:
        return {
            "status": "not_found",
            "message": (
                f"No company matching '{company_name}' found in the SEC EDGAR ticker "
                "registry. Try the exact SEC-registered name (e.g. 'APPLE INC' not 'Apple')."
            ),
            "filings": [],
        }

    cik_padded, matched_title = resolved
    cik_int = str(int(cik_padded))  # un-padded for archive URLs

    await asyncio.sleep(0.11)  # stay under EDGAR's 10 req/s limit
    async with httpx.AsyncClient(headers=_HEADERS, timeout=30.0) as client:
        r = await client.get(f"{_SUBMISSIONS_BASE}/CIK{cik_padded}.json")
        r.raise_for_status()
        data = r.json()

    recent = data.get("filings", {}).get("recent", {})
    target_forms = {f.upper() for f in form_types}

    filings: list[dict] = []
    rows = zip(
        recent.get("accessionNumber", []),
        recent.get("form", []),
        recent.get("filingDate", []),
        recent.get("reportDate", []),
        recent.get("primaryDocument", []),
        recent.get("primaryDocDescription", []),
    )
    for acc, form, filed, reported, doc, doc_desc in rows:
        if form.upper() not in target_forms:
            continue
        acc_nodash = acc.replace("-", "")
        filings.append({
            "accession_number": acc,
            "form": form,
            "filing_date": filed,
            "report_date": reported,
            "primary_document": doc,
            "primary_doc_description": doc_desc,
            "document_url": f"{_ARCHIVES_BASE}/{cik_int}/{acc_nodash}/{doc}",
        })

    return {
        "status": "success",
        "cik": cik_padded,
        "matched_company": matched_title,
        "total_found": len(filings),
        "filings": filings[:20],  # most recent 20 matching filings
    }


# ---------------------------------------------------------------------------
# Tool 2 — get_filing_document
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_filing_document(document_url: str) -> dict:
    """Fetch the text of an SEC EDGAR filing document.

    Use the document_url returned by search_company_filings. The tool strips
    HTML tags and returns the first 50,000 characters of readable text.

    Args:
        document_url: Full EDGAR Archives URL for the primary filing document,
                      e.g. https://www.sec.gov/Archives/edgar/data/789019/...

    Returns:
        dict with status, document_url, document_text (up to 50,000 chars),
        truncated (bool), and total_characters.
    """
    if not document_url:
        return {
            "status": "error",
            "message": "document_url is required and cannot be empty.",
        }
    if not document_url.startswith("https://www.sec.gov/Archives/edgar/"):
        return {
            "status": "error",
            "message": "document_url must be a www.sec.gov/Archives/edgar/ URL.",
        }

    await asyncio.sleep(0.11)
    async with httpx.AsyncClient(
        headers=_HEADERS, timeout=60.0, follow_redirects=True
    ) as client:
        r = await client.get(document_url)
        r.raise_for_status()

    clean = _strip_html(r.text)
    return {
        "status": "success",
        "document_url": document_url,
        "document_text": clean[:50_000],
        "truncated": len(clean) > 50_000,
        "total_characters": len(clean),
    }


# ---------------------------------------------------------------------------
# Tool 3 — search_enforcement_actions
# ---------------------------------------------------------------------------

@mcp.tool()
async def search_enforcement_actions(company_name: str) -> dict:
    """Search SEC EDGAR for enforcement actions against a company.

    Uses the EDGAR full-text search index (efts.sec.gov) to find Administrative
    Proceedings (form type AP) and enforcement-related filings from the past
    seven years.

    Args:
        company_name: Company name to search (e.g. "Theranos").

    Returns:
        dict with status, company_name, total_found, and enforcement_actions list.
        Each action has form_type, entity_name, file_date, accession_number,
        and filing_index_url.
    """
    seven_years_ago = (datetime.now() - timedelta(days=365 * 7)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")
    quoted = f'"{company_name}"'

    # Two passes: AP form type first, then broader keyword search.
    search_passes = [
        {
            "q": quoted,
            "forms": "AP,34-12G4",
            "dateRange": "custom",
            "startdt": seven_years_ago,
            "enddt": today,
        },
        {
            "q": f"{quoted} enforcement OR penalty OR violation OR sanctions OR fraud",
            "dateRange": "custom",
            "startdt": seven_years_ago,
            "enddt": today,
        },
    ]

    seen_adsh: set[str] = set()
    actions: list[dict] = []

    async with httpx.AsyncClient(headers=_HEADERS, timeout=30.0) as client:
        for params in search_passes:
            await asyncio.sleep(0.15)
            try:
                r = await client.get(_SEARCH_BASE, params=params)
                r.raise_for_status()
            except httpx.HTTPStatusError:
                continue

            for hit in r.json().get("hits", {}).get("hits", []):
                src = hit.get("_source", {})
                adsh: str = src.get("adsh", "")
                if not adsh or adsh in seen_adsh:
                    continue
                seen_adsh.add(adsh)

                ciks: list[str] = src.get("ciks", [])
                cik_int = str(int(ciks[0])) if ciks else ""
                acc_nodash = adsh.replace("-", "")

                display = src.get("display_names", [""])
                entity_name = display[0].split("(CIK")[0].strip() if display else ""

                actions.append({
                    "form_type": (src.get("root_forms") or [src.get("form", "")])[0],
                    "entity_name": entity_name,
                    "file_date": src.get("file_date", ""),
                    "accession_number": adsh,
                    "filing_index_url": (
                        f"{_ARCHIVES_BASE}/{cik_int}/{acc_nodash}/"
                        if cik_int
                        else f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&filenum={adsh}"
                    ),
                })

    return {
        "status": "success",
        "company_name": company_name,
        "total_found": len(actions),
        "enforcement_actions": actions[:20],
        "source": "SEC EDGAR full-text search — Administrative Proceedings and enforcement-related filings",
    }


if __name__ == "__main__":
    mcp.run()
