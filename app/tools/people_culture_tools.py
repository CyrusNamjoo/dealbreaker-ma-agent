"""
People and culture due diligence tools for the PeopleCultureAgent.

Five tools covering employee sentiment, executive backgrounds, controversy
screening, culture fit assessment, and review theme analysis. All data from
public sources — EDGAR proxy statements, GDELT news, and Reddit public API.

CONTRACT (applies to every tool in this module):
  - data_available=False when a required source returns no usable data.
  - Glassdoor, LinkedIn, and Blind numeric scores are NOT fetched by these
    tools (no public API exists). Tools return platform search URLs for the
    agent to access via load_web_page.
  - No characterisation of an individual is made without a cited public source.
  - Every finding is traceable to a specific URL, filing, or post.
"""

import asyncio
import os
import re
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
_EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_HEADERS = {"User-Agent": _UA, "Accept": "application/json"}
_GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
_REDDIT_HEADERS = {
    "User-Agent": f"python:dealbreaker-due-diligence:v0.1 (contact:{_CONTACT_EMAIL})",
    "Accept": "application/json",
}

# ── Employee-review theme keyword sets ───────────────────────────────────────
_REVIEW_THEMES: dict[str, frozenset] = {
    "compensation":     frozenset({"salary", "pay", "compensation", "bonus", "stock", "equity", "underpaid", "raise", "benefits", "rsu"}),
    "work_life_balance":frozenset({"work life", "hours", "overtime", "remote", "flexible", "burnout", "balance", "wlb", "vacation", "pto"}),
    "management":       frozenset({"manager", "management", "leadership", "boss", "micromanage", "toxic", "feedback", "directors", "vp"}),
    "culture":          frozenset({"culture", "toxic", "inclusive", "diversity", "values", "mission", "team", "morale", "environment"}),
    "career_growth":    frozenset({"growth", "promotion", "career", "learning", "opportunity", "title", "advance", "mentor", "training"}),
    "layoffs_stability":frozenset({"layoff", "fired", "restructuring", "rif", "unstable", "downsizing", "headcount", "cuts", "reduction"}),
    "product_direction":frozenset({"direction", "roadmap", "product", "strategy", "vision", "pivot", "chaos", "unclear", "focus"}),
}

# ── Culture fit dimensions and comparison questions ───────────────────────────
_CULTURE_DIMENSIONS: list[dict[str, Any]] = [
    {
        "dimension": "Remote / hybrid work policy",
        "target_source": "10-K Item 2 (Properties) and Job postings; DEF 14A human capital section",
        "verification_url_template": "https://efts.sec.gov/LATEST/search-index?q=%22{company}%22+%22remote%22+%22hybrid%22&forms=DEF+14A",
        "acquirer_keywords": ["remote", "hybrid", "office", "in-person", "distributed", "work from home"],
    },
    {
        "dimension": "Diversity, equity & inclusion (DEI)",
        "target_source": "DEF 14A human capital section; published diversity reports; EEOC filings",
        "verification_url_template": "https://efts.sec.gov/LATEST/search-index?q=%22{company}%22+%22diversity%22+%22inclusion%22&forms=DEF+14A",
        "acquirer_keywords": ["diversity", "inclusion", "dei", "erg", "equity"],
    },
    {
        "dimension": "Compensation philosophy",
        "target_source": "DEF 14A Compensation Discussion & Analysis (CD&A); job postings",
        "verification_url_template": "https://efts.sec.gov/LATEST/search-index?q=%22{company}%22+%22compensation+philosophy%22&forms=DEF+14A",
        "acquirer_keywords": ["equity", "bonus", "base salary", "performance", "comp"],
    },
    {
        "dimension": "Leadership style (flat vs hierarchical)",
        "target_source": "Glassdoor/Comparably culture ratings; press releases; CEO public statements",
        "verification_url_template": "https://www.comparably.com/companies/{slug}/culture",
        "acquirer_keywords": ["flat", "hierarchy", "decentralised", "autonomous", "top-down"],
    },
    {
        "dimension": "Innovation and R&D culture",
        "target_source": "10-K R&D expense disclosure; patent filing activity; engineering blog",
        "verification_url_template": "https://efts.sec.gov/LATEST/search-index?q=%22{company}%22+%22research+and+development%22&forms=10-K",
        "acquirer_keywords": ["innovation", "r&d", "engineering", "tech", "research", "patents"],
    },
    {
        "dimension": "Employee tenure and attrition",
        "target_source": "LinkedIn workforce analytics (public); Glassdoor tenure data; DEF 14A human capital",
        "verification_url_template": "https://www.linkedin.com/company/{slug}/people/",
        "acquirer_keywords": ["retention", "attrition", "turnover", "tenure", "stability"],
    },
    {
        "dimension": "Geographic concentration",
        "target_source": "10-K Item 2 (Properties); job postings geography; LinkedIn location data",
        "verification_url_template": "https://efts.sec.gov/LATEST/search-index?q=%22{company}%22+%22principal+offices%22+%22employees%22&forms=10-K",
        "acquirer_keywords": ["geography", "location", "hub", "global", "international", "offshore"],
    },
]


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _reddit_search(query: str) -> dict:
    return {
        "q": query,
        "sort": "relevance",
        "t": "year",
        "limit": "25",
        "type": "link",
    }


def _score_review_themes(posts: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {t: 0 for t in _REVIEW_THEMES}
    for post in posts:
        text = (post.get("title") or "").lower()
        words = set(re.findall(r"\b[a-z]+\b", text))
        for theme, keywords in _REVIEW_THEMES.items():
            if words & keywords or any(kw in text for kw in keywords if " " in kw):
                counts[theme] += 1
    return counts


# ── Tool 1: fetch_glassdoor_data ─────────────────────────────────────────────

async def fetch_glassdoor_data(company_name: str) -> dict:
    """Search Reddit for employee sentiment and return review platform search URLs.

    Glassdoor, Indeed, and Comparably do not provide public unauthenticated APIs.
    This tool searches Reddit — which has a free public API — for employee
    discussions as a proxy for workplace sentiment, and returns direct platform
    URLs for the agent to visit with load_web_page.

    The agent must use load_web_page on platform_urls.glassdoor_search,
    platform_urls.comparably, and platform_urls.indeed to retrieve actual
    review scores. These pages are often publicly accessible without login.

    Args:
        company_name: Company name (e.g. "Salesforce Inc").

    Returns:
        dict with data_available, reddit_employee_posts, reddit_sentiment_proxy,
        platform_urls, and source.
    """
    params = _reddit_search(
        f'"{company_name}" (glassdoor OR "work culture" OR employees OR salary OR "work life")'
    )

    await _rate_limit("https://www.reddit.com/search.json")
    try:
        async with httpx.AsyncClient(headers=_REDDIT_HEADERS, timeout=30.0) as client:
            r = await client.get("https://www.reddit.com/search.json", params=params)
        r.raise_for_status()
        children = r.json().get("data", {}).get("children", [])
    except httpx.HTTPError as exc:
        return {
            "data_available": False,
            "message": f"Reddit search failed: {exc}. Use load_web_page on platform_urls directly.",
            "reddit_employee_posts": [],
            "reddit_sentiment_proxy": None,
            "platform_urls": _review_platform_urls(company_name),
            "source": "Reddit public API + review platform URLs",
        }

    posts = []
    for child in children:
        d = child.get("data", {})
        posts.append({
            "title": d.get("title", ""),
            "subreddit": d.get("subreddit", ""),
            "score": d.get("score", 0),
            "num_comments": d.get("num_comments", 0),
            "url": f"https://www.reddit.com{d.get('permalink', '')}",
            "created_utc": d.get("created_utc", 0),
        })

    theme_counts = _score_review_themes(posts)
    top_themes = sorted([(t, c) for t, c in theme_counts.items() if c > 0],
                        key=lambda x: x[1], reverse=True)

    pos_words = frozenset({"great", "good", "love", "excellent", "best", "positive", "recommend"})
    neg_words = frozenset({"toxic", "terrible", "awful", "bad", "worst", "avoid", "quit", "fired", "layoff"})
    sentiment_signals: list[str] = []
    for p in posts:
        words = set(re.findall(r"\b[a-z]+\b", p["title"].lower()))
        if words & neg_words:
            sentiment_signals.append("negative")
        elif words & pos_words:
            sentiment_signals.append("positive")

    if sentiment_signals:
        neg_count = sentiment_signals.count("negative")
        pos_count = sentiment_signals.count("positive")
        proxy = "negative" if neg_count > pos_count * 1.5 else ("positive" if pos_count > neg_count * 1.5 else "mixed")
    else:
        proxy = None

    return {
        "data_available": len(posts) > 0,
        "company_name": company_name,
        "reddit_posts_found": len(posts),
        "reddit_employee_posts": posts,
        "reddit_sentiment_proxy": proxy,
        "top_employee_themes_from_reddit": [t for t, _ in top_themes[:4]],
        "platform_urls": _review_platform_urls(company_name),
        "note": (
            f"{len(posts)} Reddit posts found related to employee experience at '{company_name}'. "
            "reddit_sentiment_proxy is a rough keyword signal — NOT a Glassdoor score. "
            "Use load_web_page on platform_urls.glassdoor_search and platform_urls.comparably "
            "to retrieve actual numeric ratings. Glassdoor may require login; Comparably is "
            "typically accessible without authentication."
        ),
        "source": "Reddit public search API (https://www.reddit.com/search.json) + review platform URLs",
    }


def _review_platform_urls(company_name: str) -> dict[str, str]:
    encoded = company_name.replace(" ", "+")
    s = _slug(company_name)
    return {
        "glassdoor_search": f"https://www.glassdoor.com/Reviews/index.htm?action=ovrSearch&sc.keyword={encoded}",
        "comparably": f"https://www.comparably.com/companies/{s}",
        "comparably_culture": f"https://www.comparably.com/companies/{s}/culture",
        "comparably_ceo": f"https://www.comparably.com/companies/{s}/executive-team",
        "indeed_company": f"https://www.indeed.com/cmp/{s}",
        "blind_company": f"https://www.teamblind.com/company/{encoded}",
        "levels_fyi": f"https://www.levels.fyi/companies/{s}/",
        "reddit_employees": f"https://www.reddit.com/search/?q={encoded}+employees&sort=relevance&t=year",
    }


# ── Tool 2: research_executive_backgrounds ───────────────────────────────────

async def research_executive_backgrounds(company_name: str) -> dict:
    """Search EDGAR for the proxy statement (DEF 14A) containing executive biographies.

    The SEC requires public companies to disclose executive officer backgrounds,
    tenure, and prior roles in the annual proxy statement (DEF 14A). This tool
    returns the most recent proxy filing URL and structures the research framework
    for the agent to extract bios via load_web_page. Returns LinkedIn and news
    search URLs for individual executive research.

    Args:
        company_name: Company name (e.g. "Salesforce Inc").

    Returns:
        dict with data_available, proxy_filings, research_framework,
        linkedin_search_url, and source.
    """
    params = {
        "q": f'"{company_name}"',
        "forms": "DEF 14A,DEFA14A,DEF 14A/A",
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
            "message": f"EDGAR DEF 14A search failed: {exc}",
            "proxy_filings": [],
            "research_framework": _exec_research_framework(company_name),
            "source": "SEC EDGAR DEF 14A proxy statements",
        }

    filings = []
    for hit in hits[:5]:
        src = hit.get("_source", {})
        ciks = src.get("ciks", [])
        cik_int = str(int(ciks[0])).strip() if ciks else ""
        adsh = src.get("adsh", "").replace("-", "")
        display = src.get("display_names", [""])
        filings.append({
            "entity_name": (display[0] or "").split("(CIK")[0].strip(),
            "form": (src.get("root_forms") or [src.get("form", "")])[0],
            "file_date": src.get("file_date", ""),
            "accession_number": src.get("adsh", ""),
            "filing_index_url": (
                f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{adsh}/"
                if cik_int and adsh else ""
            ),
        })

    encoded = company_name.replace(" ", "+")
    s = _slug(company_name)
    return {
        "data_available": len(filings) > 0,
        "company_name": company_name,
        "proxy_filings_found": len(filings),
        "proxy_filings": filings,
        "research_framework": _exec_research_framework(company_name),
        "linkedin_company_people_url": f"https://www.linkedin.com/company/{s}/people/",
        "note": (
            f"{len(filings)} DEF 14A proxy filings found for '{company_name}'. "
            "Use load_web_page on proxy_filings[0].filing_index_url to locate the primary "
            "proxy document, then search for 'Executive Officers', 'Directors', or "
            "'Biographical Information' sections. Extract: name, title, tenure start year, "
            "prior companies listed in the bio. Then call check_executive_controversies "
            "with the extracted executive_names list."
        ),
        "source": "SEC EDGAR DEF 14A proxy statements (https://www.sec.gov/)",
    }


def _exec_research_framework(company_name: str) -> list[dict[str, str]]:
    encoded = company_name.replace(" ", "+")
    return [
        {"step": "1. Extract names", "action": "Read DEF 14A 'Executive Officers' section — list all C-suite and VP-level names and titles."},
        {"step": "2. Tenure", "action": "Record years in current role from bio. Flag any executive with < 18 months tenure (instability signal)."},
        {"step": "3. Prior outcomes", "action": "For each exec, note prior company names from bio. Check those companies' outcomes (acquisition, bankruptcy, success)."},
        {"step": "4. LinkedIn verification", "action": f"Search https://www.linkedin.com/search/results/people/?keywords={encoded}+CEO to verify public tenure claims."},
        {"step": "5. Controversy check", "action": "Pass extracted names to check_executive_controversies."},
    ]


# ── Tool 3: check_executive_controversies ────────────────────────────────────

async def check_executive_controversies(executive_names: list) -> dict:
    """Search GDELT and Reddit for public controversies involving named executives.

    Queries GDELT (free, no auth) for news articles mentioning each executive
    alongside controversy-related terms. Also searches Reddit for individual-level
    discussions. Limits to the first 5 executives to avoid rate-limiting.

    Args:
        executive_names: List of executive full names to screen
                         (e.g. ["Marc Benioff", "Amy Weaver"]).
                         Obtain from research_executive_backgrounds output.

    Returns:
        dict with data_available, results_by_executive (dict keyed by name),
        total_articles_found, and source.
    """
    if not executive_names:
        return {
            "data_available": False,
            "message": (
                "executive_names list is empty. "
                "Call research_executive_backgrounds first to extract executive names "
                "from the EDGAR DEF 14A proxy statement, then pass them here."
            ),
            "results_by_executive": {},
            "total_articles_found": 0,
            "source": "GDELT Project API + Reddit public API",
        }

    names_to_check = [str(n).strip() for n in executive_names if n][:5]
    results: dict[str, dict] = {}
    total_articles = 0

    controversy_terms = (
        "investigation OR lawsuit OR fraud OR misconduct OR allegation OR "
        "fired OR scandal OR harassment OR discrimination OR settlement"
    )

    async with httpx.AsyncClient(headers={"User-Agent": _UA}, timeout=30.0) as gdelt_client:
        async with httpx.AsyncClient(headers=_REDDIT_HEADERS, timeout=30.0) as reddit_client:
            for name in names_to_check:
                exec_result: dict[str, Any] = {
                    "name": name,
                    "gdelt_articles": [],
                    "reddit_posts": [],
                    "controversy_signals": [],
                }

                # GDELT search
                try:
                    await _rate_limit(_GDELT)
                    gdelt_params = {
                        "query": f'"{name}" ({controversy_terms}) sourcelang:english',
                        "mode": "artlist",
                        "maxrecords": "10",
                        "format": "json",
                        "timespan": "5years",
                    }
                    gr = await gdelt_client.get(_GDELT, params=gdelt_params)
                    gr.raise_for_status()
                    articles = gr.json().get("articles") or []
                    exec_result["gdelt_articles"] = [
                        {
                            "title": a.get("title", ""),
                            "url": a.get("url", ""),
                            "domain": a.get("domain", ""),
                            "date": a.get("seendate", "")[:8],
                        }
                        for a in articles if a.get("url") and a.get("title")
                    ]
                    total_articles += len(exec_result["gdelt_articles"])
                except httpx.HTTPError:
                    exec_result["gdelt_articles"] = []

                # Reddit search
                try:
                    await _rate_limit("https://www.reddit.com/search.json")
                    rr = await reddit_client.get(
                        "https://www.reddit.com/search.json",
                        params=_reddit_search(f'"{name}" (misconduct OR lawsuit OR fired OR scandal OR controversy)'),
                    )
                    rr.raise_for_status()
                    r_children = rr.json().get("data", {}).get("children", [])
                    exec_result["reddit_posts"] = [
                        {
                            "title": c["data"].get("title", ""),
                            "subreddit": c["data"].get("subreddit", ""),
                            "url": f"https://www.reddit.com{c['data'].get('permalink', '')}",
                            "score": c["data"].get("score", 0),
                        }
                        for c in r_children if c.get("data")
                    ]
                except httpx.HTTPError:
                    exec_result["reddit_posts"] = []

                # Flag strong signals
                for art in exec_result["gdelt_articles"]:
                    title_lower = art["title"].lower()
                    hits = [kw for kw in ["fraud", "lawsuit", "indictment", "arrested", "misconduct",
                                          "harassment", "discrimination", "convicted", "settlement"]
                            if kw in title_lower]
                    if hits:
                        exec_result["controversy_signals"].append({
                            "source": "GDELT",
                            "title": art["title"],
                            "url": art["url"],
                            "date": art["date"],
                            "keywords_matched": hits,
                        })

                exec_result["articles_found"] = len(exec_result["gdelt_articles"]) + len(exec_result["reddit_posts"])
                results[name] = exec_result

    any_found = any(v["articles_found"] > 0 for v in results.values())
    any_signals = any(v["controversy_signals"] for v in results.values())

    return {
        "data_available": any_found,
        "executives_screened": len(names_to_check),
        "total_articles_found": total_articles,
        "controversy_signals_found": any_signals,
        "results_by_executive": results,
        "note": (
            "Use load_web_page on each controversy_signals[].url to read the full article "
            "before reporting a controversy as confirmed. GDELT may surface articles where "
            "the executive is mentioned alongside a third party — verify the executive is "
            "the subject before creating a Finding."
        ),
        "source": "GDELT Project API (https://api.gdeltproject.org/) + Reddit public API",
    }


# ── Tool 4: assess_culture_fit ───────────────────────────────────────────────

async def assess_culture_fit(target_company: str, acquirer_description: str) -> dict:
    """Search EDGAR for human capital disclosures and generate a culture fit framework.

    Retrieves the target's most recent DEF 14A and 10-K human capital disclosures
    from EDGAR to surface workforce size, diversity commitments, and stated values.
    Generates a structured set of culture comparison dimensions tailored to keywords
    identified in the acquirer_description. Each dimension includes a specific
    public data source for the agent to verify.

    Args:
        target_company: Target company name (e.g. "Salesforce Inc").
        acquirer_description: Plain-English description of the acquirer's culture,
                              work model, and integration priorities
                              (e.g. "remote-first SaaS acquirer with strong DEI focus").

    Returns:
        dict with data_available, edgar_human_capital_filings,
        culture_dimensions (prioritised by acquirer relevance), and source.
    """
    params = {
        "q": f'"{target_company}" "human capital" OR "culture" OR "diversity" OR "employees"',
        "forms": "DEF 14A,10-K",
        "dateRange": "custom",
        "startdt": "2020-01-01",
    }

    await _rate_limit(_EDGAR_SEARCH)
    try:
        async with httpx.AsyncClient(headers=_EDGAR_HEADERS, timeout=30.0) as client:
            r = await client.get(_EDGAR_SEARCH, params=params)
        r.raise_for_status()
        hits = r.json().get("hits", {}).get("hits", [])
    except httpx.HTTPError as exc:
        hits = []

    filings = []
    for hit in hits[:6]:
        src = hit.get("_source", {})
        ciks = src.get("ciks", [])
        cik_int = str(int(ciks[0])).strip() if ciks else ""
        adsh = src.get("adsh", "").replace("-", "")
        display = src.get("display_names", [""])
        filings.append({
            "entity_name": (display[0] or "").split("(CIK")[0].strip(),
            "form": (src.get("root_forms") or [src.get("form", "")])[0],
            "file_date": src.get("file_date", ""),
            "filing_index_url": (
                f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{adsh}/"
                if cik_int and adsh else ""
            ),
        })

    # Prioritise culture dimensions relevant to the acquirer
    desc_lower = acquirer_description.lower()
    s = _slug(target_company)
    comp = target_company.replace(" ", "%20")

    prioritised: list[dict] = []
    remainder: list[dict] = []
    for dim in _CULTURE_DIMENSIONS:
        url = dim["verification_url_template"].format(company=comp, slug=s)
        entry = {**dim, "verification_url": url}
        entry.pop("verification_url_template", None)
        entry.pop("acquirer_keywords", None)
        if any(kw in desc_lower for kw in dim["acquirer_keywords"]):
            prioritised.append(entry)
        else:
            remainder.append(entry)

    ordered_dimensions = prioritised + remainder

    return {
        "data_available": True,
        "target_company": target_company,
        "acquirer_description_summary": acquirer_description[:200],
        "edgar_human_capital_filings_found": len(filings),
        "edgar_human_capital_filings": filings,
        "culture_dimensions": ordered_dimensions,
        "priority_dimensions": [d["dimension"] for d in prioritised],
        "note": (
            f"Culture dimensions are ordered by relevance to the acquirer description. "
            f"Priority dimensions (most relevant to acquirer): {[d['dimension'] for d in prioritised] or 'none identified — use full list'}. "
            "Use load_web_page on each verification_url and on edgar_human_capital_filings "
            "filing_index_url entries to gather evidence for each dimension. "
            "Do not assert culture compatibility without citing a specific public source."
        ),
        "source": "SEC EDGAR DEF 14A / 10-K human capital disclosures + structured assessment framework",
    }


# ── Tool 5: analyze_employee_review_themes ───────────────────────────────────

async def analyze_employee_review_themes(company_name: str) -> dict:
    """Search Reddit for employee experience discussions and extract recurring themes.

    Queries Reddit's public search API using employee-focused terms and applies
    keyword-based theme extraction to post titles. Returns review platform search
    URLs (Glassdoor, Indeed, Blind, Comparably) for supplementary research via
    load_web_page. Actual platform review scores must be retrieved separately.

    Args:
        company_name: Company name (e.g. "Salesforce Inc").

    Returns:
        dict with data_available, reddit_posts, theme_breakdown, top_themes,
        negative_signals, review_platform_urls, and source.
    """
    params = _reddit_search(
        f'"{company_name}" (employees OR "work culture" OR salary OR management '
        f'OR "work life" OR layoffs OR glassdoor OR toxic OR "great place")'
    )

    await _rate_limit("https://www.reddit.com/search.json")
    try:
        async with httpx.AsyncClient(headers=_REDDIT_HEADERS, timeout=30.0) as client:
            r = await client.get("https://www.reddit.com/search.json", params=params)
        r.raise_for_status()
        children = r.json().get("data", {}).get("children", [])
    except httpx.HTTPError as exc:
        return {
            "data_available": False,
            "message": f"Reddit search failed: {exc}. Use load_web_page on review_platform_urls directly.",
            "reddit_posts": [],
            "theme_breakdown": {},
            "top_themes": [],
            "negative_signals": [],
            "review_platform_urls": _review_platform_urls(company_name),
            "source": "Reddit public API + review platform URLs",
        }

    posts = []
    for child in children:
        d = child.get("data", {})
        posts.append({
            "title": d.get("title", ""),
            "subreddit": d.get("subreddit", ""),
            "score": d.get("score", 0),
            "num_comments": d.get("num_comments", 0),
            "url": f"https://www.reddit.com{d.get('permalink', '')}",
            "created_utc": d.get("created_utc", 0),
        })

    theme_counts = _score_review_themes(posts)
    top_themes = sorted(
        [(t, c) for t, c in theme_counts.items() if c > 0],
        key=lambda x: x[1], reverse=True,
    )

    # Identify high-engagement negative posts
    negative_kws = frozenset({"toxic", "terrible", "awful", "layoff", "fired", "avoid", "worst",
                               "burnout", "hostile", "discrimination", "harassment", "quit"})
    negative_signals = [
        {
            "title": p["title"],
            "subreddit": p["subreddit"],
            "score": p["score"],
            "num_comments": p["num_comments"],
            "url": p["url"],
            "matched_keywords": sorted(
                set(re.findall(r"\b[a-z]+\b", p["title"].lower())) & negative_kws
            ),
        }
        for p in posts
        if set(re.findall(r"\b[a-z]+\b", p["title"].lower())) & negative_kws
    ]
    negative_signals.sort(key=lambda x: x["score"], reverse=True)

    return {
        "data_available": len(posts) > 0,
        "company_name": company_name,
        "reddit_posts_found": len(posts),
        "reddit_posts": posts,
        "theme_breakdown": {t: c for t, c in top_themes},
        "top_themes": [t for t, _ in top_themes[:4]],
        "negative_signals": negative_signals[:5],
        "review_platform_urls": _review_platform_urls(company_name),
        "note": (
            f"{len(posts)} Reddit posts found about employee experience at '{company_name}'. "
            "theme_breakdown reflects keyword frequency in post titles only — not verified "
            "review platform data. Use load_web_page on review_platform_urls.comparably and "
            "review_platform_urls.glassdoor_search to retrieve actual review scores and "
            "the most-cited pros/cons. Use load_web_page on high-score negative_signals "
            "post urls to read comment threads for detail."
        ),
        "source": "Reddit public search API (https://www.reddit.com/search.json) + review platform URLs",
    }
