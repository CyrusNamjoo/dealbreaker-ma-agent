"""
News and media sentiment tools for the NewsSentimentAgent.

Five tools covering news discovery, sentiment analysis, ESG ratings,
regulatory press search, and social media signals. All data from public sources.

CONTRACT (applies to every tool in this module):
  - data_available=False when a required source returns no usable data.
  - Sentiment scores are derived from observable keywords in article titles
    and excerpts — never attributed to proprietary models or inferred from
    incomplete data without disclosure.
  - Every article, post, and regulatory action is cited with its source URL
    and publication date.
  - No article content is fetched inside these tools; load_web_page handles
    full-text retrieval. Tools operate on title/metadata or agent-supplied excerpts.
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
_GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
_EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_HEADERS = {"User-Agent": _UA, "Accept": "application/json"}
_REDDIT_HEADERS = {
    "User-Agent": f"python:dealbreaker-due-diligence:v0.1 (contact:{_CONTACT_EMAIL})",
    "Accept": "application/json",
}

# ── Business-news sentiment lexicon ──────────────────────────────────────────
_POS = frozenset({
    "growth", "record", "profit", "beat", "exceeded", "innovation", "partnership",
    "launch", "success", "award", "strong", "raised", "expansion", "milestone",
    "breakthrough", "leading", "profitable", "hire", "promotion", "win",
    "recovery", "increase", "accelerate", "momentum", "outperform", "upgrade",
    "renewed", "approved", "cleared", "resolved", "settled", "won",
})
_NEG = frozenset({
    "lawsuit", "fraud", "investigation", "breach", "loss", "layoff", "scandal",
    "fine", "penalty", "recall", "decline", "miss", "crisis", "bankruptcy",
    "violation", "misconduct", "settlement", "charges", "resign", "ousted",
    "discrimination", "harassment", "toxic", "hack", "cyberattack", "restatement",
    "subpoena", "indictment", "complaint", "downgrade", "shortfall", "delay",
    "warning", "dispute", "fired", "terminated", "controversy", "allegation",
    "probe", "accused", "failed", "collapse", "suspended", "revoked",
})

# ── Theme keyword sets ────────────────────────────────────────────────────────
_THEMES: dict[str, frozenset] = {
    "financial_performance": frozenset({
        "revenue", "earnings", "profit", "growth", "margin", "loss", "forecast",
        "guidance", "quarter", "annual", "ebitda", "ipo", "valuation", "funding",
    }),
    "legal_regulatory": frozenset({
        "lawsuit", "investigation", "fine", "penalty", "settlement", "compliance",
        "violation", "sec", "ftc", "doj", "regulator", "enforcement", "probe",
        "subpoena", "indictment", "sanctions", "consent", "order",
    }),
    "leadership_changes": frozenset({
        "ceo", "cfo", "cto", "founder", "executive", "board", "appointed",
        "resign", "departure", "hire", "fired", "named", "promoted", "replaced",
    }),
    "product_technology": frozenset({
        "launch", "product", "release", "feature", "ai", "platform", "technology",
        "patent", "innovation", "update", "partnership", "integration", "api",
    }),
    "esg_reputation": frozenset({
        "sustainability", "esg", "carbon", "diversity", "inclusion", "environmental",
        "backlash", "controversy", "award", "recall", "safety", "ethics",
        "greenwashing", "discrimination", "harassment",
    }),
    "market_competitive": frozenset({
        "competitor", "acquisition", "merger", "partnership", "deal",
        "market share", "competitive", "rival", "disruptor", "expansion",
    }),
}

_GDELT_TIMESPAN_MAP = [
    (14,   "14days"),
    (30,   "1month"),
    (90,   "3months"),
    (180,  "6months"),
    (365,  "1year"),
    (9999, "5years"),
]


def _gdelt_timespan(days_back: int) -> str:
    for threshold, label in _GDELT_TIMESPAN_MAP:
        if days_back <= threshold:
            return label
    return "5years"


def _score_text(text: str) -> float:
    """Return sentiment score -1.0..1.0 from business-news lexicon."""
    words = set(re.findall(r"\b[a-z]+\b", text.lower()))
    pos = len(words & _POS)
    neg = len(words & _NEG)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 3)


def _classify_score(avg: float, spread: float) -> str:
    if spread > 0.4 and avg > -0.25:
        return "mixed"
    if avg <= -0.2:
        return "negative"
    if avg >= 0.25:
        return "positive"
    return "neutral"


# ── Tool 1: search_news_coverage ─────────────────────────────────────────────

async def search_news_coverage(company_name: str, days_back: int) -> dict:
    """Search for recent news articles about a company using the GDELT Project API.

    GDELT (Global Database of Events, Language, and Tone) indexes news from
    thousands of outlets worldwide and is freely accessible without authentication.
    Returns article metadata (title, url, domain, date) but not full text.
    The agent should use load_web_page on article urls to read full content.

    Args:
        company_name: Company name to search (e.g. "Salesforce Inc").
        days_back: Number of days of news to retrieve (e.g. 90, 180, 730).
                   GDELT supports up to ~5 years; values above 1825 are capped.

    Returns:
        dict with data_available, articles (list), article_count, timespan,
        search_urls for supplementary sources, and source.
    """
    days_back = min(max(int(days_back), 1), 1825)
    timespan = _gdelt_timespan(days_back)
    encoded = company_name.replace(" ", "%20").replace('"', '%22')

    params = {
        "query": f'"{company_name}" sourcelang:english',
        "mode": "artlist",
        "maxrecords": "25",
        "format": "json",
        "timespan": timespan,
    }

    await _rate_limit(_GDELT)
    try:
        async with httpx.AsyncClient(headers={"User-Agent": _UA}, timeout=30.0) as client:
            r = await client.get(_GDELT, params=params)
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPError as exc:
        return {
            "data_available": False,
            "message": f"GDELT API request failed: {exc}",
            "articles": [],
            "article_count": 0,
            "timespan": timespan,
            "search_urls": _news_search_urls(company_name),
            "source": "GDELT Project (https://api.gdeltproject.org/)",
        }

    raw = data.get("articles") or []
    articles = [
        {
            "title": a.get("title", ""),
            "url": a.get("url", ""),
            "domain": a.get("domain", ""),
            "date": a.get("seendate", "")[:8],  # YYYYMMDD
            "source_country": a.get("sourcecountry", ""),
            "language": a.get("language", ""),
        }
        for a in raw
        if a.get("url") and a.get("title")
    ]

    return {
        "data_available": len(articles) > 0,
        "company_name": company_name,
        "article_count": len(articles),
        "timespan_searched": timespan,
        "articles": articles,
        "search_urls": _news_search_urls(company_name),
        "note": (
            f"{len(articles)} articles found via GDELT for '{company_name}' over {timespan}. "
            "Articles contain title and URL only — use load_web_page on article.url to read "
            "full content before passing excerpts to analyze_media_sentiment. "
            "Use search_urls to supplement with Google News and Bing News results."
        ),
        "source": "GDELT Project v2 Article List API (https://api.gdeltproject.org/api/v2/doc/doc)",
    }


def _news_search_urls(company_name: str) -> dict[str, str]:
    encoded = company_name.replace(" ", "+")
    return {
        "google_news": f"https://news.google.com/search?q={encoded}&hl=en-US&gl=US",
        "bing_news": f"https://www.bing.com/news/search?q={encoded}&FORM=HDRSC6",
        "sec_news": f"https://efts.sec.gov/LATEST/search-index?q=%22{encoded}%22&forms=8-K&dateRange=custom&startdt=2022-01-01",
        "reuters": f"https://www.reuters.com/search/news?blob={encoded}",
        "ap_news": f"https://apnews.com/search?q={encoded}",
    }


# ── Tool 2: analyze_media_sentiment ──────────────────────────────────────────

def analyze_media_sentiment(articles: list) -> dict:
    """Score sentiment and extract themes from news article titles and excerpts.

    Applies a curated business-news lexicon to score each article -1.0 to 1.0,
    then aggregates across all articles. Extracts recurring themes by keyword
    frequency. Works on title-only data from search_news_coverage, or on richer
    content_excerpt fields if the agent has loaded article text via load_web_page.

    Args:
        articles: List of dicts from search_news_coverage (or agent-enriched).
                  Each entry must have a 'title' key. Optional keys:
                  'content_excerpt' (str), 'date' (str), 'url' (str).

    Returns:
        dict with data_available, sentiment_score, overall_sentiment,
        theme_breakdown, flagged_articles (critical negative signals),
        and source.
    """
    if not articles:
        return {
            "data_available": False,
            "message": "articles list is empty. Call search_news_coverage first.",
            "sentiment_score": 0.0,
            "overall_sentiment": "neutral",
            "theme_breakdown": {},
            "flagged_articles": [],
            "source": "Lexicon-based sentiment analysis on article titles and excerpts",
        }

    scores: list[float] = []
    theme_counts: dict[str, int] = {t: 0 for t in _THEMES}
    flagged: list[dict] = []

    for art in articles:
        text = (art.get("title") or "") + " " + (art.get("content_excerpt") or "")
        score = _score_text(text)
        scores.append(score)

        text_lower = text.lower()
        words = set(re.findall(r"\b[a-z]+\b", text_lower))
        for theme, keywords in _THEMES.items():
            if words & keywords:
                theme_counts[theme] += 1

        # Flag articles with strong negative signals (lawsuit, fraud, bankruptcy)
        critical_hits = words & {"lawsuit", "fraud", "bankruptcy", "indictment",
                                  "subpoena", "restatement", "hack", "cyberattack",
                                  "violation", "misconduct", "discrimination", "harassment"}
        if critical_hits:
            flagged.append({
                "title": art.get("title", ""),
                "url": art.get("url", ""),
                "date": art.get("date", ""),
                "flags": sorted(critical_hits),
                "sentiment_score": score,
            })

    avg = round(sum(scores) / len(scores), 3)
    spread = round(max(scores) - min(scores), 3) if len(scores) > 1 else 0.0
    sentiment_label = _classify_score(avg, spread)

    top_themes = sorted(
        [(t, c) for t, c in theme_counts.items() if c > 0],
        key=lambda x: x[1], reverse=True,
    )

    return {
        "data_available": True,
        "articles_analysed": len(articles),
        "sentiment_score": avg,
        "overall_sentiment": sentiment_label,
        "sentiment_score_spread": spread,
        "theme_breakdown": {t: c for t, c in top_themes},
        "top_themes": [t for t, _ in top_themes[:4]],
        "flagged_articles": flagged[:10],
        "flagged_count": len(flagged),
        "note": (
            "Scores derived from a business-news keyword lexicon applied to article titles "
            "and any content_excerpt fields provided. Title-only analysis has lower precision; "
            "enrich articles with content_excerpt via load_web_page for higher-confidence scoring."
        ),
        "source": "Lexicon-based sentiment analysis — titles and excerpts only; no ML model inference",
    }


# ── Tool 3: check_esg_ratings ────────────────────────────────────────────────

async def check_esg_ratings(company_name: str) -> dict:
    """Search for publicly available ESG disclosures and sustainability filings.

    Searches SEC EDGAR for climate, ESG, and sustainability disclosures in
    10-K and DEF 14A (proxy) filings. Returns direct links to public ESG
    databases (CDP, B Corp, GRI, SASB) for supplementary research.
    Paid ESG rating services (Sustainalytics, MSCI ESG) are not accessed.

    Args:
        company_name: Company name (e.g. "Salesforce Inc").

    Returns:
        dict with data_available, edgar_esg_filings, public_esg_database_urls,
        b_corp_search_url, and source.
    """
    params = {
        "q": f'"{company_name}" "sustainability" OR "ESG" OR "climate" OR "carbon"',
        "forms": "10-K,DEF 14A,10-K/A",
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
        return {
            "data_available": False,
            "message": f"EDGAR search failed: {exc}",
            "edgar_esg_filings": [],
            "public_esg_database_urls": _esg_urls(company_name),
            "source": "SEC EDGAR (https://efts.sec.gov/LATEST/search-index)",
        }

    filings = []
    for hit in hits[:8]:
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

    encoded = company_name.replace(" ", "+")
    return {
        "data_available": len(filings) > 0,
        "company_name": company_name,
        "edgar_esg_filings_found": len(filings),
        "edgar_esg_filings": filings,
        "public_esg_database_urls": _esg_urls(company_name),
        "note": (
            f"{len(filings)} EDGAR filings found mentioning ESG/sustainability for "
            f"'{company_name}'. Use load_web_page on filing_index_url to read the "
            "sustainability or ESG section. Use public_esg_database_urls to check "
            "third-party ESG ratings — note these are public summaries only; paid "
            "detailed reports (Sustainalytics, MSCI ESG) are not accessed."
        ),
        "source": "SEC EDGAR full-text search + public ESG database links",
    }


def _esg_urls(company_name: str) -> dict[str, str]:
    encoded = company_name.replace(" ", "+")
    slug = company_name.lower().replace(" ", "-").replace(",", "").replace(".", "")
    return {
        "cdp_search": f"https://www.cdp.net/en/responses?utf8=%E2%9C%93&queries%5Bname%5D={encoded}",
        "b_corp_search": f"https://www.bcorporation.net/en-us/find-a-b-corp/?query={encoded}",
        "gri_search": f"https://database.globalreporting.org/search/?q={encoded}",
        "sasb_standards": "https://www.sasb.org/standards/",
        "sec_climate_search": f"https://efts.sec.gov/LATEST/search-index?q=%22{encoded}%22+%22climate%22&forms=10-K",
        "sustainalytics_summary": f"https://www.sustainalytics.com/esg-rating/{slug}",
        "msci_esg_summary": f"https://www.msci.com/our-solutions/esg-investing/esg-ratings-climate-search-tool",
    }


# ── Tool 4: search_regulatory_press ──────────────────────────────────────────

async def search_regulatory_press(company_name: str) -> dict:
    """Search SEC EDGAR for enforcement actions and return regulator press release URLs.

    Searches EDGAR for SEC administrative proceedings (AP), litigation releases
    (LR), and other enforcement forms. Returns direct search URLs for FTC, DOJ,
    CFPB, CFTC, and EU Commission enforcement databases.

    Args:
        company_name: Company name to search (e.g. "Theranos Inc").

    Returns:
        dict with data_available, sec_enforcement_filings, regulator_search_urls,
        and source.
    """
    params = {
        "q": f'"{company_name}"',
        "forms": "AP,LR,34-12G4,IC-34,IA-34",
        "dateRange": "custom",
        "startdt": "2019-01-01",
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
            "message": f"EDGAR enforcement search failed: {exc}",
            "sec_enforcement_filings": [],
            "regulator_search_urls": _regulator_urls(company_name),
            "source": "SEC EDGAR enforcement search (https://efts.sec.gov/LATEST/search-index)",
        }

    filings = []
    for hit in hits[:10]:
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
            "filing_url": (
                f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{adsh}/"
                if cik_int and adsh else ""
            ),
        })

    return {
        "data_available": True,
        "company_name": company_name,
        "sec_enforcement_filings_found": len(filings),
        "sec_enforcement_filings": filings,
        "regulator_search_urls": _regulator_urls(company_name),
        "note": (
            f"{len(filings)} SEC enforcement-type filings found for '{company_name}'. "
            "Use load_web_page on filing_url to read each action. "
            "Also use load_web_page on each URL in regulator_search_urls to search FTC, "
            "DOJ, CFPB, and EU enforcement databases."
        ),
        "source": "SEC EDGAR enforcement search + regulator press release database links",
    }


def _regulator_urls(company_name: str) -> dict[str, str]:
    encoded = company_name.replace(" ", "+")
    q = company_name.replace(" ", "%20")
    return {
        "sec_lit_releases": f"https://www.sec.gov/litigation/litreleases.htm",
        "sec_admin_proceedings": f"https://www.sec.gov/litigation/admin.shtml",
        "sec_enforcement_search": f"https://efts.sec.gov/LATEST/search-index?q=%22{q}%22&forms=AP,LR",
        "ftc_actions": f"https://www.ftc.gov/enforcement/cases-proceedings?search_api_fulltext={encoded}",
        "doj_press": f"https://www.justice.gov/news?keys={encoded}&items_per_page=10",
        "cfpb_actions": f"https://www.consumerfinance.gov/enforcement/actions/?search_term={encoded}",
        "cftc_actions": f"https://www.cftc.gov/LawRegulation/Enforcement/enforcementactions.html",
        "eu_commission_cases": f"https://ec.europa.eu/competition/elojade/isef/case_details.cfm",
        "oig_exclusions": f"https://oig.hhs.gov/exclusions/exclusions_list.asp",
    }


# ── Tool 5: scan_social_media_signals ────────────────────────────────────────

async def scan_social_media_signals(company_name: str) -> dict:
    """Search Reddit for public mentions and return social media search URLs.

    Queries Reddit's public search API for recent posts mentioning the company.
    Returns direct search URLs for Twitter/X, LinkedIn, and Blind for the agent
    to access with load_web_page. Reddit is the only platform with a free,
    unauthenticated public search API; other platforms require authentication.

    Args:
        company_name: Company name (e.g. "Salesforce Inc").

    Returns:
        dict with data_available, reddit_posts, subreddit_breakdown,
        social_search_urls, and source.
    """
    reddit_params = {
        "q": company_name,
        "sort": "relevance",
        "t": "year",
        "limit": "25",
        "type": "link",
    }

    await _rate_limit("https://www.reddit.com/search.json")
    try:
        async with httpx.AsyncClient(headers=_REDDIT_HEADERS, timeout=30.0) as client:
            r = await client.get("https://www.reddit.com/search.json", params=reddit_params)
        r.raise_for_status()
        children = r.json().get("data", {}).get("children", [])
    except httpx.HTTPError as exc:
        return {
            "data_available": False,
            "message": f"Reddit search failed: {exc}",
            "reddit_posts": [],
            "subreddit_breakdown": {},
            "social_search_urls": _social_urls(company_name),
            "source": "Reddit public search API + social platform search URLs",
        }

    posts = []
    subreddit_counts: dict[str, int] = {}
    for child in children:
        d = child.get("data", {})
        subreddit = d.get("subreddit", "")
        subreddit_counts[subreddit] = subreddit_counts.get(subreddit, 0) + 1
        posts.append({
            "title": d.get("title", ""),
            "subreddit": subreddit,
            "score": d.get("score", 0),
            "num_comments": d.get("num_comments", 0),
            "created_utc": d.get("created_utc", 0),
            "url": f"https://www.reddit.com{d.get('permalink', '')}",
            "domain": d.get("domain", ""),
        })

    # Quick sentiment pass on post titles
    title_scores = [_score_text(p["title"]) for p in posts if p["title"]]
    avg_reddit_sentiment = round(sum(title_scores) / len(title_scores), 3) if title_scores else 0.0

    return {
        "data_available": len(posts) > 0,
        "company_name": company_name,
        "reddit_posts_found": len(posts),
        "reddit_posts": posts,
        "subreddit_breakdown": dict(sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True)),
        "reddit_title_sentiment_avg": avg_reddit_sentiment,
        "social_search_urls": _social_urls(company_name),
        "note": (
            f"{len(posts)} Reddit posts found for '{company_name}'. "
            "reddit_title_sentiment_avg is a rough lexicon signal from post titles only. "
            "Use load_web_page on individual post urls for comment thread analysis. "
            "Use social_search_urls for Twitter/X, LinkedIn, and Blind — these platforms "
            "require the agent to use load_web_page as they have no free public API."
        ),
        "source": "Reddit public search API (https://www.reddit.com/search.json) + social platform search URLs",
    }


def _social_urls(company_name: str) -> dict[str, str]:
    encoded = company_name.replace(" ", "+")
    slug = company_name.lower().replace(" ", "-").replace(",", "").replace(".", "")
    return {
        "twitter_x": f"https://twitter.com/search?q=%22{encoded}%22&src=typed_query&f=top",
        "linkedin_company": f"https://www.linkedin.com/company/{slug}/",
        "linkedin_posts": f"https://www.linkedin.com/search/results/content/?keywords={encoded}",
        "blind": f"https://www.teamblind.com/search/{encoded}",
        "reddit_company": f"https://www.reddit.com/search/?q={encoded}&sort=relevance&t=year",
        "youtube_news": f"https://www.youtube.com/results?search_query={encoded}+news&sp=EgIIAg%3D%3D",
    }
