"""
app/tools — package-level utilities.

Exports load_web_page_cached: a session-state-backed caching wrapper for web
page fetching.  Agents that want request de-duplication within a session should
import this and pass it as a tool instead of (or alongside) the ADK built-in
load_web_page.

Cache key: "_lwp_cache:<url>" stored in ADK session state.  The cached result
survives for the lifetime of the InMemorySessionService session and is discarded
when the session ends — no cross-session persistence.
"""

from __future__ import annotations

import httpx

_WEB_CACHE_PREFIX = "_lwp_cache:"
_DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DealBreakerBot/1.0)",
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
_MAX_CHARS = 50_000


async def load_web_page_cached(url: str, tool_context=None) -> dict:
    """Fetch a web page with session-level caching.

    First call for a URL fetches via httpx and stores the result in session
    state.  Subsequent calls within the same session return the stored result
    instantly — no network round-trip.

    Args:
        url:          Full URL to retrieve (must start with http:// or https://).
        tool_context: ADK ToolContext injected by the framework; provides
                      state read/write.  When None (unit-test mode) caching is
                      disabled and a fresh fetch is always made.

    Returns:
        dict with 'status' ('success' or 'error'), 'url', 'content' (str, up
        to 50,000 chars), 'content_type', 'status_code', 'truncated', and
        optionally '_from_cache' (True when returned from session state).
    """
    if not url or not url.startswith(("http://", "https://")):
        return {
            "status": "error",
            "url": url,
            "message": "URL must start with http:// or https://",
        }

    cache_key = f"{_WEB_CACHE_PREFIX}{url}"
    if tool_context is not None:
        cached = tool_context.state.get(cache_key)
        if cached is not None:
            return {**cached, "_from_cache": True}

    try:
        async with httpx.AsyncClient(
            headers=_DEFAULT_HEADERS,
            timeout=30.0,
            follow_redirects=True,
        ) as client:
            r = await client.get(url)
        r.raise_for_status()
        body = r.text
        result: dict = {
            "status": "success",
            "url": str(r.url),
            "content": body[:_MAX_CHARS],
            "content_type": r.headers.get("content-type", ""),
            "status_code": r.status_code,
            "truncated": len(body) > _MAX_CHARS,
        }
    except httpx.HTTPStatusError as exc:
        result = {
            "status": "error",
            "url": url,
            "message": f"HTTP {exc.response.status_code}: {exc}",
            "status_code": exc.response.status_code,
        }
    except httpx.HTTPError as exc:
        result = {
            "status": "error",
            "url": url,
            "message": str(exc),
        }

    if tool_context is not None and result.get("status") == "success":
        tool_context.state[cache_key] = result

    return result
