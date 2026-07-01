"""
app/tools — package-level utilities.

Exports fetch_url: an httpx-based URL fetcher with SEC-compliant User-Agent,
redirect following, and a data_available/content dict return format.

Contact email is read from CONTACT_EMAIL in app/.env at import time.
User-Agent format: "DealBreaker/1.0 <contact_email>" per SEC EDGAR requirements.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

_CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "").strip()
_USER_AGENT = f"DealBreaker/1.0 {_CONTACT_EMAIL}" if _CONTACT_EMAIL else "DealBreaker/1.0"
_MAX_CHARS = 8_000


async def fetch_url(url: str) -> dict:
    """Fetch a public URL and return its text content.

    Uses httpx with redirect following, a 30-second timeout, and an SEC-compliant
    User-Agent derived from CONTACT_EMAIL in app/.env. On success, returns the raw
    response text (up to 8,000 characters). On failure, returns data_available=False
    with an error description so callers can fall back gracefully.

    Args:
        url: Full URL to retrieve. Must start with http:// or https://.

    Returns:
        dict with keys:
          data_available (bool)  — True when content was retrieved successfully.
          content        (str)   — Response text, truncated to 8,000 chars.
          url            (str)   — Final URL after any redirects.
          status_code    (int)   — HTTP status code; 0 on connection/timeout error.
          truncated      (bool)  — True when response exceeded 8,000 chars.
          error          (str)   — Present only when data_available=False.
    """
    if not url or not url.startswith(("http://", "https://")):
        return {
            "data_available": False,
            "content": "",
            "url": url,
            "status_code": 0,
            "truncated": False,
            "error": "URL must start with http:// or https://",
        }

    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/json,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    try:
        async with httpx.AsyncClient(
            headers=headers,
            timeout=30.0,
            follow_redirects=True,
        ) as client:
            r = await client.get(url)
        r.raise_for_status()
        body = r.text
        return {
            "data_available": True,
            "content": body[:_MAX_CHARS],
            "url": str(r.url),
            "status_code": r.status_code,
            "truncated": len(body) > _MAX_CHARS,
        }
    except httpx.HTTPStatusError as exc:
        return {
            "data_available": False,
            "content": "",
            "url": url,
            "status_code": exc.response.status_code,
            "truncated": False,
            "error": f"HTTP {exc.response.status_code}: {exc.response.reason_phrase}",
        }
    except httpx.TimeoutException:
        return {
            "data_available": False,
            "content": "",
            "url": url,
            "status_code": 0,
            "truncated": False,
            "error": "Request timed out after 30 seconds",
        }
    except httpx.HTTPError as exc:
        return {
            "data_available": False,
            "content": "",
            "url": url,
            "status_code": 0,
            "truncated": False,
            "error": str(exc),
        }
