"""
OllamaAid - Trends scraping
Fetch model trend data from ollama.com/search.
"""

import logging
import re
from typing import List, Optional

from .models import ToolResult, TrendData

log = logging.getLogger(__name__)

_EXCLUDED_NAMES = frozenset({
    "models", "turbo", "signin", "download", "blog", "library",
    "search", "docs", "api", "pricing", "about", "contact",
    "login", "register", "dashboard", "settings", "profile",
    "help", "support", "terms", "privacy", "home", "index",
})

_TAG_KEYWORDS = ("tools", "vision", "embedding", "thinking", "chat", "code")

_DEFAULT_LIMIT = 100


def _parse_number(num_str: str) -> float:
    """Convert K/M/B suffixed string to a float."""
    if not num_str:
        return 0.0
    s = num_str.upper().replace(",", "").strip()
    try:
        if "K" in s:
            return float(s.replace("K", "")) * 1_000
        if "M" in s:
            return float(s.replace("M", "")) * 1_000_000
        if "B" in s:
            return float(s.replace("B", "")) * 1_000_000_000
        return float(s)
    except ValueError:
        return 0.0


def _parse_time_to_days(text: str) -> int:
    """Convert time text such as '3 months ago' to approximate days."""
    if not text:
        return 999_999
    t = text.lower().strip()
    patterns = [
        (r"(\d+)\s*days?\s*ago", 1),
        (r"(\d+)\s*weeks?\s*ago", 7),
        (r"(\d+)\s*months?\s*ago", 30),
        (r"(\d+)\s*years?\s*ago", 365),
        (r"(\d+)\s*天前", 1),
        (r"(\d+)\s*周前", 7),
        (r"(\d+)\s*月前", 30),
        (r"(\d+)\s*年前", 365),
    ]
    for pat, mult in patterns:
        m = re.search(pat, t)
        if m:
            return int(m.group(1)) * mult
    if "version" in t or "版本" in t:
        return 0
    return 999_999


def _parse_model_card(card, seen_names: set) -> Optional[TrendData]:
    """Parse a single model card element and return TrendData or None."""
    try:
        href = card.get("href", "")
        if not href.startswith("/library/"):
            return None

        name = href.replace("/library/", "").strip("/")
        if not name or len(name) < 2:
            return None
        if name.lower() in _EXCLUDED_NAMES:
            return None
        if name in seen_names:
            return None

        card_text = card.get_text()

        pulls = 0.0
        for pat in (
            r"([\d,]+(?:\.\d+)?[KMB]?)\s*[Pp]ulls",
            r"([\d,]+(?:\.\d+)?[KMB]?)\s*下载",
            r"Downloads[:\s]*([\d,]+(?:\.\d+)?[KMB]?)",
        ):
            m = re.search(pat, card_text, re.IGNORECASE)
            if m:
                pulls = _parse_number(m.group(1))
                break

        param_values: list[float] = []
        param_details: list[str] = []
        for pat in (r"(\d+(?:\.\d+)?)[bB](?!\w)", r"(\d+(?:\.\d+)?)[mM](?![a-zA-Z]|\s*[Pp]ulls|\s*[Dd]ownload)"):
            for m in re.finditer(pat, card_text):
                orig = m.group(1)
                val = float(orig)
                if "m" in pat.lower():
                    param_details.append(f"{orig}M")
                    val /= 1000
                else:
                    param_details.append(f"{orig}B")
                if val > 0.01:
                    param_values.append(val)

        min_p = min(param_values) if param_values else 0.0
        max_p = max(param_values) if param_values else 0.0
        unique_details = sorted(set(param_details), key=lambda s: float(re.match(r"[\d.]+", s).group()) if re.match(r"[\d.]+", s) else 0)
        pd_str = ", ".join(unique_details) if unique_details else ""

        text_lower = card_text.lower()
        tags = [kw for kw in _TAG_KEYWORDS if kw in text_lower]

        updated = ""
        time_pats = [
            r"(\d+\s+(?:month|week|day|year)s?\s+ago)",
            r"(to\s+version\s+[\d.]+[^\s]*)",
            r"(\d+\s+(?:周|月|天|年)前)",
        ]
        for tp in time_pats:
            tm = re.search(tp, card_text, re.IGNORECASE)
            if tm:
                updated = re.sub(r"\s+", " ", tm.group(1).strip())
                break

        desc_elem = card.find("p")
        desc = ""
        if desc_elem:
            desc = desc_elem.get_text(strip=True)[:100]
            if len(desc) == 100:
                desc += "..."

        if not (pulls > 0 or max_p > 0):
            return None

        return TrendData(
            name=name,
            pulls=pulls,
            min_params=min_p,
            max_params=max_p,
            param_details=pd_str,
            tags=tags,
            description=desc,
            updated=updated,
            url=f"https://ollama.com{href}",
        )
    except Exception:
        return None


def fetch_trends(*, limit: int = _DEFAULT_LIMIT) -> ToolResult:
    """Scrape ollama.com/search and return a list of ``TrendData``.
    
    Args:
        limit: Maximum number of models to return (default: 100, use -1 for all)
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as exc:
        return ToolResult(
            success=False,
            error=f"Missing dependency: {exc}. Install with: pip install requests beautifulsoup4 lxml",
        )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    results: List[TrendData] = []
    seen_names: set = set()

    urls = [
        "https://ollama.com/search",
        "https://ollama.com/search?o=newest",
    ]

    for url in urls:
        if limit > 0 and len(results) >= limit:
            break
        try:
            log.debug("Fetching %s", url)
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
        except Exception as exc:
            log.warning("Failed to fetch %s: %s", url, exc)
            continue

        soup = BeautifulSoup(resp.content, "html.parser")
        cards = soup.find_all("a", href=True)

        for card in cards:
            if limit > 0 and len(results) >= limit:
                break
            trend = _parse_model_card(card, seen_names)
            if trend:
                seen_names.add(trend.name)
                results.append(trend)

    results.sort(key=lambda x: x.pulls, reverse=True)

    if limit > 0:
        results = results[:limit]

    if not results:
        return ToolResult(success=False, error="No models found on ollama.com")
    return ToolResult(success=True, data=results, metadata={"count": len(results)})