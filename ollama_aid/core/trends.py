"""
OllamaAid - Trends scraping
Fetch model trend data from ollama.com/search.
"""

import logging
import re
from typing import List

from .models import ToolResult, TrendData

log = logging.getLogger(__name__)

_EXCLUDED_NAMES = frozenset({
    "models", "turbo", "signin", "download", "blog", "library",
    "search", "docs", "api", "pricing", "about", "contact",
    "login", "register", "dashboard", "settings", "profile",
    "help", "support", "terms", "privacy", "home", "index",
})

_TAG_KEYWORDS = ("tools", "vision", "embedding", "thinking", "chat", "code")


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


def fetch_trends() -> ToolResult:
    """Scrape ollama.com/search and return a list of ``TrendData``."""
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
        )
    }

    try:
        resp = requests.get("https://ollama.com/search", headers=headers, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        return ToolResult(success=False, error=f"Network error: {exc}")

    soup = BeautifulSoup(resp.content, "html.parser")
    cards = soup.find_all("a", href=True)
    results: List[TrendData] = []

    for card in cards:
        try:
            href = card.get("href", "")
            if not href.startswith("/library/") and not re.match(r"^/[^/]+$", href):
                continue

            name_elem = card.find("h3") or card.find("h2")
            name = name_elem.get_text(strip=True) if name_elem else href.strip("/").split("/")[-1]
            if not name or len(name) < 3:
                continue
            if name.lower() in _EXCLUDED_NAMES or name.lower() in ("new", "hot", "top", "best", "latest"):
                continue

            card_text = card.get_text()

            # Pulls
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

            # Parameters
            param_values: list[float] = []
            param_details: list[str] = []
            for pat in (r"(\d+(?:\.\d+)?)[bB](?!\w)", r"(\d+(?:\.\d+)?)[mM](?!\w)"):
                for m in re.finditer(pat, card_text):
                    val = float(m.group(1))
                    if "m" in pat.lower():
                        param_details.append(f"{val}M")
                        val /= 1000
                    else:
                        param_details.append(f"{val}B")
                    if val > 0.01:
                        param_values.append(val)

            min_p = min(param_values) if param_values else 0.0
            max_p = max(param_values) if param_values else 0.0
            pd_str = ", ".join(sorted(set(param_details))) if param_details else ""

            # Tags
            text_lower = card_text.lower()
            tags = [kw for kw in _TAG_KEYWORDS if kw in text_lower]

            # Update time
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

            # Description
            desc_elem = card.find("p")
            desc = ""
            if desc_elem:
                desc = desc_elem.get_text(strip=True)[:100]
                if len(desc) == 100:
                    desc += "..."

            if not (pulls > 0 or max_p > 0):
                continue

            results.append(TrendData(
                name=name,
                pulls=pulls,
                min_params=min_p,
                max_params=max_p,
                param_details=pd_str,
                tags=tags,
                description=desc,
                updated=updated,
                url=f"https://ollama.com{href}",
            ))
        except Exception:
            continue

    if not results:
        return ToolResult(success=False, error="No models found on ollama.com")
    return ToolResult(success=True, data=results, metadata={"count": len(results)})
