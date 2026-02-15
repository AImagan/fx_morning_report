from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import requests


GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"


@dataclass(frozen=True)
class Headline:
    title: str
    url: str
    source: Optional[str]
    published: Optional[str]


def fetch_gdelt_headlines(query: str, max_records: int = 8) -> List[Headline]:
    """Fetch recent headlines using GDELT 2.1 DOC API (free, no key)."""
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": max_records,
        "sort": "HybridRel",
    }
    try:
        r = requests.get(GDELT_DOC_API, params=params, timeout=20, headers={"User-Agent": "fxmorning/0.1"})
        r.raise_for_status()
        j = r.json()
    except Exception:
        return []

    out: List[Headline] = []
    for a in (j.get("articles") or [])[:max_records]:
        out.append(
            Headline(
                title=a.get("title") or "",
                url=a.get("url") or "",
                source=a.get("sourceCountry") or a.get("sourceCollection") or None,
                published=a.get("seendate") or a.get("datetime") or None,
            )
        )
    return [h for h in out if h.title and h.url]
