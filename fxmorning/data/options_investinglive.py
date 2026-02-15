from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Tuple

import requests
from bs4 import BeautifulSoup


ORDERS_URL = "https://investinglive.com/Orders/"


@dataclass(frozen=True)
class OptionsExpirySheet:
    as_of: date
    article_url: str
    image_path: Optional[Path]
    note: Optional[str]


def _get(url: str, timeout: int = 20) -> str:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "fxmorning/0.1"})
    r.raise_for_status()
    return r.text


def fetch_options_expiries_sheet(as_of: date, out_dir: Path) -> Optional[OptionsExpirySheet]:
    """Best-effort: download the InvestingLive 'FX option expiries' image for the NY cut."""
    try:
        html = _get(ORDERS_URL)
    except Exception:
        return None

    soup = BeautifulSoup(html, "lxml")
    links = []
    import re

    # Slug often contains '-YYYYMMDD/'
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "fx-option-expiries-for" not in href:
            continue
        m = re.search(r"-(\d{8})/?$", href.rstrip("/"))
        if not m:
            continue
        ymd = m.group(1)
        try:
            d = datetime.strptime(ymd, "%Y%m%d").date()
        except Exception:
            continue
        links.append((d, href))

    if not links:
        return None

    links.sort(key=lambda x: x[0])
    # Pick exact date if available, else most recent prior
    chosen = None
    for d, href in reversed(links):
        if d <= as_of:
            chosen = (d, href)
            break
    if chosen is None:
        chosen = links[-1]

    chosen_date, article_url = chosen
    if article_url.startswith("/"):
        article_url = "https://investinglive.com" + article_url

    try:
        article_html = _get(article_url)
    except Exception:
        return None

    soup2 = BeautifulSoup(article_html, "lxml")

    # Find the FXO image (the expiries list is often embedded as an image)
    img_url = None
    for img in soup2.find_all("img", src=True):
        alt = (img.get("alt") or "").lower()
        src = img["src"]
        if "fxo" in alt or "fx option" in alt or "option" in alt:
            img_url = src
            break
    if img_url is None:
        # fallback: first image in article body
        first = soup2.find("article")
        if first:
            img = first.find("img", src=True)
            if img:
                img_url = img["src"]

    if img_url is None:
        return OptionsExpirySheet(as_of=chosen_date, article_url=article_url, image_path=None, note="No image found")

    if img_url.startswith("//"):
        img_url = "https:" + img_url
    elif img_url.startswith("/"):
        img_url = "https://investinglive.com" + img_url

    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ".png"
    for e in [".png", ".jpg", ".jpeg", ".webp"]:
        if e in img_url.lower():
            ext = e
            break
    out_path = out_dir / f"options_expiries_{chosen_date.strftime('%Y%m%d')}{ext}"

    try:
        r = requests.get(img_url, timeout=20, headers={"User-Agent": "fxmorning/0.1"})
        r.raise_for_status()
        out_path.write_bytes(r.content)
    except Exception:
        out_path = None

    # Keep a short note (do not reproduce full article)
    note = None
    try:
        # first paragraph text
        p = soup2.find("p")
        if p:
            note = p.get_text(" ", strip=True)
            if len(note) > 240:
                note = note[:237] + "..."
    except Exception:
        note = None

    return OptionsExpirySheet(as_of=chosen_date, article_url=article_url, image_path=out_path, note=note)
