from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


@dataclass(frozen=True)
class ForecastResult:
    source: str
    as_of_text: Optional[str]
    url: str
    table: pd.DataFrame  # index: pair code (e.g., USDCAD), columns: tenors in source


SCOTIA_URL = "https://www.scotiabank.com/ca/en/about/economics/forecast-snapshot.html"
TD_URL = "https://economics.td.com/ca-forecast-tables"
MUFG_FORECASTS_URL = "https://www.mufgresearch.com/forecasts/"
NBF_FOREX_PDF_URL = "https://www.nbc.ca/content/dam/bnc/taux-analyses/analyse-eco/mensuel/forex.pdf"


def _get_text(url: str) -> str:
    r = requests.get(url, timeout=25, headers={"User-Agent": "fxmorning/0.1"})
    r.raise_for_status()
    return r.text


def fetch_scotiabank_fx() -> Optional[ForecastResult]:
    """Parse the 'Foreign Exchange Rates' block from Scotiabank Forecast Snapshot.

    Expected columns often include year/forecast labels (e.g., 2024, 2025F, 2026F, 2027F).
    """
    try:
        html = _get_text(SCOTIA_URL)
    except Exception:
        return None

    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text("\n", strip=True)

    # As-of line (best effort)
    as_of_text = None
    import re

    m = re.search(r"Forecasts as of ([^\n]+)", text)
    if m:
        as_of_text = m.group(1).strip()

    # Extract block between Foreign Exchange Rates and Commodities
    m2 = re.search(r"Foreign Exchange Rates(.*?)Commodities", text, flags=re.S)
    if not m2:
        return None
    block = m2.group(1)
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]

    # Infer column labels from header-like lines
    col_labels: Optional[List[str]] = None
    for ln in lines[:10]:
        toks = ln.split()
        yrs = [t for t in toks if re.fullmatch(r"20\d{2}F?", t)]
        if len(yrs) >= 3:
            col_labels = yrs
            break

    rows = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 3:
            continue
        pair = parts[0].strip()
        if not pair.isalpha() and not pair.isalnum():
            continue
        # Skip lines that look like header years
        if re.fullmatch(r"20\d{2}F?", pair):
            continue

        nums = parts[1:]
        vals = []
        for x in nums:
            try:
                vals.append(float(x.replace(",", "")))
            except Exception:
                pass

        if len(vals) >= 2:
            rows.append((pair, vals))

    if not rows:
        return None

    max_len = max(len(v) for _, v in rows)
    if col_labels and len(col_labels) == max_len:
        cols = col_labels
    else:
        cols = [f"t{i+1}" for i in range(max_len)]

    data = {pair: (vals + [None] * (max_len - len(vals))) for pair, vals in rows}
    df = pd.DataFrame.from_dict(data, orient="index", columns=cols)
    df.index = [str(i).replace("/", "").replace(" ", "").upper() for i in df.index]

    return ForecastResult(source="Scotiabank", as_of_text=as_of_text, url=SCOTIA_URL, table=df)


def fetch_td_fx() -> Optional[ForecastResult]:
    """Parse TD Economics 'Foreign Exchange Outlook' table.

    TD provides quarterly forecast paths. The page layout can change; parsing is best-effort.
    """
    try:
        html = _get_text(TD_URL)
    except Exception:
        return None
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text("\n", strip=True)

    import re

    # Extract block after 'Foreign Exchange Outlook' up to next section
    m = re.search(r"Foreign Exchange Outlook(.*?)Commodity Price Outlook", text, flags=re.S)
    if not m:
        return None
    block = m.group(1)
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]

    # TD table typically has 12 quarterly columns: 2025Q1..2027Q4
    td_cols = [f"{y}Q{q}" for y in [2025, 2026, 2027] for q in [1, 2, 3, 4]]

    rows: Dict[str, List[float]] = {}

    def parse_row(prefix: str, key: str, invert: bool = False):
        for ln in lines:
            if not ln.lower().startswith(prefix.lower()):
                continue
            nums = re.findall(r"-?\d+(?:\.\d+)?", ln)
            if len(nums) < 6:
                continue
            # Keep the last 12 numeric values if present
            vals = [float(x) for x in nums[-12:]] if len(nums) >= 12 else [float(x) for x in nums]
            if invert:
                vals = [None if v == 0 else 1.0 / v for v in vals]
            rows[key] = vals
            break

    parse_row("Euro USD per EUR", "EURUSD")
    parse_row("UK Pound USD per GBP", "GBPUSD")
    parse_row("Japanese Yen JPY per USD", "USDJPY")
    parse_row("Chinese Renminbi CNY per USD", "USDCNY")
    # USD per CAD -> invert to get USDCAD
    parse_row("U.S. Dollar USD per CAD", "USDCAD", invert=True)

    if not rows:
        return None

    max_len = max(len(v) for v in rows.values())
    cols = td_cols[-max_len:] if max_len <= len(td_cols) else [f"t{i+1}" for i in range(max_len)]
    df = pd.DataFrame.from_dict({k: (v + [None] * (max_len - len(v))) for k, v in rows.items()}, orient="index", columns=cols)

    as_of_text = None
    m2 = re.search(r"Forecast by TD Economics, ([^.]+)\.", text)
    if m2:
        as_of_text = m2.group(1).strip()

    return ForecastResult(source="TD Economics", as_of_text=as_of_text, url=TD_URL, table=df)


def fetch_mufg_fx() -> Optional[ForecastResult]:
    """Parse MUFG Research 'Forecasts' FX Rates table."""
    try:
        html = _get_text(MUFG_FORECASTS_URL)
    except Exception:
        return None

    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text("\n", strip=True)
    import re

    as_of_text = None
    m0 = re.search(r"Latest FX forecast update:\s*([^\n]+)", text)
    if m0:
        as_of_text = m0.group(1).strip()

    # Infer quarter labels like 'Q1 2026'
    quarter_labels = re.findall(r"Q[1-4]\s*20\d{2}", text)
    quarter_labels = [q.replace(" ", "") for q in quarter_labels]
    # Many duplicates; keep first 4 unique
    seen = []
    for q in quarter_labels:
        if q not in seen:
            seen.append(q)
    quarter_labels = seen[:4] if seen else None

    m = re.search(r"FX Rates(.*?)(?:Growth \(GDP\)|Inflation \(CPI\)|Policy Rates|Govt\. Bond Yields)", text, flags=re.S)
    if not m:
        return None
    block = m.group(1)
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]

    rows = {}
    for ln in lines:
        if "/" not in ln:
            continue
        parts = ln.split()
        if len(parts) < 3:
            continue
        pair = parts[0].replace("/", "").upper()
        nums = []
        for x in parts[1:]:
            try:
                nums.append(float(x))
            except Exception:
                pass
        if len(nums) >= 3:
            rows[pair] = nums

    if not rows:
        return None

    max_len = max(len(v) for v in rows.values())
    if quarter_labels and len(quarter_labels) == max_len:
        cols = quarter_labels
    else:
        cols = [f"Q{i+1}" for i in range(max_len)]

    df = pd.DataFrame.from_dict({k: (v + [None] * (max_len - len(v))) for k, v in rows.items()}, orient="index", columns=cols)
    return ForecastResult(source="MUFG Research", as_of_text=as_of_text, url=MUFG_FORECASTS_URL, table=df)


def fetch_nbf_fx_pdf() -> Optional[ForecastResult]:
    """Parse National Bank of Canada forex.pdf (Current Forward Estimates).

    Table includes spot + quarterly forecasts (usually Q1..Q4 for current year).
    """
    try:
        r = requests.get(NBF_FOREX_PDF_URL, timeout=30, headers={"User-Agent": "fxmorning/0.1"})
        r.raise_for_status()
        content = r.content
    except Exception:
        return None

    reader = PdfReader(BytesIO(content))
    if not reader.pages:
        return None

    text = reader.pages[0].extract_text() or ""

    import re

    # Infer Q labels from the header line
    qlabels = re.findall(r"Q[1-4]\s*20\d{2}", text)
    qlabels = [q.replace(" ", "") for q in qlabels]
    seen = []
    for q in qlabels:
        if q not in seen:
            seen.append(q)
    qlabels = seen[:4]
    cols = ["Spot"] + (qlabels if qlabels else ["Q1", "Q2", "Q3", "Q4"])

    rows: Dict[str, List[float]] = {}
    for ln in [l.strip() for l in text.splitlines() if l.strip()]:
        m = re.search(
            r"\(([^)]+)\)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$",
            ln,
        )
        if not m:
            continue
        pair_raw = m.group(1)
        nums = [float(m.group(i)) for i in range(2, 7)]
        pair_key = pair_raw.replace(" ", "").replace("/", "").upper()
        rows[pair_key] = nums

    if not rows:
        return None

    df = pd.DataFrame.from_dict(rows, orient="index", columns=cols)
    return ForecastResult(source="National Bank (NBF)", as_of_text=None, url=NBF_FOREX_PDF_URL, table=df)


def combine_forecast_sources(results: List[ForecastResult]) -> pd.DataFrame:
    """Outer-join forecast tables from multiple sources into one multi-source table.

    Output columns are MultiIndex: (source, tenor)
    """
    if not results:
        return pd.DataFrame()

    frames = []
    for res in results:
        df = res.table.copy()
        df.columns = pd.MultiIndex.from_product([[res.source], df.columns.astype(str).tolist()])
        frames.append(df)

    out = pd.concat(frames, axis=1, join="outer")
    out.index.name = "pair"
    return out
