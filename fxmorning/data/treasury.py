from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional, Tuple

import pandas as pd
import requests


TREASURY_CSV_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value={year}&_format=csv"
)

# Fallback: download all years (bigger) — last resort.
TREASURY_CSV_ALL_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/all/all?type=daily_treasury_yield_curve&field_tdr_date_value=all&_format=csv"
)


@dataclass(frozen=True)
class TreasuryYields:
    as_of: date
    y2: Optional[float]
    y10: Optional[float]
    curve: Dict[str, Optional[float]]  # maturity label -> yield


def _download_csv(url: str, timeout: int = 20) -> pd.DataFrame:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "fxmorning/0.1"})
    r.raise_for_status()
    # Treasury CSV columns often include spaces; pandas handles.
    from io import StringIO

    df = pd.read_csv(StringIO(r.text))
    return df


def fetch_treasury_yield_curve(as_of: date) -> TreasuryYields:
    """Fetch U.S. Treasury daily par yield curve for a given date.

    Uses Treasury's published CSV (official source).
    """
    year = as_of.year
    url = TREASURY_CSV_URL.format(year=year)

    try:
        df = _download_csv(url)
    except Exception:
        # fallback to all/all
        df = _download_csv(TREASURY_CSV_ALL_URL)

    # Normalize date
    if "Date" not in df.columns:
        raise ValueError(f"Unexpected Treasury CSV columns: {list(df.columns)[:10]}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date

    row = df.loc[df["Date"] == as_of]
    if row.empty:
        # If holiday/weekend, caller should pass a business day; still return empty-ish
        return TreasuryYields(as_of=as_of, y2=None, y10=None, curve={})

    row = row.iloc[0].to_dict()

    def pick(col: str) -> Optional[float]:
        v = row.get(col)
        try:
            if pd.isna(v):
                return None
            return float(v)
        except Exception:
            return None

    # Treasury columns are like '2 Yr', '10 Yr'
    y2 = pick("2 Yr") if "2 Yr" in df.columns else None
    y10 = pick("10 Yr") if "10 Yr" in df.columns else None

    curve: Dict[str, Optional[float]] = {}
    for c in df.columns:
        if c == "Date":
            continue
        curve[c] = pick(c)

    return TreasuryYields(as_of=as_of, y2=y2, y10=y10, curve=curve)


def fetch_yields_with_delta(as_of: date, prev_date: date) -> Tuple[TreasuryYields, TreasuryYields, Dict[str, Optional[float]]]:
    """Fetch yields for as_of and prev_date, plus delta in bp for 2Y/10Y."""
    a = fetch_treasury_yield_curve(as_of)
    p = fetch_treasury_yield_curve(prev_date)
    delta = {
        "2Y_bp": None if (a.y2 is None or p.y2 is None) else (a.y2 - p.y2) * 100.0,
        "10Y_bp": None if (a.y10 is None or p.y10 is None) else (a.y10 - p.y10) * 100.0,
    }
    return a, p, delta
