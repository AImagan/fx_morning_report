from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TechnicalLevels:
    pivot: Optional[float]
    r1: Optional[float]
    r2: Optional[float]
    s1: Optional[float]
    s2: Optional[float]
    ma50: Optional[float]
    ma100: Optional[float]
    ma200: Optional[float]
    rsi14: Optional[float]


def _safe_last(series: pd.Series) -> Optional[float]:
    try:
        v = float(series.dropna().iloc[-1])
        return v
    except Exception:
        return None


def compute_pivots(daily: pd.DataFrame) -> Dict[str, Optional[float]]:
    """Classic floor pivots from prior day's H/L/C."""
    if daily is None or daily.empty or len(daily) < 2:
        return {"pivot": None, "r1": None, "r2": None, "s1": None, "s2": None}
    # Use prior completed day = second last row
    prior = daily.dropna().iloc[-2]
    H, L, C = prior.get("High"), prior.get("Low"), prior.get("Close")
    try:
        H, L, C = float(H), float(L), float(C)
    except Exception:
        return {"pivot": None, "r1": None, "r2": None, "s1": None, "s2": None}

    P = (H + L + C) / 3.0
    R1 = 2 * P - L
    S1 = 2 * P - H
    R2 = P + (H - L)
    S2 = P - (H - L)
    return {"pivot": P, "r1": R1, "r2": R2, "s1": S1, "s2": S2}


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_technicals(daily: pd.DataFrame) -> TechnicalLevels:
    if daily is None or daily.empty:
        return TechnicalLevels(None, None, None, None, None, None, None, None, None)

    piv = compute_pivots(daily)

    close = daily["Close"].astype(float)
    ma50 = _safe_last(close.rolling(50).mean())
    ma100 = _safe_last(close.rolling(100).mean())
    ma200 = _safe_last(close.rolling(200).mean())

    rsi14 = None
    try:
        rsi14 = _safe_last(compute_rsi(close, 14))
    except Exception:
        rsi14 = None

    return TechnicalLevels(
        pivot=piv.get("pivot"),
        r1=piv.get("r1"),
        r2=piv.get("r2"),
        s1=piv.get("s1"),
        s2=piv.get("s2"),
        ma50=ma50,
        ma100=ma100,
        ma200=ma200,
        rsi14=rsi14,
    )
