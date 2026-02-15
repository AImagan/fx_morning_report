from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import pandas as pd


SESSIONS_UTC = {
    "Asia": (0, 7),       # 00:00 - 07:59 UTC
    "Europe": (8, 12),    # 08:00 - 12:59 UTC
    "Americas": (13, 21), # 13:00 - 21:59 UTC
}


def _session_name(hour: int) -> str:
    for name, (h0, h1) in SESSIONS_UTC.items():
        if h0 <= hour <= h1:
            return name
    return "Off"


def session_moves(intraday: pd.DataFrame, lookback_hours: int = 30) -> Dict[str, Optional[float]]:
    """Compute session % moves over a recent window using intraday Close prices."""
    if intraday is None or intraday.empty or "Close" not in intraday.columns:
        return {"Asia": None, "Europe": None, "Americas": None}

    df = intraday.copy()
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    end = df.index.max()
    start = end - timedelta(hours=lookback_hours)
    df = df.loc[df.index >= start].copy()
    if df.empty:
        return {"Asia": None, "Europe": None, "Americas": None}

    df["session"] = df.index.hour.map(_session_name)
    out: Dict[str, Optional[float]] = {}
    for sess in ["Asia", "Europe", "Americas"]:
        sub = df[df["session"] == sess]
        if sub.empty:
            out[sess] = None
            continue
        try:
            p0 = float(sub["Close"].iloc[0])
            p1 = float(sub["Close"].iloc[-1])
            out[sess] = (p1 / p0 - 1.0) * 100.0
        except Exception:
            out[sess] = None
    return out
