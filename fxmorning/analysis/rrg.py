from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RRGPoint:
    label: str
    rs: float
    mom: float


# Map pair code to "USD strength" sign convention
# +1 means pair return = USD strength (USD/XXX)
# -1 means pair return = -USD strength (XXX/USD)
USD_STRENGTH_SIGN = {
    "EURUSD": -1.0,
    "GBPUSD": -1.0,
    "AUDUSD": -1.0,
    "NZDUSD": -1.0,
    "USDJPY": +1.0,
    "USDCAD": +1.0,
    "USDMXN": +1.0,
    "USDCNH": +1.0,
    "USDCNY": +1.0,
}


def _pct_change(series: pd.Series, n: int) -> Optional[float]:
    try:
        s = series.dropna()
        if len(s) <= n:
            return None
        return float(s.iloc[-1] / s.iloc[-1 - n] - 1.0) * 100.0
    except Exception:
        return None


def build_rrg(points_by_pair: Dict[str, pd.DataFrame], lookback: int = 20, mom_lookback: int = 10) -> List[RRGPoint]:
    """Compute simple RRG-style points (relative strength vs USD + momentum).

    - rs: % change over `lookback` days expressed as USD strength for the quote
    - mom: rs(lookback) - rs(lookback + mom_lookback)
    """
    pts: List[RRGPoint] = []
    for code, df in points_by_pair.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        sign = USD_STRENGTH_SIGN.get(code.upper(), 1.0)
        close = df["Close"].astype(float)

        rs_now = _pct_change(close, lookback)
        rs_prev = _pct_change(close, lookback + mom_lookback)
        if rs_now is None or rs_prev is None:
            continue

        rs = 100.0 + (sign * rs_now)
        mom = 100.0 + ((sign * rs_now) - (sign * rs_prev))
        pts.append(RRGPoint(label=code.upper(), rs=rs, mom=mom))
    return pts
