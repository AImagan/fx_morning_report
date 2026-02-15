from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def realized_vol(close: pd.Series, window: int, ann_factor: int = 252) -> pd.Series:
    r = np.log(close).diff()
    return r.rolling(window).std() * np.sqrt(ann_factor)


def latest_realized_vols(daily: pd.DataFrame) -> dict:
    if daily is None or daily.empty:
        return {"rv10": None, "rv20": None, "rv60": None}
    close = daily["Close"].astype(float)
    out = {}
    for w, key in [(10, "rv10"), (20, "rv20"), (60, "rv60")]:
        try:
            v = realized_vol(close, w).dropna().iloc[-1]
            out[key] = float(v) * 100.0  # percent
        except Exception:
            out[key] = None
    return out
