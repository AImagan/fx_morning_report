from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _fmt_bp(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    s = f"{x:+.1f}bp"
    return s


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.2f}%"


def build_blitz(
    usd_bias_pct: Optional[float],
    big_mover: Optional[Tuple[str, float]],
    top_event: Optional[str],
    rates_delta_bp: Optional[Dict[str, Optional[float]]] = None,
) -> List[str]:
    """Create 3 bullets: narrative, big mover, key driver."""
    narrative = "USD tone mixed"
    if usd_bias_pct is not None:
        if usd_bias_pct > 0.15:
            narrative = "USD bid: broad USD strength across majors"
        elif usd_bias_pct < -0.15:
            narrative = "USD offered: broad USD weakness across majors"
        else:
            narrative = "USD rangebound: no clear directional impulse"

    if rates_delta_bp:
        d2 = rates_delta_bp.get("2Y_bp")
        d10 = rates_delta_bp.get("10Y_bp")
        if d2 is not None or d10 is not None:
            narrative += f"; UST 2Y {_fmt_bp(d2)}, 10Y {_fmt_bp(d10)} d/d"

    mover_line = "Big mover: n/a"
    if big_mover:
        mover_line = f"Big mover: {big_mover[0]} ({_fmt_pct(big_mover[1])})"

    key_driver = "Key driver today: watch top-tier data + central bank speakers"
    if top_event:
        key_driver = f"Key driver today: {top_event}"

    return [narrative, mover_line, key_driver]


def usd_bias_from_pairs(dashboard: pd.DataFrame) -> Optional[float]:
    """Compute a simple USD strength proxy from a dashboard table.

    Expects rows to include: EURUSD, GBPUSD, AUDUSD, USDJPY, USDCAD, USDMXN, USDCNH/USDCNY.
    We transform returns into 'USD strength' convention and average.
    """
    if dashboard is None or dashboard.empty:
        return None

    sign = {
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
    vals = []
    for idx, row in dashboard.iterrows():
        code = str(idx).upper()
        if code not in sign:
            continue
        chg = row.get("pct_change")
        try:
            if chg is None or pd.isna(chg):
                continue
            vals.append(float(chg) * sign[code])
        except Exception:
            continue
    if not vals:
        return None
    return float(np.mean(vals))


def big_mover_from_dashboard(dashboard: pd.DataFrame) -> Optional[Tuple[str, float]]:
    if dashboard is None or dashboard.empty:
        return None
    try:
        s = dashboard["pct_change"].astype(float).dropna()
        if s.empty:
            return None
        code = s.abs().idxmax()
        return (str(code), float(s.loc[code]))
    except Exception:
        return None
