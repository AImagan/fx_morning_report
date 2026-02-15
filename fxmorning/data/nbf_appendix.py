from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from io import BytesIO
from typing import Dict, List, Optional

import pandas as pd
import requests
from pypdf import PdfReader

from .forecasts import NBF_FOREX_PDF_URL


@dataclass(frozen=True)
class PolicyRow:
    country: str
    ccy: str
    pair: Optional[str]
    spot: Optional[float]
    prev: Optional[float]
    perf_pct: Optional[float]
    y2: Optional[float]
    y2_spread_bp: Optional[float]
    policy_rate: Optional[float]
    next_meeting: Optional[str]


def fetch_nbf_policy_table() -> pd.DataFrame:
    """Best-effort: parse the 'Performance / 2Y Yield / Policy Rate / Next MP Meeting' table from NBF forex.pdf."""
    try:
        r = requests.get(NBF_FOREX_PDF_URL, timeout=30, headers={"User-Agent": "fxmorning/0.1"})
        r.raise_for_status()
        content = r.content
    except Exception:
        return pd.DataFrame()

    reader = PdfReader(BytesIO(content))
    texts = []
    for p in reader.pages:
        try:
            texts.append(p.extract_text() or "")
        except Exception:
            texts.append("")

    full = "\n".join(texts)
    import re

    # Capture lines that look like:
    # Canada Canadian Dollar CAD (USD/CAD) 1.36 1.37 -0.87% 2.56 -97.2 2.25 3/18/26
    line_re = re.compile(
        r"^(?P<country>[A-Za-z ]+?)\s+"  # country
        r"(?P<curname>[A-Za-z ]+?)\s+"  # currency name
        r"(?P<ccy>[A-Z]{3})\s+"         # code
        r"(?P<pair>\([^)]*\)|-)\s+"   # (USD/CAD) or -
        r"(?P<spot>-?\d+(?:\.\d+)?)\s+"  # spot
        r"(?P<prev>-?\d+(?:\.\d+)?)\s+"  # prev
        r"(?P<perf>-?\d+(?:\.\d+)?)%\s+" # perf %
        r"(?P<y2>-?\d+(?:\.\d+)?)\s+"    # 2Y
        r"(?P<spread>-?\d+(?:\.\d+)?)\s+"# 2Y spread (bp?)
        r"(?P<policy>-?\d+(?:\.\d+)?)\s+"# policy
        r"(?P<next>\d{1,2}/\d{1,2}/\d{2})$",  # next meeting
        flags=re.M,
    )

    rows = []
    for m in line_re.finditer(full):
        gd = m.groupdict()
        pair = gd["pair"]
        if pair and pair != "-":
            pair = pair.strip("()").replace(" ", "")
        else:
            pair = None
        rows.append(
            {
                "country": gd["country"].strip(),
                "ccy": gd["ccy"].strip(),
                "pair": pair,
                "spot": float(gd["spot"]),
                "prev": float(gd["prev"]),
                "perf_pct": float(gd["perf"]),
                "y2": float(gd["y2"]),
                "y2_spread_bp": float(gd["spread"]),
                "policy_rate": float(gd["policy"]),
                "next_meeting": gd["next"],
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df
