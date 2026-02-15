from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import zipfile


TFF_ZIP_URL = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"


@dataclass(frozen=True)
class PositioningSnapshot:
    as_of: date
    market: str
    asset_mgr_net: Optional[int]
    lev_money_net: Optional[int]
    open_interest: Optional[int]


def _download_zip(url: str, timeout: int = 30) -> bytes:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "fxmorning/0.1"})
    r.raise_for_status()
    return r.content


def fetch_tff_history(years: List[int]) -> pd.DataFrame:
    """Download and concatenate CFTC Traders in Financial Futures (TFF) futures-only data."""
    frames: List[pd.DataFrame] = []
    for y in years:
        url = TFF_ZIP_URL.format(year=y)
        try:
            content = _download_zip(url)
        except Exception:
            continue

        z = zipfile.ZipFile(BytesIO(content))
        # Pick the first text-like file inside
        name = None
        for n in z.namelist():
            if n.lower().endswith((".txt", ".csv")):
                name = n
                break
        if not name:
            continue
        with z.open(name) as f:
            df = pd.read_csv(f, low_memory=False)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    # Normalize report date
    if "Report_Date_as_MM_DD_YYYY" in df.columns:
        df["Report_Date"] = pd.to_datetime(df["Report_Date_as_MM_DD_YYYY"], errors="coerce").dt.date
    elif "Report_Date_as_YYYY-MM-DD" in df.columns:
        df["Report_Date"] = pd.to_datetime(df["Report_Date_as_YYYY-MM-DD"], errors="coerce").dt.date
    else:
        # As_of_Date_In_Form_YYMMDD is like 260210
        df["Report_Date"] = pd.to_datetime(df.get("As_of_Date_In_Form_YYMMDD"), errors="coerce").dt.date

    return df


def latest_positioning_snapshots(df: pd.DataFrame, markets: List[str]) -> List[PositioningSnapshot]:
    """Return latest available TFF snapshots for a list of market name substrings."""
    if df is None or df.empty:
        return []

    out: List[PositioningSnapshot] = []
    for m in markets:
        sub = df[df["Market_and_Exchange_Names"].astype(str).str.contains(m, case=False, na=False)].copy()
        if sub.empty:
            continue
        sub = sub.dropna(subset=["Report_Date"])
        if sub.empty:
            continue
        latest_date = sub["Report_Date"].max()
        row = sub.loc[sub["Report_Date"] == latest_date].iloc[0]

        def i(col: str) -> Optional[int]:
            try:
                v = row.get(col)
                if pd.isna(v):
                    return None
                return int(v)
            except Exception:
                return None

        asset_net = None
        lev_net = None
        if "Asset_Mgr_Positions_Long_All" in df.columns and "Asset_Mgr_Positions_Short_All" in df.columns:
            asset_net = i("Asset_Mgr_Positions_Long_All")
            s = i("Asset_Mgr_Positions_Short_All")
            if asset_net is not None and s is not None:
                asset_net = asset_net - s
        if "Lev_Money_Positions_Long_All" in df.columns and "Lev_Money_Positions_Short_All" in df.columns:
            lev_net = i("Lev_Money_Positions_Long_All")
            s = i("Lev_Money_Positions_Short_All")
            if lev_net is not None and s is not None:
                lev_net = lev_net - s

        out.append(
            PositioningSnapshot(
                as_of=latest_date,
                market=str(row.get("Market_and_Exchange_Names")),
                asset_mgr_net=asset_net,
                lev_money_net=lev_net,
                open_interest=i("Open_Interest_All"),
            )
        )
    return out
