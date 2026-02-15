from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from ..utils import stable_hash, ensure_dirs


@dataclass(frozen=True)
class OHLC:
    df: pd.DataFrame  # index: datetime/date, columns: Open High Low Close (Adj Close optional)


def _cache_path(cache_dir: Path, key: Dict) -> Path:
    ensure_dirs(cache_dir)
    h = stable_hash(key)
    return cache_dir / f"yf_{h}.parquet"


def download_ohlc(
    tickers: List[str],
    *,
    period: str = "400d",
    interval: str = "1d",
    cache_dir: Optional[Path] = None,
    max_age_minutes: int = 30,
) -> pd.DataFrame:
    """Download OHLC for a list of tickers using yfinance.

    Returns a DataFrame:
      - if single ticker: columns Open/High/Low/Close/... directly
      - if multiple tickers: a MultiIndex column (field, ticker) as yfinance returns
    """
    tickers_key = ",".join(tickers)
    key = {"tickers": tickers_key, "period": period, "interval": interval}

    if cache_dir is not None:
        p = _cache_path(cache_dir, key)
        if p.exists():
            age = datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)
            if age <= timedelta(minutes=max_age_minutes):
                return pd.read_parquet(p)

    df = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=False,
        threads=True,
        group_by="column",
        progress=False,
    )
    if cache_dir is not None:
        try:
            df.to_parquet(p)
        except Exception:
            # Parquet may fail on some environments; silently ignore.
            pass
    return df


def download_single(
    ticker: str,
    *,
    fallback: Optional[str] = None,
    period: str = "400d",
    interval: str = "1d",
    cache_dir: Optional[Path] = None,
) -> Tuple[str, pd.DataFrame]:
    """Download a single ticker; fallback to an alternate ticker if empty."""
    df = download_ohlc([ticker], period=period, interval=interval, cache_dir=cache_dir)
    # yfinance returns columns even for single tickers; normalize
    if isinstance(df.columns, pd.MultiIndex):
        # ('Open', 'EURUSD=X') style
        df2 = df.xs(ticker, axis=1, level=1, drop_level=True)
    else:
        df2 = df

    if df2 is not None and not df2.empty:
        return ticker, df2

    if fallback:
        df_fb = download_ohlc([fallback], period=period, interval=interval, cache_dir=cache_dir)
        if isinstance(df_fb.columns, pd.MultiIndex):
            df_fb = df_fb.xs(fallback, axis=1, level=1, drop_level=True)
        if df_fb is not None and not df_fb.empty:
            return fallback, df_fb

    return ticker, df2


def get_yahoo_news(ticker: str, max_items: int = 8) -> List[Dict]:
    """Best-effort: fetch Yahoo Finance related news via yfinance."""
    try:
        t = yf.Ticker(ticker)
        items = t.news or []
        cleaned: List[Dict] = []
        for it in items[:max_items]:
            cleaned.append(
                {
                    "title": it.get("title"),
                    "publisher": it.get("publisher"),
                    "link": it.get("link"),
                    "providerPublishTime": it.get("providerPublishTime"),
                }
            )
        return cleaned
    except Exception:
        return []


def get_fundamentals(ticker: str) -> Dict:
    """Fetch key fundamentals (Market Cap, P/E, etc.) via yfinance info."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "marketCap": info.get("marketCap"),
            "forwardPE": info.get("forwardPE"),
            "trailingPE": info.get("trailingPE"),
            "dividendYield": info.get("dividendYield"),
            "priceToBook": info.get("priceToBook"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        }
    except Exception:
        return {}


def get_options_chain(ticker: str, expiration: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch calls and puts for a specific expiration (or next available)."""
    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            return pd.DataFrame(), pd.DataFrame()

        target_exp = expiration
        if not target_exp or target_exp not in expirations:
            # default to first available
            target_exp = expirations[0]

        opt = t.option_chain(target_exp)
        return opt.calls, opt.puts
    except Exception:
        return pd.DataFrame(), pd.DataFrame()
