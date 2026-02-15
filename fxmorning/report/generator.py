from __future__ import annotations

import shutil
from dataclasses import asdict
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytz
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..config import AppConfig, PairConfig, AssetConfig
from ..utils import ensure_dirs, parse_as_of_date, previous_business_day
from ..data.yfinance_client import download_ohlc, get_fundamentals, get_options_chain
from ..data.treasury import fetch_yields_with_delta
from ..data.econoday import fetch_econoday_week
from ..data.options_investinglive import fetch_options_expiries_sheet
from ..data.gdelt import fetch_gdelt_headlines
from ..data.cftc import fetch_tff_history, latest_positioning_snapshots
from ..data.forecasts import (
    fetch_scotiabank_fx,
    fetch_td_fx,
    fetch_mufg_fx,
    fetch_nbf_fx_pdf,
    combine_forecast_sources,
    ForecastResult,
)
from ..data.nbf_appendix import fetch_nbf_policy_table
from ..analysis.technicals import compute_technicals
from ..analysis.volatility import latest_realized_vols
from ..analysis.session_recap import session_moves
from ..analysis.narrative import usd_bias_from_pairs, big_mover_from_dashboard, build_blitz
from ..analysis.rrg import build_rrg
from ..charts import generate_line_chart_html, generate_rrg_chart_html
from ..data.ib_client import IBClient, make_contract


def _fmt(x: Optional[float], decimals: int = 4) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "–"
    return f"{x:.{decimals}f}"


def _fmt_pct(x: Optional[float], decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "–"
    return f"{x:+.{decimals}f}%"


def _decimals_for_pair(code: str) -> int:
    code = code.upper()
    if "JPY" in code:
        return 2
    if "MXN" in code:
        return 2
    if "CNH" in code or "CNY" in code:
        return 4
    return 4


def _format_price(code: str, x: Optional[float]) -> str:
    return _fmt(x, decimals=_decimals_for_pair(code))


def _build_dashboard(pairs: List[PairConfig], daily_data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p in pairs:
        ticker = p.yf
        if isinstance(daily_data.columns, pd.MultiIndex):
            if ticker in daily_data.columns.get_level_values(1):
                df = daily_data.xs(ticker, axis=1, level=1, drop_level=True).dropna()
            else:
                # try fallback
                fb = p.fallback_yf
                if fb and fb in daily_data.columns.get_level_values(1):
                    df = daily_data.xs(fb, axis=1, level=1, drop_level=True).dropna()
                else:
                    df = pd.DataFrame()
        else:
            df = daily_data.dropna()

        if df is None or df.empty:
            rows.append(
                {
                    "code": p.code,
                    "label": p.label,
                    "open": None,
                    "high": None,
                    "low": None,
                    "last": None,
                    "pct_change": None,
                    "range": None,
                }
            )
            continue

        df = df.dropna(subset=["Close"])
        if len(df) < 2:
            prev_close = None
        else:
            prev_close = float(df["Close"].iloc[-2])

        last = float(df["Close"].iloc[-1])
        open_ = float(df["Open"].iloc[-1]) if "Open" in df.columns else None
        high = float(df["High"].iloc[-1]) if "High" in df.columns else None
        low = float(df["Low"].iloc[-1]) if "Low" in df.columns else None
        pct = None if prev_close is None else (last / prev_close - 1.0) * 100.0
        rng = None if (high is None or low is None) else (high - low)

        rows.append(
            {
                "code": p.code,
                "label": p.label,
                "open": open_,
                "high": high,
                "low": low,
                "last": last,
                "pct_change": pct,
                "range": rng,
            }
        )

    df = pd.DataFrame(rows).set_index("code")
    return df


def _select_events_for_day(events_by_day: Dict[str, list], as_of: date) -> List[str]:
    if not events_by_day:
        return []
    month_abbr = as_of.strftime("%b")  # Feb
    day_str = str(as_of.day)
    # try to find a key that contains month + day
    for k in events_by_day.keys():
        if month_abbr in k and day_str in k:
            events = events_by_day[k]
            return [_fmt_event(e) for e in events[:8]]
    # fallback first key
    k0 = list(events_by_day.keys())[0]
    return [_fmt_event(e) for e in events_by_day[k0][:8]]


def _fmt_event(ev) -> str:
    # CalendarEvent dataclass
    if getattr(ev, "time_et", None):
        local = getattr(ev, "time_local", None)
        if local:
            return f"{ev.time_et} ({local}) — {ev.event}"
        return f"{ev.time_et} — {ev.event}"
    return ev.event


def _build_cross_cards(assets: List[AssetConfig], daily_data: pd.DataFrame) -> List[Dict]:
    cards = []
    for a in assets:
        ticker = a.yf
        if isinstance(daily_data.columns, pd.MultiIndex):
            if ticker not in daily_data.columns.get_level_values(1):
                continue
            df = daily_data.xs(ticker, axis=1, level=1, drop_level=True).dropna()
        else:
            df = daily_data.dropna()
        if df.empty:
            continue
        df = df.dropna(subset=["Close"])
        if len(df) < 2:
            prev = None
        else:
            prev = float(df["Close"].iloc[-2])
        last = float(df["Close"].iloc[-1])
        pct = None if prev is None else (last / prev - 1) * 100.0

        # Format: indices often no decimals; commodities 2; DXY 2
        dec = 2
        if a.code.upper() in {"SPX", "VIX"}:
            dec = 2
        cards.append(
            {
                "code": a.code,
                "label": a.label,
                "last": _fmt(last, decimals=dec),
                "pct_change": pct,
                "pct_change_str": _fmt_pct(pct, decimals=2),
            }
        )
    return cards


def _copy_static_assets(cfg: AppConfig) -> None:
    # Copy packaged CSS into site/static/css
    pkg_static = Path(__file__).resolve().parent / "static"
    site_static = cfg.site_dir / "static"
    ensure_dirs(site_static / "css", site_static / "generated")
    # Copy CSS folder (overwrite)
    src_css = pkg_static / "css" / "style.css"
    dst_css = site_static / "css" / "style.css"
    dst_css.write_text(src_css.read_text())


def generate_site(cfg: AppConfig) -> None:
    ensure_dirs(cfg.site_dir, cfg.cache_dir)
    _copy_static_assets(cfg)

    tz = cfg.timezone
    as_of = parse_as_of_date(cfg.report.as_of_date, tz)
    generated_at = datetime.now(pytz.timezone(tz)).strftime("%Y-%m-%d %I:%M %p %Z")

    # -------- Market data (yfinance) --------
    pair_tickers = [p.yf for p in cfg.pairs]
    # include fallbacks too so one bulk download can cover
    for p in cfg.pairs:
        if p.fallback_yf and p.fallback_yf not in pair_tickers:
            pair_tickers.append(p.fallback_yf)

    cross_tickers = [a.yf for a in cfg.cross_assets]
    iv_tickers = [a.yf for a in cfg.fx_vol_indices]

    # Initialize dataframes
    daily_fx = pd.DataFrame()
    intraday_fx = pd.DataFrame()
    daily_cross = pd.DataFrame()
    daily_iv = pd.DataFrame()

    if cfg.data_source == "ibkr" and cfg.ib:
        print("Using IBKR Data Source...")
        ib = IBClient(cfg.ib['host'], cfg.ib['port'], cfg.ib['client_id'])
        if ib.connect_and_start():
            try:
                # Helper to fetch and convert
                def fetch_ib_df(tickers: List[str], duration: str, bar_size: str, what: str = "MIDPOINT", use_rth: int = 1) -> pd.DataFrame:
                    dfs = []
                    today_str = datetime.now().strftime("%Y%m%d %H:%M:%S")
                    for t in tickers:
                        # Simple heuristic for contract mapping
                        symbol = t
                        sec_type = "STK"
                        exchange = "SMART"
                        currency = "USD"
                        
                        if len(t) == 6 and t.isalpha(): # FX
                            symbol = t[:3]
                            currency = t[3:]
                            sec_type = "CASH"
                            exchange = "IDEALPRO"
                        elif t in ["SPY", "QQQ", "IWM", "GLD", "USO", "NVDA"]:
                            sec_type = "STK"
                        elif t == "^GSPC" or t == "SPX":
                            symbol = "SPX"
                            sec_type = "IND"
                            exchange = "CBOE"
                        elif t == "^VIX" or t == "VIX":
                            symbol = "VIX"
                            sec_type = "IND"
                            exchange = "CBOE"
                        elif t == "DX=F" or t == "DXY": # DXY futures
                            symbol = "DX"
                            sec_type = "FUT"
                            exchange = "ICEUS"
                            continue 
                        
                        contract = make_contract(symbol, sec_type, exchange, currency)
                        bars = ib.get_historical_data(contract, "", duration, bar_size, what, use_rth)
                        if not bars:
                            continue
                            
                        data = []
                        for b in bars:
                            d = b.date
                            if len(d) == 8:
                                dt = datetime.strptime(d, "%Y%m%d")
                            elif len(d) > 10: # YYYYMMDD  HH:MM:SS
                                dt = datetime.strptime(d, "%Y%m%d  %H:%M:%S")
                            else:
                                dt = datetime.fromtimestamp(int(d))
                                
                            data.append({
                                "Date": dt,
                                "Open": b.open,
                                "High": b.high,
                                "Low": b.low,
                                "Close": b.close,
                                "Volume": b.volume
                            })
                        
                        df = pd.DataFrame(data).set_index("Date")
                        # Add MultiIndex level for compatibility
                        df.columns = pd.MultiIndex.from_product([df.columns, [t]])
                        dfs.append(df)
                    
                    if not dfs:
                        return pd.DataFrame()
                    return pd.concat(dfs, axis=1)

                # Fetching
                daily_fx = fetch_ib_df(pair_tickers, "1 Y", "1 day", "MIDPOINT", 1)
                intraday_fx = fetch_ib_df(pair_tickers, "2 D", "1 hour", "MIDPOINT", 1)
                daily_cross = fetch_ib_df(cross_tickers, "3 M", "1 day", "TRADES", 1) 
                daily_iv = fetch_ib_df(iv_tickers, "2 M", "1 day", "TRADES", 1)

            except Exception as e:
                print(f"IBKR Fetch Error: {e}")
            finally:
                ib.disconnect_and_stop()
        else:
             print("Could not connect to IBKR (check TWS).")

    # Fallback to yfinance if data is missing (handles IB failure, empty result, or config=yfinance)
    if daily_fx.empty:
        print("Fetching data from Yahoo Finance...")
        daily_fx = download_ohlc(pair_tickers, period="420d", interval="1d", cache_dir=cfg.cache_dir)
        intraday_fx = download_ohlc(pair_tickers, period="7d", interval="60m", cache_dir=cfg.cache_dir)
        daily_cross = download_ohlc(cross_tickers, period="120d", interval="1d", cache_dir=cfg.cache_dir)
        daily_iv = download_ohlc(iv_tickers, period="60d", interval="1d", cache_dir=cfg.cache_dir)

    dashboard = _build_dashboard(cfg.pairs, daily_fx)

    # -------- Fundamentals (Cross Assets) --------
    fundamentals_data = []
    for a in cfg.cross_assets:
        # Check if it's likely an equity/index that has fundamentals
        # Heuristic: anything not explicitly a currency future often has some info
        try:
            f = get_fundamentals(a.yf)
            # Only keep if we have main fields
            if f and (f.get("marketCap") or f.get("sector")):
                f["label"] = a.label
                f["code"] = a.code
                # Format big numbers
                if f.get("marketCap"):
                    val = f["marketCap"]
                    if val > 1e12:
                        f["marketCapStr"] = f"{val/1e12:.2f}T"
                    elif val > 1e9:
                        f["marketCapStr"] = f"{val/1e9:.2f}B"
                    else:
                        f["marketCapStr"] = f"{val/1e6:.2f}M"
                else:
                    f["marketCapStr"] = "–"
                fundamentals_data.append(f)
        except Exception:
            pass

    # -------- Options Chain (Featured) --------
    # Try to get options for the first equity cross-asset (e.g. SPX or NVDA if added)
    featured_options = None
    featured_options_ticker = None
    for a in cfg.cross_assets:
        code_up = a.code.upper()
        if code_up in ["SPY", "NVDA", "QQQ", "IWM", "SPX"] or "Stock" in a.label or "ETF" in a.label:
            # Try to fetch
            calls, puts = get_options_chain(a.yf)
            if not calls.empty and not puts.empty:
                featured_options = {
                    "label": a.label,
                    "calls": calls.head(5).to_dict(orient="records"),
                    "puts": puts.head(5).to_dict(orient="records"),
                }
                featured_options_ticker = a.code
                break

    # -------- U.S. Rates --------
    ust2y = ust10y = ust2y_chg = ust10y_chg = "–"
    rates_delta = {"2Y_bp": None, "10Y_bp": None}
    if cfg.sources.treasury_yield_curve:
        y_asof = as_of
        if y_asof.weekday() >= 5:
            y_asof = previous_business_day(y_asof)
        prev = previous_business_day(y_asof)
        try:
            a, p, delta = fetch_yields_with_delta(y_asof, prev)
            rates_delta = delta
            ust2y = _fmt(a.y2, 2) if a.y2 is not None else "–"
            ust10y = _fmt(a.y10, 2) if a.y10 is not None else "–"
            ust2y_chg = _fmt(delta.get("2Y_bp"), 1) + " bp" if delta.get("2Y_bp") is not None else "–"
            ust10y_chg = _fmt(delta.get("10Y_bp"), 1) + " bp" if delta.get("10Y_bp") is not None else "–"
        except Exception:
            pass

    # -------- Calendar --------
    todays_events: List[str] = []
    week_events: List[str] = []
    if cfg.sources.econoday_calendar:
        try:
            week = fetch_econoday_week(as_of, tz)
            todays_events = _select_events_for_day(week, as_of)
            # flatten for weekly
            flat = []
            for k, evs in week.items():
                for ev in evs:
                    flat.append(f"{k}: {_fmt_event(ev)}")
            week_events = flat[:14]
        except Exception:
            pass

    # -------- Narrative Blitz --------
    usd_bias = usd_bias_from_pairs(dashboard)
    big_mover = big_mover_from_dashboard(dashboard)
    top_event = todays_events[0] if todays_events else None
    blitz = build_blitz(usd_bias, big_mover, top_event, rates_delta_bp=rates_delta)

    # -------- Session recap --------
    session_best = {}
    for sess in ["Asia", "Europe", "Americas"]:
        session_best[sess] = (None, None)  # (pair, move)
    for p in cfg.pairs:
        ticker = p.yf
        if isinstance(intraday_fx.columns, pd.MultiIndex):
            if ticker in intraday_fx.columns.get_level_values(1):
                df = intraday_fx.xs(ticker, axis=1, level=1, drop_level=True).dropna()
            else:
                fb = p.fallback_yf
                if fb and fb in intraday_fx.columns.get_level_values(1):
                    df = intraday_fx.xs(fb, axis=1, level=1, drop_level=True).dropna()
                else:
                    df = pd.DataFrame()
        else:
            df = intraday_fx.dropna()

        moves = session_moves(df, lookback_hours=36)
        for sess, mv in moves.items():
            if mv is None:
                continue
            cur_pair, cur_mv = session_best[sess]
            if cur_mv is None or abs(mv) > abs(cur_mv):
                session_best[sess] = (p.label, mv)

    def sess_str(sess: str) -> str:
        pair, mv = session_best.get(sess, (None, None))
        if pair is None or mv is None:
            return "–"
        return f"{pair} {_fmt_pct(mv, 2)}"

    asia_move = sess_str("Asia")
    europe_move = sess_str("Europe")
    americas_move = sess_str("Americas")

    asia_comment = "Largest move during Asia session (proxy). Watch CNH fix / Japan headlines if JPY-led."
    europe_comment = "Largest move during Europe session (proxy). Watch EZ/UK data + ECB/BoE commentary."
    americas_comment = "Largest move during Americas session (proxy). Watch US data + Treasury/yield moves."

    # -------- Charts --------
    usd_chart_html = ""
    # Use DXY proxy if configured; otherwise use first cross asset.
    dxy_ticker = None
    for a in cfg.cross_assets:
        if a.code.upper() == "DXY":
            dxy_ticker = a.yf
            break
    if dxy_ticker and isinstance(daily_cross.columns, pd.MultiIndex) and dxy_ticker in daily_cross.columns.get_level_values(1):
        dxy = daily_cross.xs(dxy_ticker, axis=1, level=1, drop_level=True).dropna()
        s = dxy["Close"].tail(90)
        try:
            usd_chart_html = generate_line_chart_html(s, "USD Index proxy (3M)")
        except Exception:
            pass

    # -------- Technicals + Vol --------
    tech_rows = []
    vol_rows = []
    for p in cfg.pairs:
        ticker = p.yf
        if isinstance(daily_fx.columns, pd.MultiIndex):
            if ticker in daily_fx.columns.get_level_values(1):
                df = daily_fx.xs(ticker, axis=1, level=1, drop_level=True).dropna()
            else:
                fb = p.fallback_yf
                if fb and fb in daily_fx.columns.get_level_values(1):
                    df = daily_fx.xs(fb, axis=1, level=1, drop_level=True).dropna()
                else:
                    df = pd.DataFrame()
        else:
            df = daily_fx.dropna()

        t = compute_technicals(df)
        tech_rows.append(
            {
                "label": p.label,
                "s2": _format_price(p.code, t.s2),
                "s1": _format_price(p.code, t.s1),
                "pivot": _format_price(p.code, t.pivot),
                "r1": _format_price(p.code, t.r1),
                "r2": _format_price(p.code, t.r2),
                "ma50": _format_price(p.code, t.ma50),
                "ma200": _format_price(p.code, t.ma200),
                "rsi14": _fmt(t.rsi14, 1) if t.rsi14 is not None else "–",
            }
        )

        vols = latest_realized_vols(df)
        vol_rows.append(
            {
                "label": p.label,
                "rv10": _fmt(vols.get("rv10"), 1) if vols.get("rv10") is not None else "–",
                "rv20": _fmt(vols.get("rv20"), 1) if vols.get("rv20") is not None else "–",
                "rv60": _fmt(vols.get("rv60"), 1) if vols.get("rv60") is not None else "–",
            }
        )

    # -------- Implied vol proxies --------
    implied_rows = []
    for a in cfg.fx_vol_indices:
        ticker = a.yf
        if isinstance(daily_iv.columns, pd.MultiIndex):
            if ticker not in daily_iv.columns.get_level_values(1):
                continue
            df = daily_iv.xs(ticker, axis=1, level=1, drop_level=True).dropna()
        else:
            df = daily_iv.dropna()
        if df.empty:
            continue
        if len(df) < 2:
            prev = None
        else:
            prev = float(df["Close"].iloc[-2])
        last = float(df["Close"].iloc[-1])
        pct = None if prev is None else (last / prev - 1) * 100.0
        implied_rows.append(
            {"label": a.label, "last": _fmt(last, 2), "pct_change": pct, "pct_change_str": _fmt_pct(pct, 2)}
        )

    # -------- Options expiries image --------
    options_sheet_img = None
    options_note = None
    if cfg.sources.investinglive_options:
        sheet = fetch_options_expiries_sheet(as_of, cfg.site_dir / "static" / "generated")
        if sheet and sheet.image_path:
            options_sheet_img = f"static/generated/{sheet.image_path.name}"
            options_note = sheet.note

    # -------- CFTC positioning --------
    cot_rows = []
    if cfg.sources.cftc_cot:
        try:
            years = sorted({as_of.year, as_of.year - 1})
            df_cot = fetch_tff_history(years)
            snaps = latest_positioning_snapshots(
                df_cot,
                markets=["EURO FX", "JAPANESE YEN", "BRITISH POUND", "CANADIAN DOLLAR", "MEXICAN PESO"],
            )
            market_map = {
                "EURO FX": "EUR",
                "JAPANESE YEN": "JPY",
                "BRITISH POUND": "GBP",
                "CANADIAN DOLLAR": "CAD",
                "MEXICAN PESO": "MXN",
            }
            for s in snaps:
                code = "N/A"
                for m_key, c_val in market_map.items():
                    if m_key in s.market:
                        code = c_val
                        break
                cot_rows.append(
                    {
                        "code": code,
                        "net": s.lev_money_net or 0,
                        "change": 0,  # Placeholder as we only fetch latest
                    }
                )
        except Exception:
            cot_rows = []

    # -------- Headlines (GDELT) --------
    headlines = []
    if cfg.sources.gdelt_headlines:
        try:
            hs = fetch_gdelt_headlines("(USD OR dollar) AND (yields OR Fed OR inflation OR oil)", max_records=6)
            headlines = [{"title": h.title, "url": h.url} for h in hs]
        except Exception:
            headlines = []

    # -------- Render templates --------
    env = Environment(
        loader=FileSystemLoader((Path(__file__).resolve().parent / "templates")),
        autoescape=select_autoescape(["html"]),
    )

    fx_rows = []
    for code, row in dashboard.iterrows():
        fx_rows.append(
            {
                "code": code,
                "label": row["label"],
                "open": _format_price(code, row["open"]),
                "high": _format_price(code, row["high"]),
                "low": _format_price(code, row["low"]),
                "last": _format_price(code, row["last"]),
                "pct_change": row["pct_change"],
                "pct_change_str": _fmt_pct(row["pct_change"], 2),
                "range": _format_price(code, row["range"]),
            }
        )

    top_mover_str = "–"
    if big_mover:
        top_mover_str = f"{big_mover[0]} {_fmt_pct(big_mover[1], 2)}"

    # Home page
    # Crypto Data
    crypto_rows = []
    if cfg.crypto:
        try:
            # Reusing download_ohlc for crypto
            c_data = download_ohlc([c.yf for c in cfg.crypto], period="5d")
            # Build similar to dashboard
            # We can reuse _build_dashboard if we duck-type or if AssetConfig has same fields as PairConfig
            # AssetConfig has code, label, yf. PairConfig has code, label, yf, fallback_yf.
            # We can construct temporary PairConfig objects
            fake_pairs = [PairConfig(code=c.code, label=c.label, yf=c.yf) for c in cfg.crypto]
            c_df = _build_dashboard(fake_pairs, c_data)
            # convert to list of dicts
            for code, row in c_df.iterrows():
                crypto_rows.append({
                    "code": code,
                    "label": row["label"],
                    "open": _format_price(code, row["open"]),
                    "high": _format_price(code, row["high"]),
                    "low": _format_price(code, row["low"]),
                    "last": _format_price(code, row["last"]),
                    "pct_change": row["pct_change"],
                    "pct_change_str": _fmt_pct(row["pct_change"], 2),
                    "range": _format_price(code, row["range"]),
                })
        except Exception as e:
            print(f"Crypto fetch failed: {e}")
            crypto_rows = []

    # Common editorial data
    ed = cfg.editorial

    cross_assets_df = _build_dashboard(cfg.cross_assets, daily_cross)

    # 1. Markets Report (Index) - Stocks, ETFs, Futures, Options
    home_ctx = {
        "title": "Markets | Institutional Intelligence",
        "active_page": "markets",
        "as_of": as_of.isoformat(),
        "generated_at": generated_at,
        "performance_widgets": [
            {
                "label": "S&P 500",
                "value": _format_price("SPX", dashboard.loc["^GSPC", "last"] if "^GSPC" in dashboard.index else 0),
                "change": (dashboard.loc["^GSPC", "pct_change"] if "^GSPC" in dashboard.index else 0),
                "change_str": _fmt_pct(dashboard.loc["^GSPC", "pct_change"] if "^GSPC" in dashboard.index else 0, 2)
            },
            {
                "label": "UST 10Y",
                "value": f"{ust10y}%",
                "change": 0, # Placeholder or use actual int if possible
                "change_str": f"{ust10y_chg}bp"
            },
            {
                "label": "USD Index Proxy",
                "value": _format_price("EURUSD", dashboard.loc["EURUSD", "last"] if "EURUSD" in dashboard.index else 0),
                "change": (dashboard.loc["EURUSD", "pct_change"] if "EURUSD" in dashboard.index else 0),
                "change_str": _fmt_pct(dashboard.loc["EURUSD", "pct_change"] if "EURUSD" in dashboard.index else 0, 2)
            },
        ],
        "rates_data": [
            {"label": "U.S. 2-Year", "value": f"{ust2y}%", "change": 0, "change_str": f"{ust2y_chg}bp"},
            {"label": "U.S. 10-Year", "value": f"{ust10y}%", "change": 0, "change_str": f"{ust10y_chg}bp"},
        ],
        "performance_rows": [
            {
                "code": code,
                "open": _format_price(code, row["open"]),
                "high": _format_price(code, row["high"]),
                "low": _format_price(code, row["low"]),
                "last": _format_price(code, row["last"]),
                "pct_change": row["pct_change"],
                "pct_change_str": _fmt_pct(row["pct_change"], 2),
                "range": _format_price(code, row["range"]),
            }
            for code, row in cross_assets_df.iterrows()
        ] if not cross_assets_df.empty else [],
        "headlines": headlines,
        "ed": ed,
    }
    (cfg.site_dir / "index.html").write_text(env.get_template("index.html").render(**home_ctx), encoding="utf-8")

    # 2. Daily FX Report
    if cfg.report.generate_daily:
        daily_ctx = {
            "title": "Daily FX Update | FX Morning Report",
            "active_page": "daily",
            "as_of": as_of.isoformat(),
            "blitz": blitz,
            "fx_rows": fx_rows,
            "ust2y": ust2y,
            "ust10y": ust10y,
            "ust2y_chg": ust2y_chg,
            "ust10y_chg": ust10y_chg,
            "todays_events": todays_events,
            "asia_move": asia_move,
            "europe_move": europe_move,
            "americas_move": americas_move,
            "asia_comment": asia_comment,
            "europe_comment": europe_comment,
            "americas_comment": americas_comment,
            "usd_chart_html": usd_chart_html,
            "tech_rows": tech_rows,
            "vol_rows": vol_rows,
            "implied_vol_rows": implied_rows,
            "options_sheet_img": options_sheet_img,
            "options_note": options_note,
            "cot_rows": cot_rows,
            "main_chart": usd_chart_html, # Renamed to match template
            "headlines": headlines,
            "ed": ed,
        }
        (cfg.site_dir / "daily.html").write_text(env.get_template("daily.html").render(**daily_ctx), encoding="utf-8")

    # 3. Crypto Report
    crypto_ctx = {
        "title": "Cryptocurrency | FX Morning Report",
        "active_page": "crypto",
        "as_of": as_of.isoformat(),
        "generated_at": generated_at,
        "crypto_rows": crypto_rows,
        "headlines": headlines, # Sharing headlines
        "ed": ed,
    }
    (cfg.site_dir / "crypto.html").write_text(env.get_template("crypto.html").render(**crypto_ctx), encoding="utf-8")

    if cfg.report.generate_weekly:
        # Forecast sources
        forecast_results: List[ForecastResult] = []
        if cfg.sources.forecasts.scotiabank:
            r = fetch_scotiabank_fx()
            if r:
                forecast_results.append(r)
        if cfg.sources.forecasts.td_economics:
            r = fetch_td_fx()
            if r:
                forecast_results.append(r)
        if cfg.sources.forecasts.mufg:
            r = fetch_mufg_fx()
            if r:
                forecast_results.append(r)
        if cfg.sources.forecasts.national_bank_canada_pdf:
            r = fetch_nbf_fx_pdf()
            if r:
                forecast_results.append(r)

        combined = combine_forecast_sources(forecast_results)
        # Filter to common pairs
        wanted = ["EURUSD", "USDJPY", "GBPUSD", "USDCAD", "AUDUSD", "USDMXN", "USDCNH", "USDCNY"]
        if not combined.empty:
            # Some sources use USDCNY; keep either
            combined = combined.loc[[i for i in combined.index if i in wanted]]
        forecast_table_html = None
        if combined is not None and not combined.empty:
            # Format nicely
            df_disp = combined.copy()
            for col in df_disp.columns:
                # per-pair formatting
                for idx in df_disp.index:
                    val = df_disp.loc[idx, col]
                    if pd.isna(val):
                        continue
                    try:
                        dec = _decimals_for_pair(idx)
                        df_disp.loc[idx, col] = f"{float(val):.{dec}f}"
                    except Exception:
                        pass
            forecast_table_html = '<div class="table-wrap">' + df_disp.to_html(escape=False) + "</div>"

        # Policy table (from NBF appendix)
        policy_table_rows = []
        try:
            pol = fetch_nbf_policy_table()
            if not pol.empty:
                # keep a subset relevant to our FX grid
                keep = pol[pol["ccy"].isin(["CAD", "USD", "EUR", "JPY", "AUD", "GBP", "CNY", "MXN"])].copy()
                for _, r in keep.iterrows():
                    policy_table_rows.append(
                        {
                            "code": r["ccy"],
                            "policy_rate": _fmt(r["policy_rate"], 2),
                            "yield_2y": _fmt(r["y2"], 2),
                            "spread_2y": r["y2_spread_bp"], # Keep as float for comparison
                            "spread_2y_str": _fmt(r["y2_spread_bp"], 1),
                            "next_meeting": r["next_meeting"],
                        }
                    )
        except Exception:
            policy_table_rows = []

        # Macro theme paragraphs (rule-based)
        macro_theme = []
        if usd_bias is not None:
            if usd_bias > 0.2:
                macro_theme.append(
                    "The dominant theme remains USD resilience: the broad USD basket is firm, consistent with a rates-driven bid and a market that continues to reward USD carry."
                )
            elif usd_bias < -0.2:
                macro_theme.append(
                    "The dominant theme is USD consolidation/softness: USD has lost momentum versus majors, typically consistent with easing rate expectations or a risk-on impulse reducing demand for USD liquidity."
                )
            else:
                macro_theme.append(
                    "The dominant theme is divergence: currencies are rotating rather than moving in unison, suggesting positioning is driving price action as much as macro data."
                )

        if rates_delta.get("2Y_bp") is not None:
            macro_theme.append(
                f"Rates are still the transmission mechanism for FX: the U.S. 2Y moved {_fmt(rates_delta['2Y_bp'],1)}bp d/d and the 10Y moved {_fmt(rates_delta['10Y_bp'],1)}bp, which typically maps to intraday USD beta and higher sensitivity to tier-1 data."
            )

        macro_theme.append(
            "Focus this week: central bank reaction functions (Fed/BoC/ECB/BoJ) and whether inflation data keeps front-end yield differentials wide — the key driver of G10 FX trends in a high-carry environment."
        )

        # RRG chart
        # Build close series dict for pairs
        pair_daily_map = {}
        for p in cfg.pairs:
            ticker = p.yf
            if isinstance(daily_fx.columns, pd.MultiIndex):
                if ticker in daily_fx.columns.get_level_values(1):
                    dfp = daily_fx.xs(ticker, axis=1, level=1, drop_level=True).dropna()
                else:
                    fb = p.fallback_yf
                    if fb and fb in daily_fx.columns.get_level_values(1):
                        dfp = daily_fx.xs(fb, axis=1, level=1, drop_level=True).dropna()
                    else:
                        dfp = pd.DataFrame()
            else:
                dfp = daily_fx.dropna()
            pair_daily_map[p.code] = dfp

        rrg_points = build_rrg(pair_daily_map, lookback=10, mom_lookback=5)
        rrg_chart_html = ""
        try:
            if rrg_points:
                rrg_chart_html = generate_rrg_chart_html(rrg_points, "FX rotation vs USD (RRG-style)")
        except Exception:
            pass

        # Trade ideas (very simple)
        trade_ideas = []
        # Example 1: MXN vs JPY carry if both available
        if dashboard.loc["USDMXN", "last"] is not None and dashboard.loc["USDJPY", "last"] is not None:
            mxnjpy = float(dashboard.loc["USDJPY", "last"]) / float(dashboard.loc["USDMXN", "last"])
            trade_ideas.append(
                {
                    "trade": "Long MXN/JPY (carry)",
                    "entry": f"{mxnjpy:.3f}",
                    "stop": f"{mxnjpy*0.97:.3f}",
                    "target": f"{mxnjpy*1.04:.3f}",
                    "rationale": "High-rate MXN versus low-rate JPY historically screens well on carry; monitor risk sentiment and MXN volatility.",
                }
            )
        # Example 2: USD/CAD technical mean reversion around MA200
        for t in tech_rows:
            if t["label"].startswith("USD/CAD") or t["label"] == "USD/CAD":
                trade_ideas.append(
                    {
                        "trade": "USD/CAD tactical fade / mean reversion",
                        "entry": "Near pivot / MA50",
                        "stop": "Above R2",
                        "target": "Back to pivot / S1",
                        "rationale": "USD/CAD is typically sensitive to oil + rate spreads. Use pivot framework intraday; reassess on US/Canada data prints.",
                    }
                )
                break

        # Fill with generic ideas
        while len(trade_ideas) < 4:
            trade_ideas.append(
                {
                    "trade": "EUR/USD momentum break",
                    "entry": "On break above R1",
                    "stop": "Below pivot",
                    "target": "R2 / prior swing high",
                    "rationale": "If US yields stabilize and Europe data surprises to the upside, EUR/USD can extend; watch ECB communication and US inflation prints.",
                }
            )
            if len(trade_ideas) >= 4:
                break

        weekly_ctx = {
            "title": "Weekly FX Outlook | FX Morning Report",
            "active_page": "weekly",
            "macro_theme": macro_theme,
            "rates_rows": policy_table_rows,
            "week_events": week_events,
            "rrg_chart": rrg_chart_html,
            "cross_cards": _build_cross_cards(cfg.cross_assets, daily_cross),
            "trade_ideas": trade_ideas[:4],
            "forecast_table_html": forecast_table_html,
            "cot_rows": cot_rows, # Added for consistency
            "headlines": headlines,
            "ed": ed,
        }
        (cfg.site_dir / "weekly.html").write_text(env.get_template("weekly.html").render(**weekly_ctx), encoding="utf-8")
