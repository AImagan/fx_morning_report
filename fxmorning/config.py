from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class PairConfig:
    code: str
    label: str
    yf: str
    fallback_yf: Optional[str] = None


@dataclass(frozen=True)
class AssetConfig:
    code: str
    label: str
    yf: str


@dataclass(frozen=True)
class ForecastSourcesConfig:
    scotiabank: bool = True
    td_economics: bool = True
    mufg: bool = True
    national_bank_canada_pdf: bool = True


@dataclass(frozen=True)
class SourcesConfig:
    treasury_yield_curve: bool = True
    econoday_calendar: bool = True
    investinglive_options: bool = True
    cftc_cot: bool = True

    forecasts: ForecastSourcesConfig = field(default_factory=ForecastSourcesConfig)

    yahoo_news_via_yfinance: bool = True
    gdelt_headlines: bool = True


@dataclass(frozen=True)
class OutputConfig:
    site_dir: str = "site"
    cache_dir: str = "cache"
    keep_history_days: int = 14


@dataclass(frozen=True)
class ReportConfig:
    as_of_date: Optional[str] = None  # YYYY-MM-DD or null = today
    generate_daily: bool = True
    generate_weekly: bool = True


@dataclass(frozen=True)
class AppConfig:
    timezone: str = "America/Edmonton"
    output: OutputConfig = field(default_factory=OutputConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    pairs: List[PairConfig] = field(default_factory=list)
    cross_assets: List[AssetConfig] = field(default_factory=list)
    fx_vol_indices: List[AssetConfig] = field(default_factory=list)
    crypto: List[AssetConfig] = field(default_factory=list)
    data_source: str = "yfinance"  # or "ibkr"
    ib: Optional[Dict[str, int | str]] = None
    editorial: Dict[str, Any] = field(default_factory=dict)

    sources: SourcesConfig = field(default_factory=SourcesConfig)

    # resolved paths (filled in by loader)
    project_dir: Path = Path(".")
    site_dir: Path = Path("site")
    cache_dir: Path = Path("cache")


def _coerce_dataclass(dc_cls, data: Dict[str, Any]):
    # A tiny helper to build nested dataclasses without pulling extra deps.
    if dc_cls is ForecastSourcesConfig:
        return ForecastSourcesConfig(**(data or {}))
    if dc_cls is SourcesConfig:
        forecasts = _coerce_dataclass(ForecastSourcesConfig, (data or {}).get("forecasts", {}))
        rest = {k: v for k, v in (data or {}).items() if k != "forecasts"}
        return SourcesConfig(forecasts=forecasts, **rest)
    if dc_cls is OutputConfig:
        return OutputConfig(**(data or {}))
    if dc_cls is ReportConfig:
        return ReportConfig(**(data or {}))
    raise ValueError(f"Unsupported dataclass coercion for: {dc_cls}")


def load_config(path: str | Path) -> AppConfig:
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text()) or {}

    timezone = raw.get("timezone", "America/Edmonton")

    output = _coerce_dataclass(OutputConfig, raw.get("output", {}))
    report = _coerce_dataclass(ReportConfig, raw.get("report", {}))
    sources = _coerce_dataclass(SourcesConfig, raw.get("sources", {}))

    pairs = [PairConfig(**p) for p in (raw.get("pairs", []) or [])]
    cross_assets = [AssetConfig(**a) for a in (raw.get("cross_assets", []) or [])]
    fx_vol_indices = [AssetConfig(**a) for a in (raw.get("fx_vol_indices", []) or [])]
    crypto = [AssetConfig(**a) for a in (raw.get("crypto", []) or [])]

    data_source = raw.get("data_source", "yfinance")
    ib = raw.get("ib")
    editorial = raw.get("editorial", {})

    project_dir = path.parent
    site_dir = (project_dir / output.site_dir).resolve()
    cache_dir = (project_dir / output.cache_dir).resolve()

    return AppConfig(
        timezone=timezone,
        output=output,
        report=report,
        pairs=pairs,
        cross_assets=cross_assets,
        fx_vol_indices=fx_vol_indices,
        crypto=crypto,
        data_source=data_source,
        ib=ib,
        editorial=editorial,
        sources=sources,
        project_dir=project_dir,
        site_dir=site_dir,
        cache_dir=cache_dir,
    )
