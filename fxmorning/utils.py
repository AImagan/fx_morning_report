from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pytz


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def local_today(tz_name: str) -> date:
    tz = pytz.timezone(tz_name)
    return datetime.now(tz).date()


def parse_as_of_date(value: Optional[str], tz_name: str) -> date:
    if not value:
        return local_today(tz_name)
    return datetime.strptime(value, "%Y-%m-%d").date()


def previous_business_day(d: date) -> date:
    # Simple weekend-aware business day helper.
    # For FX, Sunday is typically thin; this is “good enough” for daily reports.
    one = timedelta(days=1)
    d = d - one
    while d.weekday() >= 5:  # Sat=5, Sun=6
        d = d - one
    return d


def stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def coalesce(*values):
    for v in values:
        if v is not None:
            return v
    return None
