from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional

import pandas as pd
import pytz
import requests


@dataclass(frozen=True)
class CalendarEvent:
    day_label: str
    event: str
    time_et: Optional[str]
    time_local: Optional[str]
    url: Optional[str] = None


ECONODAY_WEEK_URL = "https://us.econoday.com/byweek?day={day}&month={month}&year={year}&lid=0"


def fetch_econoday_week(as_of: date, local_tz: str) -> Dict[str, List[CalendarEvent]]:
    """Fetch Econoday weekly calendar (best-effort HTML parse).

    Returns dict: day_label -> list of events.
    """
    url = ECONODAY_WEEK_URL.format(day=as_of.day, month=as_of.month, year=as_of.year)
    r = requests.get(url, timeout=20, headers={"User-Agent": "fxmorning/0.1"})
    r.raise_for_status()
    html = r.text

    # 1) Prefer parsing the day-column table via read_html
    try:
        dfs = pd.read_html(html)
    except Exception:
        dfs = []

    schedule_df: Optional[pd.DataFrame] = None
    for df in dfs:
        cols = [str(c) for c in df.columns]
        if any("Monday" in c for c in cols) and any("Friday" in c for c in cols) and len(cols) >= 5:
            schedule_df = df
            break

    if schedule_df is None and dfs:
        # fallback: pick the widest table
        schedule_df = max(dfs, key=lambda d: d.shape[1])

    if schedule_df is None:
        return {}

    out: Dict[str, List[CalendarEvent]] = {}
    local = pytz.timezone(local_tz)

    def convert_et_to_local(time_et: str) -> Optional[str]:
        try:
            # time_et like '8:30 AM ET' (no date) — assume 'as_of' date for conversion
            t = datetime.strptime(time_et.replace(" ET", ""), "%I:%M %p").time()
            dt_et = pytz.timezone("US/Eastern").localize(datetime.combine(as_of, t))
            dt_local = dt_et.astimezone(local)
            return dt_local.strftime("%I:%M %p %Z").lstrip("0")
        except Exception:
            return None

    import re

    time_re = re.compile(r"(\d{1,2}:\d{2}\s*(?:AM|PM)\s*ET)")
    for col in schedule_df.columns:
        day_label = str(col)
        out.setdefault(day_label, [])
        series = schedule_df[col].dropna().astype(str)
        for cell in series.tolist():
            # Cells may contain multiple lines — split heuristically
            parts = [p.strip() for p in re.split(r"\n|\r|\|", cell) if p.strip()]
            # If no explicit line breaks, split on double spaces
            if len(parts) == 1 and "  " in parts[0]:
                parts = [p.strip() for p in parts[0].split("  ") if p.strip()]

            for p in parts:
                m = time_re.search(p)
                if m:
                    time_et = m.group(1)
                    event = p.replace(time_et, "").strip(" -–—\t")
                    out[day_label].append(
                        CalendarEvent(
                            day_label=day_label,
                            event=event,
                            time_et=time_et,
                            time_local=convert_et_to_local(time_et),
                            url=None,
                        )
                    )
                else:
                    # all-day/no-time entry
                    out[day_label].append(
                        CalendarEvent(
                            day_label=day_label,
                            event=p.strip(" -–—\t"),
                            time_et=None,
                            time_local=None,
                            url=None,
                        )
                    )

    # Clean out empty day columns
    out = {k: v for k, v in out.items() if v}
    return out
