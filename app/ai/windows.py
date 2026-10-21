"""The months the data covers, read from the data, for every text the model is given.

The AI's prompts used to name the months outright ("Jun–Aug 2026 ONLY", "Dec-2025 … May-2026",
DATE '2026-05-31'), so a new month of data would have left the model insisting it did not
exist. The texts now carry tokens, and fill() puts in what the data says today:

  <<REPORTS>>        HCG's formulary / spend reports: "Jun–Aug 2026"   (formulary_summary.period)
  <<REPORTS_RANGE>>  the same as '2026-06..2026-08'
  <<REPORTS_LATEST>> the newest report month: "Aug-2026"
  <<SALES>>          SAP sales / purchase / consumption: "Dec-2025…May-2026"   (sales_monthly.month)
  <<SALES_SHORT>>    "Dec–May"
  <<SALES_MONTHS>>   how many months: "6"
  <<SALES_START>>    first day of the sales window: '2025-12-01'
  <<SALES_END>>      first day after it: '2026-06-01'
  <<SALES_YEARS>>    how those months fall in calendar years: "'2025' covers Dec only (1 month),
                     '2026' covers Jan–May (5 months)"
  <<SNAPSHOT>>       the inventory snapshot date: 2026-05-31   (fact_inventory.snapshot_date)
  <<SNAPSHOT_SQL>>   the same as a SQL value: DATE '2026-05-31'

fill() runs where a text is handed over: the two model-call wrappers (orchestrator._chat and
deep.llm.chat) fill every message and tool description, and the few texts shown to the user
directly fill themselves. A source that cannot be read gives a neutral phrase ("the report
months"), never a guessed date. Each value is cached against its file's mtime, so a rebuilt
table changes the text on the next call.
"""
from __future__ import annotations

import calendar
import os
import re
from functools import lru_cache
from pathlib import Path

from app.core.config import settings

_YM = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
TOKEN = re.compile(r"<<([A-Z_]+)>>")


def _key(path: Path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return str(path), st.st_mtime_ns, st.st_size


@lru_cache(maxsize=8)
def _months_at(path: str, _mtime: int, _size: int, column: str) -> tuple[str, ...]:
    try:
        import pyarrow.parquet as pq
        values = pq.read_table(path, columns=[column]).column(column).to_pylist()
    except Exception:  # unreadable or wrong schema: the window is unknown
        return ()
    return tuple(sorted({str(v)[:7] for v in values if v is not None and _YM.match(str(v)[:7])}))


@lru_cache(maxsize=4)
def _max_date_at(path: str, _mtime: int, _size: int, column: str) -> str | None:
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        hi = pc.max(pq.read_table(path, columns=[column]).column(column)).as_py()
    except Exception:
        return None
    return str(hi)[:10] if hi is not None else None


def report_months() -> tuple[str, ...]:
    """Months of HCG's formulary / spend reports ('2026-06', ...), from the published summary."""
    k = _key(Path(settings.FORMULARY_DATA_DIR) / "formulary_summary.parquet")
    return _months_at(*k, "period") if k else ()


def sales_months() -> tuple[str, ...]:
    """Months of the SAP sales / purchase / consumption tables, from sales_monthly."""
    k = _key(Path(settings.KPI_DIR) / "sales_monthly.parquet")
    return _months_at(*k, "month") if k else ()


def snapshot_date() -> str | None:
    """The inventory snapshot date ('2026-05-31'), from fact_inventory."""
    k = _key(Path(settings.CURATED_DIR) / "fact_inventory.parquet")
    return _max_date_at(*k, "snapshot_date") if k else None


# ---------- wording ----------

def _abbr(ym: str) -> str:
    return calendar.month_abbr[int(ym[5:7])]


def month_label(ym: str) -> str:
    """'2026-05' -> 'May-2026'."""
    return f"{_abbr(ym)}-{ym[:4]}"


def span(months) -> str:
    """('2026-06', ..., '2026-08') -> 'Jun–Aug 2026'; across years 'Dec 2025–May 2026'."""
    if not months:
        return ""
    lo, hi = months[0], months[-1]
    if lo == hi:
        return f"{_abbr(lo)} {lo[:4]}"
    if lo[:4] == hi[:4]:
        return f"{_abbr(lo)}–{_abbr(hi)} {hi[:4]}"
    return f"{_abbr(lo)} {lo[:4]}–{_abbr(hi)} {hi[:4]}"


def _next_month(ym: str) -> str:
    y, m = int(ym[:4]), int(ym[5:7])
    return f"{y + (m == 12)}-{m % 12 + 1:02d}"


def _years(months) -> str:
    by: dict[str, list[str]] = {}
    for m in months:
        by.setdefault(m[:4], []).append(m)
    parts = []
    for y, ms in by.items():
        names = _abbr(ms[0]) if len(ms) == 1 else f"{_abbr(ms[0])}–{_abbr(ms[-1])}"
        parts.append(f"'{y}' covers {names} only ({len(ms)} month{'s' if len(ms) != 1 else ''})")
    return ", ".join(parts)


def values() -> dict[str, str]:
    """Every token's text for the data as it is now."""
    r, s, snap = report_months(), sales_months(), snapshot_date()
    return {
        "REPORTS": span(r) or "the report months",
        "REPORTS_RANGE": f"{r[0]}..{r[-1]}" if r else "the report months",
        "REPORTS_LATEST": month_label(r[-1]) if r else "the latest report month",
        "SALES": f"{month_label(s[0])}…{month_label(s[-1])}" if s else "the sales months",
        "SALES_SHORT": f"{_abbr(s[0])}–{_abbr(s[-1])}" if s else "sales-window",
        "SALES_MONTHS": str(len(s)) if s else "few",
        "SALES_START": f"{s[0]}-01" if s else "",
        "SALES_END": f"{_next_month(s[-1])}-01" if s else "",
        "SALES_YEARS": _years(s) if s else "check which months each year holds",
        "SNAPSHOT": snap or "the inventory snapshot date",
        "SNAPSHOT_SQL": f"DATE '{snap}'" if snap else "(SELECT MAX(snapshot_date)::DATE FROM fact_inventory)",
    }


def fill(text):
    """The text with every <<TOKEN>> replaced; anything that is not a string comes back as is."""
    if not isinstance(text, str) or "<<" not in text:
        return text
    v = values()
    return TOKEN.sub(lambda m: v.get(m.group(1), m.group(0)), text)


def fill_request(kwargs: dict) -> dict:
    """A chat.completions request with its messages and tool descriptions filled (a copy)."""
    out = dict(kwargs)
    msgs = kwargs.get("messages")
    if isinstance(msgs, list):
        out["messages"] = [_fill_message(m) for m in msgs]
    tools = kwargs.get("tools")
    if isinstance(tools, list):
        out["tools"] = [_fill_deep(t) for t in tools]
    return out


def _fill_message(m):
    if not isinstance(m, dict):
        return m
    c = m.get("content")
    if isinstance(c, str) and "<<" in c:
        return {**m, "content": fill(c)}
    if isinstance(c, list):
        return {**m, "content": [_fill_deep(p) for p in c]}
    return m


def _fill_deep(x):
    if isinstance(x, str):
        return fill(x)
    if isinstance(x, dict):
        return {k: _fill_deep(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_fill_deep(v) for v in x]
    return x
