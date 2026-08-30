"""What a good answer LOOKS LIKE, per kind of question.

WHY THIS EXISTS
---------------
The engine could explore the warehouse competently and still write this:

    "The purchasing trend shows variability across hospitals and months, with examples
     including 80 units purchased by HC05 in December 2025 and 50 units in April 2026.
     WHAT DRIVES IT: This purchasing data is grouped by year, month, and hospital.
     The material ID for Keytruda is 101313."

That is not analysis. It is two cells read aloud and a description of the SQL. The engine
retrieved fine and reasoned not at all, because nothing in it knew what a TREND answer
owes the reader: one series over time, a direction, a magnitude, where it turned.

A human analyst does not rediscover that per question — they know the shape of the answer
before they open the data, and the shape tells them which numbers they still owe. That
knowledge was the missing piece, and it is knowledge, not intelligence, so it belongs in
code rather than in a prompt.

HOW IT SCALES
-------------
Each shape declares the SLOTS a complete answer needs and a `derive()` that computes the
facts deterministically from the rows — direction, % change, concentration, share. The
planner fills slots instead of inventing sub-questions; the synthesiser is handed computed
facts instead of raw tables and SQL. Adding a shape adds a whole class of question, and no
derived number is ever produced by the model.
"""
from __future__ import annotations

import re
from typing import Any, Callable


# ── helpers ──────────────────────────────────────────────────────────────────
_MONTHS = ["january", "february", "march", "april", "may", "june",
           "july", "august", "september", "october", "november", "december"]
_MIDX = {m: i for i, m in enumerate(_MONTHS)}
_MIDX.update({m[:3]: i for i, m in enumerate(_MONTHS)})


def _num(v) -> float | None:
    try:
        f = float(v)
        return f if f == f else None      # NaN check
    except (TypeError, ValueError):
        return None


def _measure_col(cols: list[str], rows: list[dict]) -> str | None:
    """The column being measured — never an identifier, never the time axis."""
    ident = re.compile(r"^(year|yr|month|month_num|month_name|period|quarter|week|id|.*_id|.*_code|"
                       r"material|plant|hospital|vendor.*|manufacturer.*|category|.*_desc|.*_name)$", re.I)
    best, best_n = None, -1
    for c in cols:
        if ident.match(c):
            continue
        vals = [_num(r.get(c)) for r in rows]
        n = sum(1 for v in vals if v is not None)
        if n > best_n:
            best, best_n = c, n
    return best if best_n > 0 else None


def _time_col(cols: list[str]) -> str | None:
    for c in cols:
        if c.lower() in ("month", "month_name", "period", "posting_date", "gr_date", "date"):
            return c
    return None


def _label_col(cols: list[str], rows: list[dict]) -> str | None:
    for c in cols:
        if any(k in c.lower() for k in ("desc", "name", "hospital", "plant", "vendor",
                                        "manufacturer", "category", "group", "bucket", "band")):
            return c
    for c in cols:
        if rows and isinstance(rows[0].get(c), str):
            return c
    return None


def _chrono(rows: list[dict], tcol: str, ycol: str | None) -> list[dict]:
    def key(r):
        raw = str(r.get(tcol) or "").strip().lower()
        idx = _MIDX.get(raw, _MIDX.get(raw[:3], 99))
        if idx == 99:
            return (0, raw)
        y = _num(r.get(ycol)) if ycol else 0
        return (y or 0, idx)
    try:
        return sorted(rows, key=key)
    except Exception:
        return rows


def _pct(a: float, b: float) -> float | None:
    return None if not b else (a - b) / abs(b) * 100.0


# ── the derivations ──────────────────────────────────────────────────────────
def derive_trend(res: dict) -> dict:
    """Direction, size and turning point of ONE series — computed, never narrated.

    The engine previously handed the writer a hospital-by-month grid and let it improvise,
    which is how "shows variability" became the headline of a trend answer.
    """
    cols, rows = res.get("columns") or [], res.get("rows") or []
    tcol = _time_col(cols)
    mcol = _measure_col(cols, rows)
    if not (tcol and mcol and len(rows) >= 2):
        return {}
    ycol = next((c for c in cols if c.lower() in ("year", "yr")), None)
    # collapse to ONE point per period: a trend is a single series, and any other
    # dimension in the result has to be summed away before it is one
    agg: dict[str, float] = {}
    order: list[str] = []
    for r in _chrono(rows, tcol, ycol):
        k = f"{str(r.get(ycol)).split('.')[0] + ' ' if ycol else ''}{r.get(tcol)}"
        v = _num(r.get(mcol))
        if v is None:
            continue
        if k not in agg:
            agg[k] = 0.0
            order.append(k)
        agg[k] += v
    if len(order) < 2:
        return {}
    series = [(k, agg[k]) for k in order]
    first, last = series[0], series[-1]
    hi = max(series, key=lambda x: x[1])
    lo = min(series, key=lambda x: x[1])
    change = _pct(last[1], first[1])
    direction = ("flat" if change is None or abs(change) < 5
                 else "rising" if change > 0 else "falling")
    return {"measure": mcol, "periods": len(series),
            "series": [{"period": k, "value": v} for k, v in series],
            "first": {"period": first[0], "value": first[1]},
            "last": {"period": last[0], "value": last[1]},
            "peak": {"period": hi[0], "value": hi[1]},
            "trough": {"period": lo[0], "value": lo[1]},
            "change_pct_first_to_last": change, "direction": direction,
            "swing_pct_trough_to_peak": _pct(hi[1], lo[1])}


def derive_ranking(res: dict) -> dict:
    """Concentration facts: what the top one and top three actually account for."""
    cols, rows = res.get("columns") or [], res.get("rows") or []
    mcol = _measure_col(cols, rows)
    lcol = _label_col(cols, rows)
    if not (mcol and lcol and len(rows) >= 2):
        return {}
    pairs = [(str(r.get(lcol)), _num(r.get(mcol))) for r in rows]
    pairs = [(l, v) for l, v in pairs if v is not None]
    if len(pairs) < 2:
        return {}
    pairs.sort(key=lambda x: -x[1])
    total = sum(v for _, v in pairs) or 1.0
    return {"measure": mcol, "dimension": lcol, "n": len(pairs),
            "top": [{"label": l, "value": v, "share_pct": v / total * 100} for l, v in pairs[:5]],
            "top1_share_pct": pairs[0][1] / total * 100,
            "top3_share_pct": sum(v for _, v in pairs[:3]) / total * 100,
            "tail_n": max(0, len(pairs) - 3)}


def derive_exposure(res: dict) -> dict:
    """Totals and the worst band — for risk/expiry/aging style questions."""
    cols, rows = res.get("columns") or [], res.get("rows") or []
    mcol = _measure_col(cols, rows)
    lcol = _label_col(cols, rows)
    if not mcol or not rows:
        return {}
    vals = [(str(r.get(lcol)) if lcol else "", _num(r.get(mcol))) for r in rows]
    vals = [(l, v) for l, v in vals if v is not None]
    if not vals:
        return {}
    total = sum(v for _, v in vals)
    worst = max(vals, key=lambda x: x[1])
    return {"measure": mcol, "total": total, "bands": len(vals),
            "worst": {"label": worst[0], "value": worst[1],
                      "share_pct": (worst[1] / total * 100) if total else None}}


SHAPES: dict[str, dict[str, Any]] = {
    "trend": {
        "when": "how something changed over time (trend, over months, growth, decline)",
        "slots": [
            {"id": "series", "need": "the measure for the entity, one row per PERIOD, "
                                     "summed across every other dimension", "required": True},
            {"id": "breakdown", "need": "the same measure split by hospital, category or vendor, "
                                        "to say WHERE the movement came from", "required": False},
        ],
        "derive": derive_trend,
        "answer_must": "state the direction, the change from first period to last, the peak "
                       "and the trough by name, and where the movement came from",
    },
    "ranking": {
        "when": "who or what is biggest/smallest (top N, most, least, concentration)",
        "slots": [
            {"id": "ranked", "need": "the measure by the entity being ranked, ordered, "
                                     "enough rows to see the tail", "required": True},
            {"id": "context", "need": "what the leader is made of, or how it compares on a "
                                      "second measure", "required": False},
        ],
        "derive": derive_ranking,
        "answer_must": "name the leader with its value AND its share of the total, say what "
                       "the top three account for, and whether the tail matters",
    },
    "exposure": {
        "when": "how much is at risk (expiring, aging, stuck, out of stock, overstocked)",
        "slots": [
            {"id": "total", "need": "the exposure split into bands or buckets", "required": True},
            {"id": "worst", "need": "the specific items or sites carrying most of it", "required": False},
        ],
        "derive": derive_exposure,
        "answer_must": "give the total, the worst band with its share, and who is carrying it",
    },
    "comparison": {
        "when": "how two or more things compare (A vs B, this site vs the rest)",
        "slots": [
            {"id": "sides", "need": "the measure for each side of the comparison, one row each",
             "required": True},
            {"id": "baseline", "need": "the overall figure the sides should be judged against",
             "required": False},
        ],
        "derive": derive_ranking,
        "answer_must": "give both sides, the gap between them, and the gap against the baseline",
    },
    "diagnosis": {
        "when": "why something happened, what is driving it, what would fix it",
        "slots": [
            {"id": "headline", "need": "the figure the question is about", "required": True},
            {"id": "decomposition", "need": "that figure split by the most likely driver",
             "required": True},
            {"id": "alternative", "need": "a COMPETING explanation, so the answer can rule "
                                          "something out rather than only confirm", "required": False},
        ],
        "derive": derive_ranking,
        "answer_must": "rank the drivers by contribution and explicitly rule out at least one "
                       "competing explanation with the figure that rules it out",
    },
    "lookup": {
        "when": "a single fact or figure with no analysis required",
        "slots": [{"id": "value", "need": "the figure asked for", "required": True}],
        "derive": lambda res: {},
        "answer_must": "give the figure and its scope, in one or two sentences, and stop",
    },
}


def catalog() -> str:
    return "\n".join(f"- {k}: {v['when']}" for k, v in SHAPES.items())


def get(name: str) -> dict[str, Any]:
    return SHAPES.get((name or "").strip().lower(), SHAPES["lookup"])


def derive_all(shape_name: str, findings: list[dict]) -> list[dict]:
    """Run the shape's derivation over every finding, keeping whatever it can compute."""
    fn: Callable[[dict], dict] = get(shape_name)["derive"]
    out = []
    for f in findings:
        try:
            d = fn(f.get("res") or {})
        except Exception:
            d = {}
        if d:
            out.append({"from": f.get("purpose", "")[:80], **d})
    return out
