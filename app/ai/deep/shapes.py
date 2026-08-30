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


# Measures you can add up, and measures you cannot. Days, percentages, averages, medians,
# rates and scores are NOT additive: summing them produces a number with no meaning, and
# taking a share of that number produces a statistic that sounds precise and says nothing.
_NON_ADDITIVE = re.compile(
    r"(days?|_pct|percent|share|rate|ratio|avg|average|median|mean|score|index|per_)", re.I)


def _is_additive(measure: str) -> bool:
    return not _NON_ADDITIVE.search(measure or "")


def derive_ranking(res: dict) -> dict:
    """Concentration facts — but only where concentration is a real idea.

    A live brief said "these top three vendors account for 43.2% of the total lead time".
    Lead time is measured in days: the sum of everyone's days is not a quantity anyone
    holds, and a share of it is arithmetic without meaning. Concentration applies to value,
    quantity and counts; for a rate or an average the useful facts are the spread — the
    slowest, the fastest, the middle.
    """
    cols, rows = res.get("columns") or [], res.get("rows") or []
    mcol = _measure_col(cols, rows)
    lcol = _label_col(cols, rows)
    if not (mcol and lcol and rows):
        return {}
    pairs = [(str(r.get(lcol)), _num(r.get(mcol))) for r in rows]
    pairs = [(l, v) for l, v in pairs if v is not None]
    # A single row still gets a derivation — precisely so it can carry the warning that
    # there is nothing to take a percentage of. Returning {} left the model to compute the
    # share itself, which is where "100% of ₹649.91 Cr from one vendor" came from.
    if not pairs:
        return {}
    pairs.sort(key=lambda x: -x[1])
    total = sum(v for _, v in pairs) or 1.0
    out: dict[str, Any] = {"measure": mcol, "dimension": lcol, "n": len(pairs),
                           "top": [{"label": l, "value": v} for l, v in pairs[:5]]}

    # A SHARE NEEDS A REAL DENOMINATOR.
    #
    # These percentages were computed over the rows the query happened to RETURN, so a
    # single-row result scored 100% and the brief said "100% of the total procurement value
    # of ₹649.91 Cr is sourced from one vendor" (the true leader is 45.8%) and "GLASS PAPER
    # accounts for 100.0% of out-of-stock demand". Both are false, both sound authoritative,
    # and both came from dividing a number by itself. Below three rows there is nothing to
    # take a share OF, and even above it the denominator is only the returned set — which is
    # said out loud rather than implied.
    if not _is_additive(mcol):
        vals = sorted(v for _, v in pairs)
        mid = vals[len(vals) // 2]
        out["spread"] = {"highest": {"label": pairs[0][0], "value": pairs[0][1]},
                         "lowest": {"label": pairs[-1][0], "value": pairs[-1][1]},
                         "median": mid}
        out["share_note"] = (f"'{mcol}' is a rate/average, not an additive quantity — do NOT "
                             f"sum it or express any share of it. Compare the values, the "
                             f"spread and the median instead.")
        return out

    if len(pairs) >= 3:
        for i, item in enumerate(out["top"]):
            item["share_of_returned_pct"] = pairs[i][1] / total * 100
        out["top1_share_of_returned_pct"] = pairs[0][1] / total * 100
        out["top3_share_of_returned_pct"] = sum(v for _, v in pairs[:3]) / total * 100
        out["tail_n"] = max(0, len(pairs) - 3)
        out["share_note"] = (f"Percentages are shares of the {len(pairs)} rows returned, NOT of the "
                             f"company total, unless this query covered everything. Say 'of the top "
                             f"{len(pairs)}' or run an unfiltered total before claiming a share of all.")
    else:
        out["share_note"] = (f"Only {len(pairs)} row(s) returned — there is no denominator here, so do "
                             f"NOT state any percentage share. Report the values alone.")
    return out


_DAY_BAND = re.compile(r"^(\d+)\s*[-–]\s*(\d+)\s*d?$|^(\d+)\s*d?\+$", re.I)


def _cumulative_bands(rows: list[dict], lcol: str, cols: list[str]) -> dict:
    """Cumulative totals for day-banded results, computed rather than left to the model.

    "How much is expiring in the next 90 days" is answered by 0-30d + 31-90d — and NOT by
    the 'Expired' band, which is stock that expired months ago. Handed the four canonical
    buckets, the model summed all of them and reported 101,005 units against a true 45,223,
    then 87,775 on the next run. The arithmetic is trivial and the judgement is not; doing
    it here removes both the error and the variance, and makes 'expired' vs 'expiring' a
    property of the answer rather than of the model's mood.
    """
    out: dict[str, Any] = {}
    bands: list[tuple[int, dict]] = []
    expired = None
    for r in rows:
        label = str(r.get(lcol) or "").strip()
        if label.lower().startswith("expired"):
            expired = r
            continue
        m = _DAY_BAND.match(label.replace(" ", ""))
        if m:
            upper = int(m.group(2) or m.group(3) or 0)
            bands.append((upper, r))
    if not bands:
        return out
    bands.sort(key=lambda x: x[0])
    measures = [c for c in cols if c != lcol and any(isinstance(r.get(c), (int, float)) for r in rows)]
    for upper, _ in bands:
        agg = {m: sum(_num(r.get(m)) or 0 for u, r in bands if u <= upper) for m in measures}
        out[f"within_{upper}_days"] = agg
    if expired is not None:
        out["already_expired"] = {m: _num(expired.get(m)) or 0 for m in measures}
        out["note"] = ("'within_N_days' EXCLUDES already-expired stock, which is reported "
                       "separately as already_expired. Do not add them together.")
    return out


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
    out = {"measure": mcol, "total": total, "bands": len(vals),
           "worst": {"label": worst[0], "value": worst[1],
                     "share_pct": (worst[1] / total * 100) if total else None}}
    if lcol:
        cum = _cumulative_bands(rows, lcol, cols)
        if cum:
            out["cumulative"] = cum
    return out


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
