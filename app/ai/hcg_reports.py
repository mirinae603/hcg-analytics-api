# app/ai/hcg_reports.py — the Jun–Aug 2026 HCG reports (formulary compliance, brand rules,
# scheme / Attis brand savings, cash & TPA margin loss) shaped for the chatbot's get_kpi tool.
#
# WHY THIS EXISTS: the dashboard handlers in app/api/formulary.py, spend.py and brand_rules.py
# already compute every one of these figures exactly as HCG's monthly report does, so the AI
# must quote THEM, never re-derive them. But their answers are built for a page, not a model:
#   - 9–18K characters each, where get_kpi's tool result is cut at 6,000 (mid-JSON) in fast
#     mode and 2,600 in deep mode, so the scope, the hospital ranking and the notes were lost;
#   - money keys like `formulary`, `total`, `saving_missed` or `topline_increase` escape the
#     ₹ formatter (orchestrator.infer_kind reads them as plain numbers), and a raw 4347123.45
#     is exactly what the model once mis-converted by 10x;
#   - list views need a starlette Request, a doctor_id (a name matched nothing — a confident
#     zero) and a material code ('Keytruda' matched nothing), and Month only takes YYYY-MM.
#
# So every function here calls the SAME handlers in-process (every figure stays the
# dashboard's own) and reshapes the answer into one small, flat payload:
#
#     {"totals": {flat scalars}, "scope": {...}, "view": "summary", "rows": [...],
#      "rows_total": n, "notes": [<= 2 short], "data_window": "2026-06..2026-08"}
#
# Money keys end in _cost or _value and percentages in _pct (numbers, 2 dp), so the ₹ / %
# formatter always catches them. The formatted JSON is kept under BUDGET by trimming rows from
# the tail (rows come most-important first) and saying how many there were in rows_total.
#
# Errors: data not built (503) -> {"error": "<plain sentence>"} with no CLI text in it; a bad
# argument (400 / 404) -> ValueError(detail), which reaches the model as a tool error it can
# correct from. No patient identifier exists in any of these tables; doctor names appear only
# in the doctor views, as on the dashboard.
from __future__ import annotations

import functools
import json
import re
from typing import Callable, Optional
from urllib.parse import urlencode

from fastapi import HTTPException
from starlette.requests import Request

from app.ai import hospitals as _hospitals
from app.ai import windows
from app.api import brand_rules as B
from app.api import formulary as F
from app.api import spend as S

BUDGET = 3800                   # formatted JSON chars; fast mode's cut and deep mode's 4,000 sit above it
PAGE = 60                       # rows asked of a list handler before the budget trims them
ALL_ROWS = 5000                 # rows asked when a list is summed (one doctor's or one item's rows)

FORMULARY_VIEWS = ("summary", "trend", "hospitals", "segments", "types", "payers", "classes", "doctors", "items")
BRAND_RULE_VIEWS = ("measures", "trend", "hospitals", "doctors", "items", "book")
SAVING_VIEWS = ("summary", "trend", "hospitals", "doctors", "items")
ATTIS_VIEWS = SAVING_VIEWS + ("consumption",)
CASH_VIEWS = SAVING_VIEWS


# ── Plumbing ──────────────────────────────────────────────────────────────────
def _report(label: str):
    """Map the handlers' HTTP errors to what a tool caller can use. A 503 is 'not built':
    a plain sentence, because the handler's own text names a CLI command the model would
    repeat to the user. Anything else is the caller's argument: a ValueError it can fix."""
    def wrap(fn):
        @functools.wraps(fn)
        def run(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except HTTPException as e:
                if e.status_code == 503:
                    return {"error": f"The {label} report data is not loaded on this server."}
                raise ValueError(str(e.detail)) from None
        return run
    return wrap


def _text(v) -> Optional[str]:
    """A plain string for an argument the model may send as a list, a number or blank."""
    if v is None:
        return None
    if isinstance(v, (list, tuple, set)):
        v = ",".join(str(x).strip() for x in v if x is not None and str(x).strip())
    s = str(v).strip()
    return s or None


def _req(**query) -> Request:
    # The list handlers read paging / search from a starlette Request; a synthetic one is all
    # they need (verified on fastapi 0.143 / starlette 1.7).
    q = {k: v for k, v in query.items() if v not in (None, "")}
    return Request({"type": "http", "query_string": urlencode(q).encode(), "headers": []})


def _page(fn: Callable, *, size: int = PAGE, search: Optional[str] = None, **kwargs) -> tuple[list, int]:
    # global_filter is a literal search (app.core.table_query), so the text goes as it is.
    out = fn(_req(page=0, page_size=size, global_filter=search), **kwargs)
    return out["data"], int(out["total"])


def _rows_for(fn: Callable, col: str, value: str, **kwargs) -> list:
    """Every row of a list handler whose `col` is exactly `value` (the handlers only offer a
    contains-search across all columns, so search first and keep the exact matches)."""
    data, _ = _page(fn, size=ALL_ROWS, search=value, **kwargs)
    return [r for r in data if str(r.get(col)) == value]


def _p(v) -> Optional[float]:
    """A percentage as the dashboard shows it: 2 dp, None when there is no denominator."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else round(f, 2)


def _m(v) -> float:
    try:
        f = float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return round(f, 2) + 0.0                # + 0.0 turns -0.0 into 0.0


def _ratio(num, den) -> Optional[float]:
    # Every report % is a ratio of SUMS; this is only ever called on summed amounts.
    return _p(float(num) / float(den) * 100) if den else None


def _view(view, allowed: tuple, key: str) -> tuple[str, bool]:
    """(view, given?) — given matters for the doctor / item drill-downs, whose natural
    default differs from the report's."""
    s = _text(view)
    if s is None:
        return allowed[0], False
    s = s.lower()
    if s not in allowed:
        raise ValueError(f"view '{view}' is not available for {key}. Use one of: {', '.join(allowed)}.")
    return s, True


_PAYER_IN = {"CASH": "CASH", "CASH(TPA)": "CASH(TPA)", "CASH (TPA)": "CASH(TPA)", "TPA": "CASH(TPA)",
             "INSURANCE": "CASH(TPA)", "GOV": "GOV", "GOVT": "GOV", "GOVERNMENT": "GOV", "SCHEME": "GOV"}


def _payer(v) -> Optional[str]:
    s = _text(v)
    if s is None or s.upper() == "ALL":
        return None
    return _PAYER_IN.get(re.sub(r"\s+", " ", s.upper()), s)     # unknown: the handler's 400 lists the valid ones


# ── Month ─────────────────────────────────────────────────────────────────────
# The handlers take YYYY-MM, a comma list or ALL, and answer 400 to 'August'. The model and
# the user say 'Aug', 'August 2026', 'Aug-26', 'jul,aug', 'Jun to Aug', 'last 2 months'.
_FULL = ("january", "february", "march", "april", "may", "june", "july", "august", "september",
         "october", "november", "december")
_YM = re.compile(r"^(\d{4})[-/.](\d{1,2})$")
_MY = re.compile(r"^(\d{1,2})[-/.](\d{4})$")
_NAMED = re.compile(r"^([a-z]{3,9})\.?(?:[\s'’-]*(\d{4}|\d{2}))?$")
_RANGE = re.compile(r"\s*(?:\.\.|–|—|\bto\b|\bthrough\b|\btill\b|\buntil\b)\s*")
_ALL = re.compile(r"^(all|all months|every month|all (three|3) months|(the |this |whole |entire )?quarter|"
                  r"(the )?(whole|entire) (period|window)|(three|3) months|overall)$")
_LATEST = re.compile(r"^(latest|latest month|current month|this month|last month|most recent( month)?)$")
_LAST_N = re.compile(r"^(?:last|past|previous|recent) (\d{1,2}|two|three) months$")


def _available(periods: list) -> str:
    return ("Available months: " + ", ".join(F._month_label(p) for p in periods)
            + " (or 'all'). These reports cover " + windows.span(periods) + " only.")


def _month_no(word: str) -> Optional[int]:
    w = word.lower().rstrip(".")
    if len(w) < 3:
        return None
    return next((i + 1 for i, full in enumerate(_FULL) if full.startswith(w)), None)


def _one_month(tok: str, periods: list) -> Optional[str]:
    """'YYYY-MM' for one month written any common way; None if it is not a month at all.
    A month without a year takes the latest year the data has for it."""
    t = tok.strip().strip(".").strip()
    m = _YM.match(t)
    if m:
        y, mo = int(m[1]), int(m[2])
    elif _MY.match(t):
        m = _MY.match(t)
        mo, y = int(m[1]), int(m[2])
    else:
        m = _NAMED.match(t)
        if not m or _month_no(m[1]) is None:
            return None
        mo = _month_no(m[1])
        y = None if not m[2] else (2000 + int(m[2]) if len(m[2]) == 2 else int(m[2]))
    if not 1 <= mo <= 12:
        return None
    if y is None:
        hits = [p for p in periods if int(p[5:7]) == mo]
        if not hits:
            raise ValueError(f"No report data for {_FULL[mo - 1].title()}. {_available(periods)}")
        return hits[-1]
    return f"{y:04d}-{mo:02d}"


def _month_token(tok: str, periods: list) -> list:
    one = _one_month(tok, periods)
    if one:
        return [one]
    parts = _RANGE.split(tok)
    if len(parts) != 2:
        parts = tok.split("-")              # 'jun-aug' (after 'aug-26' failed as one month)
    if len(parts) == 2:
        a, b = _one_month(parts[0], periods), _one_month(parts[1], periods)
        if a and b:
            a, b = min(a, b), max(a, b)
            # endpoints outside the data go to the handler, whose 400 names the months it has
            return [p for p in periods if a <= p <= b] or [a, b]
    raise ValueError(f"Could not read month '{tok}'. Use a month name ('Aug', 'August 2026'), YYYY-MM, "
                     f"a comma list or 'all'. {_available(periods)}")


def _month_arg(month, periods: list) -> Optional[str]:
    """The handlers' Month= for what the caller wrote: None = the latest month (the
    dashboard's default), 'ALL', or a comma list of YYYY-MM."""
    s = _text(month)
    if s is None:
        return None
    low = re.sub(r"\s+", " ", s.lower()).strip()
    if _LATEST.match(low):
        return None
    if _ALL.match(low):
        return "ALL"
    m = _LAST_N.match(low)
    if m:
        n = {"two": 2, "three": 3}.get(m[1]) or int(m[1])
        return "ALL" if n >= len(periods) else ",".join(periods[-n:])
    low = re.sub(r"([a-z])\s*,\s*(\d{4})\b", r"\1 \2", low)      # 'Aug, 2026' is one month, not two
    out: list = []
    for tok in re.split(r"\s*(?:,|;|&|\+|\band\b)\s*", low):
        if tok:
            out += _month_token(tok, periods)
    if not out:
        raise ValueError(f"Could not read month '{s}'. {_available(periods)}")
    return ",".join(sorted(set(out)))


# ── Hospital, doctor, item ────────────────────────────────────────────────────
def _plants(plant) -> Optional[list]:
    """SAP codes for Plant=: None = all hospitals, one code, or several ('HC05,HC06') whose
    figure is pooled. The whole string is tried first: hospital names contain commas."""
    s = _text(plant)
    if s is None or s.upper() in ("ALL", "ALL PLANTS", "ALL HOSPITALS"):
        return None
    try:
        return [F._plant_arg(s)]
    except HTTPException as first:
        # a name as people say it: "KR", "North Bangalore", "Nagpur" (a city with several
        # hospitals gives all of them, pooled)
        alias = _hospitals.resolve(s)
        if alias:
            return list(alias)
        parts = [p for p in re.split(r"\s*(?:,|;|&|\band\b)\s*", s) if p]
        if len(parts) < 2:
            raise ValueError(f"{first.detail} Short names work too: {_hospitals.prompt_line()}.") from None
    codes = []
    for p in parts:
        alias = _hospitals.resolve(p)
        if alias:
            codes.extend(alias)
            continue
        try:
            codes.append(F._plant_arg(p))
        except HTTPException as e:
            raise ValueError(str(e.detail)) from None
    return list(dict.fromkeys(c for c in codes if c)) or None


def _norm(s) -> str:
    s = re.sub(r"\b(dr|doctor|prof|mr|mrs|ms)\b\.?", " ", str(s or "").lower())
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def _doctor(name, frame, plants: Optional[list]) -> tuple[str, str]:
    """(doctor_id, doctor_name) for a name or an id. The handlers filter on an exact
    doctor_id, so a name passed straight through matched nothing and read as a zero."""
    q = _text(name)
    d = frame[["doctor_id", "doctor_name", "plant", "hospital_name"]].astype(str).drop_duplicates()
    if plants:
        d = d[d["plant"].isin(plants)]
    exact = d[d["doctor_id"].str.lower() == q.lower()]
    if len(exact):
        return exact["doctor_id"].iloc[0], exact["doctor_name"].iloc[0]
    toks = _norm(q).split()
    if not toks:
        raise ValueError("Give the doctor's name or doctor_id.")
    words = d["doctor_name"].map(lambda n: _norm(n).split())
    hit = d[words.map(lambda ws: all(any(w.startswith(t) for w in ws) for t in toks))]
    ids = hit.drop_duplicates("doctor_id")
    if len(ids) > 1:
        same = ids[ids["doctor_name"].map(_norm) == _norm(q)]
        ids = same if len(same) == 1 else ids
    if len(ids) == 1:
        return ids["doctor_id"].iloc[0], ids["doctor_name"].iloc[0]
    where = " at the selected hospital(s)" if plants else ""
    if ids.empty:
        raise ValueError(f"No doctor matching '{q}' in this report{where}. Check the spelling, or list "
                         f"doctors with view='doctors'.")
    shown = "; ".join(f"{r.doctor_name} ({r.hospital_name}, doctor_id {r.doctor_id})"
                      for r in ids.head(5).itertuples())
    raise ValueError(f"'{q}' matches {len(ids)} doctors{where}: {shown}. Pass the doctor_id.")


def _material(name, frames: list) -> tuple[str, str]:
    """(code, description) for an item name or a material code ('KEYTRUDA' or '101313')."""
    q = _text(name)
    cols = ["material", "material_desc", "generic_name"]
    items = None
    for f in frames:
        part = f[[c for c in cols if c in f.columns]].astype(str).drop_duplicates()
        items = part if items is None else items.merge(part, how="outer")
    items = items.drop_duplicates()
    if re.fullmatch(r"\d+(\.0+)?", q):
        code = F._code(q)
        hit = items[items["material"] == code]
        return code, (hit["material_desc"].iloc[0] if len(hit) else code)
    toks = _norm(q).split()
    if not toks:
        raise ValueError("Give the item's name or material code.")

    def has_all(text) -> bool:
        t = _norm(text)
        return all(x in t for x in toks)
    m = items["material_desc"].map(has_all)
    if "generic_name" in items.columns:
        m |= items["generic_name"].map(has_all)
    codes = items[m].drop_duplicates("material")
    if len(codes) > 1:
        same = codes[codes["material_desc"].map(_norm) == _norm(q)]
        codes = same if len(same) == 1 else codes
    if len(codes) == 1:
        return codes["material"].iloc[0], codes["material_desc"].iloc[0]
    if codes.empty:
        raise ValueError(f"No item matching '{q}' in this report. Check the spelling or pass the material code.")
    shown = "; ".join(f"{r.material_desc} ({r.material})" for r in codes.head(5).itertuples())
    raise ValueError(f"'{q}' matches {len(codes)} items: {shown}. Pass the material code.")


# ── Payload ───────────────────────────────────────────────────────────────────
_MONEY = re.compile(r"_(cost|value)$")


def _shown(key: str, v):
    """How get_kpi's formatter prints a number (orchestrator._fmt: ₹ Cr/L/K for money keys, a
    % for _pct keys). Mirrored here only to MEASURE the payload; the payload stays numeric."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return v
    if key.endswith("_pct"):
        return f"{v:.2f}%"
    if _MONEY.search(key):
        a = abs(v)
        if a >= 1e7:
            return f"₹{v / 1e7:.2f} Cr"
        if a >= 1e5:
            return f"₹{v / 1e5:.2f} L"
        if a >= 1e3:
            return f"₹{v / 1e3:.1f} K"
        return f"₹{v:.2f}"
    return v


def _as_shown(o, key: str = ""):
    if isinstance(o, dict):
        return {k: _as_shown(v, k) for k, v in o.items()}
    if isinstance(o, list):
        return [_as_shown(x, key) for x in o]
    return _shown(key, o)


def _size(o) -> int:
    return len(json.dumps(_as_shown(o), ensure_ascii=False, default=str))


def _fit(out: dict, budget: int = BUDGET) -> dict:
    """Trim rows from the tail until the formatted payload fits. A cut mid-JSON (what the
    tool result used to get) loses the closing keys; a shorter list with rows_total loses
    nothing the model cannot ask for again."""
    rows = out["rows"]
    over = _size(out) - budget
    while over > 0 and rows:
        over -= _size(rows.pop()) + 1
    if rows and len(rows) < out["rows_total"] and len(out["notes"]) < 2:
        out["notes"].append(f"Showing the first {len(rows)} of {out['rows_total']} rows.")
        while rows and _size(out) > budget:
            rows.pop()
            out["notes"][-1] = f"Showing the first {len(rows)} of {out['rows_total']} rows."
    while _size(out) > budget and out["notes"]:
        out["notes"].pop()
    return out


def _finish(view: str, totals: dict, scope: dict, rows: list, periods: list,
            rows_total: Optional[int] = None, notes=()) -> dict:
    return _fit({
        "totals": totals,
        "scope": scope,
        "view": view,
        "rows": list(rows),
        "rows_total": int(len(rows) if rows_total is None else rows_total),
        "notes": [n for n in notes if n][:2],
        "data_window": f"{periods[0]}..{periods[-1]}" if periods else "",
    })


def _scope(echo: dict, **extra) -> dict:
    """The handler's own scope echo, renamed and without the defaults, plus the drill-downs."""
    s = {"plant": echo.get("plant") or "ALL",
         "hospital": echo.get("his_name") or echo.get("plant_name") or "All hospitals",
         "months": list(echo.get("months") or [])}
    for out_key, in_key in (("patient_type", "type"), ("segment", "segment"), ("segment", "category"),
                            ("payer", "payer")):
        v = echo.get(in_key)
        if v and v != "ALL":
            s[out_key] = v
    s.update({k: v for k, v in extra.items() if v})
    return s


def _handler_note(notes: list) -> Optional[str]:
    """The one handler note that changes what a figure means here: an empty selection
    ('No ... in this selection') or several months pooled."""
    for n in notes or []:
        if n.startswith("No ") or "never an average" in n:
            return n
    return None


def _hospital_name(h: dict) -> str:
    return h.get("his_name") or h.get("name") or h.get("plant")


def _no_drill(view: str, what: str, allowed: str):
    raise ValueError(f"view '{view}' has no {what} filter. With {what}= use view {allowed}.")


# ── Formulary compliance ──────────────────────────────────────────────────────
NOTE_COMPLIANCE = ("compliance_pct = Formulary ÷ (Formulary + Out of Formulary) drug cost: HCG's MIS compliance "
                   "(pooled method, HCG's current one). formulary_consumption_pct = Formulary ÷ ALL drug cost incl. "
                   "Non Formulary: a different, lower figure; never call it compliance.")
NOTE_COMPLIANCE_SHORT = "compliance_pct = Formulary ÷ (Formulary + Out of Formulary) drug cost (HCG's MIS figure)."
NOTE_GROUPS = ("Group A = formulary, B = out of formulary, C = non formulary. Consumables are outside "
               "compliance.")


def _f_split(r: dict) -> dict:
    return {"compliance_pct": _p(r.get("compliance_pct")), "out_of_formulary_cost": _m(r.get("out_of_formulary")),
            "formulary_cost": _m(r.get("formulary")), "non_formulary_cost": _m(r.get("non_formulary"))}


def _f_core(a: float, b: float, c: float, compliance) -> dict:
    return {"compliance_pct": _p(compliance), "formulary_consumption_pct": _ratio(a, a + b + c),
            "formulary_cost": _m(a), "out_of_formulary_cost": _m(b), "non_formulary_cost": _m(c)}


def _f_totals(ov: dict, full: bool) -> dict:
    h = ov["headline"]
    t = {**_f_core(h["formulary"], h["out_of_formulary"], h["non_formulary"], h["compliance_pct"]),
         "drug_cost": _m(h["total"])}
    if full:
        t["onco_compliance_pct"] = _p(h["onco_pct"])
        t["non_onco_compliance_pct"] = _p(h["non_onco_pct"])
        # HCG's brand figures from the same page: GOV-billed restricted / preferred brands and
        # restricted brands billed to cash & TPA patients, each as a share of ALL drug spend.
        for src, name in (("restricted", "restricted_scheme"), ("preferred", "preferred_scheme"),
                          ("restricted_cash_tpa", "restricted_cash_tpa")):
            blk = (ov.get("scheme") or {}).get(src) or {}
            t[f"{name}_cost"] = _m(blk.get("cost"))
            t[f"{name}_share_pct"] = _p(blk.get("share_pct"))
    return t


def _f_doctor_totals(K: dict, doctor_id: str) -> dict:
    rows = _rows_for(F.formulary_doctors, "doctor_id", doctor_id, **K)
    a, b, c = (sum(float(r[k] or 0) for r in rows) for k in ("formulary", "out_of_formulary", "non_formulary"))
    # one row: the handler's own figure (it follows Method); a doctor at several hospitals: pooled
    return _f_core(a, b, c, rows[0]["compliance_pct"] if len(rows) == 1 else _ratio(a, a + b))



@_report("formulary compliance")
def formulary_compliance(Plant=None, Category=None, month=None, payer=None, patient_type=None, segment=None,
                         method=None, view=None, doctor=None, material=None, formulary_group=None) -> dict:
    periods = F._periods()                  # 503 first: nothing below means anything without data
    view, _ = _view(view, FORMULARY_VIEWS, "formulary-compliance")
    plants = _plants(Plant)
    K = {"Month": _month_arg(month, periods), "Type": _text(patient_type), "Segment": _text(segment),
         "Category": _text(Category), "Payer": _payer(payer), "Method": _text(method)}
    if plants and len(plants) > 1:
        return _f_pooled(plants, view, K, periods)
    K["Plant"] = plants[0] if plants else None
    doc = _doctor(doctor, F._table("formulary_by_doctor"), plants) if _text(doctor) else None
    mat = _material(material, [F._table("formulary_by_item")]) if _text(material) else None
    if doc and mat:
        raise ValueError("Pass doctor or material, not both: doctor= with view='items' lists that doctor's items.")
    group = (_text(formulary_group) or "B").upper()
    ov = F.formulary_overview(**K)
    scope = _scope(ov["scope"], method=ov["scope"]["method"])
    no_method = {k: v for k, v in K.items() if k != "Method"}
    item_note = NOTE_GROUPS + (" Items default to group B." if not _text(formulary_group) else "")

    if doc:
        did, dname = doc
        scope.update(doctor=dname, doctor_id=did)
        if view in ("summary", "items"):
            rows, total = _page(F.formulary_items, Group=group, Doctor=did, **no_method)
            scope["formulary_group"] = group
            return _finish("items", _f_doctor_totals(K, did), scope, [_f_item(r) for r in rows], periods, total,
                           [NOTE_COMPLIANCE_SHORT, item_note])
        if view == "doctors":
            rows = _rows_for(F.formulary_doctors, "doctor_id", did, **K)
            return _finish("doctors", _f_doctor_totals(K, did), scope, [_f_doctor_row(r, None) for r in rows],
                           periods, notes=[NOTE_COMPLIANCE_SHORT])
        if view == "trend":
            rows = [{"month": F._month_label(p), **_f_doctor_totals({**K, "Month": p}, did)} for p in periods]
            return _finish("trend", _f_doctor_totals(K, did), scope, rows, periods,
                           notes=[NOTE_COMPLIANCE_SHORT, "The trend shows every month whatever the month filter."])
        _no_drill(view, "doctor", "'summary', 'items', 'doctors' or 'trend'")

    if mat:
        code, desc = mat
        scope.update(item=desc, material=code)
        if view not in ("summary", "doctors"):
            _no_drill(view, "material", "'doctors' (the doctors behind the item)")
        # The item's own figures come from the item table, every group: the doctor x item table
        # is built for groups B and C only, so a formulary (A) item such as KEYTRUDA has no
        # doctor rows and summing them read as a confident zero.
        items = [r for g in F.GROUPS for r in _rows_for(F.formulary_items, "material", code, Group=g, **no_method)]
        groups = sorted({str(r["formulary_group"]) for r in items})
        totals = {"item": desc, "material": code, "formulary_group": "/".join(groups) or None,
                  "formulary_class": "/".join(sorted({str(r["formulary_class"]) for r in items})) or None,
                  "drug_cost": _m(sum(float(r["cost"] or 0) for r in items)),
                  "qty": round(sum(float(r["qty"] or 0) for r in items), 3)}
        rows = []
        if set(groups) & {"B", "C"}:
            rows, _ = _page(F.formulary_doctors, size=ALL_ROWS, Material=code, **K)
            totals["doctors"] = len({(r["plant"], r["doctor_id"]) for r in rows})
        out = [{"doctor": r["doctor_name"], "doctor_id": r["doctor_id"], "hospital": r["hospital_name"],
                "formulary_class": r["formulary_class"], "drug_cost": _m(r["cost"]), "qty": r["qty"]}
               for r in rows[:PAGE]]
        why = ("Doctors who used this item, highest cost first (doctor rows cover its out-of-formulary / "
               "non-formulary use only)." if rows or not items else
               "A formulary (group A) item: it is compliant use, and the report keeps no doctor rows for it.")
        return _finish("doctors", totals, scope, out, periods, len(rows), [NOTE_GROUPS, why])

    note = _handler_note(ov.get("notes"))
    if view == "summary":
        rows = ([{"breakdown": s["label"], **_f_split(s)} for s in ov["segments"]]
                + [{"breakdown": t["type"], **_f_split(t)} for t in ov["by_type"]])
        return _finish("summary", _f_totals(ov, True), scope, rows, periods, notes=[NOTE_COMPLIANCE, note])
    totals = _f_totals(ov, False)
    if view == "trend":
        rows = [{"month": t["label"], **_f_split(t)} for t in ov["trend"]]
        return _finish("trend", totals, scope, rows, periods,
                       notes=[NOTE_COMPLIANCE_SHORT, "The trend shows every month whatever the month filter."])
    if view == "hospitals":
        # two figures per hospital, so every hospital fits the budget: a ranking cut short is
        # exactly the list the model is told to quote site by site
        rows = [{"hospital": _hospital_name(h), "plant": h["plant"], "compliance_pct": _p(h["compliance_pct"]),
                 "out_of_formulary_cost": _m(h["out_of_formulary"])} for h in ov["by_hospital"]]
        return _finish("hospitals", totals, scope, rows, periods, notes=[
            "Every hospital, worst compliance first, whatever the hospital filter. Never add or average these "
            "percentages: a group's figure is its summed Formulary ÷ (Formulary + Out of Formulary).", note])
    if view == "segments":
        rows = [{"segment": s["label"], **_f_split(s)} for s in ov["segments"]]
        return _finish("segments", totals, scope, rows, periods, notes=[NOTE_COMPLIANCE_SHORT, F.NOTE_SEGMENT])
    if view == "types":
        rows = [{"patient_type": t["type"], **_f_split(t)} for t in ov["by_type"]]
        return _finish("types", totals, scope, rows, periods, notes=[NOTE_COMPLIANCE_SHORT, note])
    if view == "payers":
        rows = [{"payer": p, **_f_split(F.formulary_overview(**{**K, "Payer": p})["headline"])} for p in F.PAYERS]
        return _finish("payers", totals, scope, rows, periods, notes=[
            NOTE_COMPLIANCE_SHORT, "One row per payer bucket (GOV = scheme patients), whatever the payer filter."])
    if view == "classes":
        rows = [{"formulary_class": c["formulary_class"], "group": c["formulary_group"], "drug_cost": _m(c["cost"]),
                 "share_pct": _p(c["share_pct"])} for c in ov["by_class"]]
        return _finish("classes", totals, scope, rows, periods,
                       notes=[NOTE_GROUPS + " share_pct = share of all drug cost in scope.", note])
    if view == "doctors":
        rows, total = _page(F.formulary_doctors, **K)
        return _finish("doctors", totals, scope, [_f_doctor_row(r, plants) for r in rows], periods, total,
                       [NOTE_COMPLIANCE_SHORT, "Doctors with the most out-of-formulary cost first."])
    # items
    rows, total = _page(F.formulary_items, Group=group, **no_method)
    scope["formulary_group"] = group
    return _finish("items", totals, scope, [_f_item(r) for r in rows], periods, total, [item_note])


def _f_doctor_row(r: dict, plants: Optional[list]) -> dict:
    row = {"doctor": r["doctor_name"], "doctor_id": r["doctor_id"]}
    if not plants:                      # one hospital in scope: its name on every row is noise
        row["hospital"] = r["hospital_name"]
    return {**row, **_f_split(r)}


def _f_item(r: dict) -> dict:
    return {"item": r["material_desc"], "material": r["material"], "generic": r["generic_name"],
            "formulary_class": r["formulary_class"], "drug_cost": _m(r["cost"]),
            "list_share_pct": _p(r["share_pct"]), "hospitals": r["hospitals"]}


def _f_pooled(plants: list, view: str, K: dict, periods: list) -> dict:
    """Several hospitals at once: amounts summed first, the % taken once. Adding or averaging
    the hospitals' own percentages (what the model did when asked per site) is never HCG's
    figure."""
    if view not in ("summary", "hospitals"):
        raise ValueError(f"Several hospitals are pooled for view 'summary' or 'hospitals' only; call once "
                         f"per hospital for view '{view}'.")
    per = [F.formulary_overview(**K, Plant=code) for code in plants]
    heads = [ov["headline"] for ov in per]
    a, b, c = (sum(float(h[k]) for h in heads) for k in ("formulary", "out_of_formulary", "non_formulary"))
    seg = {k: [next(s for s in ov["segments"] if s["key"] == k) for ov in per] for k in F.SEGMENTS}
    seg_pct = {k: _ratio(sum(float(s["formulary"]) for s in v),
                         sum(float(s["formulary"]) + float(s["out_of_formulary"]) for s in v)) for k, v in seg.items()}
    method = per[0]["scope"]["method"]
    present = [x for x in seg_pct.values() if x is not None]
    compliance = (sum(present) / len(present) if present else None) if method == "mean" else _ratio(a, a + b)
    totals = {**_f_core(a, b, c, compliance), "drug_cost": _m(sum(float(h["total"]) for h in heads)),
              "onco_compliance_pct": seg_pct["onco"], "non_onco_compliance_pct": seg_pct["non-onco"]}
    rows = [{"hospital": ov["scope"]["his_name"] or ov["scope"]["plant_name"], "plant": code,
             **_f_split(ov["headline"])} for code, ov in zip(plants, per)]
    scope = _scope({**per[0]["scope"], "plant": ",".join(plants), "his_name": " + ".join(r["hospital"] for r in rows)},
                   method=method)
    return _finish(view, totals, scope, rows, periods, notes=[
        NOTE_COMPLIANCE_SHORT, f"Pooled over {len(plants)} hospitals: amounts added first, the % taken once "
                               f"(never an average of the hospitals' percentages)."])


# ── Brand rules ───────────────────────────────────────────────────────────────
NOTE_SHARE = ("share_pct = the measure's cost ÷ ALL drug spend in scope (every payer), as HCG's report divides "
              "it. A sale can count in several measures, so never add measures up.")
DEFAULT_MEASURE = {"GOV": "restricted_scheme", "CASH": "restricted_cash_tpa", "CASH(TPA)": "restricted_cash_tpa"}


def _b_measure(measure, payer: Optional[str], rules: list) -> Optional[str]:
    """The rule to report. The rules carry the payer (restricted FOR SCHEME patients...), so
    a payer picks the measure rather than filtering one."""
    s = _text(measure)
    by_name = {r["measure"].lower(): r for r in rules}
    if s:
        r = by_name.get(s.lower().replace(" ", "_").replace("-", "_"))
        if r is None:
            raise ValueError(f"Invalid measure '{s}'. Use one of: {', '.join(r['measure'] for r in rules)}")
        if payer and r["payers"] != ["*"] and payer not in r["payers"]:
            fits = [x["measure"] for x in rules if x["payers"] == ["*"] or payer in x["payers"]]
            raise ValueError(f"Measure '{r['measure']}' counts {'/'.join(r['payers'])} patients, not {payer}. "
                             f"Measures for {payer}: {', '.join(fits)}")
        return r["measure"]
    if payer:
        return DEFAULT_MEASURE.get(payer) if DEFAULT_MEASURE.get(payer) in by_name else None
    return None


def _b_selected(ov: dict, measure: str) -> dict:
    m = next(x for x in ov["measures"] if x["measure"] == measure)
    t = {"measure": measure, "measure_cost": _m(m["cost"]), "share_pct": _p(m["share_pct"]),
         "onco_share_pct": _p(m["onco"]["share_pct"]), "non_onco_share_pct": _p(m["non_onco"]["share_pct"]),
         "drug_cost": _m(ov["drug_spend"]["total"])}
    alt = m.get("alternative")
    if alt:
        # does the brand's Main Sheet block hold another brand the rule accepts?
        t.update(alt_yes_cost=_m(alt.get("yes")), alt_no_cost=_m(alt.get("no")),
                 alt_unlisted_cost=_m(alt.get("not in list")))
    return t


def _b_item(r: dict) -> dict:
    return {"item": r["material_desc"], "material": r["material"], "letter": r["letter"],
            "measure_cost": _m(r["cost"]), "share_of_measure_pct": _p(r["share_pct"]),
            "alternative": r["alternative"], "p1_brand": r.get("p1_desc")}


def _b_doctor_row(r: dict, plants: Optional[list]) -> dict:
    row = {"doctor": r["doctor_name"], "doctor_id": r["doctor_id"]}
    if not plants:
        row["hospital"] = r["hospital_name"]
    return {**row, "measure_cost": _m(r["cost"]), "brands": r["brands"], "share_of_measure_pct": _p(r["share_pct"])}


@_report("brand rules")
def brand_rules(Plant=None, Category=None, month=None, patient_type=None, segment=None, measure=None, view=None,
                doctor=None, material=None, payer=None) -> dict:
    rules = B._rules()                      # 503 when the rules were not built
    periods = F._periods()
    view, given = _view(view, BRAND_RULE_VIEWS, "brand-rules")
    pay = _payer(payer)
    named = _b_measure(measure, pay, rules)
    meas = named or rules[0]["measure"]
    by_doctor = next(r["by_doctor"] for r in rules if r["measure"] == meas)
    plants = _plants(Plant)
    K = {"Month": _month_arg(month, periods), "Type": _text(patient_type), "Segment": _text(segment),
         "Category": _text(Category)}
    if plants and len(plants) > 1:
        return _b_pooled(plants, view, K, meas, periods)
    K["Plant"] = plants[0] if plants else None
    doc = _doctor(doctor, F._table(B.BREACH), plants) if _text(doctor) else None
    mat = (_material(material, [F._table("formulary_by_item"), F._table(B.BOOK)]) if _text(material) else None)
    if doc and mat:
        raise ValueError("Pass doctor or material, not both: doctor= with view='items' lists that doctor's brands.")
    ov = B.rules_overview(Measure=meas, **K)
    scope = _scope(ov["scope"], measure=meas, payer=pay)
    sel = _b_selected(ov, meas)

    if doc:
        did, dname = doc
        scope.update(doctor=dname, doctor_id=did)
        mine = _rows_for(B.rules_doctors, "doctor_id", did, Measure=meas, **K)
        totals = {"measure": meas, "measure_cost": _m(sum(float(r["cost"] or 0) for r in mine)),
                  "brands": mine[0]["brands"] if len(mine) == 1 else None}
        if view in ("measures", "items"):
            rows, total = _page(B.rules_items, Measure=meas, Doctor=did, **K)
            return _finish("items", totals, scope, [_b_item(r) for r in rows], periods, total,
                           ["This doctor's brands counted by the measure, highest cost first."])
        if view == "doctors":
            return _finish("doctors", totals, scope, [_b_doctor_row(r, None) for r in mine], periods)
        if view == "trend":
            rows = [{"month": F._month_label(p), "measure_cost": _m(sum(
                float(r["cost"] or 0) for r in _rows_for(B.rules_doctors, "doctor_id", did, Measure=meas,
                                                         **{**K, "Month": p})))} for p in periods]
            return _finish("trend", totals, scope, rows, periods,
                           notes=["The trend shows every month whatever the month filter."])
        _no_drill(view, "doctor", "'items', 'doctors' or 'trend'")

    if mat:
        code, desc = mat
        scope.update(item=desc, material=code)
        if view == "book":
            rows, _ = _page(B.rules_book, size=ALL_ROWS, search=code, Month=K["Month"], Segment=K["Segment"],
                            Category=K["Category"])
            rows = [_b_book(r) for r in rows if str(r["material"]) == code]
            return _finish("book", {"item": desc, "material": code}, scope, rows, periods,
                           notes=["What each payer may get under HCG's brand rules (ok / breach / watch / info)."])
        if not given and not named:
            # 'Is brand X a problem?' names no rule: report every measure the brand counts in,
            # rather than the first rule's (usually empty) doctor list.
            rows = []
            for r in rules:
                got = _rows_for(B.rules_items, "material", code, Measure=r["measure"], **K)
                cost = sum(float(x["cost"] or 0) for x in got)
                if cost:
                    rows.append({"measure": r["measure"], "label": r["label"], "verdict": r["verdict"],
                                 "measure_cost": _m(cost), "letter": got[0]["letter"]})
            scope.pop("measure", None)
            return _finish("measures", {"item": desc, "material": code, "measures": len(rows)}, scope, rows,
                           periods, notes=[
                               "The rule measures this brand counts in, in this selection (none = it breaks no "
                               "rule). A sale can count in several measures, so never add them up.",
                               "Pass measure= with view='doctors' for the doctors behind one measure."])
        if view == "doctors" or (not given and by_doctor):
            rows, total = _page(B.rules_doctors, size=ALL_ROWS, Measure=meas, Material=code, **K)
            totals = {"item": desc, "material": code, "measure": meas,
                      "measure_cost": _m(sum(float(r["cost"] or 0) for r in rows)), "doctors": total}
            return _finish("doctors", totals, scope, [_b_doctor_row(r, plants) for r in rows[:PAGE]], periods,
                           total, ["Doctors who used this brand under the measure, highest cost first."])
        if view in ("measures", "items"):
            rows = _rows_for(B.rules_items, "material", code, Measure=meas, **K)
            totals = {"item": desc, "material": code, "measure": meas,
                      "measure_cost": _m(sum(float(r["cost"] or 0) for r in rows))}
            return _finish("items", totals, scope, [_b_item(r) for r in rows], periods, notes=[NOTE_SHARE])
        _no_drill(view, "material", "'doctors', 'items' or 'book'")

    note = _handler_note(ov.get("notes"))
    if view == "measures":
        fits = [m for m in ov["measures"] if not pay or m["payers"] == ["*"] or pay in m["payers"]]
        rows = [{"measure": m["measure"], "label": m["label"], "verdict": m["verdict"],
                 "measure_cost": _m(m["cost"]), "share_pct": _p(m["share_pct"]),
                 "onco_cost": _m(m["onco"]["cost"]), "onco_share_pct": _p(m["onco"]["share_pct"]),
                 "non_onco_cost": _m(m["non_onco"]["cost"]), "non_onco_share_pct": _p(m["non_onco"]["share_pct"])}
                for m in fits]
        d = ov["drug_spend"]
        totals = {"drug_cost": _m(d["total"]), "onco_drug_cost": _m(d["onco"]), "non_onco_drug_cost": _m(d["non_onco"])}
        return _finish("measures", totals, scope, rows, periods, notes=[NOTE_SHARE, note])
    if view == "trend":
        rows = [{"month": t["label"], "measure_cost": _m(t["cost"]), "share_pct": _p(t["share_pct"])}
                for t in ov["trend"]]
        return _finish("trend", sel, scope, rows, periods,
                       notes=[NOTE_SHARE, "The trend shows every month whatever the month filter."])
    if view == "hospitals":
        rows = [{"hospital": _hospital_name(h), "plant": h["plant"], "measure_cost": _m(h["cost"]),
                 "share_pct": _p(h["share_pct"])} for h in ov["by_hospital"]]
        return _finish("hospitals", sel, scope, rows, periods, notes=[
            "Every hospital, highest cost first, whatever the hospital filter. share_pct = share of THAT "
            "hospital's drug spend; never add or average them.", note])
    if view == "doctors":
        rows, total = _page(B.rules_doctors, Measure=meas, **K)
        return _finish("doctors", sel, scope, [_b_doctor_row(r, plants) for r in rows], periods, total,
                       ["share_of_measure_pct = the doctor's part of this measure's cost.", note])
    if view == "items":
        rows, total = _page(B.rules_items, Measure=meas, **K)
        return _finish("items", sel, scope, [_b_item(r) for r in rows], periods, total, [
            "Brands counted by the measure, highest cost first. alternative = the brand's Main Sheet block has "
            "another brand the rule accepts; p1_brand = that block's first-priority brand.", note])
    # book
    rows, total = _page(B.rules_book, Month=K["Month"], Segment=K["Segment"], Category=K["Category"])
    return _finish("book", {"brands": total}, scope, [_b_book(r) for r in rows], periods, total, [
        "The month's rule book: every formulary brand, its letter, and what each payer may get (ok / breach / "
        "watch / info). No hospital or patient-type filter.", "Ask with material=<brand> for one brand."])


def _b_book(r: dict) -> dict:
    return {"item": r["material_desc"], "material": r["material"], "letter": r["letter"], "gov": r["gov"],
            "cash": r["cash"], "tpa": r["tpa"], "p_rank": r.get("p_rank")}


def _b_pooled(plants: list, view: str, K: dict, measure: str, periods: list) -> dict:
    if view not in ("measures", "hospitals"):
        raise ValueError(f"Several hospitals are pooled for view 'measures' or 'hospitals' only; call once per "
                         f"hospital for view '{view}'.")
    per = [B.rules_overview(Measure=measure, Plant=code, **K) for code in plants]
    sel = [_b_selected(ov, measure) for ov in per]
    cost, total = sum(s["measure_cost"] for s in sel), sum(s["drug_cost"] for s in sel)
    totals = {"measure": measure, "measure_cost": _m(cost), "share_pct": _ratio(cost, total), "drug_cost": _m(total)}
    rows = [{"hospital": ov["scope"]["his_name"] or ov["scope"]["plant_name"], "plant": code,
             "measure_cost": s["measure_cost"], "share_pct": s["share_pct"]} for code, ov, s in zip(plants, per, sel)]
    scope = _scope({**per[0]["scope"], "plant": ",".join(plants), "his_name": " + ".join(r["hospital"] for r in rows)},
                   measure=measure)
    return _finish("hospitals", totals, scope, rows, periods, notes=[
        NOTE_SHARE, f"Pooled over {len(plants)} hospitals: costs added first, the share taken once (never an "
                    f"average of the hospitals' shares)."])


# ── Scheme / Attis savings and cash & TPA margin loss ──────────────────────────
# One driver for the three spend reports: they share filters, views and drill-downs and
# differ only in which amounts make up their one ratio.
SAVING = {"pct": ("saving_pct", "saving_pct"), "num": "saving_missed_value", "den": "used_value",
          "money": {"saving_missed_value": "saving_missed", "used_value": "used_value",
                    "suggested_value": "suggested_value"}}
CASH = {"pct": ("margin_loss_pct", "loss_pct"), "num": "margin_loss_value", "den": "potential_margin_value",
        "money": {"margin_loss_value": "margin_loss", "potential_margin_value": "potential_margin",
                  "actual_margin_value": "actual_margin"}}

NOTE_SCHEME = ("Scheme (GOV) patients given a brand for which HCG's scheme list suggests a cheaper one. saving "
               "missed = spend on the brand used − the same quantity at the suggested brand's FY list price; "
               "saving_pct = saving ÷ spend on the brands used.")
NOTE_ATTIS = ("Scheme (GOV) patients given a brand for which HCG's scheme list suggests an Attis brand. saving "
              "missed = spend on the brand used − the same quantity at the Attis brand's FY list price; "
              "saving_pct = saving ÷ spend on the brands used.")
NOTE_CASH = ("Cash & TPA patients given a brand for which HCG's list suggests a higher-margin one. margin loss = "
             "the suggested brands' margin − the margin made; margin_loss_pct = loss ÷ the suggested brands' "
             "margin (HCG's denominator).")


def _sp_row(spec: dict, r: dict, full: bool = False, amounts: int = 2) -> dict:
    """The report's % and its amounts. A row carries the % and the `amounts` first amounts (the
    loss, then its base), so long lists fit the budget; totals (full) carry every amount."""
    out = {spec["pct"][0]: _p(r.get(spec["pct"][1]))}
    money = list(spec["money"].items())
    out.update({k: _m(r.get(src)) for k, src in (money if full else money[:amounts])})
    return out


def _sp_pool(spec: dict, rows: list) -> dict:
    """Sum the amounts of several rows and take the ratio once."""
    t = {k: _m(sum(float(r.get(k) or 0) for r in rows)) for k in spec["money"]}
    return {spec["pct"][0]: _ratio(t[spec["num"]], t[spec["den"]]), **t}


def _spend(kind: str, Plant, Category, month, patient_type, segment, view, doctor, material,
           payer=None, scheme_group=None) -> dict:
    periods = S._periods()                  # 503 first
    spec = CASH if kind == "cash" else SAVING
    key = {"cash": "cash-tpa-margin-loss", "attis": "attis-saving", "scheme": "scheme-brand-saving"}[kind]
    view, _ = _view(view, ATTIS_VIEWS if kind == "attis" else SAVING_VIEWS, key)
    pay = _payer(payer)
    if kind == "cash" and pay == "GOV":
        raise ValueError("Cash & TPA margin loss covers CASH and CASH(TPA) patients only. Scheme (GOV) patients "
                         "are in scheme-brand-saving and attis-saving.")
    if kind == "attis" and pay and pay != "GOV" and view != "consumption":
        raise ValueError("Attis saving counts scheme (GOV) patients only; payer= filters only view='consumption' "
                         "(Attis products billed to every payer).")
    group = _text(scheme_group)
    if group and view in ("trend", "hospitals"):
        raise ValueError("scheme_group narrows only the summary, items and doctors views.")
    plants = _plants(Plant)
    K = {"Month": _month_arg(month, periods), "Type": _text(patient_type), "Segment": _text(segment),
         "Category": _text(Category)}
    if kind == "cash":
        K["Payer"] = pay
    if kind == "cash":
        overview, items_fn, doctors_fn = S.spend_cash_overview, S.spend_cash_items, S.spend_cash_doctors
    elif kind == "attis":
        overview, items_fn, doctors_fn = S.spend_attis_overview, S.spend_attis_items, S.spend_attis_doctors
    else:
        overview, items_fn, doctors_fn = S.spend_scheme_overview, S.spend_scheme_items, S.spend_scheme_doctors
    GK = {"Group": group} if kind == "scheme" else {}
    note_def = {"cash": NOTE_CASH, "attis": NOTE_ATTIS, "scheme": NOTE_SCHEME}[kind]

    if plants and len(plants) > 1:
        if view not in ("summary", "hospitals") or group:
            raise ValueError(f"Several hospitals are pooled for view 'summary' or 'hospitals' only (no "
                             f"scheme_group); call once per hospital for view '{view}'.")
        per = [overview(Plant=code, **K) for code in plants]
        rows = [{"hospital": ov["scope"]["his_name"] or ov["scope"]["plant_name"], "plant": code,
                 **_sp_row(spec, ov["headline"])} for code, ov in zip(plants, per)]
        totals = _sp_pool(spec, [_sp_row(spec, ov["headline"], True) for ov in per])
        scope = _scope({**per[0]["scope"], "plant": ",".join(plants),
                        "his_name": " + ".join(r["hospital"] for r in rows)})
        return _finish(view, totals, scope, rows, periods, notes=[
            note_def, f"Pooled over {len(plants)} hospitals: amounts added first, the % taken once."])

    K["Plant"] = plants[0] if plants else None
    table = S.CASH_SWITCH if kind == "cash" else S.SWITCH
    doc = _doctor(doctor, S._table(table), plants) if _text(doctor) else None
    mat = None
    if _text(material):
        frames = [S._table(table)] + ([S._table(S.ATTIS_USE)] if kind == "attis" else [])
        mat = _material(material, frames)
    if doc and mat:
        raise ValueError("Pass doctor or material, not both: doctor= with view='items' lists that doctor's brands.")
    ov = overview(**K)
    scope = _scope(ov["scope"], scheme_group=group)
    item_kw = {**K, **GK}

    def item_row(r: dict) -> dict:
        row = {"item": r["material_desc"], "material": r["material"], "suggested_item": r["suggested_desc"],
               **_sp_row(spec, r), "qty": r["qty"]}
        if kind == "cash":
            row["bifurcated"] = bool(r.get("bifurcated"))
        return row

    def doctor_row(r: dict, show_hospital: bool) -> dict:
        row = {"doctor": r["doctor_name"], "doctor_id": r["doctor_id"]}
        if show_hospital:
            row["hospital"] = r["hospital_name"]
        return {**row, **_sp_row(spec, r), "brands": r["brands"]}

    if doc:
        did, dname = doc
        scope.update(doctor=dname, doctor_id=did)

        def doctor_totals(kw: dict) -> dict:
            return _sp_pool(spec, [_sp_row(spec, r, True) for r in _rows_for(doctors_fn, "doctor_id", did, **kw)])
        if view in ("summary", "items"):
            rows, total = _page(items_fn, Doctor=did, **item_kw)
            return _finish("items", doctor_totals(item_kw), scope, [item_row(r) for r in rows], periods, total,
                           [note_def, "This doctor's brands, most money missed first."])
        if view == "doctors":
            mine = _rows_for(doctors_fn, "doctor_id", did, **item_kw)
            return _finish("doctors", doctor_totals(item_kw), scope, [doctor_row(r, True) for r in mine], periods,
                           notes=[note_def])
        if view == "trend":
            rows = [{"month": F._month_label(p), **doctor_totals({**item_kw, "Month": p})} for p in periods]
            return _finish("trend", doctor_totals(item_kw), scope, rows, periods,
                           notes=[note_def, "The trend shows every month whatever the month filter."])
        _no_drill(view, "doctor", "'summary', 'items', 'doctors' or 'trend'")

    if mat:
        code, desc = mat
        scope.update(item=desc, material=code)
        if view == "consumption":
            rows = _rows_for(S.spend_attis_consumption, "material", code, Payer=pay, **K)
            return _finish("consumption", {"item": desc, "material": code}, scope,
                           [_consumption_row(r) for r in rows], periods, notes=[S.NOTE_CONSUMPTION])
        if view not in ("summary", "doctors", "items"):
            _no_drill(view, "material", "'doctors' (the doctors behind the brand) or 'items'")
        mine = _rows_for(items_fn, "material", code, **item_kw)
        totals = {"item": desc, "material": code, **(
            {"suggested_item": mine[0]["suggested_desc"], **_sp_row(spec, mine[0], True), "qty": mine[0]["qty"]}
            if mine else _sp_pool(spec, []))}
        if view == "items":
            return _finish("items", totals, scope, [item_row(r) for r in mine], periods, notes=[note_def])
        rows, total = _page(doctors_fn, Material=code, **item_kw)
        return _finish("doctors", totals, scope, [doctor_row(r, not plants) for r in rows], periods, total,
                       [note_def, "Doctors who used this brand, most money lost first."])

    h = ov["headline"]
    note = _handler_note(ov.get("notes"))
    if view == "summary" and group:
        blk = ov.get("groups", {}).get(group.lower())
        if blk is None:
            raise ValueError("scheme_group must be attis or healthium.")
        rows, total = _page(items_fn, **item_kw)
        totals = {**_sp_row(spec, blk, True), "group_share_pct": _p(blk.get("share_pct"))}
        return _finish("summary", totals, scope, [item_row(r) for r in rows], periods, total,
                       [note_def, f"{blk['label']}: group_share_pct = its part of the whole scheme list's saving "
                                  f"missed. Rows: its brands, most money missed first."])
    if view == "summary":
        totals = {**_sp_row(spec, h, True), "brands": h["brands"], "doctors": h["doctors"],
                  "hospitals": h["hospitals"]}
        if kind == "cash":
            totals.update(used_mrp_value=_m(h["used_mrp_value"]), suggested_mrp_value=_m(h["suggested_mrp_value"]),
                          topline_increase_value=_m(h["topline_increase"]))
        elif kind == "attis":
            c = ov.get("consumption") or {}
            totals.update(scheme_saving_missed_value=_m(h["scheme_saving_missed"]),
                          share_of_scheme_pct=_p(h["share_of_scheme_pct"]),
                          attis_consumption_cost=_m(c.get("cost")))
        else:
            for g, blk in (ov.get("groups") or {}).items():
                totals[f"{g}_saving_missed_value"] = _m(blk.get("saving_missed"))
                totals[f"{g}_share_pct"] = _p(blk.get("share_pct"))
        splits = ([("payer", ov.get("by_payer") or [])] if kind == "cash" else []) + \
                 [("label", ov["by_category"]), ("type", ov["by_type"])]
        rows = [{"breakdown": r[k], **_sp_row(spec, r)} for k, part in splits for r in part]
        extra = {"cash": "Rows: by payer, material category and IP/OP.",
                 "attis": "attis_consumption_cost = Attis products billed to every payer, at bill cost.",
                 "scheme": "healthium = HCG's 'J&J & Covidien → Healthium' line (sutures and endo rows)."}[kind]
        return _finish("summary", totals, scope, rows, periods, notes=[note_def, note or extra])
    totals = _sp_row(spec, h, True)
    if view == "trend":
        rows = [{"month": t["label"], **_sp_row(spec, t)} for t in ov["trend"]]
        return _finish("trend", totals, scope, rows, periods,
                       notes=[note_def, "The trend shows every month whatever the month filter."])
    if view == "hospitals":
        # the % and the amount lost only, so every hospital fits (see the formulary ranking)
        rows = [{"hospital": _hospital_name(x), "plant": x["plant"], **_sp_row(spec, x, amounts=1)}
                for x in ov["by_hospital"]]
        return _finish("hospitals", totals, scope, rows, periods, notes=[
            "Every hospital, most money lost first, whatever the hospital filter. Never add or average the "
            "percentages: a group's % is its summed amounts' ratio.", note])
    if view == "items":
        rows, total = _page(items_fn, **item_kw)
        return _finish("items", totals, scope, [item_row(r) for r in rows], periods, total,
                       [note_def, "Brands used, most money lost first."])
    if view == "doctors":
        rows, total = _page(doctors_fn, **item_kw)
        return _finish("doctors", totals, scope, [doctor_row(r, not plants) for r in rows], periods, total,
                       [note_def, "Doctors, most money lost first."])
    # consumption (Attis only)
    c = ov.get("consumption") or {}
    by_payer = {p["payer"]: p["cost"] for p in c.get("by_payer") or []}
    rows, total = _page(S.spend_attis_consumption, Payer=pay, **K)
    totals = {"attis_consumption_cost": _m(by_payer.get(pay) if pay else c.get("cost")), "products": total}
    if pay:
        scope["payer"] = pay
    return _finish("consumption", totals, scope, [_consumption_row(r) for r in rows], periods, total,
                   [S.NOTE_CONSUMPTION, "Highest cost first."])


def _consumption_row(r: dict) -> dict:
    return {"item": r["material_desc"], "material": r["material"], "consumption_cost": _m(r["cost"]),
            "gov_cost": _m(r["gov_cost"]), "qty": r["qty"], "list_share_pct": _p(r["share_pct"])}


@_report("scheme brand saving")
def scheme_saving(Plant=None, Category=None, month=None, patient_type=None, segment=None, view=None, doctor=None,
                  material=None, scheme_group=None) -> dict:
    return _spend("scheme", Plant, Category, month, patient_type, segment, view, doctor,
                  material, scheme_group=scheme_group)


@_report("Attis saving")
def attis_saving(Plant=None, Category=None, month=None, patient_type=None, segment=None, view=None, doctor=None,
                 material=None, payer=None) -> dict:
    return _spend("attis", Plant, Category, month, patient_type, segment, view, doctor, material,
                  payer=payer)


@_report("cash & TPA margin loss")
def cash_tpa_margin_loss(Plant=None, Category=None, month=None, payer=None, patient_type=None, segment=None,
                         view=None, doctor=None, material=None) -> dict:
    return _spend("cash", Plant, Category, month, patient_type, segment, view, doctor,
                  material, payer=payer)
