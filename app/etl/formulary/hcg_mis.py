"""HCG's monthly PAN formulary-compliance MIS workbook -> reference tables the app shows beside
its own figures: each unit's budget, HCG's printed compliance, HCG's remarks, the regions, and
the months the app has no sales lines for.

HCG's SCM team circulates one workbook a month ('Onco and Non Onco Formulary Compliance Sep 26 -
MIS.xlsx'). It holds unit-level totals only, never patient lines:

  'FY 26-27 Formulary Compliance %'   per unit and month of the financial year: onco and
                                      non-onco Formulary and Approved-Out-of-formulary cost,
                                      the unit's Budget, HCG's printed Actual and one Remark
                                      per unit; the 'Overall' row is the PAN HCG total.
  'Onco Formulary Consumption'        per unit and month since FY 22-23: Formulary, OOF, Non
  'Non Onco Formulary Consumption'    formulary, restricted / preferred for scheme, restricted
                                      for cash & TPA and OPD brands given to inpatients, in
                                      rupees, plus the unit's Region. The same figures as %
                                      blocks are not read: every % is recomputed from rupees.

Published to FORMULARY_DATA_DIR:
  hcg_mis_units        plant, HCG's unit name, region, PAN order, in_pan, HCG's remark
  hcg_mis_compliance   plant x period: budget, HCG's printed actual and the formula behind it,
                       and the pooled % of the corrected rupees
  hcg_mis_monthly      plant x period x segment: the rupee measures (the history sheets where
                       they have the month, the FY sheet otherwise, e.g. Jul-26)
  hcg_mis_pan          period: HCG's printed PAN %, the same recomputed from HCG's unit rows,
                       and from the corrected rupees
  hcg_mis_validation.json   what was read, and every correction or inference made

HCG's sheet has quirks; each is handled by rule and reported, never patched by hand:
  - Apr-26: the FY sheet's 'Non Onco Approved-Out of formulary' repeats the onco value for 21
    units. The history sheet has the true value, and history wins wherever both have a month.
  - Unit Actuals for Apr-Jun-26 are 100% - OOF/Formulary; from Jul-26, Formulary/(Formulary +
    OOF). The formula behind every printed Actual is detected and stored.
  - History month labels are written many ways ('June %', 'August 23 in Value', 'Sep 2025 in
    Value', a date), and some carry the wrong year. The month names set the order and the
    years most labels agree with set the placement (see _resolve_months).
  - Jul-26 has no block in the history sheets.

Run:  python -m app.etl.formulary.hcg_mis           (reads the latest MIS workbook)
      python -m app.etl.formulary.hcg_mis --list    (shows which file it would read)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from app.etl.formulary import config as C
from app.etl.formulary import excel as X
from app.etl.formulary.build import _check_columns
from app.etl.formulary.discover import BuildError

UNIT_RE = re.compile(r"^[A-Z]{2}\d{2}$")
SEGMENTS = ("onco", "non-onco")
MEASURES = ("formulary", "out_of_formulary", "non_formulary", "total", "restricted_scheme",
            "restricted_scheme_monopoly", "restricted_scheme_alternative", "preferred_scheme",
            "restricted_cash_tpa", "opd_gov_ip", "opd_cash_tpa_ip")
# Two printed values agree when they differ by less than this (a fraction of 1).
SAME_RATIO = 1e-6
# Rupee values agree when they differ by less than this.
SAME_RUPEES = 0.5

# History-sheet column headers (letters and digits only, upper case) -> measure.
_HISTORY_COLUMNS = {
    "FORMULARYCONSUMPTION": "formulary",
    "FORMULARYBRANDCONSUMPTION": "formulary",
    "APPROVEDOUTOFFORMULARYCONSUMPTION": "out_of_formulary",
    "NONFORMULARYCONSUMPTION": "non_formulary",
    "APPROVEDNONFORMULARYCONSUMPTION": "non_formulary",
    "TOTALCONSUMPTION": "total",
    "RESTRICTEDBRANDCONSUMPTIONFORSCHEMEPATIENTS": "restricted_scheme",
    "RESTRICTEDBRANDCONSUMPTIONFORSCHEMEPATIENTSMONOPOLY": "restricted_scheme_monopoly",
    "RESTRICTEDBRANDCONSUMPTIONFORSCHEMEPATIENTSALTRANATIVEBRAND": "restricted_scheme_alternative",
    "RESTRICTEDBRANDCONSUMPTIONFORSCHEMEPATIENTSALTERNATIVEBRAND": "restricted_scheme_alternative",
    "PREFERREDBRANDCONSUMPTIONFORSCHEMEPATIENTS": "preferred_scheme",
    "PREFERREDFORMULARYBRANDCONSUMPTIONFORSCHEMEPATIENTS": "preferred_scheme",
    "PREFERREDFORMULARYBRANDCONSUMPTIONFORSCHEMEPATIENTSINVALUE": "preferred_scheme",
    "RESTRICTEDBRANDCONSUMPTIONFORCASHTPAPATIENTS": "restricted_cash_tpa",
    "FORMULARYOPDGOVIP": "opd_gov_ip",
    "FORMULARYOPDCASHTPAIP": "opd_cash_tpa_ip",
}
# FY-sheet column headers -> role in a month block.
_FY_COLUMNS = {
    "FORMULARYCONSUMPTION": "formulary",
    "APPROVEDOUTOFFORMULARYCONSUMPTION": "out_of_formulary",
    "ONCONONONCOFORMULARYCONSUMPTION": "total_formulary",
    "ONCONONONCOAPPROVEDOUTOFFORMULARYCONSUMPTION": "total_out_of_formulary",
    "ONCOANDNONONCOFORMULARYAPPROVEDOUTOFFORMULARYCONSUMPTION": "total_in_ratio",
    "ACTUAL": "actual",
}
_SEGMENT_LABELS = {"ONCO": "onco", "NONONCO": "non-onco"}
_MONTH_RE = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?(?:\s*[-']?\s*(\d{4}|\d{2})\b)?", re.I)


def _key(v) -> str:
    """A header as letters and digits only: 'Approved-Out of formulary  consumption ' ->
    'APPROVEDOUTOFFORMULARYCONSUMPTION'. '%' is kept so a % column never passes for rupees."""
    return re.sub(r"[^A-Z0-9%]", "", X.header_key(v))


def _period(n: int) -> str:
    """Months since year 0 -> 'YYYY-MM'."""
    return f"{n // 12:04d}-{n % 12 + 1:02d}"


def _months(period: str) -> int:
    y, m = period.split("-")
    return int(y) * 12 + int(m) - 1


def _cell_month(v) -> Optional[tuple]:
    """(month, year or None, year_is_certain) from a header cell, or None when it names no
    month. A date, or a 4-digit year, is certain; a 2-digit year is only a hint."""
    if isinstance(v, (datetime, date)):
        return v.month, v.year, True
    s = X.text(v)
    if not s:
        return None
    try:
        d = datetime.fromisoformat(s.strip())
        return d.month, d.year, True
    except ValueError:
        pass
    m = _MONTH_RE.search(s)
    if not m:
        return None
    month = C.MONTHS[m.group(1).lower()[:3]]
    y = m.group(2)
    if y is None:
        return month, None, False
    return month, int(y) + (2000 if len(y) == 2 else 0), len(y) == 4


def _fraction_pct(v) -> Optional[float]:
    """HCG writes budgets and actuals as fractions (0.97); a few cells carry 97. -> 97.0"""
    f = X.number(v)
    if f is None:
        return None
    return f * 100 if abs(f) <= 1.5 else f


# --------------------------------------------------------------------------- #
# Finding the workbook
# --------------------------------------------------------------------------- #
@dataclass
class MisFile:
    path: Path
    period: str                       # the month in the file name


def find_mis(raw=None) -> tuple:
    """(the latest MIS workbook, every one found) under FORMULARY_RAW_DIR."""
    root = C.raw_dir(raw)
    if not root.is_dir():
        raise BuildError(f"FORMULARY_RAW_DIR does not exist: {root}")
    found, errors = [], []
    for d in sorted(p for p in root.rglob("*") if p.is_dir() and C.DASHBOARD_DIR_RE.match(p.name)):
        for sub in sorted(p for p in d.iterdir() if p.is_dir() and C.MIS_DIR_RE.match(p.name.strip())):
            for f in sorted(sub.iterdir()):
                if (not f.is_file() or f.name.startswith(("~$", ".~lock"))
                        or f.suffix.lower() not in C.EXCEL_SUFFIXES or not C.MIS_FILE_RE.search(f.stem)):
                    continue
                period = C.month_from_name(f.name)
                if period is None:
                    errors.append(f"cannot read a month from the MIS file name {f.name!r}")
                else:
                    found.append(MisFile(f, period))
    by_period: dict = {}
    for m in found:
        if m.period in by_period:
            errors.append(f"two MIS workbooks for {m.period}: {by_period[m.period].path.name!r} and {m.path.name!r}")
        by_period[m.period] = m
    if errors:
        raise BuildError("\n  ".join(["MIS discovery failed:"] + errors))
    if not found:
        raise BuildError(f"no MIS workbook found: put HCG's '... Formulary Compliance <Mon YY> - MIS.xlsx' in "
                         f"'<{C.DASHBOARD_DIR_RE.pattern.lstrip('^')} ...>/HCG MIS/' under {root}")
    ordered = sorted(found, key=lambda m: m.period)
    return ordered[-1], ordered


# --------------------------------------------------------------------------- #
# The FY sheet: budgets, HCG's printed actuals, remarks, the PAN row
# --------------------------------------------------------------------------- #
@dataclass
class FyBlock:
    period: str
    cols: dict = field(default_factory=dict)        # role -> column ('onco.formulary', 'budget', ...)


def _header_row(rows: list, must: set, sheet: str) -> int:
    for i, r in enumerate(rows[:X.HEADER_SCAN_ROWS]):
        if must <= {_key(v) for v in (r or ())}:
            return i
    raise BuildError(f"sheet {sheet!r}: no header row with {sorted(must)}")


def read_fy_sheet(rows: list, sheet: str) -> dict:
    h = _header_row(rows, {"UNITCODE", "UNITNAME"}, sheet)
    top, sub = list(rows[h]), list(rows[h + 1]) if h + 1 < len(rows) else []
    width = max(len(top), len(sub))
    top += [None] * (width - len(top))
    sub += [None] * (width - len(sub))
    keys_top = [_key(v) for v in top]
    code_col, name_col = keys_top.index("UNITCODE"), keys_top.index("UNITNAME")

    blocks, remark_col, cur, seg, start = [], None, FyBlock(""), None, name_col + 1
    for j in range(name_col + 1, width):
        if keys_top[j] in _SEGMENT_LABELS:
            seg = _SEGMENT_LABELS[keys_top[j]]
        k = _key(sub[j])
        if not k:
            continue
        if k.startswith("REMARKS"):
            remark_col = j
            continue
        if k.startswith("BUDGETFY"):
            cur.cols["budget"] = j
            continue
        role = _FY_COLUMNS.get(k)
        if role is None:
            raise BuildError(f"sheet {sheet!r}: unknown column {sub[j]!r} (column {j + 1})")
        if role in ("formulary", "out_of_formulary"):
            if seg is None:
                raise BuildError(f"sheet {sheet!r}: column {j + 1} ({sub[j]!r}) is not under 'Onco' or 'Non Onco'")
            role = f"{seg}.{role}"
        if role in cur.cols:
            raise BuildError(f"sheet {sheet!r}: two {role!r} columns in one month block (column {j + 1})")
        cur.cols[role] = j
        if role == "actual":
            # The month is the date written above the block (over its Budget column).
            dates = [_cell_month(top[c]) for c in range(start, j + 1)]
            dates = [d for d in dates if d and d[2]]
            if len(dates) != 1:
                raise BuildError(f"sheet {sheet!r}: the month block ending at column {j + 1} has "
                                 f"{len(dates)} dates above it, expected 1")
            cur.period = f"{dates[0][1]:04d}-{dates[0][0]:02d}"
            need = {"onco.formulary", "onco.out_of_formulary", "non-onco.formulary",
                    "non-onco.out_of_formulary", "budget", "actual"}
            if not need <= set(cur.cols):
                raise BuildError(f"sheet {sheet!r}: block {cur.period} lacks {sorted(need - set(cur.cols))}")
            blocks.append(cur)
            cur, seg, start = FyBlock(""), None, j + 1
    if not blocks:
        raise BuildError(f"sheet {sheet!r}: no month blocks found")
    periods = [b.period for b in blocks]
    if periods != sorted(set(periods)):
        raise BuildError(f"sheet {sheet!r}: months out of order or repeated: {periods}")

    units, pan = [], None
    for r in rows[h + 2:]:
        r = list(r or ()) + [None] * width
        label = X.text(r[code_col]) or ""
        code = (X.code(r[code_col]) or "").upper()
        vals = {b.period: {role: r[c] for role, c in b.cols.items()} for b in blocks}
        if UNIT_RE.match(code):
            units.append({"plant": code, "unit_name": X.text(r[name_col]),
                          "remark": X.text(r[remark_col]) if remark_col is not None else None, "months": vals})
        elif label.upper().startswith("OVERALL"):
            pan = vals
    if not units:
        raise BuildError(f"sheet {sheet!r}: no unit rows (unit codes like 'HC05')")
    dup = sorted({u["plant"] for u in units if sum(v["plant"] == u["plant"] for v in units) > 1})
    if dup:
        raise BuildError(f"sheet {sheet!r}: units listed twice: {dup}")
    return {"periods": periods, "units": units, "pan": pan}


def _formula(printed: Optional[float], f: float, oof: float) -> Optional[str]:
    """Which formula gives HCG's printed actual (all fractions of 1)."""
    if printed is None or f <= 0:
        return None
    if abs(printed - f / (f + oof)) < SAME_RATIO:
        return "pooled"
    if abs(printed - (1 - oof / f)) < SAME_RATIO:
        return "oof_over_formulary"
    return "other"


# --------------------------------------------------------------------------- #
# The history sheets: rupee measures per unit, month and segment, and regions
# --------------------------------------------------------------------------- #
def _resolve_months(labels: list, sheet: str, notes: list) -> list:
    """labels: [(column, (month, year, certain), text)] in column order -> month numbers.

    HCG appends one block per month, left to right, and names the month reliably but not the
    year: some labels have none ('June -Value'), and some have the wrong one, even with four
    digits ('August 23 in Value' and 'September 2022 in Value' sit between Jul-24 and Oct-24).
    So the month names set the sequence (each block is the first month of that name after the
    previous block, which also steps over a missing month such as Jul-26), and the years
    written on the labels set where the sequence sits: the placement most labels agree with
    (a date or 4-digit year counts twice a 2-digit one). Every label that disagrees is
    reported; when most labels disagree, nothing is published."""
    rel, cur = [], None
    for _, (month, _, _), _ in labels:
        v = month - 1 if cur is None else cur + 1
        while v % 12 != month - 1:
            v += 1
        rel.append(v)
        cur = v
    dated = [(lab[1][1], 2 if lab[1][2] else 1, r) for lab, r in zip(labels, rel) if lab[1][1]]
    if not dated:
        raise BuildError(f"sheet {sheet!r}: no month block carries a year, so none can be placed")
    weight = sum(w for _, w, _ in dated)

    def agree(base: int) -> int:
        return sum(w for y, w, r in dated if base + r // 12 == y)

    years = [y for y, _, _ in dated]
    base = max(range(min(years) - 6, max(years) + 1), key=lambda b: (agree(b), b))
    if agree(base) * 2 < weight:
        raise BuildError(f"sheet {sheet!r}: the years on the month labels contradict each other; "
                         f"at best {agree(base)} of {weight} agree")
    out = [base * 12 + r for r in rel]
    for lab, v in zip(labels, out):
        if lab[1][1] != v // 12:
            said = f"says {lab[1][1]}" if lab[1][1] else "has no year"
            notes.append(f"{sheet}: block {lab[2]!r} read as {_period(v)} (its label {said})")
    return out


def read_history_sheet(rows: list, sheet: str, segment: str, notes: list) -> dict:
    h = _header_row(rows, {"CODE", "REGION", "PLANTNAME"}, sheet)
    if h == 0:
        raise BuildError(f"sheet {sheet!r}: no month-label row above the header")
    lab_row, hdr = list(rows[h - 1]), list(rows[h])
    width = max(len(lab_row), len(hdr))
    lab_row += [None] * (width - len(lab_row))
    hdr += [None] * (width - len(hdr))
    keys = [_key(v) for v in hdr]
    code_col, region_col, name_col = keys.index("CODE"), keys.index("REGION"), keys.index("PLANTNAME")
    data = [list(r or ()) + [None] * width for r in rows[h + 1:]]
    data = [r for r in data if UNIT_RE.match((X.code(r[code_col]) or "").upper())]

    starts = [j for j in range(name_col + 1, width) if X.text(lab_row[j])]
    groups = []
    for i, j in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else width
        text = X.text(lab_row[j])
        lk = _key(lab_row[j])
        kind = "value" if "VALUE" in lk else "percent" if "%" in lk else None
        if kind is None:
            # An unmarked block (a bare date or month): a % column in it, or every figure a
            # fraction of 1, makes it a % block.
            if any("%" in keys[c] for c in range(j, end)):
                kind = "percent"
            else:
                nums = [X.number(r[c]) for r in data for c in range(j, end)]
                nums = [v for v in nums if v is not None]
                kind = "percent" if nums and all(abs(v) <= 1.0 for v in nums) else "value"
            notes.append(f"{sheet}: unmarked block {text!r} read as {kind}")
        month = _cell_month(lab_row[j])
        if month is None:
            raise BuildError(f"sheet {sheet!r}: block label {text!r} names no month")
        groups.append({"start": j, "end": end, "kind": kind, "month": month, "text": text})

    # A block pasted twice (the non-onco sheet repeats Aug-22..Oct-23 after Oct-23: same labels,
    # same rupees for every unit) is read once.
    values, seen = [], {}
    for g in (g for g in groups if g["kind"] == "value"):
        sig = (_key(g["text"]), tuple(X.number(r[c]) for r in data for c in range(g["start"], g["end"])))
        if sig in seen:
            notes.append(f"{sheet}: block {g['text']!r} at column {g['start'] + 1} repeats the one at "
                         f"column {seen[sig] + 1}; read once")
            continue
        seen[sig] = g["start"]
        values.append(g)
    resolved = _resolve_months([(g["start"], g["month"], g["text"]) for g in values], sheet, notes)
    records, units = [], {}
    for g, m in zip(values, resolved):
        period = _period(m)
        cols = {}
        for c in range(g["start"], g["end"]):
            if not keys[c]:
                continue
            measure = _HISTORY_COLUMNS.get(keys[c])
            if measure is None:
                raise BuildError(f"sheet {sheet!r}: unknown column {hdr[c]!r} in block {g['text']!r}")
            if measure in cols:
                raise BuildError(f"sheet {sheet!r}: two {measure!r} columns in block {g['text']!r}")
            cols[measure] = c
        for r in data:
            plant = X.code(r[code_col]).upper()
            vals = {m_: X.number(r[c]) for m_, c in cols.items()}
            if all(v is None for v in vals.values()):
                continue
            records.append({"plant": plant, "period": period, "segment": segment, **vals})
    for r in data:
        plant = X.code(r[code_col]).upper()
        units[plant] = {"region": X.text(r[region_col]), "name": X.text(r[name_col])}
    return {"records": records, "units": units, "periods": sorted({_period(m) for m in resolved})}


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def read_mis(path) -> tuple:
    """(tables, report) from one MIS workbook."""
    path = Path(path)
    notes, corrections = [], []
    with X.Workbook(path) as wb:
        by_key = {X.header_key(n): n for n in wb.sheet_names}
        fy_names = [n for k, n in by_key.items() if C.MIS_FY_SHEET_RE.match(k)]
        if len(fy_names) != 1:
            raise BuildError(f"{path.name}: expected one 'FY yy-yy Formulary Compliance %' sheet, found {fy_names}")
        missing = [k for k in C.MIS_HISTORY_SHEETS if k not in by_key]
        if missing:
            raise BuildError(f"{path.name}: missing sheets {missing}")
        fy = read_fy_sheet(list(wb.rows(fy_names[0])), fy_names[0])
        hist = {seg: read_history_sheet(list(wb.rows(by_key[k])), by_key[k], seg, notes)
                for k, seg in C.MIS_HISTORY_SHEETS.items()}

    # Units: the FY sheet's list (the PAN set, in HCG's order) plus any unit only in history.
    regions = {}
    for h in hist.values():
        for plant, u in h["units"].items():
            regions.setdefault(plant, u["region"])
    units = [{"plant": u["plant"], "unit_name": u["unit_name"], "region": regions.get(u["plant"]),
              "pan_order": i + 1, "in_pan": True, "remark": u["remark"]} for i, u in enumerate(fy["units"])]
    pan_plants = {u["plant"] for u in units}
    for h in hist.values():
        for plant, u in h["units"].items():
            if plant not in pan_plants:
                pan_plants.add(plant)
                units.append({"plant": plant, "unit_name": u["name"], "region": u["region"],
                              "pan_order": None, "in_pan": False, "remark": None})

    # Rupees: history first; the FY sheet where history has no block for the month.
    hist_rows = {(r["plant"], r["period"], r["segment"]): r for h in hist.values() for r in h["records"]}
    monthly, compliance = [], []
    for u in fy["units"]:
        for period, v in u["months"].items():
            fy_vals = {seg: {m: X.number(v[f"{seg}.{m}"]) for m in ("formulary", "out_of_formulary")}
                       for seg in SEGMENTS}
            copied = (fy_vals["onco"]["out_of_formulary"] or 0) > 0 and \
                fy_vals["onco"]["out_of_formulary"] == fy_vals["non-onco"]["out_of_formulary"]
            for seg in SEGMENTS:
                key = (u["plant"], period, seg)
                fv = fy_vals[seg]
                if key in hist_rows:
                    hv = hist_rows.pop(key)
                    for m in ("formulary", "out_of_formulary"):
                        a, b = fv[m], hv.get(m)
                        if a is not None and b is not None and abs(a - b) >= SAME_RUPEES:
                            corrections.append({
                                "plant": u["plant"], "period": period, "segment": seg, "measure": m,
                                "fy_sheet": round(a, 2), "history": round(b, 2), "used": "history",
                                "reason": ("non-onco OOF repeats the onco OOF in the FY sheet"
                                           if copied and seg == "non-onco" and m == "out_of_formulary"
                                           else "the FY sheet and the history sheet differ")})
                    monthly.append({**{k: hv.get(k) for k in MEASURES}, "plant": u["plant"], "period": period,
                                    "segment": seg, "source": "history"})
                else:
                    if copied and seg == "non-onco":
                        notes.append(f"{u['plant']} {period}: the FY sheet's non-onco OOF repeats the onco OOF "
                                     f"and history has no {period} block to correct it")
                    monthly.append({**{k: None for k in MEASURES}, **fv, "plant": u["plant"], "period": period,
                                    "segment": seg, "source": "fy_sheet"})
            f_tot = sum(fy_vals[s]["formulary"] or 0 for s in SEGMENTS)
            o_tot = sum(fy_vals[s]["out_of_formulary"] or 0 for s in SEGMENTS)
            printed = X.number(v["actual"])
            compliance.append({"plant": u["plant"], "period": period, "budget_pct": _fraction_pct(v["budget"]),
                               "hcg_actual_pct": None if printed is None else printed * 100,
                               "hcg_formula": _formula(printed, f_tot, o_tot)})
    # Months only in history (earlier FYs, units outside the PAN list).
    for (plant, period, seg), hv in sorted(hist_rows.items()):
        monthly.append({**{k: hv.get(k) for k in MEASURES}, "plant": plant, "period": period,
                        "segment": seg, "source": "history"})

    mdf = pd.DataFrame(monthly, columns=["plant", "period", "segment", *MEASURES, "source"])
    # Pooled % of the corrected rupees, per unit and month (both segments present).
    both = (mdf.groupby(["plant", "period"])
            .agg(f=("formulary", lambda s: s.sum(min_count=1)), o=("out_of_formulary", lambda s: s.sum(min_count=1)),
                 n=("segment", "nunique")).reset_index())
    both["pooled_pct"] = [(100 * f / (f + o)) if n == 2 and pd.notna(f) and pd.notna(o) and f + o > 0 else None
                          for f, o, n in zip(both["f"], both["o"], both["n"])]
    cdf = pd.DataFrame(compliance).merge(both[["plant", "period", "pooled_pct"]], on=["plant", "period"], how="left")

    # PAN: HCG's printed row, the same from HCG's own unit rows, and from the corrected rupees.
    pan = []
    in_pan = [u["plant"] for u in units if u["in_pan"]]
    for period in fy["periods"]:
        printed = X.number((fy["pan"] or {}).get(period, {}).get("actual")) if fy["pan"] else None
        f_sheet = sum(X.number(u["months"][period][f"{s}.formulary"]) or 0 for u in fy["units"] for s in SEGMENTS)
        o_sheet = sum(X.number(u["months"][period][f"{s}.out_of_formulary"]) or 0 for u in fy["units"] for s in SEGMENTS)
        cur = mdf[(mdf["period"] == period) & mdf["plant"].isin(in_pan)]
        f_c, o_c = cur["formulary"].sum(), cur["out_of_formulary"].sum()
        pan.append({"period": period, "printed_pct": None if printed is None else printed * 100,
                    "sheet_pooled_pct": 100 * f_sheet / (f_sheet + o_sheet) if f_sheet + o_sheet else None,
                    "corrected_pooled_pct": 100 * f_c / (f_c + o_c) if f_c + o_c else None,
                    "units": int(cur["plant"].nunique())})
    for p in pan:
        if p["printed_pct"] is not None and p["sheet_pooled_pct"] is not None \
                and abs(p["printed_pct"] - p["sheet_pooled_pct"]) >= 0.005:
            notes.append(f"PAN {p['period']}: HCG prints {p['printed_pct']:.4f}% but its unit rows give "
                         f"{p['sheet_pooled_pct']:.4f}%")

    tables = {
        C.MIS_UNITS: pd.DataFrame(units, columns=["plant", "unit_name", "region", "pan_order", "in_pan", "remark"])
                     .assign(pan_order=lambda d: d["pan_order"].astype("Int64"),
                             remark_period=fy["periods"][-1]),
        C.MIS_COMPLIANCE: cdf[["plant", "period", "budget_pct", "hcg_actual_pct", "hcg_formula", "pooled_pct"]],
        C.MIS_MONTHLY: mdf,
        C.MIS_PAN: pd.DataFrame(pan, columns=["period", "printed_pct", "sheet_pooled_pct", "corrected_pooled_pct", "units"]),
    }
    formula = (cdf.dropna(subset=["hcg_formula"]).groupby(["period", "hcg_formula"]).size()
               .unstack(fill_value=0).to_dict("index"))
    report = {
        "fy_sheet": fy_names[0],
        "fy_periods": fy["periods"],
        "units": len(units), "pan_units": len(in_pan),
        "history": {seg: {"first": h["periods"][0], "last": h["periods"][-1],
                          "missing_inside": [_period(m) for m in range(_months(h["periods"][0]), _months(h["periods"][-1]))
                                             if _period(m) not in set(h["periods"])]}
                    for seg, h in hist.items()},
        "hcg_formula_by_month": {p: {k: int(n) for k, n in v.items()} for p, v in formula.items()},
        "pan": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()} for p in pan],
        "corrections": corrections,
        "notes": notes,
    }
    return tables, report


# --------------------------------------------------------------------------- #
# Publishing
# --------------------------------------------------------------------------- #
_SORT = {C.MIS_UNITS: ["in_pan", "pan_order", "plant"], C.MIS_COMPLIANCE: ["plant", "period"],
         C.MIS_MONTHLY: ["plant", "period", "segment"], C.MIS_PAN: ["period"]}


def publish(tables: dict, report: dict, data=None) -> dict:
    d = C.data_dir(data)
    d.mkdir(parents=True, exist_ok=True)
    staged = []
    try:
        for name, df in tables.items():
            _check_columns(name, df)
            asc = [False] + [True] * (len(_SORT[name]) - 1) if name == C.MIS_UNITS else True
            df = df.sort_values(_SORT[name], ascending=asc, kind="stable").reset_index(drop=True)
            tmp = d / f".{name}.parquet.tmp"
            df.to_parquet(tmp, index=False, compression="zstd")
            staged.append((tmp, d / f"{name}.parquet"))
        for tmp, final in staged:
            os.replace(tmp, final)
    finally:
        for tmp, _ in staged:
            if tmp.exists():
                tmp.unlink()
    report["tables"] = {name: int(len(df)) for name, df in tables.items()}
    tmp = d / f".{C.MIS_VALIDATION_FILE}.tmp"
    tmp.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, d / C.MIS_VALIDATION_FILE)
    return report["tables"]


def build(*, raw=None, data=None, log=print) -> dict:
    latest, every = find_mis(raw)
    log(f"MIS workbook: {latest.path.name} ({latest.period}); {len(every)} found")
    tables, report = read_mis(latest.path)
    if latest.period not in report["fy_periods"]:
        report["notes"].append(f"the file name says {latest.period} but the FY sheet's last month is "
                               f"{report['fy_periods'][-1]}")
    report = {"built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "generator": "app.etl.formulary.hcg_mis", "source_file": latest.path.name,
              "file_period": latest.period, **report}
    sizes = publish(tables, report, data)
    pan = {p["period"]: p for p in report["pan"]}
    log(f"published {', '.join(f'{k} {v} rows' for k, v in sizes.items())}")
    log(f"PAN printed / corrected: " + ", ".join(
        f"{C.period_label(p)} {v['printed_pct']}/{v['corrected_pooled_pct']}" for p, v in pan.items()))
    log(f"{len(report['corrections'])} values corrected from the history sheets; {len(report['notes'])} notes "
        f"in {C.MIS_VALIDATION_FILE}")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.etl.formulary.hcg_mis",
                                 description="Read HCG's latest PAN formulary-compliance MIS workbook into "
                                             "reference tables (budgets, HCG's printed figures, remarks, history).")
    ap.add_argument("--list", action="store_true", help="show the MIS workbooks found and exit")
    ap.add_argument("--raw", help="override FORMULARY_RAW_DIR")
    ap.add_argument("--data", help="override FORMULARY_DATA_DIR")
    a = ap.parse_args(argv)
    try:
        latest, every = find_mis(a.raw)
        if a.list:
            for m in every:
                print(f"{m.period}  {m.path}{'   <- read' if m is latest else ''}")
            return 0
        build(raw=a.raw, data=a.data)
        return 0
    except BuildError as e:
        print(f"MIS build aborted, nothing published:\n  {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
