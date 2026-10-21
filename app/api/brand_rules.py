"""Brand rules — HCG's brand rules per payer (Task 4).

HCG's Item Master gives every formulary brand a Category letter (P, P-SCH, SCH, R, R-OPD, OPD,
P-OPD, SCH-OPD) and HCG's monthly emails say which letters are wrong for which patients. The
formulary build applies app/data/reference/brand_rules.csv to every sale and publishes:
formulary_rule_breach (per doctor x brand, for the per-doctor measures) and formulary_rules
(each month's rule book). The rules themselves are served from formulary_validation.json —
exactly the rules the published tables were built with. P1/P2 ranks come from the spend
build's brand_blocks (the Main Sheet's blocks of interchangeable brands and their SHARING).

    GET /formulary/rules/overview  -> every measure: cost, share of drug spend (all payers, HCG's
                                      denominator), onco / non-onco, with vs without an
                                      alternative brand; trend and hospital ranking of Measure=
    GET /formulary/rules/items     -> {data,total} brands one Measure counts; with Doctor=, one
                                      doctor's (per-doctor measures only)
    GET /formulary/rules/doctors   -> {data,total} doctors of a per-doctor Measure; with
                                      Material=, the doctors behind one brand
    GET /formulary/rules/book      -> {data,total} the month's rule book: every formulary brand,
                                      its letter, what each payer may get, its block and P1/P2

A line can count in more than one measure (e.g. GOV R-OPD on an inpatient): measures are
reported side by side and never added up. Every share is a ratio of sums, and its base is
returned beside it (drug_spend on the overview, its trend and hospital rows; `sums` under the
brands and doctors tables).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request

from app.api import formulary as F
from app.api import spend as S
from app.core import table_query as tq

router = APIRouter()

BREACH, BOOK, BLOCKS = "formulary_rule_breach", "formulary_rules", "brand_blocks"
NOT_BUILT = ("Brand rules have not been built yet — rebuild the formulary tables with "
             "python -m app.etl.formulary --from-cache")
PAYERS = ("GOV", "CASH", "CASH(TPA)")
VERDICT_ORDER = {"breach": 3, "watch": 2, "info": 1}
NOTE_SHARE = ("Each share is the measure's cost ÷ ALL drug spend in the same scope (every payer), as HCG's "
              "report divides it.")
NOTE_OVERLAP = ("A sale can count in more than one measure (e.g. a restricted -OPD brand given to a scheme "
                "inpatient), so the measures are shown side by side and never added up.")
NOTE_P1 = ("P1/P2 = rank by '2026-27 SHARING' within a block of interchangeable brands in HCG's FY Main Sheet "
           "(CASH&TPA). Proposed until HCG confirms it or sends its Brand Priority Master.")
NOTE_ALT = ("'Alternative' = the brand's block in HCG's Main Sheet has another brand with a letter the rule "
            "accepts for these patients; 'no alternative' = the brand is the only choice HCG lists.")


# ── Data ──────────────────────────────────────────────────────────────────────
def _rules() -> list:
    v = F._validation() or {}
    rules = v.get("brand_rules")
    if not rules:
        raise HTTPException(status_code=503, detail=NOT_BUILT)
    return rules


def _blocks() -> pd.DataFrame:
    """The Main Sheet blocks (spend build); empty when not built, so P1/P2 show as blank."""
    try:
        return S._table(BLOCKS)
    except HTTPException:
        return pd.DataFrame(columns=["material", "material_desc", "ms_category", "sharing", "block_id",
                                     "block_size", "block_sharing_sum", "p_rank", "tie", "cash_suggested_desc",
                                     "scheme_suggested_desc"])


def _version() -> tuple:
    return F._version() + (F._stamp(S._data_dir() / f"{BLOCKS}.parquet"),)


def _rule_arg(measure, rules: list, need_doctor: bool = False) -> dict:
    s = F._str(measure)
    if s is None:
        return rules[0]
    for r in rules:
        if r["measure"].lower() == s.lower():
            if need_doctor and not r["by_doctor"]:
                F._bad(f"Measure '{r['measure']}' is not kept per doctor. Per-doctor measures: "
                       + ", ".join(x["measure"] for x in rules if x["by_doctor"]))
            return r
    F._bad(f"Invalid Measure '{s}'. Use one of: {', '.join(r['measure'] for r in rules)}")


def _mask(df: pd.DataFrame, rule: dict) -> pd.Series:
    m = df["formulary_class"].astype(str).isin(rule["classes"])
    if rule["payers"] != ["*"]:
        m &= df["payer_bucket"].astype(str).isin(rule["payers"])
    if rule["patient_types"] != ["*"] and "patient_type" in df.columns:
        m &= df["patient_type"].astype(str).isin(rule["patient_types"])
    return m


def _letter(cls: str) -> str:
    return cls[len("FORMULARY"):] if cls.startswith("FORMULARY") else cls


@F._cache
def _block_info(version: tuple) -> pd.DataFrame:
    """Per material: its block, P-rank, SHARING, the block's brands as one readable line and its
    P1 brand."""
    b = _blocks().copy()
    if b.empty:
        return pd.DataFrame(columns=["material", "block_id", "block_size", "sharing", "p_rank", "tie",
                                     "ms_category", "block_brands", "p1_desc", "cash_suggested_desc",
                                     "scheme_suggested_desc", "block_letters"])
    b["material"] = b["material"].astype(str)
    b = b.sort_values(["block_id", "p_rank", "material"])
    label = b["material_desc"].fillna(b["material"]) + " (" + b["ms_category"].fillna("?") + " " \
        + (b["sharing"].astype(float) * 100).round(0).astype("Int64").astype(str) + "%)"
    brands = label.groupby(b["block_id"]).agg(" · ".join)
    p1 = b[b["p_rank"] == 1].groupby("block_id")["material_desc"].agg(lambda x: " / ".join(x.dropna()))
    letters = {bid: list(zip(g["material"], g["ms_category"])) for bid, g in b.groupby("block_id")}
    out = b[["material", "block_id", "block_size", "sharing", "p_rank", "tie", "ms_category",
             "cash_suggested_desc", "scheme_suggested_desc"]].copy()
    out["block_brands"] = out["block_id"].map(brands)
    out["p1_desc"] = out["block_id"].map(p1)
    out["block_letters"] = out["block_id"].map(letters)
    return out.reset_index(drop=True)


def _alternative(info: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """material -> 'yes' / 'no' / 'not in list': does its block hold another brand the rule accepts?"""
    alts = set(rule.get("alternatives") or [])
    if info.empty:
        return pd.DataFrame(columns=["material", "alternative"])
    def one(row):
        if not alts:
            return "—"
        return "yes" if any(m != row["material"] and c in alts for m, c in row["block_letters"]) else "no"
    return pd.DataFrame({"material": info["material"], "alternative": info.apply(one, axis=1)})


def _scope(Plant, Month, Type, Segment, Category) -> F.Scope:
    _rules()                                         # 503 before anything else when not built
    return F._scope(Plant, Month, Type, Segment, Category, None)


def _rule_meta(r: dict) -> dict:
    return {"measure": r["measure"], "label": r["label"], "verdict": r["verdict"], "by_doctor": r["by_doctor"],
            "letters": [_letter(c) for c in r["classes"]], "payers": r["payers"],
            "patient_types": r["patient_types"], "alternatives": r.get("alternatives") or [],
            "rule_ref": r.get("rule_ref"), "note": r.get("note")}


# ── Overview ──────────────────────────────────────────────────────────────────
@router.get("/formulary/rules/overview")
def rules_overview(Plant: str = Query(None), Month: str = Query(None), Type: str = Query(None),
                   Segment: str = Query(None), Category: str = Query(None), Measure: str = Query(None)):
    sc = _scope(Plant, Month, Type, Segment, Category)
    rule = _rule_arg(Measure, _rules())
    return _overview_cached(sc, rule["measure"], _version())


@F._cache
def _overview_cached(sc: F.Scope, measure: str, version: tuple) -> dict:
    rules = _rules()
    rule = next(r for r in rules if r["measure"] == measure)
    s = F._table("formulary_summary")
    items = F._table("formulary_by_item")
    names = F._sap_names()
    drugs = F._cut(s, sc)                                          # drug spend, every payer
    cat = F._category(drugs)
    total = float(drugs["cost"].sum())
    seg_total = {k: float(drugs.loc[cat == c, "cost"].sum()) for k, c in F.SEGMENTS.items()}
    info = _block_info(version)
    it = F._cut(items, sc).astype({"material": str})

    measures = []
    for r in rules:
        m = drugs[_mask(drugs, r)]
        mc = F._category(m)
        cost = float(m["cost"].sum())
        row = {**_rule_meta(r), "cost": F._money(cost), "lines": int(m["lines"].sum()),
               "share_pct": F._share(cost, total)}
        for k, c in F.SEGMENTS.items():
            v = float(m.loc[mc == c, "cost"].sum())
            row[k.replace("-", "_")] = {"cost": F._money(v), "share_pct": F._share(v, seg_total[k])}
        im = it[_mask(it, r)].merge(_alternative(info, r), on="material", how="left")
        alt = im["alternative"].fillna("not in list")
        row["alternative"] = {k: F._money(im.loc[alt == k, "cost"].sum())
                              for k in ("yes", "no", "not in list")} if r.get("alternatives") else None
        measures.append(row)

    t = F._cut(s, sc, months=False)
    tm = t[_mask(t, rule)]
    tot_p = t.groupby(t["period"].astype(str))["cost"].sum()
    cost_p = tm.groupby(tm["period"].astype(str))["cost"].sum()
    # drug_spend: the month's / hospital's all-payer drug spend each share_pct divides by
    trend = [{"period": p, "label": F._month_label(p), "cost": F._money(cost_p.get(p, 0.0)),
              "share_pct": F._share(cost_p.get(p, 0.0), tot_p.get(p, 0.0)),
              "drug_spend": F._money(tot_p.get(p, 0.0))} for p in F._periods()]

    h = F._cut(s, sc, plant=False)
    hm = h[_mask(h, rule)]
    tot_h = h.groupby(h["plant"].astype(str))["cost"].sum()
    cost_h = hm.groupby(hm["plant"].astype(str)).agg(cost=("cost", "sum"), lines=("lines", "sum"))
    hosp = F._hospitals()
    by_hospital = []
    for p, r in cost_h.iterrows():
        hh = hosp.get(p, {"his_code": None, "his_name": None})
        by_hospital.append({"plant": p, "name": names.get(p, hh["his_name"] or p), "his_code": hh["his_code"],
                            "his_name": hh["his_name"], "cost": F._money(r["cost"]), "lines": int(r["lines"]),
                            "share_pct": F._share(r["cost"], tot_h.get(p, 0.0)),
                            "drug_spend": F._money(tot_h.get(p, 0.0)), "rank": None,
                            "selected": p == sc.plant})
    by_hospital.sort(key=lambda x: (-x["cost"], x["his_name"] or ""))
    for i, x in enumerate(by_hospital):
        x["rank"] = i + 1

    notes = [NOTE_SHARE, NOTE_OVERLAP, NOTE_ALT, NOTE_P1]
    if len(sc.months) > 1:
        notes.append(F.NOTE_MONTHS)
    return {"scope": {"plant": sc.plant or "ALL",
                      "plant_name": names.get(sc.plant, sc.plant) if sc.plant else "All Hospitals",
                      "his_name": F._his_name(sc.plant) if sc.plant else None, "months": list(sc.months),
                      "type": sc.ptype or "ALL", "segment": sc.segment or "ALL"},
            "drug_spend": {"total": F._money(total), "onco": F._money(seg_total["onco"]),
                           "non_onco": F._money(seg_total["non-onco"])},
            "measures": measures, "selected": measure, "trend": trend, "by_hospital": by_hospital,
            "notes": notes, "data_as_of": F._built_at()}


# ── Items and doctors ─────────────────────────────────────────────────────────
# Totals row of the brands and doctors tables (data_access.paginate sums). Not `brands` (a
# brand counts once per doctor, so a sum over doctors double-counts) and nothing on the rule
# book, which has no amounts.
SUMS = ["cost", "qty", "lines"]
ITEM_COLUMNS = ["material", "material_desc", "generic_name", "manufacturer", "category", "letter", "cost", "qty",
                "lines", "share_pct", "hospitals", "alternative", "p_rank", "sharing", "p1_desc", "block_brands"]
# The brands tables' own filters (Hospital and Category come from their columns).
_LETTER = tq.Facet("letter", "Letter", order="label")
_MAKER = tq.Facet("manufacturer", "Manufacturer")
ITEM_FACETS = (_LETTER, tq.Facet("alternative", "Alternative in block"), _MAKER)


def _measure_chip(rule: dict) -> tq.ScopeChip:
    return tq.chip("Measure", "Measure", rule["measure"], rule["label"])


def _finish_items(r: pd.DataFrame, rule: dict, version: tuple) -> pd.DataFrame:
    info = _block_info(version)
    r = r.merge(_alternative(info, rule), on="material", how="left")
    r["alternative"] = r["alternative"].fillna("not in list") if rule.get("alternatives") else "—"
    r = r.merge(info[["material", "p_rank", "sharing", "p1_desc", "block_brands"]], on="material", how="left")
    tot = float(r["cost"].sum())
    r["share_pct"] = r["cost"] / tot * 100 if tot else np.nan
    r["letter"] = r["formulary_class"].astype(str).map(_letter)
    r["sharing"] = r["sharing"].astype(float) * 100
    r = F._round(r, money=("cost",), pct=("share_pct", "sharing"), qty=("qty",))
    r = r.sort_values(["cost", "material"], ascending=[False, True], kind="mergesort")
    return r[ITEM_COLUMNS].reset_index(drop=True)


@router.get("/formulary/rules/items")
def rules_items(request: Request, Plant: str = Query(None), Month: str = Query(None), Type: str = Query(None),
                Segment: str = Query(None), Category: str = Query(None), Measure: str = Query(None),
                Doctor: str = Query(None)):
    sc = _scope(Plant, Month, Type, Segment, Category)
    doctor = F._str(Doctor)
    rule = _rule_arg(Measure, _rules(), need_doctor=bool(doctor))
    chips = (_measure_chip(rule), *([tq.chip("Doctor", "Doctor", doctor)] if doctor else []))
    return F._page(request, _items_cached(sc, rule["measure"], doctor, _version()), sum_cols=SUMS,
                   facets=ITEM_FACETS, scope=F.scope_chips(sc, *chips))


@F._cache
def _items_cached(sc: F.Scope, measure: str, doctor: Optional[str], version: tuple) -> pd.DataFrame:
    rule = next(r for r in _rules() if r["measure"] == measure)
    if doctor:
        d = F._cut(F._table(BREACH), sc)
        d = d[(d["measure"].astype(str) == measure) & (d["doctor_id"].astype(str) == doctor)]
    else:
        d = F._cut(F._table("formulary_by_item"), sc)
        d = d[_mask(d, rule)]
    d = d.astype({"material": str, "plant": str})
    r = d.groupby(["material", "formulary_class"], as_index=False).agg(
        cost=("cost", "sum"), qty=("qty", "sum"), lines=("lines", "sum"), hospitals=("plant", "nunique"))
    r = r.merge(F._last(d, ["material"], ["material_desc", "generic_name", "category"]), on="material", how="left")
    if doctor:   # the per-doctor table has no manufacturer: take it from the item table
        maker = F._describe_all(F._version())[["material", "manufacturer"]].astype({"material": str})
    else:
        maker = F._last(d, ["material"], ["manufacturer"])
    r = r.merge(maker, on="material", how="left")
    return _finish_items(r, rule, version)


DOCTOR_COLUMNS = ["doctor_id", "doctor_name", "plant", "hospital_name", "brands", "cost", "qty", "lines",
                  "share_pct"]


@router.get("/formulary/rules/doctors")
def rules_doctors(request: Request, Plant: str = Query(None), Month: str = Query(None), Type: str = Query(None),
                  Segment: str = Query(None), Category: str = Query(None), Measure: str = Query(None),
                  Material: str = Query(None)):
    sc = _scope(Plant, Month, Type, Segment, Category)
    rule = _rule_arg(Measure, _rules(), need_doctor=True)
    material = F._code(Material)
    chips = (_measure_chip(rule), *([tq.chip("Material", "Item", material)] if material else []))
    return F._page(request, _doctors_cached(sc, rule["measure"], material, _version()), sum_cols=SUMS,
                   scope=F.scope_chips(sc, *chips))


@F._cache
def _doctors_cached(sc: F.Scope, measure: str, material: Optional[str], version: tuple) -> pd.DataFrame:
    d = F._cut(F._table(BREACH), sc).astype({"plant": str, "doctor_id": str, "material": str})
    d = d[d["measure"].astype(str) == measure]
    if material:
        d = d[d["material"] == material]
    keys = ["plant", "doctor_id"]
    r = d.groupby(keys, as_index=False).agg(cost=("cost", "sum"), qty=("qty", "sum"), lines=("lines", "sum"),
                                            brands=("material", "nunique"))
    r = r.merge(F._last(d, keys, ["doctor_name", "hospital_name"]), on=keys, how="left")
    tot = float(r["cost"].sum())
    r["share_pct"] = r["cost"] / tot * 100 if tot else np.nan
    r = F._round(r, money=("cost",), pct=("share_pct",), qty=("qty",))
    r = r.sort_values(["cost", "doctor_id"], ascending=[False, True], kind="mergesort")
    return r[DOCTOR_COLUMNS].reset_index(drop=True)


# ── Rule book ─────────────────────────────────────────────────────────────────
BOOK_COLUMNS = ["material", "material_desc", "molecule", "manufacturer", "category", "letter", "gov", "cash",
                "tpa", "sold", "p_rank", "sharing", "tie", "block_size", "ms_category", "conflict", "p1_desc",
                "block_brands", "cash_suggested_desc", "scheme_suggested_desc"]
BOOK_FACETS = (_LETTER, tq.Facet("gov", "GOV patients"), tq.Facet("cash", "Cash patients"),
               tq.Facet("tpa", "TPA patients"), tq.Facet("sold", "Sold this month", value_label=tq.yes_no),
               tq.Facet("conflict", "Letter differs from Main Sheet", value_label=tq.yes_no),
               tq.Facet("ms_category", "Main Sheet letter", order="label"), _MAKER,
               tq.Facet("molecule", "Molecule"))


def _verdict(cls: str, payer: str, rules: list) -> str:
    """What the rules say about this class for this payer: the strongest verdict, with the
    patient type when a rule is limited to one; 'ok' when no rule counts it."""
    hits = [r for r in rules if cls in r["classes"] and (r["payers"] == ["*"] or payer in r["payers"])]
    if not hits:
        return "ok"
    top = max(hits, key=lambda r: VERDICT_ORDER.get(r["verdict"], 0))
    limited = top["patient_types"] != ["*"]
    return top["verdict"] + (f" ({'/'.join(top['patient_types'])})" if limited else "")


@router.get("/formulary/rules/book")
def rules_book(request: Request, Month: str = Query(None), Segment: str = Query(None),
               Category: str = Query(None), Letter: str = Query(None)):
    sc = _scope(None, Month, None, Segment, Category)
    letter = F._str(Letter)
    letter = letter.upper() if letter else None
    # The book is one month's (the latest of Month=), for every hospital.
    period = sc.months[-1]
    chips = [tq.months_chip([period], F._periods())]
    if sc.segment:
        chips.append(tq.chip("Segment", "Segment", sc.segment, F.SEGMENT_LABELS[sc.segment]))
    if letter:
        chips.append(tq.chip("Letter", "Letter", letter))
    return F._page(request, _book_cached(period, sc.segment, letter, _version()), facets=BOOK_FACETS,
                   scope=chips)


@F._cache
def _book_cached(period: str, segment: Optional[str], letter: Optional[str], version: tuple) -> pd.DataFrame:
    rules = _rules()
    b = F._table(BOOK)
    b = b[b["period"].astype(str) == period].astype({"material": str})
    if segment:
        b = b[b["category"].astype(str) == F.SEGMENTS[segment]]
    if letter:
        b = b[b["letter"].astype(str) == letter]
    b = b.merge(_block_info(version), on="material", how="left")
    for payer, col in (("GOV", "gov"), ("CASH", "cash"), ("CASH(TPA)", "tpa")):
        verdicts = {c: _verdict(c, payer, rules) for c in b["formulary_class"].astype(str).unique()}
        b[col] = b["formulary_class"].astype(str).map(verdicts)
    b["conflict"] = b["ms_category"].notna() & (b["ms_category"].astype(str) != b["letter"].astype(str))
    b["sharing"] = b["sharing"].astype(float) * 100
    b["sold"] = b["sold"].astype(bool)
    b["tie"] = b["tie"].fillna(False).astype(bool)
    b = F._round(b, pct=("sharing",))
    b = b.sort_values(["molecule", "block_id", "p_rank", "material"], kind="mergesort", na_position="last")
    return b[BOOK_COLUMNS].reset_index(drop=True)
