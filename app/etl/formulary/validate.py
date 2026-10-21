"""The build's validation report (formulary_validation.json) and its abort rules.

Aborts (nothing is published): a month whose sales dates fall outside the file's month,
published cost != read cost (> Rs 1), < 99.9% of drug lines found in the month's Item
Master, category agreement with HCG's own column < 99.5%. (0-row files, months without
a master and unmapped hospitals abort earlier.) Payer and segment agreement never abort:
they are reported, because HCG's payer column has hand edits and its Aug-26 segment column
mislabels 151 onco codes.

The report holds totals, percentages and names of payers/hospitals only — never a sales
line, invoice, patient or day-level date.
"""
from __future__ import annotations

import pandas as pd

from app.etl.formulary import classify as K
from app.etl.formulary import config as C
from app.etl.formulary.discover import BuildError


def _r(v, nd=2):
    return None if v is None else round(float(v), nd)


def compliance_block(df: pd.DataFrame) -> dict:
    """A/B/C, pooled, onco, non-onco and mean on a slice of formulary_summary."""
    d = df[df["category"].isin(K.DRUG_CATEGORIES)]
    g = d.groupby("formulary_group")["cost"].sum()
    by = d.groupby(["category", "formulary_group"])["cost"].sum()

    def get(cat, grp):
        return float(by.get((cat, grp), 0.0))

    a, b, c = (float(g.get(x, 0.0)) for x in "ABC")
    return {
        "lines": int(d["lines"].sum()),
        "formulary": _r(a), "out_of_formulary": _r(b), "non_formulary": _r(c),
        "pooled_pct": _r(K.pooled_pct(a, b), 4),
        "onco_pct": _r(K.pooled_pct(get(K.ONCO, "A"), get(K.ONCO, "B")), 4),
        "non_onco_pct": _r(K.pooled_pct(get(K.OTHER, "A"), get(K.OTHER, "B")), 4),
        "mean_pct": _r(K.mean_pct(get(K.ONCO, "A"), get(K.ONCO, "B"),
                                  get(K.OTHER, "A"), get(K.OTHER, "B")), 4),
    }


def scheme_block(df: pd.DataFrame) -> dict:
    """Restricted / preferred brands billed to GOV payers (drugs)."""
    d = df[df["category"].isin(K.DRUG_CATEGORIES) & (df["payer_bucket"] == K.GOV)]
    return {
        "restricted": _r(d.loc[d["formulary_class"].isin(K.RESTRICTED_CLASSES), "cost"].sum()),
        "preferred": _r(d.loc[d["formulary_class"].isin(K.PREFERRED_CLASSES), "cost"].sum()),
    }


def rule_mask(df: pd.DataFrame, rule) -> pd.Series:
    """Rows of a formulary_summary-like frame that a BrandRule counts (drugs only)."""
    m = df["category"].isin(K.DRUG_CATEGORIES) & df["formulary_class"].isin(rule.classes)
    if rule.payers:
        m &= df["payer_bucket"].isin(rule.payers)
    if rule.patient_types:
        m &= df["patient_type"].isin(rule.patient_types)
    return m


def rules_block(summary: pd.DataFrame, rules) -> dict:
    """Cost, lines and share of drug spend (all payers, HCG's denominator) per measure."""
    d = summary[summary["category"].isin(K.DRUG_CATEGORIES)]
    total = float(d["cost"].sum())
    out = {}
    for r in rules:
        m = d[rule_mask(d, r)]
        cost = float(m["cost"].sum())
        out[r.measure] = {"cost": _r(cost), "lines": int(m["lines"].sum()),
                          "share_pct": _r(K.pct(cost, total), 4)}
    return out


def rules_reconciliation(summary: pd.DataFrame, breach: pd.DataFrame, rules) -> list:
    """Per (plant, measure): the per-doctor table must hold exactly the summary's cost."""
    out = []
    for r in (r for r in rules if r.by_doctor):
        want = summary[rule_mask(summary, r)].groupby("plant")["cost"].sum()
        got = breach[breach["measure"] == r.measure].groupby("plant")["cost"].sum()
        for plant in sorted(set(want.index) | set(got.index)):
            a, b = float(want.get(plant, 0.0)), float(got.get(plant, 0.0))
            if abs(a - b) > C.MAX_COST_DIFF:
                out.append(f"{r.measure} at {plant}: per-doctor table {b:,.2f} vs summary {a:,.2f}")
    return out


def month_report(*, period, sidecar, item_master, result, rules=()) -> dict:
    s = result.tables[C.SUMMARY]
    ck = result.checks
    drugs = ck["drugs"]
    found_pct = K.pct(drugs["lines"] - drugs["not_in_master_lines"], drugs["lines"])
    kr = s[s["plant"] == C.KR_PLANT]
    rep = {
        "period": period,
        "label": C.period_label(period),
        "source": {
            "file": sidecar["source_file"], "sheet": sidecar["sheet"],
            "has_hcg_columns": sidecar["has_hcg_columns"],
            "rows_read": sidecar["rows"], "blank_rows_skipped": sidecar["blank_rows_skipped"],
            "cost_read": sidecar["cost"], "mrp_read": sidecar["mrp"], "qty_read": sidecar["qty"],
            "cost_empty_lines": sidecar["cost_empty_lines"],
            "dates_missing": sidecar["dates_missing"],
            "dates_outside_month": sidecar["dates_outside_month"],
            "patient_types": sidecar["patient_types"],
            "extracted_at": sidecar.get("extracted_at"),
        },
        "published": {
            "lines": int(s["lines"].sum()), "cost": _r(s["cost"].sum()),
            "mrp": _r(s["mrp"].sum()), "qty": _r(s["qty"].sum(), 3),
            "cost_diff": _r(s["cost"].sum() - sidecar["cost"]),
        },
        "item_master": {"file": item_master.path.name, "materials": len(item_master),
                        "flag_column": item_master.flag_column},
        "drugs": {"lines": int(drugs["lines"]),
                  "not_in_master_lines": int(drugs["not_in_master_lines"]),
                  "not_in_master_cost": _r(drugs["not_in_master_cost"]),
                  "found_in_master_pct": _r(found_pct, 4)},
        "unclassified_lines": ck["unclassified_lines"],
        "doctor_unknown_drug_lines": ck["doctor_unknown_drug_lines"],
        "hospitals": {"count": int(s["hospital"].nunique()), "unmapped": [],
                      "names_differ_from_map": ck["hospital_names_differ_from_map"]},
        "payers": {"by_source": ck["payers"],
                   "unmatched_names": ck["unmatched_payer_names"],
                   "unmatched_top50": ck["unmatched_payers"],
                   "conflicting_master_names_used": ck["used_conflicting_payers"]},
        "agreement": ck.get("agreement"),
        "kr": {"plant": C.KR_PLANT, **compliance_block(kr), "scheme": scheme_block(kr)},
        "all_hospitals": compliance_block(s),
        "brand_rules": {
            "kr": rules_block(kr, rules), "all_hospitals": rules_block(s, rules),
            "rule_book_items": int(len(result.tables.get(C.RULE_BOOK, ()))),
            "per_doctor_rows": int(len(result.tables.get(C.RULE_BREACH, ()))),
            "reconciliation": rules_reconciliation(s, result.tables[C.RULE_BREACH], rules)
            if C.RULE_BREACH in result.tables else [],
        },
        "warnings": list(sidecar.get("warnings") or []),
    }
    if not sidecar["has_hcg_columns"]:
        rep["warnings"].append("no HCG lookup columns in this month's sheet: agreement not checked")
    if sidecar["dates_missing"]:
        rep["warnings"].append(f"{sidecar['dates_missing']:,} lines have no SALESDATE")
    odd = {k: v for k, v in sidecar["patient_types"].items() if k not in ("IP", "OP")}
    if odd:
        rep["warnings"].append(f"Type values other than IP/OP: {odd}")
    if ck["unclassified_lines"]:
        rep["warnings"].append(f"{ck['unclassified_lines']:,} lines have a material code not "
                               f"starting with 1, 2 or 3 (category Unclassified)")
    if ck["hospital_names_differ_from_map"]:
        rep["warnings"].append("hospital names differ from hospital_map.csv: "
                               + ", ".join(r["his_code"] for r in ck["hospital_names_differ_from_map"]))
    if ck["unmatched_payer_names"]:
        u = ck["payers"].get("unmatched", {})
        rep["warnings"].append(f"{ck['unmatched_payer_names']} payer names are in neither the "
                               f"Payer Master nor payer_overrides.csv ({u.get('lines', 0):,} lines, "
                               f"cost {u.get('cost') or 0:,.2f}); counted as CASH(TPA)")
    ag = ck.get("agreement") or {}
    for what in ("payer", "segment"):
        a = ag.get(what)
        if a and (a["lines_pct"] or 0) < 100:
            rep["warnings"].append(f"{what} agrees with HCG's column on {a['lines_pct']}% of lines "
                                   f"/ {a['cost_pct']}% of cost (reported, not an error)")
    return rep


def abort_reasons(rep: dict) -> list:
    p, out = rep["period"], []
    src = rep["source"]
    if src["dates_outside_month"]:
        out.append(f"{p}: {src['dates_outside_month']:,} lines have a SALESDATE outside {p} "
                   f"(file {src['file']!r})")
    if rep["published"]["lines"] != src["rows_read"]:
        out.append(f"{p}: published {rep['published']['lines']:,} lines but read {src['rows_read']:,}")
    if abs(rep["published"]["cost_diff"] or 0) > C.MAX_COST_DIFF:
        out.append(f"{p}: published cost {rep['published']['cost']:,.2f} != read cost "
                   f"{src['cost_read']:,.2f}")
    found = rep["drugs"]["found_in_master_pct"]
    if found is not None and found < C.MIN_DRUGS_IN_MASTER_PCT:
        out.append(f"{p}: only {found}% of drug lines are in the {p} Item Master "
                   f"({rep['item_master']['file']!r}); need {C.MIN_DRUGS_IN_MASTER_PCT}%")
    for problem in (rep.get("brand_rules") or {}).get("reconciliation") or []:
        out.append(f"{p}: brand rules do not reconcile: {problem}")
    cat = (rep.get("agreement") or {}).get("category")
    if cat:
        for k in ("lines_pct", "cost_pct"):
            if cat[k] is not None and cat[k] < C.MIN_CATEGORY_AGREEMENT_PCT:
                out.append(f"{p}: our formulary class agrees with HCG's column on only "
                           f"{cat[k]}% ({k.split('_')[0]}); need {C.MIN_CATEGORY_AGREEMENT_PCT}%")
    return out


def check(reports: list) -> None:
    reasons = [r for rep in reports for r in abort_reasons(rep)]
    if reasons:
        raise BuildError("validation failed, nothing published:\n  " + "\n  ".join(reasons))


def payer_master_report(pm, overrides: dict) -> dict:
    return {
        "file": pm.path.name, "rows": pm.rows, "bucket_column": pm.bucket_column,
        "names": len(pm.names),
        "rule": "first row in sheet order per normalised CUSTOMERNAME",
        "conflicting_names": pm.conflicts,
        "overrides": sum(len(v) for v in overrides.values()),
    }
