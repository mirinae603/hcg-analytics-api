# tests/test_brand_rules_pipeline.py — Task 4: HCG's brand rules through the formulary build.
#
# Reuses the miniature HCG folder of tests/test_formulary_pipeline.py (its `env` fixture) and
# adds the committed brand_rules.csv. Every expected number is worked out by hand from that
# file's LINES (line numbers in the comments are its list positions, 1-based).
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from app.etl.formulary import classify as K
from app.etl.formulary import config as C
from app.etl.formulary import validate as V
from app.etl.formulary.discover import BuildError
from app.etl.formulary.masters import load_brand_rules
from tests.test_formulary_pipeline import env, run, tables  # noqa: F401  (pytest fixture)

RULES_CSV = Path(__file__).resolve().parents[1] / "app" / "data" / "reference" / C.BRAND_RULES_FILE


@pytest.fixture()
def ruled(env):  # noqa: F811
    shutil.copy(RULES_CSV, env["dirs"]["reference"] / C.BRAND_RULES_FILE)
    return env


def test_the_committed_rules_file_loads():
    rules = {r.measure: r for r in load_brand_rules(RULES_CSV.parent)}
    assert list(rules) == ["restricted_scheme", "preferred_scheme", "oof_scheme", "restricted_cash_tpa",
                           "opd_brand_inpatient", "scheme_brand_cash_tpa"]
    r = rules["restricted_scheme"]
    assert r.classes == {"FORMULARYR", "FORMULARYR-OPD"} and r.payers == {"GOV"} and r.patient_types is None
    assert (r.verdict, r.by_doctor, r.alternatives) == ("breach", True, {"P-SCH", "SCH", "SCH-OPD"})
    assert rules["restricted_cash_tpa"].payers == {"CASH", "CASH(TPA)"}
    assert rules["restricted_cash_tpa"].verdict == "watch" and not rules["restricted_cash_tpa"].by_doctor
    assert rules["opd_brand_inpatient"].patient_types == {"IP"} and rules["opd_brand_inpatient"].verdict == "info"
    assert rules["oof_scheme"].classes == {"OUT OF FORMULARY"}


@pytest.mark.parametrize("line, match", [
    ("x,X,FORMULARYQ,GOV,*,breach,no,,r,n", "not a formulary class"),
    ("x,X,FORMULARYR,PRIVATE,*,breach,no,,r,n", "payers must be"),
    ("x,X,FORMULARYR,GOV,ER,breach,no,,r,n", "patient_types must be"),
    ("x,X,FORMULARYR,GOV,*,bad,no,,r,n", "verdict must be"),
    ("x,X,FORMULARYR,GOV,*,breach,maybe,,r,n", "by_doctor must be"),
    ("x,X,FORMULARYR,GOV,*,breach,no,P;ZZ,r,n", "unknown alternative"),
])
def test_bad_rule_rows_abort(tmp_path, line, match):
    (tmp_path / C.BRAND_RULES_FILE).write_text(
        "measure,label,classes,payers,patient_types,verdict,by_doctor,alternatives,rule_ref,note\n" + line + "\n")
    with pytest.raises(BuildError, match=match):
        load_brand_rules(tmp_path)


def test_rule_classes_are_canonicalised(tmp_path):
    (tmp_path / C.BRAND_RULES_FILE).write_text(
        "measure,label,classes,payers,patient_types,verdict,by_doctor,alternatives,rule_ref,note\n"
        "x,X,formularyopd-r;FORMULARYP-0PD;out of formulary,CASH(TPA);gov,IP;OP,info,no,p-sch,r,n\n")
    r = load_brand_rules(tmp_path)[0]
    assert r.classes == {"FORMULARYR-OPD", "FORMULARYP-OPD", "OUT OF FORMULARY"}
    assert r.payers == {"CASH(TPA)", "GOV"} and r.patient_types == {"IP", "OP"} and r.alternatives == {"P-SCH"}
    assert load_brand_rules(tmp_path / "missing") == []
    assert K.rule_letter("FORMULARYR-OPD") == "R-OPD" and K.rule_letter("OUT OF FORMULARY") == "OUT OF FORMULARY"


def test_measures_by_hand(ruled):
    rep = run(ruled)
    kr = rep["months"]["2026-08"]["brand_rules"]["kr"]
    # line 1: GOV R 1000; line 7: GOV P 500; line 4: GOV P-OPD inpatient 400 (+ line 9 R-OPD, cost 0)
    assert kr["restricted_scheme"] == {"cost": 1000.0, "lines": 1, "share_pct": pytest.approx(1000 / 2570 * 100, abs=1e-4)}
    assert kr["preferred_scheme"]["cost"] == 500.0
    assert kr["oof_scheme"] == {"cost": 0.0, "lines": 0, "share_pct": 0.0}
    assert kr["restricted_cash_tpa"]["cost"] == 0.0 and kr["restricted_cash_tpa"]["lines"] == 1    # line 9
    assert kr["opd_brand_inpatient"]["cost"] == 400.0 and kr["opd_brand_inpatient"]["lines"] == 2
    al = rep["months"]["2026-08"]["brand_rules"]["all_hospitals"]
    assert al["restricted_cash_tpa"]["cost"] == 800.0                                            # line 11, DR
    assert rep["months"]["2026-08"]["brand_rules"]["reconciliation"] == []
    assert [r["measure"] for r in rep["brand_rules"]][:2] == ["restricted_scheme", "preferred_scheme"]


def test_per_doctor_table_holds_only_the_per_doctor_measures(ruled):
    run(ruled)
    b = tables(ruled)[C.RULE_BREACH]
    assert set(b["measure"]) == {"restricted_scheme", "preferred_scheme"}     # oof_scheme has no line
    r = b[b["measure"] == "restricted_scheme"].iloc[0]
    assert (r["plant"], r["doctor_id"], r["doctor_name"], r["material"], r["patient_type"], r["cost"]) == \
        ("HC05", "D1", "Dr A", "101001", "IP", 1000.0)
    p = b[b["measure"] == "preferred_scheme"].iloc[0]
    assert (p["doctor_id"], p["doctor_name"], p["cost"]) == (C.UNKNOWN_DOCTOR_ID, C.UNKNOWN_DOCTOR_NAME, 500.0)
    assert not {c.upper() for c in b.columns} & C.FORBIDDEN_COLUMNS


def test_rule_book_is_every_formulary_item_of_the_month(ruled):
    run(ruled)
    book = tables(ruled)[C.RULE_BOOK].set_index("material")
    assert sorted(book.index) == ["101001", "101003", "201002", "201003", "201005"]
    assert book.loc["201002", "letter"] == "P-OPD" and book.loc["201002", "formulary_class"] == "FORMULARYP-OPD"
    assert book.loc["101001", "molecule"] == "MINOR 101001" and book.loc["101001", "category"] == "Onco Drugs"
    assert bool(book["sold"].all())


def test_task1_numbers_do_not_move(ruled, env):  # noqa: F811
    with_rules = run(ruled)["months"]["2026-08"]
    s = tables(ruled)[C.SUMMARY]
    assert with_rules["kr"]["pooled_pct"] == pytest.approx(1950 / 2250 * 100, abs=1e-4)
    assert with_rules["kr"]["scheme"] == {"restricted": 1000.0, "preferred": 500.0}
    assert int(s["lines"].sum()) == 15


def test_reconciliation_catches_a_per_doctor_table_that_does_not_add_up():
    rules = load_brand_rules(RULES_CSV.parent)
    summary = pd.DataFrame([{"plant": "HC05", "category": "Onco Drugs", "formulary_class": "FORMULARYR",
                             "payer_bucket": "GOV", "patient_type": "IP", "cost": 1000.0, "lines": 1}])
    breach = pd.DataFrame([{"measure": "restricted_scheme", "plant": "HC05", "cost": 990.0}])
    out = V.rules_reconciliation(summary, breach, rules)
    assert out == ["restricted_scheme at HC05: per-doctor table 990.00 vs summary 1,000.00"]
    rep = {"period": "2026-08", "source": {"dates_outside_month": 0, "rows_read": 1, "cost_read": 1.0, "file": "f"},
           "published": {"lines": 1, "cost": 1.0, "cost_diff": 0.0}, "drugs": {"found_in_master_pct": 100.0},
           "item_master": {"file": "m"}, "brand_rules": {"reconciliation": out}}
    assert any("do not reconcile" in x for x in V.abort_reasons(rep))
