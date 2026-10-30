# tests/test_brand_rules_api.py — /formulary/rules/* (Task 4) on the hand-made formulary dataset.
#
# The formulary tables are tests/test_formulary_api.py's (15 invented lines); this file adds the
# two Task 4 tables derived from the same lines with the committed brand_rules.csv, the rules in
# the validation file, and a few Main Sheet blocks. Expected numbers are worked out by hand
# (line numbers = positions in test_formulary_api.LINES, 1-based).
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.api import formulary, spend
from app.core.config import settings
from app.etl.formulary import config as FC
from app.etl.formulary.masters import load_brand_rules
from tests.conftest import bearer, signup_and_approve
from tests.test_formulary_api import AUG, JUL, _lines, write_hospital_map, write_tables

RULES = load_brand_rules(Path(__file__).resolve().parents[1] / "app" / "data" / "reference")
# Main Sheet blocks: material, name, letter, SHARING, block
BLOCKS = [
    ("100001", "ONCO A 100MG", "R", 0.1, 1), ("100009", "ONCO A2 100MG", "P-SCH", 0.9, 1),
    ("100003", "ONCO C INJ", "R-OPD", 1.0, 2),
    ("200001", "PARA 500", "P", 0.3, 3), ("200009", "PARA PRIME", "P-SCH", 0.7, 3),
    ("200004", "SCH BRAND", "P", 1.0, 4),          # Main Sheet says P, the month's master says SCH
]
# the months' rule books: period, material, name, molecule, letter, sold
BOOK = [
    (AUG, "100001", "ONCO A 100MG", "Onco generic A", "R", True),
    (AUG, "100009", "ONCO A2 100MG", "Onco generic A", "P-SCH", False),
    (AUG, "100003", "ONCO C INJ", "Onco generic C", "R-OPD", True),
    (AUG, "200001", "PARA 500", "Paracetamol", "P", True),
    (AUG, "200004", "SCH BRAND", "Sch generic", "SCH", False),
    (JUL, "100001", "ONCO A 100MG", "Onco generic A", "R", True),
]


def _match(df: pd.DataFrame, r) -> pd.Series:
    m = df["formulary_class"].isin(r.classes) & df["category"].isin(["Onco Drugs", "Other Drugs"])
    if r.payers:
        m &= df["payer_bucket"].isin(r.payers)
    if r.patient_types:
        m &= df["patient_type"].isin(r.patient_types)
    return m


@pytest.fixture()
def rules_data(tmp_path, monkeypatch):
    f, s = tmp_path / "formulary", tmp_path / "spend"
    monkeypatch.setattr(settings, "FORMULARY_DATA_DIR", str(f))
    monkeypatch.setattr(settings, "SPEND_DATA_DIR", str(s))
    # no purchase build: the spend views these tables feed (here and in test_hcg_reports) stay
    # on the hand-made rows, without the committed out-of-formulary purchases
    monkeypatch.setattr(settings, "PURCHASE_DATA_DIR", str(tmp_path / "purchase-not-built"))
    monkeypatch.setattr(settings, "REFERENCE_DIR", str(tmp_path / "reference"))
    write_tables(f)
    write_hospital_map(tmp_path / "reference")
    lines = _lines()
    parts = [lines[_match(lines, r)].assign(measure=r.measure) for r in RULES if r.by_doctor]
    breach = pd.concat(parts, ignore_index=True)
    breach.drop(columns=["formulary_group", "mrp"]).to_parquet(f / f"{FC.RULE_BREACH}.parquet", index=False)
    pd.DataFrame([{"period": p, "material": m, "material_desc": n, "molecule": mol, "manufacturer": "Maker",
                   "category": "Onco Drugs" if m[0] == "1" else "Other Drugs", "letter": l,
                   "formulary_class": "FORMULARY" + l, "sold": sold} for p, m, n, mol, l, sold in BOOK]
                 ).to_parquet(f / f"{FC.RULE_BOOK}.parquet", index=False)
    v = json.loads((f / FC.VALIDATION_FILE).read_text())
    v["brand_rules"] = [r.as_dict() for r in RULES]
    (f / FC.VALIDATION_FILE).write_text(json.dumps(v))
    s.mkdir()
    b = pd.DataFrame([{"material": m, "material_desc": n, "company": "X", "ms_category": l, "sharing": sh,
                       "block_id": blk} for m, n, l, sh, blk in BLOCKS])
    g = b.groupby("block_id")
    b["block_size"] = g["material"].transform("size")
    b["block_sharing_sum"] = g["sharing"].transform("sum")
    b["p_rank"] = g["sharing"].rank(method="min", ascending=False).astype(int)
    b["tie"] = False
    b["cash_suggested"] = b["cash_suggested_desc"] = b["scheme_suggested"] = b["scheme_suggested_desc"] = None
    b["scheme_attis"] = False
    b.to_parquet(s / "brand_blocks.parquet", index=False)
    formulary.clear_caches()
    spend.clear_caches()
    yield tmp_path
    formulary.clear_caches()
    spend.clear_caches()


def get(client, path, **params):
    r = client.get(path, params=params)
    assert r.status_code == 200, r.text
    return r.json()


def measures(d) -> dict:
    return {m["measure"]: m for m in d["measures"]}


# ---------- overview ----------

def test_every_measure_by_hand(client, rules_data):
    d = get(client, "/formulary/rules/overview", Plant="HC05", Month=AUG)
    m = measures(d)
    assert d["drug_spend"] == {"total": 1700.0, "onco": 1000.0, "non_onco": 700.0}
    r = m["restricted_scheme"]                                        # lines 1 + 8: GOV R 800, R-OPD 100
    assert (r["cost"], r["lines"], r["verdict"], r["by_doctor"]) == (900.0, 2, "breach", True)
    assert r["share_pct"] == pytest.approx(900 / 1700 * 100, abs=1e-4)
    assert r["onco"] == {"cost": 900.0, "share_pct": 90.0} and r["non_onco"]["cost"] == 0.0
    assert r["letters"] == ["R", "R-OPD"] and r["payers"] == ["GOV"]
    assert m["preferred_scheme"]["cost"] == 50.0                      # line 4 (GOV P), not line 3 (cash)
    assert m["oof_scheme"]["cost"] == 0.0                             # only DR has one (line 15)
    assert m["restricted_cash_tpa"]["cost"] == 0.0 and m["restricted_cash_tpa"]["verdict"] == "watch"
    assert m["opd_brand_inpatient"]["cost"] == 100.0                  # line 8 again: measures overlap
    assert m["scheme_brand_cash_tpa"]["cost"] == 0.0
    assert any("never added up" in n for n in d["notes"])


def test_alternative_split_uses_the_main_sheet_blocks(client, rules_data):
    m = measures(get(client, "/formulary/rules/overview", Plant="HC05", Month=AUG))
    # 100001 (R) shares block 1 with a P-SCH brand -> yes; 100003 (R-OPD) is alone in block 2 -> no
    assert m["restricted_scheme"]["alternative"] == {"yes": 800.0, "no": 100.0, "not in list": 0.0}
    assert m["preferred_scheme"]["alternative"] == {"yes": 50.0, "no": 0.0, "not in list": 0.0}
    assert m["oof_scheme"]["alternative"] is None                     # the rule names no alternatives


def test_trend_and_hospitals_follow_the_selected_measure(client, rules_data):
    d = get(client, "/formulary/rules/overview", Plant="HC05", Month=AUG, Measure="oof_scheme")
    assert d["selected"] == "oof_scheme"
    assert [(h["plant"], h["cost"]) for h in d["by_hospital"]] == [("HC06", 40.0)]
    assert d["by_hospital"][0]["share_pct"] == pytest.approx(40 / 1040 * 100, abs=1e-4)    # DR Aug drugs 1040
    t = get(client, "/formulary/rules/overview", Plant="HC05", Month=AUG)["trend"]
    assert [(x["label"], x["cost"], x["share_pct"]) for x in t] == [("Jul-26", 0.0, 0.0),
                                                                      ("Aug-26", 900.0, pytest.approx(52.9412, abs=1e-4))]


# ---------- items and doctors ----------

def test_items_of_a_measure_with_their_block(client, rules_data):
    rows = get(client, "/formulary/rules/items", Plant="HC05", Month=AUG)["data"]
    a, c = rows
    assert (a["material"], a["letter"], a["cost"], a["alternative"], a["p_rank"], a["sharing"]) == \
        ("100001", "R", 800.0, "yes", 2, 10.0)
    assert a["p1_desc"] == "ONCO A2 100MG" and "ONCO A2 100MG (P-SCH 90%)" in a["block_brands"]
    assert (c["material"], c["alternative"], c["p_rank"]) == ("100003", "no", 1)
    assert a["share_pct"] == pytest.approx(800 / 900 * 100, abs=1e-4)
    assert a["manufacturer"] == "Maker 1" and a["generic_name"] == "Onco generic A"


def test_one_doctors_items_and_the_doctors_of_a_brand(client, rules_data):
    d1 = get(client, "/formulary/rules/items", Plant="HC05", Month=AUG, Doctor="D1")["data"]
    assert [(r["material"], r["cost"], r["manufacturer"]) for r in d1] == [("100001", 800.0, "Maker 1")]
    docs = get(client, "/formulary/rules/doctors", Plant="HC05", Month=AUG, Measure="restricted_scheme")["data"]
    assert [(r["doctor_id"], r["doctor_name"], r["cost"], r["brands"]) for r in docs] == \
        [("D1", "Dr Asha", 800.0, 1), ("UNKNOWN", "Not recorded", 100.0, 1)]
    one = get(client, "/formulary/rules/doctors", Month=AUG, Measure="preferred_scheme", Material="200001")["data"]
    assert [(r["doctor_id"], r["plant"], r["cost"]) for r in one] == [("D2", "HC05", 50.0)]


def test_measures_not_kept_per_doctor_are_a_400(client, rules_data):
    for path, params in (("/formulary/rules/doctors", {"Measure": "restricted_cash_tpa"}),
                         ("/formulary/rules/items", {"Measure": "opd_brand_inpatient", "Doctor": "D1"}),
                         ("/formulary/rules/overview", {"Measure": "made_up"})):
        r = client.get(path, params=params)
        assert r.status_code == 400, (path, r.text)


# ---------- rule book ----------

def test_rule_book_says_what_each_payer_may_get(client, rules_data):
    rows = {r["material"]: r for r in get(client, "/formulary/rules/book", Month=AUG)["data"]}
    assert sorted(rows) == ["100001", "100003", "100009", "200001", "200004"]
    assert (rows["100001"]["gov"], rows["100001"]["cash"], rows["100001"]["tpa"]) == ("breach", "watch", "watch")
    assert (rows["100003"]["gov"], rows["100003"]["cash"]) == ("breach", "watch")       # breach beats info (IP)
    assert (rows["200001"]["gov"], rows["200001"]["cash"]) == ("breach", "ok")
    assert (rows["100009"]["gov"], rows["100009"]["cash"], rows["100009"]["sold"]) == ("ok", "ok", False)
    assert (rows["200004"]["gov"], rows["200004"]["cash"]) == ("ok", "info")
    assert rows["200004"]["conflict"] is True and rows["100001"]["conflict"] is False
    assert (rows["100009"]["p_rank"], rows["100009"]["sharing"]) == (1, 90.0)


def test_rule_book_filters(client, rules_data):
    assert [r["material"] for r in get(client, "/formulary/rules/book", Month=AUG, Letter="r")["data"]] == ["100001"]
    nonc = get(client, "/formulary/rules/book", Month=AUG, Segment="non-onco")["data"]
    assert sorted(r["material"] for r in nonc) == ["200001", "200004"]
    assert [r["material"] for r in get(client, "/formulary/rules/book", Month=JUL)["data"]] == ["100001"]
    hits = get(client, "/formulary/rules/book", Month=AUG, global_filter="para")["data"]
    assert [r["material"] for r in hits] == ["200001"]


# ---------- Task 1 overview gains HCG's third figure ----------

def test_overview_scheme_block_adds_restricted_for_cash_and_tpa(client, rules_data):
    kr = get(client, "/formulary/overview", Plant="HC05", Month="ALL")["scheme"]["restricted_cash_tpa"]
    assert (kr["cost"], kr["lines"]) == (400.0, 1)                     # line 9: Jul, cash, R
    dr = get(client, "/formulary/overview", Plant="HC06", Month=AUG)["scheme"]
    assert dr["restricted_cash_tpa"]["cost"] == 500.0 and dr["restricted"]["cost"] == 0.0   # line 11


# ---------- auth, missing data ----------

def test_every_rules_route_needs_a_login(anon_client, rules_data):
    for path in ("/formulary/rules/overview", "/formulary/rules/items", "/formulary/rules/doctors",
                 "/formulary/rules/book"):
        assert anon_client.get(path).status_code == 401, path


def test_rules_not_built_are_a_clear_503(client, rules_data):
    f = Path(settings.FORMULARY_DATA_DIR) / FC.VALIDATION_FILE
    v = json.loads(f.read_text())
    v.pop("brand_rules")
    f.write_text(json.dumps(v))
    formulary.clear_caches()
    for path in ("/formulary/rules/overview", "/formulary/rules/items", "/formulary/rules/book"):
        r = client.get(path)
        assert r.status_code == 503 and "--from-cache" in r.json()["detail"], path
    assert client.get("/formulary/overview", params={"Plant": "HC05"}).status_code == 200   # Task 1 unaffected


def test_no_patient_fields(client, rules_data):
    bad = {"uhid", "patname", "ip_opnumber", "salesinvoiceno", "payer1"}
    for path in ("/formulary/rules/overview", "/formulary/rules/items", "/formulary/rules/doctors",
                 "/formulary/rules/book"):
        text = json.dumps(get(client, path)).lower()
        assert not any(f'"{b}"' in text for b in bad), path
