# tests/test_formulary_api.py — the /formulary/* endpoints, on a tiny hand-made dataset.
#
# The four published tables are derived here from 15 invented drug/consumable lines, so they
# agree with each other the way the build's tables do. Every expected number below was
# worked out by hand from those lines (comments show the arithmetic), not by re-running the
# API's own code. Nothing here needs the real data; tests/test_formulary_golden.py checks the
# real KR figures when the built tables are present.
from __future__ import annotations

import json

import pandas as pd
import pytest

from app.api import formulary
from app.core import data_access as da
from app.core.config import settings

from tests.conftest import bearer, signup_and_approve

AUG, JUL = "2026-08", "2026-07"
KR, DR, RAJKOT = "HC05", "HC06", "HS01"
HIS = {KR: ("KABHK", "HCG KR, Bangalore"), DR: ("KABHD", "HCG DR, Bangalore"),
       RAJKOT: ("GJHHR", "HCG Hospitals, Rajkot")}
CLASS_GROUP = {"FORMULARYR": "A", "FORMULARYR-OPD": "A", "FORMULARYP": "A",
               "OUT OF FORMULARY": "B", "NON FORMULARY": "C", "NOT IN MASTER": "X"}
DOCTORS = {"D1": "Dr Asha", "D2": "Dr Bhanu", "D4": "Dr Dev", "D5": "Dr Esha", "UNKNOWN": "Not recorded"}

# period, plant, type, payer, class, doctor, material, cost
LINES = [
    (AUG, KR, "IP", "GOV", "FORMULARYR", "D1", "100001", 800),          # 1
    (AUG, KR, "IP", "CASH", "OUT OF FORMULARY", "D1", "100002", 100),   # 2
    (AUG, KR, "OP", "CASH", "FORMULARYP", "D2", "200001", 300),         # 3
    (AUG, KR, "OP", "GOV", "FORMULARYP", "D2", "200001", 50),           # 4
    (AUG, KR, "OP", "CASH(TPA)", "OUT OF FORMULARY", "D2", "200002", 150),  # 5
    (AUG, KR, "IP", "GOV", "NON FORMULARY", "D1", "200003", 200),       # 6
    (AUG, KR, "IP", "CASH", "NOT IN MASTER", "D1", "300001", 999),      # 7 consumable
    (AUG, KR, "IP", "GOV", "FORMULARYR-OPD", "UNKNOWN", "100003", 100), # 8
    (JUL, KR, "IP", "CASH", "FORMULARYR", "D1", "100001", 400),         # 9
    (JUL, KR, "OP", "CASH", "NON FORMULARY", "D2", "200003", 50),       # 10
    (AUG, DR, "IP", "CASH", "FORMULARYR", "D4", "100001", 500),         # 11
    (AUG, DR, "IP", "CASH", "OUT OF FORMULARY", "D4", "100002", 500),   # 12
    (JUL, DR, "OP", "GOV", "FORMULARYP", "D4", "200001", 100),          # 13
    (AUG, RAJKOT, "OP", "CASH", "NON FORMULARY", "D5", "200003", 70),   # 14
    (AUG, DR, "IP", "GOV", "OUT OF FORMULARY", "D1", "100002", 40),     # 15
]
ITEMS = {"100001": ("ONCO A 100MG", "Onco generic A", "Maker 1"),
         "100002": ("ONCO B 50MG", "Onco generic B", "Maker 2"),
         "100003": ("ONCO C INJ", "Onco generic C", "Maker 1"),
         "200001": ("PARA 500", "Paracetamol", "Maker 3"),
         "200002": ("PANTO 40", "Pantoprazole", "Maker 4"),
         "200003": ("VIT D3", "Cholecalciferol", "Maker 5"),
         "300001": ("SYRINGE 5ML", "Syringe", "Maker 6")}
CATEGORY = {"1": "Onco Drugs", "2": "Other Drugs", "3": "Consumables"}


def _lines() -> pd.DataFrame:
    df = pd.DataFrame(LINES, columns=["period", "plant", "patient_type", "payer_bucket",
                                      "formulary_class", "doctor_id", "material", "cost"])
    df["cost"] = df["cost"].astype(float)
    df["hospital"] = df["plant"].map(lambda p: HIS[p][0])
    df["hospital_name"] = df["plant"].map(lambda p: HIS[p][1])
    df["category"] = df["material"].str[0].map(CATEGORY)
    df["formulary_group"] = df["formulary_class"].map(CLASS_GROUP)
    df["doctor_name"] = df["doctor_id"].map(DOCTORS)
    df["material_desc"] = df["material"].map(lambda m: ITEMS[m][0])
    df["generic_name"] = df["material"].map(lambda m: ITEMS[m][1])
    df["manufacturer"] = df["material"].map(lambda m: ITEMS[m][2])
    df["mrp"] = df["cost"] * 1.2
    df["qty"] = 2.0
    df["lines"] = 1
    return df


def _agg(df, keys, measures=("cost", "qty", "lines")):
    return df.groupby(keys, as_index=False)[list(measures)].sum()


def write_tables(folder) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    df = _lines()
    drugs = df[df["category"].isin(["Onco Drugs", "Other Drugs"])]
    _agg(df, ["period", "plant", "hospital", "hospital_name", "patient_type", "category", "payer_bucket",
              "formulary_class", "formulary_group"], ("cost", "mrp", "qty", "lines")
         ).to_parquet(folder / "formulary_summary.parquet", index=False)
    _agg(drugs, ["period", "plant", "hospital", "hospital_name", "patient_type", "category", "payer_bucket",
                 "formulary_group", "doctor_id", "doctor_name"]
         ).to_parquet(folder / "formulary_by_doctor.parquet", index=False)
    _agg(drugs, ["period", "plant", "hospital", "hospital_name", "patient_type", "payer_bucket", "material",
                 "material_desc", "generic_name", "manufacturer", "category", "formulary_class",
                 "formulary_group"]
         ).to_parquet(folder / "formulary_by_item.parquet", index=False)
    _agg(drugs[drugs["formulary_group"].isin(["B", "C"])],
         ["period", "plant", "hospital", "payer_bucket", "doctor_id", "doctor_name", "material",
          "material_desc", "formulary_class", "formulary_group"]
         ).to_parquet(folder / "formulary_doctor_item.parquet", index=False)
    (folder / "formulary_validation.json").write_text(json.dumps({
        "built_at": "2026-10-09T12:00:00+00:00",
        "months": {AUG: {"source_file": "IP & OP Consolidated File August 26.xlsx",
                         "sheet": "IP&OP Con WS August 26", "rows_read": 14,
                         "agreement": {"category": {"lines_pct": 100.0}}},
                   JUL: {"source_file": "IP and OP Sales Consolidated file July 2026.xlsb",
                         "sheet": "IP&OP Jul 26 WS", "rows_read": 4}}}))


def write_hospital_map(folder) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"his_code": "KABHK", "his_location_name": "HCG KR, Bangalore", "his_location_id": "10107",
         "sap_plant": "HC05", "sap_plant_name": "HCG KR, Bangalore", "confidence": "high", "note": ""},
        {"his_code": "KABHD", "his_location_name": "HCG DR, Bangalore", "his_location_id": "10108",
         "sap_plant": "HC06", "sap_plant_name": "HCG DR, Bangalore", "confidence": "high", "note": ""},
        {"his_code": "GJHHR", "his_location_name": "HCG Hospitals, Rajkot", "his_location_id": "11201",
         "sap_plant": "HS01", "sap_plant_name": "HCG SUN HOSPITALS LLP", "confidence": "high", "note": ""},
    ]).to_csv(folder / "hospital_map.csv", index=False)


@pytest.fixture()
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "FORMULARY_DATA_DIR", str(tmp_path / "formulary"))
    monkeypatch.setattr(settings, "REFERENCE_DIR", str(tmp_path / "reference"))
    write_tables(tmp_path / "formulary")
    write_hospital_map(tmp_path / "reference")
    formulary.clear_caches()
    yield tmp_path
    formulary.clear_caches()


def overview(client, **params):
    r = client.get("/formulary/overview", params=params)
    assert r.status_code == 200, r.text
    return r.json()


def table(client, path, **params):
    r = client.get(path, params=params)
    assert r.status_code == 200, r.text
    return r.json()


# ---------- headline arithmetic ----------

def test_headline_is_a_ratio_of_sums_over_drugs(client, data):
    h = overview(client, Plant="HC05", Month=AUG)["headline"]
    # A = 800 + 300 + 50 + 100, B = 100 + 150, C = 200; the consumable (999) is outside.
    assert (h["formulary"], h["out_of_formulary"], h["non_formulary"]) == (1250.0, 250.0, 200.0)
    assert h["total"] == 1700.0 and h["lines"] == 7
    assert h["pooled_pct"] == pytest.approx(83.3333, abs=1e-4)      # 1250 / 1500
    assert h["onco_pct"] == pytest.approx(90.0)                       # 900 / 1000
    assert h["non_onco_pct"] == pytest.approx(70.0)                   # 350 / 500
    assert h["mean_pct"] == pytest.approx(80.0)
    assert h["compliance_pct"] == h["pooled_pct"]                     # Method defaults to pooled
    assert list(h)[0] == "compliance_pct"


def test_mean_method_averages_the_two_segment_ratios(client, data):
    h = overview(client, Plant="HC05", Month=AUG, Method="mean")["headline"]
    assert h["compliance_pct"] == pytest.approx(80.0)
    assert h["pooled_pct"] == pytest.approx(83.3333, abs=1e-4)


def test_mean_skips_a_segment_with_no_formulary_or_out_of_formulary_spend(client, data):
    # KR July: onco A = 400, B = 0; non-onco has only Non Formulary (50) -> no ratio.
    h = overview(client, Plant="HC05", Month=JUL)["headline"]
    assert h["onco_pct"] == 100.0 and h["non_onco_pct"] is None
    assert h["mean_pct"] == 100.0 and h["pooled_pct"] == 100.0


def test_a_zero_denominator_is_null_not_zero_or_hundred(client, data):
    # Rajkot has only Non Formulary spend in August.
    h = overview(client, Plant="HS01", Month=AUG)["headline"]
    assert h["non_formulary"] == 70.0
    assert h["compliance_pct"] is None and h["pooled_pct"] is None and h["mean_pct"] is None
    assert h["onco_pct"] is None and h["non_onco_pct"] is None


def test_several_months_pool_the_amounts_never_the_percentages(client, data):
    h = overview(client, Plant="HC05", Month="ALL")["headline"]
    # (1250 + 400) / (1250 + 400 + 250) = 86.8421 — not the mean of 83.33 and 100.
    assert h["pooled_pct"] == pytest.approx(86.8421, abs=1e-4)
    assert overview(client, Plant="HC05", Month=f"{JUL},{AUG}")["headline"] == h


def test_month_defaults_to_the_latest_period(client, data):
    d = overview(client, Plant="HC05")
    assert d["scope"]["months"] == [AUG]
    assert d["headline"] == overview(client, Plant="HC05", Month=AUG)["headline"]


# ---------- filters ----------
# (Cases are looped rather than parametrized: every test pays ~3 s of admin seeding.)

def test_filters_cut_the_headline(client, data):
    for params, a, b, pct in [
        ({"Type": "IP"}, 900.0, 100.0, 90.0),                  # lines 1, 2, 8 (6 is C)
        ({"Type": "op"}, 350.0, 150.0, 70.0),                  # lines 3, 4, 5
        ({"Payer": "GOV"}, 950.0, 0.0, 100.0),                 # lines 1, 4, 8 (6 is C)
        ({"Payer": "CASH"}, 300.0, 100.0, 75.0),               # lines 2, 3 (7 is a consumable)
        ({"Payer": "CASH(TPA)"}, 0.0, 150.0, 0.0),             # line 5
        ({"Segment": "onco"}, 900.0, 100.0, 90.0),
        ({"Segment": "non-onco"}, 350.0, 150.0, 70.0),
        ({"Category": "Onco Drugs"}, 900.0, 100.0, 90.0),
        ({"Category": "Other Drugs"}, 350.0, 150.0, 70.0),
    ]:
        h = overview(client, Plant="HC05", Month=AUG, **params)["headline"]
        assert (h["formulary"], h["out_of_formulary"]) == (a, b), params
        assert h["compliance_pct"] == pytest.approx(pct, abs=1e-4), params


def test_scope_echoes_the_normalised_filters(client, data):
    s = overview(client, Plant="KABHK", Type="ip", Segment="Non-Onco", Payer="gov", Method="MEAN")["scope"]
    assert s == {"plant": "HC05", "plant_name": "HCG KR, Bangalore", "his_name": "HCG KR, Bangalore",
                 "months": [AUG], "type": "IP", "segment": "non-onco", "payer": "GOV", "method": "mean"}
    s = overview(client)["scope"]
    assert s["plant"] == "ALL" and s["his_name"] is None and s["type"] == "ALL"


def test_segments_and_types_always_carry_both_rows(client, data):
    d = overview(client, Plant="HC05", Month=AUG)
    seg = {s["key"]: s for s in d["segments"]}
    assert [s["key"] for s in d["segments"]] == ["onco", "non-onco"]
    assert seg["onco"]["label"] == "Onco" and seg["non-onco"]["label"] == "Non-Onco"
    assert (seg["onco"]["formulary"], seg["onco"]["out_of_formulary"], seg["onco"]["lines"]) == (900.0, 100.0, 3)
    assert seg["non-onco"]["compliance_pct"] == pytest.approx(70.0)
    assert seg["non-onco"]["non_formulary"] == 200.0
    types = {t["type"]: t for t in d["by_type"]}
    assert types["IP"]["compliance_pct"] == pytest.approx(90.0)
    assert types["OP"]["compliance_pct"] == pytest.approx(70.0)
    only_ip = overview(client, Plant="HC05", Month=AUG, Type="IP")["by_type"]
    assert [t["type"] for t in only_ip] == ["IP", "OP"]
    assert only_ip[1]["compliance_pct"] is None and only_ip[1]["lines"] == 0


def test_trend_shows_every_month_whatever_the_month_filter(client, data):
    for month in (JUL, AUG, "ALL"):
        trend = overview(client, Plant="HC05", Month=month)["trend"]
        assert [(t["period"], t["label"]) for t in trend] == [(JUL, "Jul-26"), (AUG, "Aug-26")]
        assert trend[0]["compliance_pct"] == 100.0
        assert trend[1]["compliance_pct"] == pytest.approx(83.3333, abs=1e-4)
        assert trend[1]["mean_pct"] == pytest.approx(80.0)


def test_by_class_shares_add_up_to_the_drug_total(client, data):
    rows = overview(client, Plant="HC05", Month=AUG)["by_class"]
    by = {r["formulary_class"]: r for r in rows}
    assert by["FORMULARYR"]["cost"] == 800.0 and by["FORMULARYR"]["formulary_group"] == "A"
    assert by["OUT OF FORMULARY"]["share_pct"] == pytest.approx(250 / 1700 * 100, abs=1e-4)
    assert "NOT IN MASTER" not in by                                  # consumables only
    assert sum(r["share_pct"] for r in rows) == pytest.approx(100.0, abs=1e-3)


# ---------- hospital ranking ----------

def test_hospital_ranking_is_worst_first_and_ignores_the_plant(client, data):
    rows = overview(client, Plant="KABHK", Month=AUG)["by_hospital"]
    assert [r["plant"] for r in rows] == [DR, KR, RAJKOT]
    dr, kr, rajkot = rows
    assert dr["compliance_pct"] == pytest.approx(500 / 1040 * 100, abs=1e-4)    # 500 / (500 + 500 + 40)
    assert (dr["rank"], kr["rank"], rajkot["rank"]) == (1, 2, None)              # no ratio, no rank
    assert [r["selected"] for r in rows] == [False, True, False]
    assert rajkot["name"] == "HCG SUN HOSPITALS LLP"                             # SAP name ...
    assert (rajkot["his_code"], rajkot["his_name"]) == ("GJHHR", "HCG Hospitals, Rajkot")  # ... and HIS
    assert overview(client, Month=AUG)["by_hospital"] == [{**r, "selected": False} for r in rows]


def test_network_is_the_all_hospitals_figure_of_the_same_filters(client, data):
    # Aug drugs, every hospital: A 1250 (KR) + 500 (DR), B 250 (KR) + 540 (DR) -> 1750 / 2540
    assert overview(client, Plant="HC05", Month=AUG)["network"] == {
        "compliance_pct": pytest.approx(1750 / 2540 * 100, abs=1e-4)}
    for q in ({"Month": AUG}, {"Month": "ALL", "Type": "IP"}, {"Month": AUG, "Method": "mean"},
              {"Month": AUG, "Segment": "onco", "Payer": "CASH"}):
        whole = overview(client, **q)["headline"]["compliance_pct"]
        for plant in (KR, DR, RAJKOT, "ALL"):
            assert overview(client, Plant=plant, **q)["network"] == {"compliance_pct": whole}, (plant, q)


# ---------- GOV scheme views ----------

def test_scheme_views_count_gov_lines_and_share_all_payers_spend(client, data):
    sc = overview(client, Plant="HC05", Month=AUG, Payer="CASH")["scheme"]    # Payer is ignored here
    r, p = sc["restricted"], sc["preferred"]
    assert (r["cost"], r["lines"]) == (900.0, 2)                    # FORMULARYR 800 + FORMULARYR-OPD 100
    assert r["share_pct"] == pytest.approx(900 / 1700 * 100, abs=1e-4)
    assert r["onco"] == {"cost": 900.0, "share_pct": pytest.approx(90.0)}       # of onco spend 1000
    assert r["non_onco"] == {"cost": 0.0, "share_pct": 0.0}
    assert (p["cost"], p["lines"]) == (50.0, 1)                     # FORMULARYP on GOV only (not line 3)
    assert p["non_onco"]["share_pct"] == pytest.approx(50 / 700 * 100, abs=1e-4)
    assert sc["gov_total"] == 1150.0                                # 800 + 50 + 200 + 100


# ---------- Plant resolution ----------

def test_plant_accepts_sap_code_his_code_and_name(client, data):
    for plant in ("HC05", "KABHK", "kabhk", "HCG KR, Bangalore", "  KABHK "):
        assert overview(client, Plant=plant, Month=AUG)["scope"]["plant"] == "HC05", plant
    # Rajkot's HIS and SAP names differ; both resolve.
    for plant in ("HS01", "GJHHR", "HCG Hospitals, Rajkot", "HCG SUN HOSPITALS LLP"):
        assert overview(client, Plant=plant, Month=AUG)["scope"]["plant"] == "HS01", plant


def test_all_means_every_hospital(client, data):
    for plant in ("", "ALL", "All Plants", "all hospitals"):
        h = overview(client, Plant=plant, Month=AUG)["headline"]
        assert (h["formulary"], h["out_of_formulary"], h["non_formulary"]) == (1750.0, 790.0, 270.0), plant


def test_an_unknown_plant_is_a_404_not_every_hospital(client, data):
    for path in ("/formulary/overview", "/formulary/doctors", "/formulary/items", "/formulary/scheme/items"):
        r = client.get(path, params={"Plant": "NOPE"})
        assert r.status_code == 404, (path, r.text)
        assert "NOPE" in r.json()["detail"]


def test_a_known_plant_without_formulary_data_answers_with_nulls(client, data):
    d = overview(client, Plant="HC01")
    assert d["headline"]["compliance_pct"] is None and d["headline"]["lines"] == 0
    assert any("No formulary drug spend" in n for n in d["notes"])


def test_his_codes_now_filter_the_existing_endpoints_too(client, data):
    # Before the hospital map, Plant=KABHK resolved to None — every plant, silently.
    assert da.resolve_plant("KABHK") == "HC05"
    def total(plant):
        r = client.get("/kpi/stock-change-table", params={"Plant": plant, "page_size": 1})
        assert r.status_code == 200, r.text
        return r.json()["total"]
    assert total("KABHK") == total("HC05") < total("ALL")


def test_without_the_hospital_map_his_codes_resolve_as_before(client, data, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "REFERENCE_DIR", str(tmp_path / "nowhere"))
    assert da.resolve_plant("KABHK") is None
    assert da.resolve_plant("HC05") == "HC05"
    assert da.resolve_plant("HCG KR, Bangalore") == "HC05"
    assert da.hospital_map().empty


# ---------- bad parameters ----------

def test_bad_parameters_are_a_400(client, data):
    for path, params in [("/formulary/overview", p) for p in (
            {"Month": "2026-8"}, {"Month": "Aug-26"}, {"Month": "2026-05"}, {"Month": ","},
            {"Type": "ER"}, {"Segment": "consumables"}, {"Category": "Consumables"}, {"Category": "Lab"},
            {"Segment": "onco", "Category": "Other Drugs"}, {"Payer": "INSURANCE"}, {"Method": "median"})] + [
            ("/formulary/items", {"Group": "D"}),
            ("/formulary/items", {"Group": "A", "Doctor": "D1"}),         # doctor drill is B / C only
            ("/formulary/items", {"Doctor": "D1", "Type": "IP"}),         # doctor x item has no IP/OP
            ("/formulary/doctors", {"Material": "100002", "Type": "OP"}),
            ("/formulary/scheme/items", {"Kind": "banned"}),
            ("/formulary/doctors", {"Payer": "TPA"})]:
        r = client.get(path, params=params)
        assert r.status_code == 400, (path, params, r.text)
        assert r.json()["detail"], (path, params)


# ---------- doctors ----------

def test_doctors_default_to_most_out_of_formulary_first(client, data):
    body = table(client, "/formulary/doctors", Plant="HC05", Month=AUG)
    assert body["total"] == 3
    d2, d1, unknown = body["data"]
    assert (d2["doctor_id"], d1["doctor_id"], unknown["doctor_id"]) == ("D2", "D1", "UNKNOWN")
    assert d1 == {"doctor_id": "D1", "doctor_name": "Dr Asha", "plant": "HC05",
                  "hospital_name": "HCG KR, Bangalore", "formulary": 800.0, "out_of_formulary": 100.0,
                  "non_formulary": 200.0, "compliance_pct": pytest.approx(88.8889, abs=1e-4),
                  "oof_share_pct": pytest.approx(40.0), "lines": 3,
                  # lines 1, 2, 6: total 800 + 100 + 200, onco 800 + 100, non-onco 200
                  "total": 1100.0, "onco": 900.0, "non_onco": 200.0}
    assert d2["oof_share_pct"] == pytest.approx(60.0) and d2["compliance_pct"] == pytest.approx(70.0)
    assert unknown["doctor_name"] == "Not recorded" and unknown["compliance_pct"] == 100.0


def test_doctors_page_sort_and_search(client, data):
    body = table(client, "/formulary/doctors", Month=AUG, sort_field="formulary", sort_order="asc",
                 page=0, page_size=2)
    assert body["total"] == 6                                        # D1 at two hospitals = two rows
    assert [r["formulary"] for r in body["data"]] == [0.0, 0.0]
    nxt = table(client, "/formulary/doctors", Month=AUG, sort_field="formulary", sort_order="asc",
                page=1, page_size=2)
    assert [r["formulary"] for r in nxt["data"]] == [100.0, 350.0]
    found = table(client, "/formulary/doctors", Month=AUG, global_filter="bhanu")
    assert found["total"] == 1 and found["data"][0]["doctor_id"] == "D2"


def test_doctors_follow_the_filters(client, data):
    body = table(client, "/formulary/doctors", Plant="HC05", Month=AUG, Type="OP")
    assert [r["doctor_id"] for r in body["data"]] == ["D2"]
    body = table(client, "/formulary/doctors", Plant="HC05", Month=AUG, Payer="GOV")
    assert {r["doctor_id"]: r["formulary"] for r in body["data"]} == {"D1": 800.0, "D2": 50.0, "UNKNOWN": 100.0}


def test_doctors_behind_one_item(client, data):
    body = table(client, "/formulary/doctors", Month=AUG, Material="100002")
    rows = [(r["doctor_id"], r["plant"], r["cost"]) for r in body["data"]]
    assert rows == [("D4", "HC06", 500.0), ("D1", "HC05", 100.0), ("D1", "HC06", 40.0)]
    assert body["data"][0] == {"doctor_id": "D4", "doctor_name": "Dr Dev", "plant": "HC06",
                               "hospital_name": "HCG DR, Bangalore", "formulary_class": "OUT OF FORMULARY",
                               "cost": 500.0, "qty": 2.0, "lines": 1}
    assert table(client, "/formulary/doctors", Plant="HC05", Month=AUG, Material="100002.0")["total"] == 1


# ---------- items ----------

def test_items_default_to_out_of_formulary_by_value(client, data):
    body = table(client, "/formulary/items", Month=AUG)
    assert [r["material"] for r in body["data"]] == ["100002", "200002"]
    top = body["data"][0]
    assert top == {"material": "100002", "material_desc": "ONCO B 50MG", "generic_name": "Onco generic B",
                   "manufacturer": "Maker 2", "category": "Onco Drugs", "formulary_class": "OUT OF FORMULARY",
                   "formulary_group": "B", "cost": 640.0, "qty": 6.0, "lines": 3,
                   "share_pct": pytest.approx(640 / 790 * 100, abs=1e-4), "hospitals": 2}


def test_items_by_group_and_filters(client, data):
    a = table(client, "/formulary/items", Plant="HC05", Month=AUG, Group="a")
    assert [(r["material"], r["formulary_class"], r["cost"]) for r in a["data"]] == [
        ("100001", "FORMULARYR", 800.0), ("200001", "FORMULARYP", 350.0), ("100003", "FORMULARYR-OPD", 100.0)]
    c = table(client, "/formulary/items", Plant="HC05", Month=AUG, Group="C", Segment="non-onco")
    assert [(r["material"], r["share_pct"]) for r in c["data"]] == [("200003", 100.0)]
    assert table(client, "/formulary/items", Plant="HC05", Month=AUG, Group="C", Segment="onco")["total"] == 0


def test_one_doctors_items(client, data):
    b = table(client, "/formulary/items", Plant="HC05", Month=AUG, Doctor="D1")
    assert b["total"] == 1
    assert b["data"][0] == {"material": "100002", "material_desc": "ONCO B 50MG", "generic_name": "Onco generic B",
                            "manufacturer": "Maker 2", "category": "Onco Drugs",
                            "formulary_class": "OUT OF FORMULARY", "formulary_group": "B", "cost": 100.0,
                            "qty": 2.0, "lines": 1, "share_pct": 100.0, "hospitals": 1}
    everywhere = table(client, "/formulary/items", Month=AUG, Doctor="D1")["data"]
    assert [(r["cost"], r["hospitals"]) for r in everywhere] == [(140.0, 2)]
    c = table(client, "/formulary/items", Plant="HC05", Month=AUG, Doctor="D1", Group="C")
    assert [(r["material"], r["category"], r["cost"]) for r in c["data"]] == [("200003", "Other Drugs", 200.0)]


def test_scheme_items(client, data):
    r = table(client, "/formulary/scheme/items", Plant="HC05", Month=AUG)
    assert [(x["material"], x["formulary_class"], x["cost"]) for x in r["data"]] == [
        ("100001", "FORMULARYR", 800.0), ("100003", "FORMULARYR-OPD", 100.0)]
    assert set(r["data"][0]) == {"material", "material_desc", "generic_name", "manufacturer", "category",
                                 "formulary_class", "cost", "qty", "lines", "hospitals"}
    p = table(client, "/formulary/scheme/items", Plant="HC05", Month=AUG, Kind="preferred", Payer="CASH")
    assert [(x["material"], x["cost"]) for x in p["data"]] == [("200001", 50.0)]      # GOV line only
    net = table(client, "/formulary/scheme/items", Month="ALL", Kind="preferred")
    assert [(x["material"], x["cost"], x["hospitals"]) for x in net["data"]] == [("200001", 150.0, 2)]


# ---------- meta, validation, auth, missing data ----------

def test_meta(client, data):
    m = table(client, "/formulary/meta")
    assert m["months"] == [JUL, AUG] and m["latest"] == AUG
    assert m["month_labels"] == {JUL: "Jul-26", AUG: "Aug-26"}
    assert "pan_months" not in m
    assert [h["his_name"] for h in m["hospitals"]] == ["HCG DR, Bangalore", "HCG Hospitals, Rajkot",
                                                         "HCG KR, Bangalore"]
    assert m["hospitals"][1] == {"plant": "HS01", "name": "HCG SUN HOSPITALS LLP", "his_code": "GJHHR",
                                 "his_name": "HCG Hospitals, Rajkot"}
    assert m["types"] == ["IP", "OP"] and m["payers"] == ["CASH", "CASH(TPA)", "GOV"]
    assert m["segments"] == [{"key": "onco", "label": "Onco"}, {"key": "non-onco", "label": "Non-Onco"}]
    assert [x["key"] for x in m["methods"]] == ["pooled", "mean"]
    assert m["target_pct"] == 100 and m["built_at"] == "2026-10-09T12:00:00+00:00"
    assert m["sources"]["months"][AUG] == {"source_file": "IP & OP Consolidated File August 26.xlsx",
                                           "sheet": "IP&OP Con WS August 26", "rows_read": 14}
    assert m["notes"]


def test_the_pan_view_is_gone(client, data):
    # The PAN units tab was removed; each unit's budget still reaches /formulary/overview.
    assert client.get("/formulary/pan").status_code == 404


def test_meta_plants_marks_the_formulary_hospitals(client, data):
    plants = {p["code"]: p["domains"] for p in client.get("/meta/plants").json()["plants"]}
    assert "formulary" in plants["ALL"]
    assert {c for c, d in plants.items() if "formulary" in d} == {"ALL", KR, DR, RAJKOT}


def test_validation_is_admin_only(client, data, anon_client, admin_headers):
    r = client.get("/formulary/validation")
    assert r.status_code == 200, r.text
    assert r.json()["months"][AUG]["sheet"] == "IP&OP Con WS August 26"
    signup_and_approve(anon_client, admin_headers, email="viewer@example.com")
    tok = anon_client.post("/signin", json={"email": "viewer@example.com", "password": "password123"}).json()["token"]
    assert anon_client.get("/formulary/overview", headers=bearer(tok)).status_code == 200
    assert anon_client.get("/formulary/validation", headers=bearer(tok)).status_code == 403


def test_every_formulary_route_needs_a_login(anon_client, data):
    for path in ("/formulary/meta", "/formulary/overview", "/formulary/doctors", "/formulary/items",
                 "/formulary/scheme/items", "/formulary/validation"):
        assert anon_client.get(path).status_code == 401, path


def test_missing_tables_are_a_clear_503(client, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "FORMULARY_DATA_DIR", str(tmp_path / "empty"))
    formulary.clear_caches()
    for path in ("/formulary/meta", "/formulary/overview", "/formulary/doctors", "/formulary/items",
                 "/formulary/scheme/items", "/formulary/validation"):
        r = client.get(path)
        assert r.status_code == 503, (path, r.text)
        assert "python -m app.etl.formulary" in r.json()["detail"]
    plants = {p["code"]: p["domains"] for p in client.get("/meta/plants").json()["plants"]}
    assert not any("formulary" in d for c, d in plants.items() if c != "ALL")


def test_a_rebuild_is_served_without_a_restart(client, data):
    assert overview(client, Plant="HC05", Month=AUG)["headline"]["formulary"] == 1250.0
    folder = data / "formulary"
    s = pd.read_parquet(folder / "formulary_summary.parquet")
    s.loc[(s["plant"] == "HC05") & (s["period"] == AUG) & (s["formulary_class"] == "FORMULARYR"), "cost"] = 1800.0
    s.to_parquet(folder / "formulary_summary.parquet", index=False)
    assert overview(client, Plant="HC05", Month=AUG)["headline"]["formulary"] == 2250.0


def test_refresh_data_clears_the_formulary_caches(client, data, monkeypatch):
    cleared = []
    monkeypatch.setattr(formulary, "clear_caches", lambda: cleared.append(True))
    r = client.post("/admin/refresh-data", headers={"x-admin-token": settings.ADMIN_REFRESH_TOKEN})
    assert r.status_code == 200 and cleared == [True]


def test_no_patient_or_invoice_fields_in_any_response(client, data):
    bodies = [client.get(p, params=q).json() for p, q in (
        ("/formulary/meta", {}), ("/formulary/overview", {"Month": "ALL"}),
        ("/formulary/doctors", {"Month": "ALL"}), ("/formulary/doctors", {"Material": "100002"}),
        ("/formulary/items", {"Group": "A"}), ("/formulary/items", {"Doctor": "D1"}),
        ("/formulary/scheme/items", {}))]
    text = json.dumps(bodies).upper()
    for banned in ("UHID", "PATNAME", "IP_OPNUMBER", "SALESINVOICENO", "BATCHCODE", "SALES_DATE"):
        assert banned not in text
