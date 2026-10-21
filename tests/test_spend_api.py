# tests/test_spend_api.py — the /spend/* endpoints, on a tiny hand-made dataset.
#
# The published tables are written here from 8 invented scheme switch rows, 5 Attis
# consumption rows and 5 cash & TPA rows (with their bifurcation allowance). Every expected number below was worked out by hand from them (comments
# show the arithmetic). tests/test_spend_golden.py checks HCG's real KR figures once the
# built tables are present.
from __future__ import annotations

import json

import pandas as pd
import pytest

from app.api import spend
from app.core.config import settings

from tests.conftest import bearer, signup_and_approve
from tests.test_formulary_api import write_hospital_map

AUG, JUL = "2026-08", "2026-07"
KR, DR = "HC05", "HC06"
HIS = {KR: ("KABHK", "HCG KR, Bangalore"), DR: ("KABHD", "HCG DR, Bangalore")}
DOCTORS = {"D1": "Dr Asha", "D2": "Dr Bhanu", "D9": "Dr Zoya"}
# used material -> (name, category, suggested, suggested name, used price, suggested price, attis)
BRANDS = {
    "101001": ("ONCO X 100MG", "Onco Drugs", "101901", "AT-X 100MG", 1000.0, 400.0, True),
    "201002": ("PARA BRAND", "Other Drugs", "201902", "ATPARA", 50.0, 30.0, True),
    "300001": ("SUTURE J&J", "Consumables", "300901", "SUTURE HEALTHIUM", 7.0, 5.0, False),
    "101005": ("ONCO Y", "Onco Drugs", "101905", "OTHER Y", 200.0, 150.0, False),
}
# period, plant, type, doctor, material, qty
SWITCHES = [
    (AUG, KR, "IP", "D1", "101001", 2),     # 1  2000 / 800  -> 1200   Attis
    (AUG, KR, "OP", "D2", "101001", 1),     # 2  1000 / 400  -> 600    Attis
    (AUG, KR, "IP", "D2", "201002", 10),    # 3  500 / 300   -> 200    Attis
    (AUG, KR, "IP", "D1", "300001", 4),     # 4  28 / 20     -> 8      other brand
    (AUG, KR, "IP", "D1", "101005", 1),     # 5  200 / 150   -> 50     other brand
    (AUG, DR, "IP", "D9", "101001", 1),     # 6  1000 / 400  -> 600    Attis
    (JUL, KR, "IP", "D1", "101001", 3),     # 7  3000 / 1200 -> 1800   Attis
    (JUL, DR, "OP", "D9", "201002", 4),     # 8  200 / 120   -> 80     Attis
]
# period, plant, type, category, payer, material, name, cost, qty
CONSUMPTION = [
    (AUG, KR, "IP", "Onco Drugs", "GOV", "101901", "AT-X 100MG", 700.0, 2),
    (AUG, KR, "OP", "Onco Drugs", "CASH", "101901", "AT-X 100MG", 350.0, 1),
    (AUG, KR, "IP", "Other Drugs", "CASH(TPA)", "201902", "ATPARA", 60.0, 2),
    (AUG, DR, "IP", "Onco Drugs", "GOV", "101901", "AT-X 100MG", 350.0, 1),
    (JUL, KR, "IP", "Onco Drugs", "GOV", "101901", "AT-X 100MG", 400.0, 1),
]
SCHEME_GROUP = {"300001": "SUTURES"}                    # J&J -> Healthium line

# Cash & TPA list: used -> (name, category, suggested, its name, price, MRP, GST, s.price, s.MRP, s.GST)
# Unit margin = MRP / 1.05 - price: 101001 1200 - 1000 = 200 vs 1500 - 900 = 600 (gap 400);
# 201002 100 - 50 = 50 vs 150 - 30 = 120 (gap 70).
CASH_BRANDS = {
    "101001": ("ONCO X 100MG", "Onco Drugs", "101950", "ONCO X PRIME", 1000.0, 1260.0, 5.0, 900.0, 1575.0, 5.0),
    "201002": ("PARA BRAND", "Other Drugs", "201902", "PARA PRIME", 50.0, 105.0, 5.0, 30.0, 157.5, 5.0),
}
# period, plant, type, payer, doctor, material, cash/TPA qty, counted qty (after the allowance)
CASH_ROWS = [
    (AUG, KR, "IP", "CASH", "D1", "101001", 5, 4.3125),        # c1 bifurcated: 8 used, 1.1 allowed
    (AUG, KR, "OP", "CASH(TPA)", "D2", "101001", 3, 2.5875),   # c2   -> 6.9 counted, shared 5 : 3
    (AUG, KR, "IP", "CASH", "D2", "201002", 6, 6.0),           # c3
    (AUG, DR, "IP", "CASH", "D9", "101001", 2, 1.7),           # c4 bifurcated: 2 used, 0.3 allowed
    (JUL, KR, "IP", "CASH(TPA)", "D1", "201002", 4, 4.0),      # c5
]
ALLOWANCE = [(AUG, KR, "101001", 11.0, 1.1, 8.0, 6.9), (AUG, DR, "101001", 3.0, 0.3, 2.0, 1.7)]


def write_tables(folder) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for period, plant, typ, doc, mat, qty in SWITCHES:
        name, cat, sug, sug_name, up, sp, attis = BRANDS[mat]
        rows.append({"period": period, "plant": plant, "hospital": HIS[plant][0],
                     "hospital_name": HIS[plant][1], "patient_type": typ, "category": cat,
                     "doctor_id": doc, "doctor_name": DOCTORS[doc], "material": mat,
                     "material_desc": name, "hcg_category": "R", "scheme_group": SCHEME_GROUP.get(mat),
                     "suggested_material": sug,
                     "suggested_desc": sug_name, "attis": attis, "used_price": up,
                     "suggested_price": sp, "qty": float(qty), "actual_cost": up * qty * 0.9,
                     "lines": 1, "used_value": up * qty, "suggested_value": sp * qty})
    pd.DataFrame(rows).to_parquet(folder / "scheme_switch.parquet", index=False)
    pd.DataFrame([{"period": p, "plant": pl, "hospital": HIS[pl][0], "hospital_name": HIS[pl][1],
                   "patient_type": t, "category": c, "payer_bucket": pay, "material": m,
                   "material_desc": n, "generic_name": f"GEN {m}", "cost": cost, "mrp": cost * 1.5,
                   "qty": float(q), "lines": 1}
                  for p, pl, t, c, pay, m, n, cost, q in CONSUMPTION]
                 ).to_parquet(folder / "attis_consumption.parquet", index=False)
    crows = []
    for period, plant, typ, payer, doc, mat, cq, q in CASH_ROWS:
        name, cat, sug, sug_name, up, um, ug, sp, sm, sg = CASH_BRANDS[mat]
        used_mrp_ex, sug_mrp_ex = um / (1 + ug / 100), sm / (1 + sg / 100)
        crows.append({"period": period, "plant": plant, "hospital": HIS[plant][0], "hospital_name": HIS[plant][1],
                      "patient_type": typ, "category": cat, "payer_bucket": payer, "doctor_id": doc,
                      "doctor_name": DOCTORS[doc], "material": mat, "material_desc": name, "hcg_category": "R",
                      "suggested_material": sug, "suggested_desc": sug_name, "used_price": up, "used_mrp": um,
                      "used_gst": ug, "suggested_price": sp, "suggested_mrp": sm, "suggested_gst": sg,
                      "cash_qty": float(cq), "actual_cost": up * cq, "lines": 1, "bifurcated": q != cq,
                      "qty": q, "used_cost_value": q * up, "used_mrp_value": q * used_mrp_ex,
                      "suggested_cost_value": q * sp, "suggested_mrp_value": q * sug_mrp_ex,
                      "actual_margin": q * (used_mrp_ex - up), "potential_margin": q * (sug_mrp_ex - sp),
                      "margin_loss": q * ((sug_mrp_ex - sp) - (used_mrp_ex - up))})
    pd.DataFrame(crows).to_parquet(folder / "cash_switch.parquet", index=False)
    pd.DataFrame([{"period": p, "plant": pl, "hospital": HIS[pl][0], "hospital_name": HIS[pl][1], "material": m,
                   "material_desc": CASH_BRANDS[m][0], "flag": "N", "sharing": 0.1, "group_qty": g,
                   "allowed_qty": a, "cash_qty": c, "excess_qty": e} for p, pl, m, g, a, c, e in ALLOWANCE]
                 ).to_parquet(folder / "cash_allowance.parquet", index=False)
    (folder / "spend_validation.json").write_text(json.dumps({
        "built_at": "2026-10-10T12:00:00+00:00",
        "scheme_list": {"file": "Spend Analysis Main Sheet.xlsx", "financial_year": "2026-27",
                        "switches": 1076, "attis_switches": 26, "attis_brands": 18},
        "cash_list": {"file": "Spend Analysis Main Sheet.xlsx", "financial_year": "2026-27", "switches": 562,
                      "bifurcation": {"brands": 41, "source": "seeded from HCG KR Aug-26"}},
        "months": {AUG: {"attis": {"saving_missed": 2600.0}}, JUL: {}}}))


@pytest.fixture()
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SPEND_DATA_DIR", str(tmp_path / "spend"))
    monkeypatch.setattr(settings, "FORMULARY_DATA_DIR", str(tmp_path / "formulary-not-built"))
    monkeypatch.setattr(settings, "REFERENCE_DIR", str(tmp_path / "reference"))
    write_tables(tmp_path / "spend")
    write_hospital_map(tmp_path / "reference")
    spend.clear_caches()
    yield tmp_path
    spend.clear_caches()


def get(client, path, **params):
    r = client.get(path, params=params)
    assert r.status_code == 200, r.text
    return r.json()


def overview(client, **params):
    return get(client, "/spend/attis/overview", **params)


def money(b) -> tuple:
    return b["used_value"], b["suggested_value"], b["saving_missed"]


# ---------- headline arithmetic ----------

def test_headline_counts_attis_suggestions_only(client, data):
    h = overview(client, Plant="HC05", Month=AUG)["headline"]
    # rows 1-3: 2000 + 1000 + 500 = 3500 used, 800 + 400 + 300 = 1500 with Attis
    assert money(h) == (3500.0, 1500.0, 2000.0)
    assert h["saving_pct"] == pytest.approx(57.1429, abs=1e-4)          # 2000 / 3500
    assert (h["qty"], h["lines"], h["brands"], h["doctors"], h["hospitals"]) == (13.0, 3, 2, 2, 1)
    assert h["bill_cost"] == 3150.0                                     # 0.9 x 3500
    # whole scheme list, same scope: + rows 4, 5 -> saving 2000 + 8 + 50
    assert h["scheme_saving_missed"] == 2058.0
    assert h["share_of_scheme_pct"] == pytest.approx(97.1817, abs=1e-4)


def test_all_hospitals_and_several_months_pool_the_amounts(client, data):
    assert money(overview(client, Month=AUG)["headline"]) == (4500.0, 1900.0, 2600.0)
    h = overview(client, Plant="HC05", Month="ALL")["headline"]
    assert money(h) == (6500.0, 2700.0, 3800.0)                         # + Jul row 7
    assert h["saving_pct"] == pytest.approx(58.4615, abs=1e-4)          # 3800 / 6500
    d = overview(client, Plant="HC05", Month=f"{JUL},{AUG}")
    assert d["headline"]["saving_pct"] == h["saving_pct"]
    assert any("never an average" in n for n in d["notes"])


def test_month_defaults_to_the_latest(client, data):
    d = overview(client, Plant="HC05")
    assert d["scope"]["months"] == [AUG] and d["headline"]["saving_missed"] == 2000.0


def test_filters_cut_the_headline(client, data):
    assert money(overview(client, Plant="HC05", Month=AUG, Type="OP")["headline"]) == (1000.0, 400.0, 600.0)
    assert money(overview(client, Plant="HC05", Month=AUG, Segment="non-onco")["headline"]) == (500.0, 300.0, 200.0)
    assert money(overview(client, Plant="HC05", Month=AUG, Category="Onco Drugs")["headline"]) == \
        (3000.0, 1200.0, 1800.0)


def test_nothing_in_scope_is_zero_with_a_null_percentage(client, data):
    d = overview(client, Plant="HC05", Month=AUG, Category="Consumables")
    h = d["headline"]
    assert money(h) == (0.0, 0.0, 0.0) and h["saving_pct"] is None
    assert h["scheme_saving_missed"] == 8.0 and h["share_of_scheme_pct"] == 0.0     # row 4
    assert any("No scheme patient" in n for n in d["notes"])


# ---------- breakdowns ----------

def test_trend_shows_every_month_whatever_the_month_filter(client, data):
    t = overview(client, Plant="HC05", Month=AUG)["trend"]
    assert [x["label"] for x in t] == ["Jul-26", "Aug-26"]
    assert money(t[0]) == (3000.0, 1200.0, 1800.0) and money(t[1]) == (3500.0, 1500.0, 2000.0)


def test_hospital_ranking_is_most_missed_first_and_ignores_the_plant(client, data):
    hb = overview(client, Plant="HC06", Month=AUG)["by_hospital"]
    assert [(x["plant"], x["saving_missed"], x["rank"], x["selected"]) for x in hb] == \
        [("HC05", 2000.0, 1, False), ("HC06", 600.0, 2, True)]
    assert hb[0]["his_name"] == "HCG KR, Bangalore"


def test_network_is_the_all_hospitals_figure_of_the_same_filters(client, data):
    # Attis, Aug, every hospital: KR used 2000 + 1000 + 500, saving 1200 + 600 + 200; DR 1000, 600
    assert overview(client, Plant="HC05", Month=AUG)["network"] == {
        "saving_pct": pytest.approx(2600 / 4500 * 100, abs=1e-4)}
    for path, key in (("/spend/attis/overview", "saving_pct"), ("/spend/scheme/overview", "saving_pct"),
                      ("/spend/cash/overview", "loss_pct")):
        for q in ({"Month": AUG}, {"Month": "ALL", "Type": "IP"}):
            whole = get(client, path, **q)["headline"][key]
            for plant in (KR, DR):
                assert get(client, path, Plant=plant, **q)["network"] == {key: whole}, (path, plant, q)


def test_category_and_type_always_carry_every_row(client, data):
    d = overview(client, Plant="HC05", Month=AUG, Type="IP")
    cats = {c["key"]: money(c) for c in d["by_category"]}
    # by_category ignores the Category filter, keeps Type=IP: rows 1 and 3
    assert cats == {"onco": (2000.0, 800.0, 1200.0), "non-onco": (500.0, 300.0, 200.0),
                    "consumables": (0.0, 0.0, 0.0)}
    types = {t["type"]: money(t) for t in d["by_type"]}
    assert types == {"IP": (2500.0, 1100.0, 1400.0), "OP": (1000.0, 400.0, 600.0)}


def test_top_items(client, data):
    top = overview(client, Plant="HC05", Month=AUG)["top_items"]
    assert [(t["material"], t["suggested_desc"], t["saving_missed"]) for t in top] == \
        [("101001", "AT-X 100MG", 1800.0), ("201002", "ATPARA", 200.0)]


def test_consumption_summary_every_payer(client, data):
    c = overview(client, Plant="HC05", Month=AUG)["consumption"]
    assert (c["cost"], c["products"], c["lines"]) == (1110.0, 2, 3)
    assert {p["payer"]: p["cost"] for p in c["by_payer"]} == {"CASH": 350.0, "CASH(TPA)": 60.0, "GOV": 700.0}
    assert {t["type"]: t["cost"] for t in c["by_type"]} == {"IP": 760.0, "OP": 350.0}
    assert {x["key"]: x["cost"] for x in c["by_category"]} == {"onco": 1050.0, "non-onco": 60.0,
                                                                "consumables": 0.0}


# ---------- tables ----------

def test_items_most_missed_first(client, data):
    d = get(client, "/spend/attis/items", Plant="HC05", Month=AUG)
    assert d["total"] == 2
    a, b = d["data"]
    assert (a["material"], a["suggested_material"], a["qty"], a["used_price"], a["suggested_price"]) == \
        ("101001", "101901", 3.0, 1000.0, 400.0)
    assert (a["used_value"], a["suggested_value"], a["saving_missed"], a["saving_pct"]) == (3000.0, 1200.0, 1800.0, 60.0)
    assert (a["share_pct"], b["share_pct"]) == (90.0, 10.0)               # of 2000
    assert (a["doctors"], a["hospitals"], b["material"]) == (2, 1, "201002")


def test_one_doctors_items(client, data):
    d = get(client, "/spend/attis/items", Plant="HC05", Month=AUG, Doctor="D2")
    assert [(r["material"], r["saving_missed"]) for r in d["data"]] == [("101001", 600.0), ("201002", 200.0)]


def test_doctors_most_missed_first(client, data):
    d = get(client, "/spend/attis/doctors", Plant="HC05", Month=AUG)
    assert [(r["doctor_id"], r["doctor_name"], r["saving_missed"], r["brands"]) for r in d["data"]] == \
        [("D1", "Dr Asha", 1200.0, 1), ("D2", "Dr Bhanu", 800.0, 2)]
    assert d["data"][0]["share_pct"] == 60.0


def test_doctors_behind_one_brand(client, data):
    d = get(client, "/spend/attis/doctors", Month=AUG, Material="101001.0")
    assert [(r["doctor_id"], r["plant"], r["saving_missed"]) for r in d["data"]] == \
        [("D1", "HC05", 1200.0), ("D2", "HC05", 600.0), ("D9", "HC06", 600.0)]


def test_tables_page_sort_and_search(client, data):
    d = get(client, "/spend/attis/doctors", Month=AUG, page=0, page_size=1, sort_field="doctor_name",
            sort_order="asc")
    assert d["total"] == 3 and d["data"][0]["doctor_name"] == "Dr Asha"
    d = get(client, "/spend/attis/items", Plant="HC05", Month=AUG, global_filter="para")
    assert [r["material"] for r in d["data"]] == ["201002"]


def test_consumption_table_and_payer_filter(client, data):
    d = get(client, "/spend/attis/consumption", Plant="HC05", Month=AUG)
    a = d["data"][0]
    assert (a["material"], a["cost"], a["gov_cost"], a["ip_cost"], a["op_cost"]) == ("101901", 1050.0, 700.0, 700.0, 350.0)
    assert a["share_pct"] == pytest.approx(94.5946, abs=1e-4)              # 1050 / 1110
    d = get(client, "/spend/attis/consumption", Plant="HC05", Month=AUG, Payer="GOV")
    assert [(r["material"], r["cost"]) for r in d["data"]] == [("101901", 700.0)]


# ---------- arguments ----------

def test_plant_accepts_sap_code_his_code_and_name(client, data):
    for p in ("HC05", "KABHK", "HCG KR, Bangalore"):
        assert overview(client, Plant=p, Month=AUG)["scope"]["plant"] == "HC05", p


def test_bad_parameters_are_a_400_and_unknown_plant_a_404(client, data):
    for params in ({"Month": "2026-13"}, {"Month": "2025-01"}, {"Type": "XX"}, {"Category": "Lab"},
                   {"Segment": "onco", "Category": "Other Drugs"}, {"Segment": "bones"}):
        r = client.get("/spend/attis/overview", params=params)
        assert r.status_code == 400, (params, r.text)
    assert client.get("/spend/attis/consumption", params={"Payer": "XYZ"}).status_code == 400
    for path in ("/spend/attis/overview", "/spend/attis/items", "/spend/attis/doctors",
                 "/spend/attis/consumption"):
        assert client.get(path, params={"Plant": "NOWHERE"}).status_code == 404, path


# ---------- meta, validation, auth, missing data ----------

def test_meta(client, data):
    m = get(client, "/spend/meta")
    assert m["months"] == [JUL, AUG] and m["latest"] == AUG
    assert m["month_labels"][AUG] == "Aug-26"
    assert [h["plant"] for h in m["hospitals"]] == ["HC06", "HC05"]          # by HIS name
    assert [c["key"] for c in m["categories"]] == ["onco", "non-onco", "consumables"]
    assert m["scheme_list"]["financial_year"] == "2026-27" and m["scheme_list"]["attis_brands"] == 18
    assert any("2026-27 price list" in n for n in m["notes"])


def test_validation_is_admin_only(client, data, anon_client, admin_headers):
    assert get(client, "/spend/validation")["scheme_list"]["switches"] == 1076
    signup_and_approve(anon_client, admin_headers, email="viewer@example.com")
    tok = anon_client.post("/signin", json={"email": "viewer@example.com", "password": "password123"}).json()["token"]
    assert anon_client.get("/spend/attis/overview", headers=bearer(tok)).status_code == 200
    assert anon_client.get("/spend/validation", headers=bearer(tok)).status_code == 403


def test_every_spend_route_needs_a_login(anon_client, data):
    for path in ("/spend/meta", "/spend/attis/overview", "/spend/attis/items", "/spend/attis/doctors",
                 "/spend/attis/consumption", "/spend/scheme/overview", "/spend/scheme/items",
                 "/spend/scheme/doctors", "/spend/cash/overview", "/spend/cash/items", "/spend/cash/doctors",
                 "/spend/brands", "/spend/validation"):
        assert anon_client.get(path).status_code == 401, path


# ---------- scheme: every suggestion (Task 3b) ----------

def test_scheme_overview_counts_every_suggestion(client, data):
    d = get(client, "/spend/scheme/overview", Plant="HC05", Month=AUG)
    h = d["headline"]
    # rows 1-5: 2000 + 1000 + 500 + 28 + 200 = 3728 used, 800 + 400 + 300 + 20 + 150 = 1670
    assert money(h) == (3728.0, 1670.0, 2058.0)
    assert h["saving_pct"] == pytest.approx(55.2039, abs=1e-4)
    g = d["groups"]
    assert money(g["attis"]) == (3500.0, 1500.0, 2000.0) and g["attis"]["share_pct"] == pytest.approx(97.1817, abs=1e-4)
    assert money(g["healthium"]) == (28.0, 20.0, 8.0) and g["healthium"]["label"].startswith("J&J")
    assert "consumption" not in d
    assert any("SUTURES and Endo" in n for n in d["notes"])


def test_scheme_tables_and_groups(client, data):
    rows = get(client, "/spend/scheme/items", Plant="HC05", Month=AUG)["data"]
    assert [r["material"] for r in rows] == ["101001", "201002", "101005", "300001"]   # 1800, 200, 50, 8
    hl = get(client, "/spend/scheme/items", Plant="HC05", Month=AUG, Group="healthium")["data"]
    assert [(r["material"], r["saving_missed"]) for r in hl] == [("300001", 8.0)]
    at = get(client, "/spend/scheme/items", Plant="HC05", Month=AUG, Group="Attis")["data"]
    assert [r["material"] for r in at] == ["101001", "201002"]
    docs = get(client, "/spend/scheme/doctors", Plant="HC05", Month=AUG)["data"]
    assert [(r["doctor_id"], r["saving_missed"]) for r in docs] == [("D1", 1258.0), ("D2", 800.0)]
    assert client.get("/spend/scheme/items", params={"Group": "bones"}).status_code == 400


# ---------- cash & TPA margin loss (Task 3a) ----------

def cash(client, **params):
    return get(client, "/spend/cash/overview", **params)


def margins(b) -> tuple:
    return b["actual_margin"], b["potential_margin"], b["margin_loss"]


def test_cash_headline_by_hand(client, data):
    h = cash(client, Plant="HC05", Month=AUG)["headline"]
    # 101001 counted 6.9: 1380 / 4140 / 2760; 201002 6: 300 / 720 / 420
    assert margins(h) == pytest.approx((1680.0, 4860.0, 3180.0))
    assert h["loss_pct"] == pytest.approx(65.4321, abs=1e-4)                  # loss / potential
    assert (h["used_mrp_value"], h["suggested_mrp_value"]) == pytest.approx((8880.0, 11250.0))
    assert h["topline_increase"] == pytest.approx(2370.0)
    assert (h["cash_qty"], h["qty"], h["allowance_qty"]) == pytest.approx((14.0, 12.9, 1.1))
    assert (h["brands"], h["doctors"], h["hospitals"]) == (2, 2, 1)


def test_cash_payer_filter_and_split(client, data):
    tpa = cash(client, Plant="HC05", Month=AUG, Payer="CASH(TPA)")
    assert margins(tpa["headline"]) == pytest.approx((517.5, 1552.5, 1035.0))   # c2: 2.5875 units
    assert tpa["scope"]["payer"] == "CASH(TPA)"
    bp = {p["payer"]: margins(p) for p in tpa["by_payer"]}                     # ignores the filter
    assert bp["CASH"] == pytest.approx((1162.5, 3307.5, 2145.0)) and bp["CASH(TPA)"] == pytest.approx((517.5, 1552.5, 1035.0))
    assert client.get("/spend/cash/overview", params={"Payer": "GOV"}).status_code == 400


def test_cash_trend_ranking_splits_and_allowance(client, data):
    d = cash(client, Plant="HC05", Month=AUG)
    assert [(t["label"], t["margin_loss"]) for t in d["trend"]] == [("Jul-26", 280.0), ("Aug-26", 3180.0)]
    assert [(x["plant"], x["margin_loss"], x["rank"]) for x in d["by_hospital"]] == [("HC05", 3180.0, 1), ("HC06", 680.0, 2)]
    assert {c["key"]: c["margin_loss"] for c in d["by_category"]} == {"onco": 2760.0, "non-onco": 420.0, "consumables": 0.0}
    assert {t["type"]: t["margin_loss"] for t in d["by_type"]} == {"IP": 2145.0, "OP": 1035.0}
    assert d["allowance"] == {"brands": 1, "cash_qty": 8.0, "allowed_qty": 1.1, "counted_qty": 6.9}
    assert [(t["material"], t["margin_loss"], t["bifurcated"]) for t in d["top_items"]] == \
        [("101001", 2760.0, True), ("201002", 420.0, False)]
    assert d["cash_list"]["bifurcated_brands"] == 41


def test_cash_items(client, data):
    rows = get(client, "/spend/cash/items", Plant="HC05", Month=AUG)["data"]
    a, b = rows
    assert (a["material"], a["cash_qty"], a["qty"], a["bifurcated"]) == ("101001", 8.0, 6.9, True)
    assert (a["used_margin"], a["suggested_margin"], a["margin_gap"]) == (200.0, 600.0, 400.0)
    assert (a["used_mrp"], a["suggested_price"]) == (1260.0, 900.0)
    assert margins(a) == pytest.approx((1380.0, 4140.0, 2760.0))
    assert a["share_pct"] == pytest.approx(86.7925, abs=1e-4)                  # 2760 / 3180
    assert (b["material"], b["bifurcated"], b["margin_loss"]) == ("201002", False, 420.0)
    d2 = get(client, "/spend/cash/items", Plant="HC05", Month=AUG, Doctor="D2")["data"]
    assert [(r["material"], r["margin_loss"]) for r in d2] == [("101001", 1035.0), ("201002", 420.0)]


def test_cash_doctors(client, data):
    rows = get(client, "/spend/cash/doctors", Plant="HC05", Month=AUG)["data"]
    assert [(r["doctor_id"], r["doctor_name"], r["margin_loss"], r["brands"]) for r in rows] == \
        [("D1", "Dr Asha", 1725.0, 1), ("D2", "Dr Bhanu", 1455.0, 2)]
    one = get(client, "/spend/cash/doctors", Month=AUG, Material="101001")["data"]
    assert [(r["doctor_id"], r["plant"], r["margin_loss"]) for r in one] == \
        [("D1", "HC05", 1725.0), ("D2", "HC05", 1035.0), ("D9", "HC06", 680.0)]


def test_cash_views_answer_503_on_tables_built_before_task_3(client, data):
    (data / "spend" / "cash_switch.parquet").unlink()
    (data / "spend" / "cash_allowance.parquet").unlink()
    spend.clear_caches()
    assert client.get("/spend/cash/overview").status_code == 503
    assert overview(client, Plant="HC05", Month=AUG)["headline"]["saving_missed"] == 2000.0   # Attis still served


def test_missing_tables_are_a_clear_503(client, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SPEND_DATA_DIR", str(tmp_path / "nothing"))
    spend.clear_caches()
    for path in ("/spend/meta", "/spend/attis/overview", "/spend/attis/items", "/spend/attis/doctors",
                 "/spend/attis/consumption", "/spend/validation"):
        r = client.get(path)
        assert r.status_code == 503, (path, r.text)
        assert "python -m app.etl.spend" in r.json()["detail"]


def test_a_rebuild_is_served_without_a_restart(client, data):
    assert overview(client, Plant="HC05", Month=AUG)["headline"]["saving_missed"] == 2000.0
    p = data / "spend" / "scheme_switch.parquet"
    df = pd.read_parquet(p)
    df.loc[0, "suggested_value"] = 0.0                    # row 1: saving 2000 instead of 1200
    df.to_parquet(p, index=False)
    assert overview(client, Plant="HC05", Month=AUG)["headline"]["saving_missed"] == 2800.0


def test_no_patient_or_invoice_fields_in_any_response(client, data):
    bad = {"uhid", "patname", "ip_opnumber", "salesinvoiceno", "payer1", "sales_date"}
    for path in ("/spend/meta", "/spend/attis/overview", "/spend/attis/items", "/spend/attis/doctors",
                 "/spend/attis/consumption", "/spend/scheme/overview", "/spend/cash/overview",
                 "/spend/cash/items", "/spend/cash/doctors"):
        text = json.dumps(get(client, path)).lower()
        assert not any(f'"{b}"' in text for b in bad), path


# ---------- the Brand filter (Material= on every view, /spend/brands) ----------

def test_brand_filter_narrows_every_view_to_one_brand(client, data):
    # scheme, KR, Aug: 101001 is rows 1 + 2 -> 3000 / 1200 -> 1800, 60% of its spend
    d = get(client, "/spend/scheme/overview", Plant="HC05", Month=AUG, Material="101001.0")
    assert d["scope"]["material"] == "101001"
    assert (d["headline"]["saving_missed"], d["headline"]["saving_pct"], d["headline"]["brands"]) == (1800.0, 60.0, 1)
    # the hospital ranking ignores the hospital, never the brand: KR 1800, DR row 6 600
    assert [(h["plant"], h["saving_missed"]) for h in d["by_hospital"]] == [("HC05", 1800.0), ("HC06", 600.0)]
    assert [(t["period"], t["saving_missed"]) for t in d["trend"]] == [(JUL, 1800.0), (AUG, 1800.0)]
    items = get(client, "/spend/scheme/items", Plant="HC05", Month=AUG, Material="201002")["data"]
    assert [(r["material"], r["saving_missed"]) for r in items] == [("201002", 200.0)]
    # cash, KR, Aug: 201002 is c3 only -> 6 x (120 - 50) = 420
    c = get(client, "/spend/cash/overview", Plant="HC05", Month=AUG, Material="201002")
    assert (c["headline"]["margin_loss"], c["headline"]["brands"]) == (420.0, 1)
    assert [r["material"] for r in get(client, "/spend/cash/items", Month=AUG, Material="201002")["data"]] == ["201002"]
    # a brand with no use in the selection is an empty view, not an error
    assert get(client, "/spend/scheme/overview", Plant="HC06", Month=AUG, Material="300001")["headline"]["lines"] == 0


def test_brand_filter_on_doctors_is_one_chip(client, data):
    # the scope chips come with the facets (facets=1)
    d = get(client, "/spend/scheme/doctors", Month=AUG, Material="101001", facets=1)
    assert [(r["doctor_id"], r["saving_missed"]) for r in d["data"]] == [("D1", 1200.0), ("D2", 600.0), ("D9", 600.0)]
    chips = [c["param"] for c in d["scope"]]
    assert chips.count("Material") == 1


def test_brands_lists_each_sections_switch_biggest_first(client, data):
    d = get(client, "/spend/brands", Plant="HC05", Month=AUG)
    # biggest single-section gain first: 101001 cash 2760 (6.9 counted x 400), 201002 cash 420,
    # 101005 scheme 50, 300001 scheme 8; sections are never added
    assert [b["material"] for b in d["brands"]] == ["101001", "201002", "101005", "300001"]
    assert d["total"] == 4
    top = d["brands"][0]
    assert top["material_desc"] == "ONCO X 100MG"
    assert (top["scheme"]["missed"], top["scheme"]["pct"], top["scheme"]["suggested_material"]) == (1800.0, 60.0, "101901")
    assert (top["cash"]["missed"], top["cash"]["suggested_material"], top["cash"]["suggested_desc"]) == (2760.0, "101950", "ONCO X PRIME")
    assert top["oof"] is None                               # no out-of-formulary table built here
    assert d["brands"][2]["cash"] is None                   # 101005 is on the scheme list only


def test_brands_search_one_brand_and_limit(client, data):
    assert [b["material"] for b in get(client, "/spend/brands", Plant="HC05", Month=AUG, q="para")["brands"]] == ["201002"]
    assert [b["material"] for b in get(client, "/spend/brands", Plant="HC05", Month=AUG, q="1010")["brands"]] == ["101001", "101005"]
    one = get(client, "/spend/brands", Plant="HC05", Month=AUG, Material="201002.0")
    assert (one["total"], one["brands"][0]["material"]) == (1, "201002")
    assert len(get(client, "/spend/brands", Month=AUG, limit=2)["brands"]) == 2
    # the payer narrows the cash side only: c3 is CASH, so CASH(TPA) leaves 201002 its scheme gain
    tpa = get(client, "/spend/brands", Plant="HC05", Month=AUG, Payer="CASH(TPA)", Material="201002")["brands"][0]
    assert (tpa["cash"], tpa["scheme"]["missed"]) == (None, 200.0)



def test_brands_for_one_section_are_its_own_ranked_by_its_gain(client, data):
    # scheme, KR, Aug: 101001 1800, 201002 200, 101005 50, 300001 8 -- every scheme brand, and only those
    sch = get(client, "/spend/brands", Plant="HC05", Month=AUG, Section="scheme")
    assert [(b["material"], b["scheme"]["missed"]) for b in sch["brands"]] == \
        [("101001", 1800.0), ("201002", 200.0), ("101005", 50.0), ("300001", 8.0)]
    # cash: only the two cash-list brands, by their cash gain; 101005 (scheme only) is not offered
    cash = get(client, "/spend/brands", Plant="HC05", Month=AUG, Section="cash")
    assert [(b["material"], b["cash"]["missed"]) for b in cash["brands"]] == [("101001", 2760.0), ("201002", 420.0)]
    assert cash["total"] == 2
    # no out-of-formulary table here: that section offers nothing
    assert get(client, "/spend/brands", Plant="HC05", Month=AUG, Section="oof")["brands"] == []
    assert client.get("/spend/brands", params={"Section": "attis"}).status_code == 400
