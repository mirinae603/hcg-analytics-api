# tests/test_consumables_departments.py — Consumables · department consumption.
#
# The Consumption page's department cards read kpi_consumption_by_department, which has no
# material column and so cannot be cut to consumables. /portfolio/consumption/consumables/*
# reads fact_consumption itself and keeps HCG's 3-series (material code starting with 3).
# Every test points settings.CURATED_DIR at a tmp dir holding a few hand-written rows, so the
# expected figures are worked out here, by hand, from those rows.
from __future__ import annotations

import os

import pandas as pd
import pytest

from app.core import drill_sources as ds
from app.core.config import settings

BASE = "/portfolio/consumption/consumables"
FACT_COLS = ["material", "material_desc", "plant", "sloc", "mvt", "cost_ctr", "qty", "amount_lc",
             "uom", "posting_date", "year", "month_num", "month"]
MONTHS = {11: "November", 12: "December", 1: "January"}
PATIENT_WORDS = ("patient", "uhid", "invoice", "bill", "ipno", "opno", "mrno")


@pytest.fixture(scope="module")
def plants():
    # Two real plant codes from dim_plant, so Plant= resolves by code AND by hospital name.
    names = ds.plant_name_map()
    a, b = sorted(names)[:2]
    return {"A": a, "B": b, "name_A": names[a]}


def _row(material, plant, cc, qty, cost, year, month, desc=None):
    return {"material": material, "material_desc": desc or f"Item {material}", "plant": plant,
            "sloc": "S001", "mvt": 201.0, "cost_ctr": cc, "qty": float(qty), "amount_lc": float(cost),
            "uom": "EA", "posting_date": pd.Timestamp(year, month, 15), "year": year,
            "month_num": month, "month": MONTHS[month]}


def _write(dirpath, rows, names=None):
    df = pd.DataFrame(rows, columns=FACT_COLS)
    df["year"] = df["year"].astype("Int64")
    df["month_num"] = df["month_num"].astype("Int64")
    df.to_parquet(dirpath / "fact_consumption.parquet", index=False)
    if names is not None:
        pd.DataFrame({"cost_ctr": list(names), "department_name": list(names.values())}).to_parquet(
            dirpath / "dim_costcenter.parquet", index=False)


@pytest.fixture()
def data(tmp_path, monkeypatch, plants):
    A, B = plants["A"], plants["B"]
    rows = [
        # plant A, cost centre CC1: two consumables
        _row("300001", A, "CC1", 10, 1000, 2025, 11),
        _row("300001", A, "CC1", 5, 500, 2025, 12),
        _row("300002", A, "CC1", 2, 400, 2025, 12),
        # plant A, cost centre CC2
        _row("300002", A, "CC2", 4, 800, 2026, 1),
        _row("300003", A, "CC2", 1, 100, 2026, 1),
        # plant B, cost centre CC3
        _row("300001", B, "CC3", 3, 300, 2025, 12),
        # NOT consumables: other series, and a code that merely CONTAINS a 3
        _row("100001", A, "CC1", 1, 99999, 2025, 11),
        _row("200001", A, "CC2", 1, 88888, 2026, 1),
        _row("430001", B, "CC3", 1, 77777, 2025, 12),
        _row("930003", A, "CC1", 1, 66666, 2025, 12),
    ]
    _write(tmp_path, rows, names={"CC1": "CC1", "CC2": "CC2", "CC3": "CC3"})
    monkeypatch.setattr(settings, "CURATED_DIR", str(tmp_path))
    return tmp_path


def _keys(obj, out=None):
    out = set() if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(str(k).lower()); _keys(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _keys(v, out)
    return out


# ── what is counted ───────────────────────────────────────────────────────────
def test_only_materials_whose_code_starts_with_3_are_counted(client, data):
    o = client.get(f"{BASE}/overview").json()
    assert o["totals"] == {"cost": 3100.0, "qty": 25.0, "items": 3, "departments": 3, "hospitals": 2}
    r = client.get(f"{BASE}/items", params={"page_size": 50}).json()
    assert sorted(x["material"] for x in r["data"]) == ["300001", "300002", "300003"]
    assert r["sums"] == {"cost": 3100.0, "qty": 25.0}


def test_months_come_from_the_data_oldest_first(client, data):
    o = client.get(f"{BASE}/overview").json()
    assert [m["key"] for m in o["months"]] == ["2025-11", "2025-12", "2026-01"]
    assert [m["label"] for m in o["months"]] == ["Nov-25", "Dec-25", "Jan-26"]
    assert [m["cost"] for m in o["months"]] == [1000.0, 1200.0, 900.0]
    assert "Nov-25 to Jan-26" in o["note"] and "starting with 3" in o["note"]


# ── departments ───────────────────────────────────────────────────────────────
def test_department_totals_and_shares_add_up(client, data, plants):
    r = client.get(f"{BASE}/departments").json()
    rows = {x["cost_ctr"]: x for x in r["data"]}
    assert [x["cost_ctr"] for x in r["data"]] == ["CC1", "CC2", "CC3"]      # most cost first
    assert rows["CC1"]["cost"] == 1900.0 and rows["CC1"]["qty"] == 17.0 and rows["CC1"]["items"] == 2
    assert rows["CC2"]["cost"] == 900.0 and rows["CC3"]["cost"] == 300.0
    assert rows["CC3"]["plant"] == plants["B"]
    assert rows["CC1"]["hospital"] == plants["name_A"]
    assert r["total"] == 3 and r["sums"] == {"cost": 3100.0, "qty": 25.0}
    assert sum(x["share_pct"] for x in r["data"]) == pytest.approx(100.0)
    assert rows["CC1"]["share_pct"] == pytest.approx(1900 / 3100 * 100, abs=1e-3)
    assert set(r["data"][0]) == {"cost_ctr", "department_name", "plant", "hospital", "cost", "qty",
                                 "items", "share_pct", "top_item_desc"}


def test_each_department_names_its_biggest_consumable(client, data):
    o = client.get(f"{BASE}/overview").json()
    top = {d["code"]: d["top_item"] for d in o["departments"]}
    assert top["CC1"] == {"material": "300001", "desc": "Item 300001", "cost": 1500.0}
    assert top["CC2"] == {"material": "300002", "desc": "Item 300002", "cost": 800.0}
    r = client.get(f"{BASE}/departments").json()
    assert {x["cost_ctr"]: x["top_item_desc"] for x in r["data"]}["CC3"] == "Item 300001"


def test_department_names_are_flagged_pending_while_they_are_only_codes(client, data):
    o = client.get(f"{BASE}/overview").json()
    assert o["names_pending"] is True
    assert {d["code"]: d["name"] for d in o["departments"]}["CC1"] == "CC1"


# ── filters ───────────────────────────────────────────────────────────────────
def test_cost_centre_narrows_the_items_to_one_department(client, data):
    r = client.get(f"{BASE}/items", params={"CostCentre": "CC1"}).json()
    assert [(x["material"], x["cost"]) for x in r["data"]] == [("300001", 1500.0), ("300002", 400.0)]
    assert r["sums"] == {"cost": 1900.0, "qty": 17.0}
    assert sum(x["share_pct"] for x in r["data"]) == pytest.approx(100.0)
    assert r["data"][0]["months"] == 2 and r["data"][0]["departments"] == 1
    assert client.get(f"{BASE}/items", params={"CostCentre": "NOPE"}).json()["total"] == 0


def test_items_across_departments_count_each_department_once(client, data):
    r = client.get(f"{BASE}/items").json()
    rows = {x["material"]: x for x in r["data"]}
    assert rows["300001"]["departments"] == 2 and rows["300001"]["months"] == 2
    assert rows["300002"]["departments"] == 2


def test_plant_filter_by_code_and_by_hospital_name(client, data, plants):
    for p in (plants["A"], plants["name_A"]):
        o = client.get(f"{BASE}/overview", params={"Plant": p}).json()
        assert o["totals"]["cost"] == 2800.0 and o["totals"]["hospitals"] == 1
        assert o["scope"] == {"plant": plants["A"], "hospital": plants["name_A"]}
        d = client.get(f"{BASE}/departments", params={"Plant": p}).json()
        assert [x["cost_ctr"] for x in d["data"]] == ["CC1", "CC2"]
        assert sum(x["share_pct"] for x in d["data"]) == pytest.approx(100.0)
    # A month with no consumables at this hospital stays on the timeline, at zero.
    o = client.get(f"{BASE}/overview", params={"Plant": plants["B"]}).json()
    assert [m["cost"] for m in o["months"]] == [0.0, 300.0, 0.0]
    allp = client.get(f"{BASE}/overview", params={"Plant": "All Plants"}).json()
    assert allp["totals"]["cost"] == 3100.0 and allp["scope"]["plant"] is None


def test_table_sort_and_search_use_the_shared_protocol(client, data):
    r = client.get(f"{BASE}/departments", params={"sort_field": "cost", "sort_order": "asc"}).json()
    assert [x["cost_ctr"] for x in r["data"]] == ["CC3", "CC2", "CC1"]
    r = client.get(f"{BASE}/items", params={"global_filter": "300003"}).json()
    assert r["total"] == 1 and r["sums"] == {"cost": 100.0, "qty": 1.0}


# ── empty / missing / rebuilt data ────────────────────────────────────────────
def test_no_rows_gives_zeros_not_errors(client, tmp_path, monkeypatch):
    _write(tmp_path, [])
    monkeypatch.setattr(settings, "CURATED_DIR", str(tmp_path))
    o = client.get(f"{BASE}/overview")
    assert o.status_code == 200, o.text
    o = o.json()
    assert o["totals"] == {"cost": 0.0, "qty": 0.0, "items": 0, "departments": 0, "hospitals": 0}
    assert o["months"] == [] and o["departments"] == [] and o["items"] == []
    for ep in ("departments", "items"):
        r = client.get(f"{BASE}/{ep}")
        assert r.status_code == 200, r.text
        assert r.json()["data"] == [] and r.json()["total"] == 0


def test_no_consumables_keeps_the_months_at_zero(client, tmp_path, monkeypatch, plants):
    _write(tmp_path, [_row("100001", plants["A"], "CC1", 1, 500, 2025, 12)])
    monkeypatch.setattr(settings, "CURATED_DIR", str(tmp_path))
    o = client.get(f"{BASE}/overview").json()
    assert o["totals"]["cost"] == 0.0 and o["months"] == [{"key": "2025-12", "label": "Dec-25", "cost": 0.0, "qty": 0.0}]
    r = client.get(f"{BASE}/departments").json()
    assert r["total"] == 0 and r["sums"] == {"cost": 0.0, "qty": 0.0}


def test_a_missing_file_is_a_clear_503(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CURATED_DIR", str(tmp_path))
    r = client.get(f"{BASE}/overview")
    assert r.status_code == 503 and "not been built" in r.json()["detail"]


def test_a_rebuilt_file_is_picked_up_without_a_restart(client, data, plants):
    assert client.get(f"{BASE}/overview").json()["totals"]["cost"] == 3100.0
    _write(data, [_row("300009", plants["A"], "CC9", 1, 42, 2026, 1)])
    st = os.stat(data / "fact_consumption.parquet")
    os.utime(data / "fact_consumption.parquet", ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    o = client.get(f"{BASE}/overview").json()
    assert o["totals"]["cost"] == 42.0 and o["departments"][0]["code"] == "CC9"


# ── privacy and auth ──────────────────────────────────────────────────────────
def test_no_patient_fields_in_any_response(client, data):
    out = [client.get(f"{BASE}/overview").json(),
           client.get(f"{BASE}/departments").json(),
           client.get(f"{BASE}/items").json(),
           client.get(f"{BASE}/items", params={"CostCentre": "CC1"}).json()]
    keys = set().union(*(_keys(o) for o in out))
    assert not [k for k in keys if any(w in k for w in PATIENT_WORDS)], keys


def test_needs_a_login(anon_client, data):
    for ep in ("overview", "departments", "items"):
        assert anon_client.get(f"{BASE}/{ep}").status_code == 401
