# tests/test_meta_windows.py — the months each source covers, and the stock count's date, come
# from the data (GET /meta/windows), and the rates built on a window divide by THAT window.
#
# The older inventory, consumption and forecast pages said "6 months" and "31 May 2026" in so many
# words, the turnover was "x 2.0 (6 months to a year)" and the billed rate "/ 182 days". All of that
# was true for the 31-May-26 stock count only. A new count moves the consumption window (the
# issues up to the count) while the legacy billed-sales tables keep Dec-25..May-26 on purpose, so
# the two parts of the turnover need their own factors. The fixture below is such a case: a
# 31-Aug-26 count over five months of issues (May missing) beside three months of billing.
# The real tests pin today's committed tables: both windows six months, x 2.0, 182 days, 31 May 26.
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.api import dashboard_summary as DS
from app.api import formulary as F
from app.api import kpi_generic, legacy_kpi
from app.api import purchase_insights as PI
from app.core import data_access as da
from app.core.config import settings
from app.etl.purchase import prices as P


def _clear():
    da.refresh_cache()
    DS.clear_caches()
    kpi_generic._plant_domains.cache_clear()
    kpi_generic.clear_kpi_caches()
    legacy_kpi.clear_result_caches()
    F.clear_caches()
    PI.clear_caches()


def _ts(*days):
    return pd.to_datetime(list(days))


@pytest.fixture()
def tables(tmp_path, monkeypatch):
    """Empty curated / kpi / formulary / purchase folders every reader is pointed at."""
    d = SimpleNamespace(**{n: tmp_path / n for n in ("cur", "kpi", "form", "pur")})
    for p in vars(d).values():
        p.mkdir()
    monkeypatch.setattr(da, "CURATED", d.cur)
    monkeypatch.setattr(da, "KPI", d.kpi)
    monkeypatch.setattr(legacy_kpi, "_KPI_DIR", str(d.kpi))
    for name, p in (("CURATED_DIR", d.cur), ("KPI_DIR", d.kpi), ("FORMULARY_DATA_DIR", d.form),
                    ("PURCHASE_DATA_DIR", d.pur)):
        monkeypatch.setattr(settings, name, str(p))
    _clear()
    yield d
    _clear()


def write_inventory(d, snapshot="2026-08-31"):
    # Two batches at P1 on one count. M1 expires 15 days after a 31-Aug count, M2 in Dec-27.
    snap = pd.NaT if snapshot is None else pd.Timestamp(snapshot)
    pd.DataFrame({
        "plant": ["P1", "P1"], "material": ["M1", "M2"], "material_desc": ["DRUG ONE", "DRUG TWO"],
        "qty": [4.0, 2.0], "total_cost": [100.0, 50.25], "total_mrp": [150.0, 60.0],
        "expiry_date": _ts("2026-09-15", "2027-12-31"), "snapshot_date": [snap, snap],
    }).to_parquet(d.cur / "fact_inventory.parquet")


def write_all(d):
    write_inventory(d)
    # Issues: Mar, Apr, Jun, Jul, Aug (on the count day, so it counts) and Sep (after the count).
    # No May extract. consumption = Mar..Sep, 6 months, gap May; up to the count = Mar..Aug, 5 months.
    days = _ts("2026-03-10", "2026-04-15", "2026-06-01", "2026-07-20", "2026-08-31", "2026-09-05")
    pd.DataFrame({
        "posting_date": days, "year": days.year, "month_num": days.month, "month": days.strftime("%B"),
        "plant": ["P1", "P2", "P1", "P2", "P1", "P1"], "material": ["M2", "M3", "M2", "M3", "M2", "M4"],
        "qty": [1.0] * 6, "amount_lc": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    }).to_parquet(d.cur / "fact_consumption.parquet")
    # Receipts in Jan and Feb only; no purchase-order table at all.
    pd.DataFrame({"plant": ["P3", "P3"], "posting_date": _ts("2026-01-05", "2026-02-10")}
                 ).to_parquet(d.cur / "fact_grn.parquet")
    pd.DataFrame({"plant": ["P1", "P2", "P3", "P4"], "plant_name": ["Unit One", "Unit Two", "Unit Three", "Unit Four"]}
                 ).to_parquet(d.cur / "dim_plant.parquet")
    # The stock count's book value: 100.00 + 50.25 = 150.25 over 2 materials.
    pd.DataFrame({"plant": ["P1", "P1"], "material": ["M1", "M2"], "material_desc": ["DRUG ONE", "DRUG TWO"],
                  "material_group": ["G1", "G1"], "stock_qty": [4.0, 2.0], "stock_value_cost": [100.0, 50.25],
                  "stock_value_mrp": [150.0, 60.0]}).to_parquet(d.kpi / "kpi_stock_value.parquet")
    # Legacy billing: Jan, Feb, Mar = 3 months, 31 + 28 + 31 = 90 days.
    pd.DataFrame({"patient": ["IP", "OP", "IP"], "month": ["2026-01", "2026-02", "2026-03"],
                  "revenue": [9.0, 9.0, 45.0], "cost": [5.0, 5.0, 30.0], "qty": [1.0, 1.0, 3.0], "lines": [1, 1, 3]}
                 ).to_parquet(d.kpi / "sales_monthly.parquet")
    pd.DataFrame({"material": ["M1", "M3"], "desc": ["DRUG ONE", "DRUG THREE"], "group": ["G1", "G2"],
                  "revenue": [45.0, 18.0], "cost": [30.0, 10.0], "qty": [3.0, 1.0], "lines": [3, 1]}
                 ).to_parquet(d.kpi / "sales_by_material.parquet")
    # Turnover inputs: internal issues per row up to the count (100 + 50 = 150), stock 400 + 100 = 500.
    pd.DataFrame({"plant": ["P1", "P1"], "material": ["M1", "M2"], "material_desc": ["DRUG ONE", "DRUG TWO"],
                  "material_group": ["G1", "G1"], "closing_stock_value": [400.0, 100.0], "aging_days": [10.0, 20.0],
                  "consumption_cost": [100.0, 50.0], "turnover_annualized": [0.0, 0.0],
                  "health_score": [50.0, 50.0], "health_tier": ["Watch", "Watch"]}
                 ).to_parquet(d.kpi / "kpi_health_score.parquet")
    pd.DataFrame({"plant": ["P1", "P1"], "material": ["M1", "M2"], "material_desc": ["DRUG ONE", "DRUG TWO"],
                  "material_group": ["G1", "G1"], "stock_qty": [9.0, 10.0], "consumption_qty": [0.0, 87.0],
                  "avg_daily_consumption": [0.0, 0.5], "doh_days": [np.nan, 20.0]}
                 ).to_parquet(d.kpi / "kpi_doh.parquet")
    # HIS sales months Jun, Jul at P1; purchase build Apr and Jun (no May GRN file).
    pd.DataFrame({"period": ["2026-06", "2026-07"], "plant": ["P1", "P1"]}).to_parquet(d.form / "formulary_summary.parquet")
    row = {c: "x" for c in P.PRICES_COLUMNS}
    row.update(lines=1, qty=1.0, value=1.0, unit_price_min=1.0, unit_price_max=1.0)
    pd.DataFrame([{**row, "period": "2026-04"}, {**row, "period": "2026-06"}]).to_parquet(d.pur / "purchase_prices.parquet")


# ── GET /meta/windows ─────────────────────────────────────────────────────────
def test_every_window_is_read_from_its_own_table(client, tables):
    write_all(tables)
    r = client.get("/meta/windows")
    assert r.status_code == 200, r.text
    b = r.json()
    assert list(b) == ["inventory_snapshot", "windows", "skus_total", "skus_in_stock", "stock_value",
                       "hospitals", "about", "data_as_of"]
    assert b["inventory_snapshot"] == "2026-08-31"
    w = b["windows"]
    assert list(w) == ["consumption", "consumption_to_snapshot", "grn", "po", "legacy_sales", "sales", "purchases"]
    assert w["consumption"] == {"from": "2026-03", "to": "2026-09", "months": 6, "gaps": ["2026-05"]}
    assert w["consumption_to_snapshot"] == {"from": "2026-03", "to": "2026-08", "months": 5, "gaps": ["2026-05"]}
    assert w["grn"] == {"from": "2026-01", "to": "2026-02", "months": 2, "gaps": []}
    assert w["po"] is None                                         # no purchase-order table
    assert w["legacy_sales"] == {"from": "2026-01", "to": "2026-03", "months": 3, "gaps": []}
    assert w["sales"] == {"from": "2026-06", "to": "2026-07", "months": 2, "gaps": []}
    assert w["purchases"] == {"from": "2026-04", "to": "2026-06", "months": 2, "gaps": ["2026-05"]}
    # held at the count {M1, M2} + issued up to it {M2, M3}: 3 (M4 only after the count)
    assert (b["skus_total"], b["skus_in_stock"], b["stock_value"]) == (3, 2, 150.25)
    # P1 stock+issues+sales, P2 issues, P3 receipts; P4 has nothing
    assert b["hospitals"] == 3
    assert set(b["about"]) == set(w)
    assert b["data_as_of"] and b["data_as_of"].endswith("+00:00")


def test_nothing_built_gives_nulls_not_errors(client, tables):
    r = client.get("/meta/windows")
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["inventory_snapshot"] is None and set(b["windows"].values()) == {None}
    assert (b["skus_total"], b["skus_in_stock"], b["stock_value"], b["hospitals"], b["data_as_of"]) == (None,) * 5
    assert da.legacy_sales_days() == 0 and da.stock_months() == ()
    assert da.yearly_cogs(100.0, 40.0) == 0.0                      # no window, no total to scale


def test_the_windows_need_a_login(anon_client):
    assert anon_client.get("/meta/windows").status_code == 401


def test_window_shape_edges():
    assert DS._window([]) is None and DS._window(None) is None
    assert DS._window(["2026-05"]) == {"from": "2026-05", "to": "2026-05", "months": 1, "gaps": []}
    # unsorted and repeated months: sorted, counted once, the hole named
    assert DS._window(["2026-07", "2026-05", "2026-07"]) == {"from": "2026-05", "to": "2026-07", "months": 2,
                                                              "gaps": ["2026-06"]}
    assert DS._window(["2025-12", "2026-01"])["gaps"] == []        # across the year end


def test_a_new_stock_count_moves_the_windows_after_a_refresh(client, tables):
    write_all(tables)
    assert client.get("/meta/windows").json()["inventory_snapshot"] == "2026-08-31"
    write_inventory(tables, snapshot="2026-09-30")                 # HCG sends a new count
    # served from the cache until the data is refreshed, like every other stock figure
    assert client.get("/meta/windows").json()["inventory_snapshot"] == "2026-08-31"
    r = client.post("/admin/refresh-data", headers={"x-admin-token": settings.ADMIN_REFRESH_TOKEN})
    assert r.status_code == 200, r.text
    b = client.get("/meta/windows").json()
    assert b["inventory_snapshot"] == "2026-09-30"
    assert b["windows"]["consumption_to_snapshot"] == {"from": "2026-03", "to": "2026-09", "months": 6, "gaps": ["2026-05"]}
    assert b["skus_total"] == 4                                    # M4, issued 5 Sep, is now inside the window


# ── the rates divide by their own window ─────────────────────────────────────
def test_each_part_of_the_turnover_is_annualised_over_its_own_window(client, tables):
    write_all(tables)
    # internal: 5 months up to the count -> x 12/5 = 2.4; billed: 3 months -> x 12/3 = 4.0
    assert (da.per_year(len(da.stock_months())), da.per_year(len(da.legacy_sales_months()))) == (2.4, 4.0)
    t = client.get("/kpi/inventory-turnover-ratio/insights").json()
    tot = t["totals"]
    # internal 100 + 50 = 150, billed 30 + 10 = 40 -> 190 over the windows;
    # a year: 150 x 2.4 + 40 x 4.0 = 360 + 160 = 520; stock 500 -> 1.04 (one x 2.0 said 0.76)
    assert tot["cogs_6mo"] == pytest.approx(190.0)
    assert tot["cogs_annual"] == pytest.approx(520.0)
    assert tot["portfolio_itr"] == pytest.approx(1.04)
    assert tot["months_on_hand"] == pytest.approx(12 / 1.04)
    # group G1: internal 150, billed (M1) 30 -> 360 + 120 = 480 / 500 = 0.96
    (g1,) = t["categories"]
    assert (g1["name"], g1["cogs"], g1["itr"]) == ("G1", pytest.approx(180.0), pytest.approx(0.96))
    # rows: M1 (100 x 2.4 + 30 x 4.0) / 400 = 0.9 (slow); M2 50 x 2.4 / 100 = 1.2 (moderate)
    hs = da.itr_scoped("both").set_index("material")
    assert hs.loc["M1", "turnover_annualized"] == pytest.approx(0.9)
    assert hs.loc["M2", "turnover_annualized"] == pytest.approx(1.2)
    bands = {b["key"]: (b["count"], b["value"]) for b in t["bands"]}
    assert bands["slow"] == (1, 400.0) and bands["moderate"] == (1, 100.0)
    # the trend stops at the count; billing folds into March, the one month both have
    assert [p["label"] for p in t["timeline"]] == ["Mar", "Apr", "Jun", "Jul", "Aug"]
    assert [p["value"] for p in t["timeline"]] == pytest.approx([10.0 + 30.0, 20.0, 30.0, 40.0, 50.0])
    # the registry chart agrees with the drill-down
    (chart,) = client.get("/kpi/inventory-turnover-ratio").json()
    assert (chart["ITR"], chart["Year"], chart["Month"]) == (0.96, 2026, "August")


def test_the_billed_daily_rate_divides_by_the_billing_days(tables):
    write_all(tables)
    assert da.legacy_sales_days() == 90                            # Jan 31 + Feb 28 + Mar 31
    doh = da.doh_scoped("billable").set_index("material")
    assert doh.loc["M1", "doh_days"] == pytest.approx(9.0 / (3.0 / 90))      # 270 days
    assert np.isnan(doh.loc["M2", "doh_days"])                     # never billed
    both = da.doh_scoped("both").set_index("material")
    assert both.loc["M2", "doh_days"] == pytest.approx(10.0 / 0.5)           # internal only: 20 days
    # value basis: stock 150.25 / (issues 150 over the days of the months that hold them + billed
    # 40 over 90 days). Those months are Mar, Apr, Jun, Jul, Aug: 31 + 30 + 30 + 31 + 31 = 153 days,
    # May (no extract) left out, as the turnover's x 12/5 leaves it out; the first-to-last posting
    # date span (10-Mar to 31-Aug, 174 days) would read May as a month of zero use.
    v = da.doh_value_scoped(None, None)
    assert v["days_span"] == da.month_days(da.stock_months()) == 153
    assert v["days_inventory_value"] == pytest.approx(150.25 / (150.0 / 153 + 40.0 / 90))


def test_without_the_billing_tables_billing_folds_in_nothing(tables):
    write_all(tables)
    (tables.kpi / "sales_monthly.parquet").unlink()
    _clear()
    assert da.legacy_sales_months() == () and da.legacy_sales_days() == 0
    assert np.isnan(da.doh_scoped("billable").set_index("material").loc["M1", "doh_days"])
    assert da.doh_value_scoped(None, None)["days_inventory_value"] == pytest.approx(150.25 / (150.0 / 153))


# ── the stock count's date on the cards and the expiry ladder ─────────────────
def test_the_count_date_labels_come_from_the_data(client, tables):
    write_all(tables)
    assert DS._count_label() == "31 Aug 2026"
    assert legacy_kpi._count_period() == (2026, "August")
    (row,) = client.get("/kpi/inventory-valuation").json()
    assert (row["Year"], row["Month"], row["Period"], row["Inventory Valuation"]) == (2026, "August", "August", 150.25)
    page = client.get("/kpi/inventory-valuation-table", params={"page": 0, "page_size": 5}).json()
    assert {(x["year"], x["period"]) for x in page["data"]} == {(2026, "August")}
    items = client.get("/kpi/near-expiry/items", params={"Plant": "P1"}).json()
    assert items["asof"] == "31 Aug 2026"
    assert {(i["material"], i["days"], i["slab"]) for i in items["items"]} == {("M1", 15, "0-30d"), ("M2", 487, "365d+")}
    # a hospital with no stock: an empty ladder, as of the network's own count (not a fixed day)
    none = client.get("/kpi/near-expiry/items", params={"Plant": "P2"}).json()
    assert (none["count"], none["items"], none["asof"]) == (0, [], "31 Aug 2026")


def test_no_stock_count_claims_no_date(client, tables):
    write_all(tables)
    write_inventory(tables, snapshot=None)
    _clear()
    assert DS._count_label() is None and legacy_kpi._count_period() == (None, None)
    items = client.get("/kpi/near-expiry/items", params={"Plant": "P1"}).json()
    assert (items["count"], items["items"], items["asof"]) == (0, [], None)
    assert client.get("/meta/windows").json()["inventory_snapshot"] is None


# ── the legacy sales ingest keeps its window on purpose ───────────────────────
def test_the_legacy_sales_window_is_a_documented_constant():
    from app.etl import ingest_sales as S
    assert S.LEGACY_SALES_WINDOW == "2025-12..2026-05"
    assert S.LEGACY_SALES_MONTHS == ("2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05")
    assert S.KEEP == {(2025, 12), (2026, 1), (2026, 2), (2026, 3), (2026, 4), (2026, 5)}   # value unchanged
    assert S.window_months("2026-11..2027-02") == ("2026-11", "2026-12", "2027-01", "2027-02")
    assert S.window_months("2026-05..2026-05") == ("2026-05",)
    for bad in ("2026-05..2025-12", "2026-13..2027-01", "2026-05", "", "May..Aug"):
        with pytest.raises(SystemExit):
            S.window_months(bad)


def test_sales_months_moves_the_window_without_a_code_change(monkeypatch):
    import importlib
    from app.etl import ingest_sales as S
    monkeypatch.setenv("SALES_MONTHS", "2026-06..2026-08")
    try:
        importlib.reload(S)
        assert S.LEGACY_SALES_MONTHS == ("2026-06", "2026-07", "2026-08")
        assert S.KEEP == {(2026, 6), (2026, 7), (2026, 8)}
    finally:
        monkeypatch.delenv("SALES_MONTHS")
        importlib.reload(S)
    assert S.LEGACY_SALES_MONTHS[0] == "2025-12"


# ── real data: today's committed tables ───────────────────────────────────────
_REAL = [da.CURATED / "fact_inventory.parquet", da.CURATED / "fact_consumption.parquet",
         da.KPI / "sales_monthly.parquet", da.KPI / "kpi_stock_value.parquet", da.KPI / "kpi_health_score.parquet"]
real = pytest.mark.skipif(not all(p.exists() for p in _REAL), reason="committed stock / sales tables not present")


def _months(dates) -> list:
    return sorted(pd.to_datetime(dates).dt.strftime("%Y-%m").unique())


@real
def test_real_windows(client):
    b = client.get("/meta/windows").json()
    snap = pd.read_parquet(da.CURATED / "fact_inventory.parquet", columns=["snapshot_date"])["snapshot_date"].max()
    assert b["inventory_snapshot"] == snap.strftime("%Y-%m-%d") == "2026-05-31"
    w = b["windows"]
    six = {"from": "2025-12", "to": "2026-05", "months": 6, "gaps": []}
    assert w["consumption_to_snapshot"] == six
    assert w["legacy_sales"] == six
    from app.etl import ingest_sales as S
    assert tuple(da.legacy_sales_months()) == S.LEGACY_SALES_MONTHS   # the tables hold the window the ingest keeps
    assert w["consumption"] == {"from": "2025-12", "to": "2026-09", "months": 9, "gaps": ["2026-08"]}
    # each window is its table's own months (these never go stale) ...
    grn = pd.read_parquet(da.CURATED / "fact_grn.parquet", columns=["posting_date"])["posting_date"]
    po = pd.read_parquet(da.CURATED / "fact_po.parquet", columns=["posting_date"])["posting_date"]
    assert (w["grn"]["from"], w["grn"]["to"], w["grn"]["months"]) == (_months(grn)[0], _months(grn)[-1], len(_months(grn)))
    assert (w["po"]["from"], w["po"]["to"], w["po"]["months"]) == (_months(po)[0], _months(po)[-1], len(_months(po)))
    # ... and today they are these
    assert (w["grn"]["from"], w["grn"]["to"], w["grn"]["months"]) == ("2025-12", "2026-08", 9)
    assert (w["po"]["from"], w["po"]["to"], w["po"]["months"]) == ("2025-12", "2026-09", 10)
    if w["sales"]:
        assert (w["sales"]["from"], w["sales"]["to"]) == (F._periods()[0], F._periods()[-1])
    # the home screen's old fixed figures (REAL.skusTotal 24,931; REAL.totalInventoryCost 604,679,341)
    assert b["skus_total"] == 24931 and b["skus_in_stock"] == 18080
    assert b["stock_value"] == 604679341.2
    rep = da.KPI / "stock_replenishment_and_aging_risk.parquet"
    if rep.exists():                                               # the universe the replenishment pages track
        assert pd.read_parquet(rep, columns=["material_id"])["material_id"].nunique() == b["skus_total"]
    plants = client.get("/meta/plants").json()["plants"]
    assert b["hospitals"] == len([p for p in plants if p["code"] != "ALL"]) == 52


@real
def test_real_rates_are_unchanged(client):
    # both windows are six months today: x 2.0 a year, 182 billing days -- the old constants exactly
    assert da.per_year(len(da.stock_months())) == 2.0
    assert da.per_year(len(da.legacy_sales_months())) == 2.0
    assert da.legacy_sales_days() == 182
    tot = client.get("/kpi/inventory-turnover-ratio/insights").json()["totals"]
    assert tot["cogs_annual"] == tot["cogs_6mo"] * 2.0
    assert tot["portfolio_itr"] == tot["cogs_6mo"] * 2.0 / tot["inventory"]
    hs = da.itr_scoped("both")                                     # consumption_cost here = internal + billed
    old = np.where(hs["closing_stock_value"] > 0, hs["consumption_cost"] * 2.0 / hs["closing_stock_value"], 0.0)
    np.testing.assert_array_equal(hs["turnover_annualized"].to_numpy(), old)
    card = client.get("/api/dashboard/all").json()
    assert card["stockAging"]["lastUpdated"] == card["kpiStockLevel"]["lastUpdated"] == "31 May 2026"
    assert card["daysOnHand"]["lastCalculated"] == "31 May 2026"
    rows = client.get("/kpi/inventory-valuation").json()
    assert {(r["Year"], r["Month"], r["Period"]) for r in rows} == {(2026, "May", "May")}


@real
def test_real_billable_counts_come_from_the_table(client):
    # the three counts the billable-consumption page used to carry as fixed numbers
    r = client.get("/kpi/billable-consumption/table", params={"facets": "only"}).json()
    (scope,) = [f for f in r["facets"] if f["field"] == "scope"]
    assert {v["value"]: v["count"] for v in scope["values"]} == {"Billed only": 13921, "Both": 1229, "Internal only": 10016}
