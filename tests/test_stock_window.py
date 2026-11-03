# tests/test_stock_window.py — the live stock measures set the stock count against the
# consumption up to it, and the billed-vs-internal views keep to the months both extracts have.
#
# fact_inventory is one stock count (31-May-26), while the consumption extracts run on past it
# (Jun, Jul and Sep 26; there is no Aug file) and the billing extract covers Dec-25..May-26. The
# ETL's stock KPIs take the issues up to the count (transforms.consumption_until); these tests
# hold the live recomputes (app/core/data_access.py, app/api/legacy_kpi.py) to the same window.
# They run on the COMMITTED curated and KPI tables. The fixtures at the end hold the labels to
# their own windows: the non-moving reason names the months it looked in ("No consumption in 6mo"
# only while that is six), and the procurement / consumption card subs the months their totals hold.
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from app.api import legacy_kpi
from app.core import data_access as da
from app.etl import transforms
from app.etl.transforms import consumption_until, no_consumption_reason

# Days Inventory (value basis) at All Plants and at KR with the issues up to the 31-May count:
# the six months that hold them, Dec-25..May-26, are 31 + 31 + 28 + 31 + 30 + 31 = 182 days (the
# same months the turnover's x 2.0 is read from; the first-to-last posting-date span, 181 days,
# gave 29.2918 / 105.0299). Counting the Jun-Sep issues as well gave 30.02 / 110.89 over 303 days,
# against a stock count four months older than those issues.
DAYS_SPAN = 182
DAYS_ALL_PLANTS = 29.3207
DAYS_KR = 105.6102


def _year_month(df: pd.DataFrame) -> pd.Series:
    return df["year"].astype(int).astype(str) + "-" + df["month_num"].astype(int).astype(str).str.zfill(2)


def test_the_stock_window_ends_on_the_stock_count():
    snap = da.stock_snapshot_date()
    assert snap == da.load("fact_inventory")["snapshot_date"].max()
    cons = da.load("fact_consumption")
    assert (cons["posting_date"] > snap).any()                 # the extracts do run on past it
    window = da.stock_consumption()
    assert window["posting_date"].max() <= snap
    assert len(window) == len(consumption_until(cons, snap))


def test_days_inventory_uses_the_consumption_up_to_the_stock_count():
    d = da.doh_value_scoped(None, None)
    assert d["days_span"] == DAYS_SPAN
    assert d["internal_consumption_value"] == pytest.approx(float(da.stock_consumption()["amount_lc"].sum()))
    assert d["days_inventory_value"] == pytest.approx(DAYS_ALL_PLANTS, abs=1e-3)
    kr = da.doh_value_scoped("HC05", None)
    assert (kr["days_span"], kr["billed_applicable"]) == (DAYS_SPAN, False)
    assert kr["days_inventory_value"] == pytest.approx(DAYS_KR, abs=1e-3)


def test_a_material_issued_only_after_the_count_does_not_move_the_counted_stock():
    snap = da.stock_snapshot_date()
    cons = da.load("fact_consumption")
    before = set(cons.loc[cons["posting_date"] <= snap, "material"].astype(str))
    billed = set(da.load("sales_by_material")["material"].astype(str))
    later_only = set(cons.loc[cons["posting_date"] > snap, "material"].astype(str)) - before - billed
    aging = da.load("kpi_inventory_aging")
    stocked = later_only & set(aging["material"].astype(str))
    assert stocked                                             # the case exists in the data
    nm = da.nonmoving_scoped("both")
    flagged = nm[nm["material"].astype(str).isin(stocked)]
    assert set(flagged["material"].astype(str)) == stocked
    # the reason names the months looked in: issues up to the count and billing, six each today
    assert len(da.stock_months()) == len(da.legacy_sales_months()) == 6
    assert set(flagged["reason"]) == {"No consumption in 6mo"}
    rc = da.risk_scoped("both")
    assert not rc[rc["material"].isin(stocked)]["consumed"].any()


def test_billed_and_internal_usage_cover_the_same_months(client):
    months = set(pd.read_parquet(da.KPI / "sales_monthly.parquet", columns=["month"])["month"].astype(str))
    cons = da.load("fact_consumption")
    assert not set(_year_month(cons)) <= months                # consumption runs on past billing
    internal = float(cons[_year_month(cons).isin(months)]["amount_lc"].sum())
    split = client.get("/consumption/billable-split").json()
    assert split["nonBillable"]["cost"] == pytest.approx(internal)
    rev = client.get("/revenue/insights").json()
    assert rev["totals"]["internal_cost"] == pytest.approx(sum(t["internal_cost"] for t in rev["timeline"]))
    assert rev["totals"]["internal_cost"] == pytest.approx(internal)


def test_the_turnover_timeline_stops_at_the_stock_count(client):
    tl = client.get("/kpi/inventory-turnover-ratio/insights").json()["timeline"]
    window = da.stock_consumption()
    expected = (window.groupby(["year", "month_num", "month"], observed=True)["amount_lc"].sum()
                .reset_index().sort_values(["year", "month_num"]))
    assert [t["month"] for t in tl] == [str(m) for m in expected["month"]]
    assert [t["label"] for t in tl] == ["Dec", "Jan", "Feb", "Mar", "Apr", "May"]


# ── The labels name their own window ─────────────────────────────────────────
def test_the_reason_counts_the_months_it_names():
    assert no_consumption_reason(6) == "No consumption in 6mo"     # today's text, unchanged
    assert no_consumption_reason(1) == "No consumption in 1mo"
    assert no_consumption_reason(0) == "No consumption recorded"   # no window to name, not "0mo"


def _issues(rows):
    f = pd.DataFrame(rows, columns=["plant", "material", "posting_date", "qty", "amount_lc"])
    f["posting_date"] = pd.to_datetime(f["posting_date"])
    f["year"], f["month"] = f["posting_date"].dt.year, f["posting_date"].dt.month_name()
    return f


def test_the_etl_reason_counts_the_issue_months_up_to_the_count(tmp_path, monkeypatch):
    monkeypatch.setattr(transforms, "KPI", tmp_path)
    snap = pd.Timestamp("2026-08-31")
    inv = pd.DataFrame({
        "plant": ["P1", "P1"], "material": ["M1", "M2"], "material_desc": ["DRUG ONE", "DRUG TWO"],
        "material_group": ["G1", "G1"], "qty": [5.0, 5.0], "total_cost": [50.0, 80.0], "total_mrp": [60.0, 90.0],
        "aging_days": [10, 20], "grn_date": [pd.Timestamp("2026-08-01")] * 2,
        "expiry_date": [pd.Timestamp("2027-12-31")] * 2, "snapshot_date": [snap] * 2, "batch": ["B1", "B2"]})
    # M2 is issued in Jun and Jul (two months up to the count); M1 only in Sep, after the count
    cons = _issues([("P1", "M2", "2026-06-10", 1.0, 10.0), ("P1", "M2", "2026-07-10", 1.0, 10.0),
                    ("P1", "M1", "2026-09-05", 1.0, 10.0)])
    dim = pd.DataFrame({"material": ["M1", "M2"], "material_desc": ["DRUG ONE", "DRUG TWO"], "material_group": ["G1", "G1"]})
    transforms.build_inventory_kpis(inv, cons, dim)
    nm = pd.read_parquet(tmp_path / "kpi_non_moving.parquet")
    assert nm[["material", "reason"]].values.tolist() == [["M1", "No consumption in 2mo"]]


@pytest.fixture()
def stock(tmp_path, monkeypatch):
    """A 31-Aug-26 stock count with five months of issues up to it (Mar, Apr, Jun, Jul, Aug: no May
    extract) and one issue after it (Sep), beside three months of billing (Jan-Mar)."""
    d = SimpleNamespace(cur=tmp_path / "cur", kpi=tmp_path / "kpi")
    d.cur.mkdir(); d.kpi.mkdir()
    monkeypatch.setattr(da, "CURATED", d.cur)
    monkeypatch.setattr(da, "KPI", d.kpi)
    da.refresh_cache()
    pd.DataFrame({"plant": ["P1"], "material": ["M1"], "snapshot_date": [pd.Timestamp("2026-08-31")]}
                 ).to_parquet(d.cur / "fact_inventory.parquet")
    _issues([("P1", "M2", day, 1.0, 10.0) for day in
             ("2026-03-10", "2026-04-15", "2026-06-01", "2026-07-20", "2026-08-31")]
            + [("P1", "M4", "2026-09-05", 1.0, 10.0)]).to_parquet(d.cur / "fact_consumption.parquet")
    # M1 never used, M2 issued, M3 billed only, M4 issued only after the count (and aged)
    pd.DataFrame({"plant": ["P1"] * 4, "material": ["M1", "M2", "M3", "M4"],
                  "material_desc": ["ONE", "TWO", "THREE", "FOUR"], "material_group": ["G1"] * 4,
                  "category": ["Other Drugs"] * 4, "closing_stock_quantity": [1.0] * 4,
                  "closing_stock_value": [10.0, 20.0, 30.0, 40.0], "aging_days": [10.0, 20.0, 30.0, 400.0],
                  "last_sale_date": [pd.NaT] * 4}).to_parquet(d.kpi / "kpi_inventory_aging.parquet")
    pd.DataFrame({"month": ["2026-01", "2026-02", "2026-03"], "cost": [1.0, 1.0, 1.0]}
                 ).to_parquet(d.kpi / "sales_monthly.parquet")
    pd.DataFrame({"material": ["M3"], "cost": [5.0], "qty": [1.0]}).to_parquet(d.kpi / "sales_by_material.parquet")
    yield d
    da.refresh_cache()


def _reasons(scope):
    nm = da.nonmoving_scoped(scope)
    return dict(zip(nm["material"], nm["reason"]))


def test_the_live_reason_names_the_months_each_scope_looked_in(stock):
    assert len(da.stock_months()) == 5 and len(da.legacy_sales_months()) == 3
    # both: billing over 3 months and issues over 5; over the shorter, neither kind of use happened
    assert _reasons("both") == {"M1": "No consumption in 3mo", "M4": "No consumption in 3mo"}
    # billed only: the billing months; M2 is issued but not billed
    assert _reasons("billable") == {"M1": "No consumption in 3mo", "M2": "No consumption in 3mo",
                                    "M4": "No consumption in 3mo"}


def test_with_no_billing_tables_the_reason_names_the_issue_months(stock):
    (stock.kpi / "sales_monthly.parquet").unlink()
    (stock.kpi / "sales_by_material.parquet").unlink()
    da.refresh_cache()
    assert _reasons("both") == {"M1": "No consumption in 5mo", "M3": "No consumption in 5mo",
                                "M4": "No consumption in 5mo"}


def test_with_no_issues_and_no_billing_nothing_names_a_window(stock):
    (stock.kpi / "sales_monthly.parquet").unlink()
    (stock.kpi / "sales_by_material.parquet").unlink()
    _issues([]).to_parquet(stock.cur / "fact_consumption.parquet")    # an empty extract
    da.refresh_cache()
    assert set(_reasons("both").values()) == {"No consumption recorded"}
    assert len(_reasons("both")) == 4


def test_a_card_sub_names_the_months_its_table_holds(tmp_path, monkeypatch):
    monkeypatch.setattr(da, "KPI", tmp_path)
    da.refresh_cache()
    # two plants over three months (Dec-25 counted once, though both plants have it)
    pd.DataFrame({"plant": ["P1", "P2", "P1", "P1"], "year": [2025, 2025, 2026, 2026],
                  "month": ["December", "December", "January", "February"], "purchase_value": [1.0] * 4}
                 ).to_parquet(tmp_path / "kpi_purchase_value.parquet")
    pd.DataFrame({"plant": [], "year": [], "month": []}).to_parquet(tmp_path / "kpi_units_consumed.parquet")
    try:
        assert legacy_kpi._months_held("kpi_purchase_value") == "3-mo"
        assert legacy_kpi._months_held("kpi_units_consumed") == ""    # nothing held: no window named
    finally:
        da.refresh_cache()


def test_the_card_subs_on_todays_tables(client):
    # orders run Dec-25..Sep-26 (10 months); issues Dec-25..Sep-26 without Aug (9 months)
    po = da.load("kpi_purchase_value")[["year", "month"]].drop_duplicates()
    uc = da.load("kpi_units_consumed")[["year", "month"]].drop_duplicates()
    assert (len(po), len(uc)) == (10, 9)
    for plant in ({}, {"Plant": "HC05"}):
        cards = client.get("/portfolio/procurement/overview", params=plant).json()["cards"]
        assert cards["purchase-value"]["sub"] == "10-mo PO spend"
        cards = client.get("/portfolio/consumption/overview", params=plant).json()["cards"]
        assert cards["consumption-by-department"]["sub"] == "9-mo cost"


def test_the_assistant_annualises_with_the_dashboards_months():
    from app.ai import windows
    assert windows.stock_months() == da.stock_months()
    assert windows.values()["STOCK_MONTHS"] == str(len(da.stock_months())) == "6"
