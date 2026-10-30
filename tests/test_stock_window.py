# tests/test_stock_window.py — the live stock measures set the stock count against the
# consumption up to it, and the billed-vs-internal views keep to the months both extracts have.
#
# fact_inventory is one stock count (31-May-26), while the consumption extracts run on past it
# (Jun, Jul and Sep 26; there is no Aug file) and the billing extract covers Dec-25..May-26. The
# ETL's stock KPIs take the issues up to the count (transforms.consumption_until); these tests
# hold the live recomputes (app/core/data_access.py, app/api/legacy_kpi.py) to the same window.
# They run on the COMMITTED curated and KPI tables.
from __future__ import annotations

import pandas as pd
import pytest

from app.core import data_access as da
from app.etl.transforms import consumption_until

# Days Inventory (value basis) at All Plants and at KR with the issues up to the 31-May count:
# Dec-25..31-May-26 is 181 days. Counting the Jun-Sep issues as well gave 30.02 / 110.89 over
# 303 days, against a stock count four months older than those issues.
DAYS_SPAN = 181
DAYS_ALL_PLANTS = 29.2918
DAYS_KR = 105.0299


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
