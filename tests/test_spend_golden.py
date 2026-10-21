# tests/test_spend_golden.py — HCG's own KR Aug-26 spend figures, through the API, on the real tables.
#
# HCG KR, Bangalore (SAP HC05) is the hospital of HCG's Spend Margin Loss email. The figures
# below are the email's / 'HC05 KR Spend Analysis August 26.xlsb' Summary rows 8-14: the whole
# scheme list, its Attis and J&J & Covidien -> Healthium slices, and the cash & TPA margin loss,
# at FY 2026-27 list prices. This is the one place they are written down: the app computes
# everything from the masters and sales; this test sets KR Aug-26 beside HCG's email.
#
# Runs only once `python -m app.etl.spend` has published the tables into
# settings.SPEND_DATA_DIR and Aug-26 is among them.
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.core.config import settings

_SWITCH = Path(settings.SPEND_DATA_DIR) / "scheme_switch.parquet"
pytestmark = pytest.mark.skipif(not _SWITCH.exists(),
                                reason="spend tables not built — run python -m app.etl.spend")

MONTH = "2026-08"
ATTIS = (59_689.63, 28_659.61, 31_030.01, 51.99)        # used, with Attis, saving missed, %
SCHEME_SAVING = 365_331.00                              # 999,434.25 - 634,103.24


@pytest.fixture()
def built():
    if MONTH not in set(pd.read_parquet(_SWITCH, columns=["period"])["period"].astype(str)):
        pytest.skip(f"not built: {MONTH}")


def test_kr_attis_matches_hcg(client, built):
    for plant in ("HC05", "KABHK"):
        r = client.get("/spend/attis/overview", params={"Plant": plant, "Month": MONTH})
        assert r.status_code == 200, r.text
        h = r.json()["headline"]
        used, sug, saving, pct = ATTIS
        assert h["used_value"] == pytest.approx(used, abs=0.5), plant
        assert h["suggested_value"] == pytest.approx(sug, abs=0.5), plant
        assert h["saving_missed"] == pytest.approx(saving, abs=0.5), plant
        assert h["saving_pct"] == pytest.approx(pct, abs=0.01), plant
        assert h["scheme_saving_missed"] == pytest.approx(SCHEME_SAVING, abs=0.5), plant


def test_kr_attis_items_add_up_to_the_headline(client, built):
    r = client.get("/spend/attis/items", params={"Plant": "HC05", "Month": MONTH, "page_size": 500})
    assert r.status_code == 200, r.text
    rows = r.json()["data"]
    assert sum(x["saving_missed"] for x in rows) == pytest.approx(ATTIS[2], abs=0.5)
    assert all(x["used_price"] > x["suggested_price"] for x in rows)


SCHEME = (999_434.25, 634_103.24, 365_331.00, 36.55)
HEALTHIUM = (79_074.59, 30_611.27, 48_463.32, 61.29)           # J&J & Covidien -> Healthium
# Cash & TPA: actual margin, potential margin, margin loss, loss % of potential, MRP uplift
CASH = (10_446_684.32, 18_255_008.42, 7_808_324.11, 42.77, 7_847_025.84)
_CASH_TABLE = Path(settings.SPEND_DATA_DIR) / "cash_switch.parquet"


def test_kr_scheme_and_healthium_match_hcg(client, built):
    if "scheme_group" not in pd.read_parquet(_SWITCH).columns:
        pytest.skip("scheme tables built before Task 3 — rebuild with python -m app.etl.spend")
    r = client.get("/spend/scheme/overview", params={"Plant": "HC05", "Month": MONTH})
    assert r.status_code == 200, r.text
    d = r.json()
    for got, want in ((d["headline"], SCHEME), (d["groups"]["healthium"], HEALTHIUM)):
        assert got["used_value"] == pytest.approx(want[0], abs=0.5)
        assert got["suggested_value"] == pytest.approx(want[1], abs=0.5)
        assert got["saving_missed"] == pytest.approx(want[2], abs=0.5)
        assert got["saving_pct"] == pytest.approx(want[3], abs=0.01)
    assert d["groups"]["attis"]["saving_missed"] == pytest.approx(ATTIS[2], abs=0.5)


def test_kr_cash_and_tpa_margin_loss_matches_hcg(client, built):
    if not _CASH_TABLE.exists():
        pytest.skip("cash & TPA tables not built — rebuild with python -m app.etl.spend")
    r = client.get("/spend/cash/overview", params={"Plant": "HC05", "Month": MONTH})
    assert r.status_code == 200, r.text
    h = r.json()["headline"]
    actual, potential, loss, pct, uplift = CASH
    assert h["actual_margin"] == pytest.approx(actual, abs=0.5)
    assert h["potential_margin"] == pytest.approx(potential, abs=0.5)
    assert h["margin_loss"] == pytest.approx(loss, abs=0.5)
    assert h["loss_pct"] == pytest.approx(pct, abs=0.01)
    assert h["topline_increase"] == pytest.approx(uplift, abs=0.5)
