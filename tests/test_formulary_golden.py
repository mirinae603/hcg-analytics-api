# tests/test_formulary_golden.py — HCG's own KR figures, through the API, on the real tables.
#
# HCG KR, Bangalore (SAP HC05, HIS KABHK) is the hospital HCG's monthly compliance emails
# report. The numbers below were reproduced independently from the raw monthly sales files
# and matched HCG's emails (see the Task 1 plan). If the published tables or the API's
# arithmetic drift from them, this fails.
#
# Runs only once `python -m app.etl.formulary` has published the tables into
# settings.FORMULARY_DATA_DIR; a month that has not been built is skipped.
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.core.config import settings

_SUMMARY = Path(settings.FORMULARY_DATA_DIR) / "formulary_summary.parquet"
pytestmark = pytest.mark.skipif(not _SUMMARY.exists(),
                                reason="formulary tables not built — run python -m app.etl.formulary")

# period -> (Formulary, Out of Formulary, Non Formulary, pooled %), KR drugs, all payers
KR = {
    "2026-06": (106_287_077.82, 1_034_728.16, 1_083_892.57, 99.0359),
    "2026-07": (114_606_892.40, 968_886.60, 4_813_423.92, 99.1617),
    "2026-08": (102_061_364.55, 1_639_445.05, 2_364_618.45, 98.4191),
}
ALL_HOSPITALS_POOLED = {"2026-06": 98.0105, "2026-07": 97.8907, "2026-08": 97.5547}


def _periods() -> set:
    return set(pd.read_parquet(_SUMMARY, columns=["period"])["period"].astype(str))


def _need(*months):
    missing = [m for m in months if m not in _periods()]
    if missing:
        pytest.skip(f"not built: {', '.join(missing)}")


def overview(client, **params):
    r = client.get("/formulary/overview", params=params)
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.parametrize("month", sorted(KR))
def test_kr_monthly_compliance_matches_hcg(client, month):
    _need(month)
    a, b, c, pct = KR[month]
    for plant in ("HC05", "KABHK"):
        d = overview(client, Plant=plant, Month=month)
        h = d["headline"]
        assert d["scope"]["plant"] == "HC05"
        assert h["formulary"] == pytest.approx(a, abs=0.5), plant
        assert h["out_of_formulary"] == pytest.approx(b, abs=0.5), plant
        assert h["non_formulary"] == pytest.approx(c, abs=0.5), plant
        assert h["pooled_pct"] == pytest.approx(pct, abs=1e-4), plant
        assert h["compliance_pct"] == h["pooled_pct"]


def test_kr_august_segments_and_mean(client):
    _need("2026-08")
    h = overview(client, Plant="HC05", Month="2026-08", Method="mean")["headline"]
    assert h["lines"] == 59_805
    assert h["onco_pct"] == pytest.approx(99.0177, abs=1e-4)
    assert h["non_onco_pct"] == pytest.approx(94.2446, abs=1e-4)
    assert h["mean_pct"] == pytest.approx(96.6312, abs=1e-4)
    assert h["compliance_pct"] == h["mean_pct"]


def test_kr_august_scheme_views(client):
    _need("2026-08")
    sc = overview(client, Plant="HC05", Month="2026-08")["scheme"]
    assert sc["restricted"]["cost"] == pytest.approx(4_347_340.55, abs=0.5)
    assert sc["preferred"]["cost"] == pytest.approx(227_385.07, abs=0.5)
    # HCG's report: onco restricted 4,215,314.63 = 4.58% of onco drug spend 91,997,056.56
    assert sc["restricted"]["onco"]["cost"] == pytest.approx(4_215_314.63, abs=0.5)
    assert round(sc["restricted"]["onco"]["share_pct"], 2) == 4.58


def test_all_hospitals_pooled_and_months_pool_as_sums(client):
    _need(*KR)
    trend = {t["period"]: t for t in overview(client, Month="ALL")["trend"]}
    for month, pct in ALL_HOSPITALS_POOLED.items():
        assert trend[month]["pooled_pct"] == pytest.approx(pct, abs=1e-4), month
    kr = overview(client, Plant="KABHK", Month=",".join(sorted(KR)))["headline"]
    a = sum(v[0] for v in KR.values())
    b = sum(v[1] for v in KR.values())
    assert kr["pooled_pct"] == pytest.approx(a / (a + b) * 100, abs=1e-4)


def test_kr_ranking_row_matches_its_headline(client):
    _need("2026-08")
    d = overview(client, Plant="HC05", Month="2026-08")
    row = next(r for r in d["by_hospital"] if r["selected"])
    assert row["plant"] == "HC05" and row["his_code"] == "KABHK"
    assert row["compliance_pct"] == d["headline"]["compliance_pct"]
    # the position chart's reference line: every hospital, the same filters
    assert d["network"]["compliance_pct"] == pytest.approx(ALL_HOSPITALS_POOLED["2026-08"], abs=1e-4)
