# tests/test_brand_rules_golden.py — HCG's own KR Aug-26 brand figures, through /formulary/rules, on
# the real tables.
#
# HCG's Spend Margin Loss email (KR Bangalore, Aug-26) shows three brand-rule columns per segment:
# restricted brands for scheme patients, preferred brands for scheme patients, and restricted
# brands for cash & TPA patients, each as value and as % of the segment's total consumption
# (all payers). Our formulary_summary reproduces all of them to the rupee; this pins the
# brand-rules API on top of the real tables.
#
# Runs only once `python -m app.etl.formulary` (with brand_rules.csv) has published the Task 4
# tables; otherwise skipped.
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.config import settings

_V = Path(settings.FORMULARY_DATA_DIR) / "formulary_validation.json"


def _built() -> bool:
    try:
        v = json.loads(_V.read_text())
    except (FileNotFoundError, ValueError):
        return False
    return bool(v.get("brand_rules")) and "2026-08" in (v.get("months") or {})


pytestmark = pytest.mark.skipif(not _built(), reason="brand rules not built — run python -m app.etl.formulary --from-cache")

# measure -> (cost, lines, onco cost, onco % of onco consumption, non-onco cost, non-onco %)
KR_AUG = {
    "restricted_scheme": (4_347_340.55, 953, 4_215_314.63, 4.58, 132_025.92, 0.94),
    "preferred_scheme": (227_385.07, 500, 169_903.61, 0.18, 57_481.46, 0.41),
    "restricted_cash_tpa": (45_192_099.74, None, 40_837_964.65, 44.39, 4_354_135.09, 30.95),
}


def overview(client):
    r = client.get("/formulary/rules/overview", params={"Plant": "HC05", "Month": "2026-08"})
    assert r.status_code == 200, r.text
    return {m["measure"]: m for m in r.json()["measures"]}


@pytest.mark.parametrize("measure", sorted(KR_AUG))
def test_kr_aug_brand_figures_match_hcg(client, measure):
    m = overview(client)[measure]
    cost, lines, onco, onco_pct, non, non_pct = KR_AUG[measure]
    assert m["cost"] == pytest.approx(cost, abs=0.5)
    if lines is not None:
        assert m["lines"] == lines
    assert m["onco"]["cost"] == pytest.approx(onco, abs=0.5)
    assert m["onco"]["share_pct"] == pytest.approx(onco_pct, abs=0.005)
    assert m["non_onco"]["cost"] == pytest.approx(non, abs=0.5)
    assert m["non_onco"]["share_pct"] == pytest.approx(non_pct, abs=0.005)


def test_kr_aug_per_doctor_table_adds_up_to_the_headline(client):
    for measure in ("restricted_scheme", "preferred_scheme", "oof_scheme"):
        head = overview(client)[measure]["cost"]
        r = client.get("/formulary/rules/doctors", params={"Plant": "HC05", "Month": "2026-08", "Measure": measure,
                                                           "page_size": 5000})
        assert r.status_code == 200, r.text
        assert sum(x["cost"] for x in r.json()["data"]) == pytest.approx(head, abs=1.0), measure


def test_the_task1_figure_does_not_move(client):
    r = client.get("/formulary/overview", params={"Plant": "HC05", "Month": "2026-08"})
    assert r.json()["headline"]["pooled_pct"] == pytest.approx(98.4191, abs=1e-4)
