# tests/test_hcg_reports.py — the Jun–Aug 2026 reports as the chatbot's get_kpi sees them
# (app/ai/hcg_reports.py), on the hand-made datasets of the API tests.
#
# The formulary and brand-rule tables are tests/test_brand_rules_api.py's fixture (built on
# test_formulary_api's 15 invented lines); the spend tables are test_spend_api's. Every expected
# number below is the same hand-worked figure those API tests assert, read here through the
# AI-facing payload: what changes is the shape (small, flat, ₹-formattable keys), never a value.
from __future__ import annotations

import json

import pandas as pd
import pytest

from app.ai import hcg_reports as H
from app.ai import kpi_registry as kr
from app.ai.orchestrator import _format_kpi_payload, infer_kind
from app.api import formulary, spend
from app.core.config import settings
from tests import test_spend_api
from tests.test_brand_rules_api import rules_data  # noqa: F401  (the fixture, used by `reports`)

AUG, JUL = "2026-08", "2026-07"
REPORTS = [(H.formulary_compliance, H.FORMULARY_VIEWS), (H.brand_rules, H.BRAND_RULE_VIEWS),
           (H.scheme_saving, H.SAVING_VIEWS), (H.attis_saving, H.ATTIS_VIEWS),
           (H.cash_tpa_margin_loss, H.CASH_VIEWS)]


@pytest.fixture()
def reports(rules_data):  # noqa: F811
    test_spend_api.write_tables(rules_data / "spend")       # next to the brand_blocks rules_data wrote
    formulary.clear_caches()
    spend.clear_caches()
    yield rules_data
    formulary.clear_caches()
    spend.clear_caches()


def shown(payload) -> str:
    """The tool message the model gets: get_kpi's ₹ / % formatting applied, as the orchestrator does."""
    return json.dumps(_format_kpi_payload(payload), ensure_ascii=False, default=str)


def numeric_keys(payload) -> set:
    keys = {k for k, v in payload["totals"].items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    for r in payload["rows"]:
        keys |= {k for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return keys


# ---------- shape, size, formatting ----------

def test_every_view_is_small_flat_and_formattable(reports):
    for fn, views in REPORTS:
        for view in views:
            out = fn(Plant="All Plants", view=view)
            assert "error" not in out, (fn.__name__, view, out)
            assert set(out) == {"totals", "scope", "view", "rows", "rows_total", "notes", "data_window"}
            assert out["view"] == view and out["data_window"] == f"{JUL}..{AUG}"
            assert all(not isinstance(v, (dict, list)) for v in out["totals"].values()), "totals must be flat"
            assert len(out["notes"]) <= 2 and out["rows_total"] >= len(out["rows"])
            assert len(shown(out)) <= 4000, (fn.__name__, view)
            for k in numeric_keys(out):
                # money must reach the model as ₹ Cr/L — a raw rupee figure is the known 10x error
                if k.endswith(("_cost", "_value")):
                    assert infer_kind(k) == "inr", (fn.__name__, view, k)
                elif k.endswith("_pct"):
                    assert infer_kind(k) == "pct", (fn.__name__, view, k)
                else:
                    assert infer_kind(k) == "num", (fn.__name__, view, k)   # counts, qty: no ₹ / % / days


def test_percentages_have_two_decimals_like_the_dashboard(reports):
    # numbers, not strings (charts and the evidence check read them), at the page's 2 dp
    for fn, views in REPORTS:
        for view in views:
            out = fn(view=view)
            for row in [out["totals"], *out["rows"]]:
                for k, v in row.items():
                    if k.endswith("_pct") and v is not None:
                        assert isinstance(v, float) and v == round(v, 2), (fn.__name__, view, k, v)
    assert H.formulary_compliance(Plant="HC05", month=AUG)["totals"]["compliance_pct"] == 83.33   # 1250 / 1500


def test_no_patient_identifiers_anywhere(reports):
    for fn, views in REPORTS:
        for view in views:
            text = shown(fn(view=view)).lower()
            assert not any(w in text for w in ("uhid", "patient_id", "patient_name", "invoice", "bill_no"))


def test_a_long_list_is_trimmed_to_the_budget_never_cut_mid_json():
    rows = [{"hospital": f"Hospital number {i}, Somewhere", "plant": f"HC{i:02d}", "compliance_pct": 91.23,
             "out_of_formulary_cost": 12345678.9} for i in range(200)]
    out = H._finish("hospitals", {"compliance_pct": 90.0, "drug_cost": 1e9}, {"plant": "ALL"}, rows, [JUL, AUG])
    assert H._size(out) <= H.BUDGET and json.loads(shown(out))
    assert out["rows_total"] == 200 and 0 < len(out["rows"]) < 200
    assert out["rows"][0]["plant"] == "HC00"                  # the tail goes, the most important rows stay
    assert out["notes"][-1] == f"Showing the first {len(out['rows'])} of 200 rows."


# ---------- the numbers (same hand-worked figures as the API tests) ----------

def test_formulary_summary_has_mis_compliance_and_consumption_share(reports):
    t = H.formulary_compliance(Plant="HC05", month=AUG)["totals"]
    # A = 1250, B = 250, C = 200 (test_formulary_api.test_headline_is_a_ratio_of_sums_over_drugs)
    assert (t["formulary_cost"], t["out_of_formulary_cost"], t["non_formulary_cost"]) == (1250.0, 250.0, 200.0)
    assert t["compliance_pct"] == 83.33                       # A / (A + B): HCG's MIS figure
    assert t["formulary_consumption_pct"] == 73.53            # A / (A + B + C): a different, lower figure
    assert (t["onco_compliance_pct"], t["non_onco_compliance_pct"]) == (90.0, 70.0)
    assert (t["restricted_scheme_cost"], t["restricted_scheme_share_pct"]) == (900.0, 52.94)   # 900 / 1700
    out = H.formulary_compliance(Plant="HC05", month=AUG)
    assert any("formulary_consumption_pct" in n for n in out["notes"])


def test_mean_method_follows_through(reports):
    assert H.formulary_compliance(Plant="HC05", month=AUG, method="mean")["totals"]["compliance_pct"] == 80.0


def test_several_hospitals_are_pooled_never_averaged(reports):
    out = H.formulary_compliance(Plant="HC05,HC06", month=AUG)
    # KR A 1250 B 250; DR A 500 B 540 -> 1750 / 2540, not the mean of 83.33% and 48.08%
    assert out["totals"]["compliance_pct"] == 68.9
    assert [r["plant"] for r in out["rows"]] == ["HC05", "HC06"]
    assert out["scope"]["plant"] == "HC05,HC06"
    with pytest.raises(ValueError, match="once per hospital"):
        H.formulary_compliance(Plant="HC05,HC06", view="doctors")


def test_spend_headlines(reports):
    a = H.attis_saving(Plant="HC05", month=AUG)["totals"]
    assert (a["used_value"], a["suggested_value"], a["saving_missed_value"], a["saving_pct"]) == \
        (3500.0, 1500.0, 2000.0, 57.14)
    assert a["scheme_saving_missed_value"] == 2058.0
    s = H.scheme_saving(Plant="HC05", month=AUG)["totals"]
    assert (s["saving_missed_value"], s["attis_saving_missed_value"], s["healthium_saving_missed_value"]) == \
        (2058.0, 2000.0, 8.0)
    c = H.cash_tpa_margin_loss(Plant="HC05", month=AUG)["totals"]
    assert (c["margin_loss_value"], c["potential_margin_value"], c["margin_loss_pct"]) == (3180.0, 4860.0, 65.43)
    assert c["topline_increase_value"] == 2370.0


def test_brand_rules_measures_and_payer(reports):
    out = H.brand_rules(Plant="HC05", month=AUG)
    m = {r["measure"]: r for r in out["rows"]}
    assert (m["restricted_scheme"]["measure_cost"], m["restricted_scheme"]["share_pct"]) == (900.0, 52.94)
    assert out["totals"]["drug_cost"] == 1700.0
    # the payer picks the rules that apply to those patients; it never filters a GOV rule to cash
    cash = H.brand_rules(Plant="HC05", month=AUG, payer="CASH")
    assert "restricted_scheme" not in {r["measure"] for r in cash["rows"]}
    assert cash["scope"]["measure"] == "restricted_cash_tpa"
    with pytest.raises(ValueError, match="not CASH"):
        H.brand_rules(measure="restricted_scheme", payer="CASH")


# ---------- arguments the model and the user actually write ----------

@pytest.mark.parametrize("month", ["Aug", "aug", "August", "August 2026", "Aug-26", "aug 26", "Aug'26",
                                   "2026-08", "08/2026", "latest"])
def test_month_spellings(reports, month):
    assert H.formulary_compliance(Plant="HC05", month=month)["scope"]["months"] == [AUG]


@pytest.mark.parametrize("month", ["jul,aug", "July and August", "Jul to Aug", "jul-aug", "2026-07..2026-08",
                                   "all", "ALL", "last 2 months", "quarter"])
def test_several_months(reports, month):
    out = H.formulary_compliance(Plant="HC05", month=month)
    assert out["scope"]["months"] == [JUL, AUG]
    assert out["totals"]["formulary_cost"] == 1650.0           # 1250 + Jul's 400: summed, then one ratio


def test_month_parser_unit():
    periods = ["2026-06", "2026-07", "2026-08"]
    assert H._month_arg("jun-aug", periods) == "2026-06,2026-07,2026-08"
    assert H._month_arg("Jun–Aug 2026", periods) == "2026-06,2026-07,2026-08"
    assert H._month_arg("last 2 months", periods) == "2026-07,2026-08"
    assert H._month_arg("Aug, 2026", periods) == "2026-08"
    assert H._month_arg(["Jul", "Aug"], periods) == "2026-07,2026-08"
    assert H._month_arg(None, periods) is None
    with pytest.raises(ValueError, match="These reports cover Jun–Aug 2026 only"):   # from `periods`
        H._month_arg("May", periods)                            # a month the reports lack: say what exists
    with pytest.raises(ValueError, match="Could not read month"):
        H._month_arg("whenever", periods)


def test_a_month_outside_the_data_is_a_value_error(reports):
    with pytest.raises(ValueError, match="No formulary data for 2026-05"):
        H.formulary_compliance(month="2026-05")


def test_doctor_name_resolves_to_the_id(reports):
    for name in ("Asha", "Dr Asha", "dr. asha", "D1"):
        out = H.formulary_compliance(Plant="HC05", month=AUG, doctor=name)
        assert out["scope"]["doctor_id"] == "D1" and out["scope"]["doctor"] == "Dr Asha"
        # D1 at KR in Aug: A 800, B 100, C 200 -> 800 / 900
        assert out["totals"]["compliance_pct"] == 88.89
        assert out["view"] == "items" and [r["material"] for r in out["rows"]] == ["100002"]   # its B item
    cash = H.cash_tpa_margin_loss(Plant="HC05", month=AUG, doctor="bhanu")
    assert cash["totals"]["margin_loss_value"] == 1455.0       # test_spend_api.test_cash_doctors
    with pytest.raises(ValueError, match="No doctor matching"):
        H.formulary_compliance(doctor="Nobody Here")


def test_ambiguous_doctor_lists_the_candidates():
    frame = pd.DataFrame({"doctor_id": ["11", "12", "13"], "doctor_name": ["Dr Ravi Kumar", "Dr Ravi Shankar", "Dr Uma"],
                          "plant": ["HC05", "HC06", "HC05"], "hospital_name": ["KR", "DR", "KR"]})
    with pytest.raises(ValueError, match=r"matches 2 doctors.*Dr Ravi Kumar \(KR, doctor_id 11\)"):
        H._doctor("ravi", frame, None)
    assert H._doctor("Ravi Kumar", frame, None) == ("11", "Dr Ravi Kumar")
    assert H._doctor("ravi", frame, ["HC06"]) == ("12", "Dr Ravi Shankar")     # the hospital narrows it


def test_material_name_or_code(reports):
    for item in ("onco b", "ONCO B 50MG", "100002", "100002.0"):
        out = H.formulary_compliance(month=AUG, material=item)
        assert out["scope"]["material"] == "100002" and out["view"] == "doctors"
        # lines 2, 12, 15: KR D1 100, DR D4 500, DR D1 40
        assert out["totals"]["drug_cost"] == 640.0
        assert [(r["doctor_id"], r["drug_cost"]) for r in out["rows"]] == [("D4", 500.0), ("D1", 100.0), ("D1", 40.0)]
    with pytest.raises(ValueError, match="matches 3 items"):
        H.formulary_compliance(material="onco")


def test_a_formulary_item_has_its_cost_even_without_doctor_rows(reports):
    # The doctor x item table holds groups B and C only. ONCO A 100MG is formulary (A): lines 1 and
    # 11 in Aug, 800 + 500. Summing its (absent) doctor rows used to report a cost of zero.
    out = H.formulary_compliance(month=AUG, material="ONCO A 100MG")
    assert out["totals"]["drug_cost"] == 1300.0 and out["totals"]["formulary_group"] == "A"
    assert out["rows"] == [] and any("no doctor rows" in n for n in out["notes"])


def test_brand_with_no_measure_lists_the_rules_it_counts_in(reports):
    out = H.brand_rules(month=AUG, material="PARA 500")
    assert out["view"] == "measures"
    assert {r["measure"]: r["measure_cost"] for r in out["rows"]} == {"preferred_scheme": 50.0}   # line 4


def test_wrong_arguments_raise_value_errors_the_model_can_fix(reports):
    with pytest.raises(ValueError, match="not available for formulary-compliance"):
        H.formulary_compliance(view="book")
    with pytest.raises(ValueError, match="CASH and CASH\\(TPA\\) patients only"):
        H.cash_tpa_margin_loss(payer="GOV")
    with pytest.raises(ValueError, match="Unknown hospital"):
        H.attis_saving(Plant="Nowhere General")
    with pytest.raises(ValueError, match="no doctor filter"):
        H.formulary_compliance(doctor="Asha", view="classes")


# ---------- missing data, and the registry path ----------

def test_missing_data_is_a_plain_error_dict(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "FORMULARY_DATA_DIR", str(tmp_path / "none"))
    monkeypatch.setattr(settings, "SPEND_DATA_DIR", str(tmp_path / "none"))
    formulary.clear_caches()
    spend.clear_caches()
    for fn, _ in REPORTS:
        out = fn()
        assert set(out) == {"error"} and out["error"].startswith("The ")
        assert "2026" not in out["error"]                       # no month written into the message
        assert out["error"].endswith("report data is not loaded on this server.")
        assert "python -m" not in out["error"]                  # the handler's CLI hint must not reach users
    formulary.clear_caches()
    spend.clear_caches()


def test_call_kpi_forwards_report_args_and_reports_the_rest(reports):
    r = kr.call_kpi("formulary-compliance", plant="HC05", month="Aug", measure="restricted_scheme")
    assert r["data"]["totals"]["compliance_pct"] == 83.33
    assert r["_ignored_args"] == ["measure"]
    r = kr.call_kpi("attis-saving", plant=None, category="Onco Drugs", month="aug", view="trend")
    assert r["data"]["view"] == "trend" and r["data"]["scope"]["segment"] == "onco"
    assert "_ignored_args" not in r
