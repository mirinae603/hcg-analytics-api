# tests/test_formulary_classify.py — the formulary rules, one at a time.
#
# These are the rules that reproduced HCG's own worksheet columns on the real Jun/Jul/Aug-26
# data (100% of lines for the item class; 100% for the payer bucket in Jul/Aug once the
# override file is applied). Each test pins one decision so a "tidy-up" cannot quietly
# move KR's compliance figure.
from __future__ import annotations

from datetime import date, datetime

import pytest

from app.core.data_access import CATEGORIES
from app.etl.formulary import classify as K
from app.etl.formulary import config as C
from app.etl.formulary import excel as X


# --- codes and categories ---------------------------------------------------------------
@pytest.mark.parametrize("raw, want", [
    (101313.0, "101313"), ("101313.0", "101313"), (" 101313 ", "101313"), (101313, "101313"),
    ("101313.00", "101313"), (None, None), ("", None), ("#N/A", None), ("0x2a", None),
    (float("nan"), None),
])
def test_material_codes_are_normalised(raw, want):
    assert K.norm_code(raw) == want


@pytest.mark.parametrize("code, want", [
    ("101313", "Onco Drugs"), (201196.0, "Other Drugs"), ("300001", "Consumables"),
    ("900001", "Unclassified"), (None, "Unclassified"),
])
def test_category_comes_from_the_first_digit(code, want):
    assert K.category(code) == want


def test_category_names_are_the_apps_own():
    # filter_category / resolve_category must keep working on the formulary tables
    for c in (K.ONCO, K.OTHER, K.CONSUMABLES, K.UNCLASSIFIED):
        assert c in CATEGORIES


# --- formulary class and group ------------------------------------------------------------
@pytest.mark.parametrize("flag, cat, want", [
    ("FORMULARY", "P-SCH", "FORMULARYP-SCH"),
    (" formulary ", " r ", "FORMULARYR"),
    ("FORMULARY", "OPD-R", "FORMULARYR-OPD"),       # Jun-26 spelling, R-OPD from Jul-26
    ("FORMULARY", "P-0PD", "FORMULARYP-OPD"),       # zero instead of the letter O
    ("FORMULARY", "CONSUMABLE", "FORMULARYCONSUMABLE"),
    ("FORMULARY", None, "FORMULARY"),
    ("NON FORMULARY", None, "NON FORMULARY"),
    ("OUT OF  FORMULARY", "", "OUT OF FORMULARY"),
    (None, None, "NOT IN MASTER"),
])
def test_formulary_class(flag, cat, want):
    assert K.formulary_class(flag, cat) == want


def test_an_unknown_flag_is_an_error_not_a_guess():
    with pytest.raises(ValueError):
        K.formulary_class("MAYBE FORMULARY", "P")


@pytest.mark.parametrize("cls, want", [
    ("FORMULARYP-SCH", "A"), ("FORMULARYR-OPD", "A"), ("FORMULARY", "A"),
    ("FORMULARYCONSUMABLE", "A"),        # HCG counts the contrast media as formulary
    ("OUT OF FORMULARY", "B"), ("NON FORMULARY", "C"), ("NOT IN MASTER", "X"), (None, "X"),
])
def test_formulary_group(cls, want):
    assert K.formulary_group(cls) == want


@pytest.mark.parametrize("hcg, want", [
    ("FORMULARYP-SCH", "FORMULARYP-SCH"), ("FORMULARYOPD-R", "FORMULARYR-OPD"),
    ("#N/A", "NOT IN MASTER"), ("0x2a", "NOT IN MASTER"), (None, "NOT IN MASTER"),
    ("NON FORMULARY ", "NON FORMULARY"),
])
def test_hcg_category_column_is_read_in_our_vocabulary(hcg, want):
    assert K.hcg_class(hcg) == want


@pytest.mark.parametrize("hcg, want", [
    ("A-Onco", "Onco"), ("B-Non Onco", "Non-Onco"), ("C-Consumables", "Consumables"),
    ("C-Consumbales", "Consumables"), ("#N/A", None), (None, None),
])
def test_hcg_segment_column(hcg, want):
    assert K.hcg_segment(hcg) == want


# --- payer bucket ---------------------------------------------------------------------------
@pytest.mark.parametrize("blank", [None, "", "   ", "Blank", "Blank ", "BLANK", "blank", "\xa0"])
def test_every_blank_payer_form_is_cash(blank):
    assert K.payer_bucket(blank, "2026-08", {"BLANK": "GOV"}, {}) == "CASH"
    assert K.resolve_payer(blank, "2026-08")[1] == "blank"


def test_payer_names_match_after_whitespace_and_case_but_keep_punctuation():
    master = {"CENTRAL GOVERNMENT HEALTH SCHEME (CGHS)": "CASH(TPA)",
              "CENTRAL GOVERNMENT HEALTH SCHEME-CGHS": "GOV"}
    assert K.payer_bucket("Central  Government\xa0Health Scheme (CGHS) ", "2026-08", master) == "CASH(TPA)"
    assert K.payer_bucket("Central Government Health Scheme-CGHS", "2026-08", master) == "GOV"
    # punctuation is NOT stripped: '(CGHS)' without the space is a different payer
    assert K.resolve_payer("Central Government Health Scheme(CGHS)", "2026-08", master) == \
        ("CASH(TPA)", "unmatched")


def test_an_unmatched_payer_defaults_to_cash_tpa():
    assert K.resolve_payer("NEW INSURER LTD", "2026-08", {}, {}) == ("CASH(TPA)", "unmatched")


def test_overrides_win_over_the_master_and_respect_their_period():
    master = {"BESCOM IP AGREEMENT": "GOV", "OIL AND NATURAL GAS CORPORATION (ONGC)": "GOV"}
    overrides = {
        "BESCOM IP AGREEMENT": [(None, "2026-06", "CASH(TPA)"), ("2026-07", None, "GOV")],
        "OIL AND NATURAL GAS CORPORATION (ONGC)": [(None, "2026-06", "GOV"),
                                                   ("2026-07", None, "CASH(TPA)")],
        "YSR AROGYASRI": [(None, None, "GOV")],
    }
    assert K.resolve_payer("BESCOM IP AGREEMENT", "2026-06", master, overrides) == ("CASH(TPA)", "override")
    assert K.resolve_payer("BESCOM IP AGREEMENT", "2026-07", master, overrides) == ("GOV", "override")
    assert K.payer_bucket("Oil and Natural Gas Corporation (ONGC)", "2026-06", master, overrides) == "GOV"
    assert K.payer_bucket("OIL AND NATURAL GAS CORPORATION (ONGC)", "2026-08", master, overrides) == "CASH(TPA)"
    assert K.payer_bucket("YSR AROGYASRI", "2025-01", {}, overrides) == "GOV"


def test_a_closed_override_range_falls_back_to_the_master_outside_it():
    overrides = {"X SCHEME": [("2026-07", "2026-07", "GOV")]}
    assert K.resolve_payer("X SCHEME", "2026-07", {"X SCHEME": "CASH(TPA)"}, overrides)[0] == "GOV"
    assert K.resolve_payer("X SCHEME", "2026-08", {"X SCHEME": "CASH(TPA)"}, overrides) == \
        ("CASH(TPA)", "master")


def test_payer_master_first_row_in_sheet_order_wins_and_conflicts_are_reported():
    rows = [("Central Government Health Scheme (CGHS)", "CASH(TPA)"),     # DR, first
            ("CENTRAL GOVERNMENT HEALTH SCHEME (CGHS)", "GOV"),           # KR
            ("Central Government Health Scheme (CGHS) ", "GOV"),          # Mumbai
            ("ECHS", "GOV"), ("ECHS", "GOV"), ("", "GOV"), ("NO BUCKET", None)]
    names, conflicts = K.first_row_buckets(rows)
    assert names["CENTRAL GOVERNMENT HEALTH SCHEME (CGHS)"] == "CASH(TPA)"
    assert names["ECHS"] == "GOV"
    assert "NO BUCKET" not in names
    assert conflicts == [{"name": "CENTRAL GOVERNMENT HEALTH SCHEME (CGHS)", "used": "CASH(TPA)",
                          "rows_by_bucket": {"CASH(TPA)": 1, "GOV": 2}}]


@pytest.mark.parametrize("raw, want", [
    ("CASH(TPA)", "CASH(TPA)"), ("Cash (TPA) ", "CASH(TPA)"), ("gov", "GOV"), ("CASH", "CASH"),
    ("TPA", None), (None, None), ("#N/A", None),
])
def test_bucket_values_are_normalised(raw, want):
    assert K.norm_bucket(raw) == want


# --- compliance --------------------------------------------------------------------------------
def test_pooled_is_a_over_a_plus_b_and_ignores_non_formulary():
    assert K.pooled_pct(102_061_364.55, 1_639_445.05) == pytest.approx(98.4191, abs=5e-5)
    assert K.pooled_pct(0, 0) is None
    assert K.pooled_pct(0, 10) == 0.0


def test_mean_averages_the_two_segment_ratios_and_skips_an_empty_one():
    assert K.mean_pct(90, 10, 80, 20) == pytest.approx(85.0)
    assert K.mean_pct(90, 10, 0, 0) == pytest.approx(90.0)     # no non-onco A+B: skipped
    assert K.mean_pct(0, 0, 0, 0) is None
    # NOT the pooled figure: pooled over the same money is 170/200
    assert K.pooled_pct(90 + 80, 10 + 20) == pytest.approx(85.0)
    assert K.mean_pct(9, 1, 80, 20) == pytest.approx(85.0)
    assert K.pooled_pct(9 + 80, 1 + 20) == pytest.approx(80.909, abs=1e-3)


def test_pct_is_none_on_a_zero_denominator():
    assert K.pct(5, 0) is None and K.pct(5, None) is None and K.pct(1, 4) == 25.0


# --- file names and cell values -----------------------------------------------------------------
@pytest.mark.parametrize("name, want", [
    ("IP&OP Consoildated File June 26 Updated Final File .xlsb", "2026-06"),
    ("IP and OP Sales Consolidated file July 2026.xlsb", "2026-07"),
    ("IP & OP Consolidated File August 26.xlsx", "2026-08"),
    ("ITEM MASTER FOR FORMULARY COMPLIENCE JUne-26.xlsx", "2026-06"),
    ("Copy of item master for formulary complience July 2026.xlsb", "2026-07"),
    ("Copy of Item master for formulary complience August -26_Updated.xlsx", "2026-08"),
    ("OP Sales 1st to 31st Dec_25 (3).xlsb", "2025-12"),
    ("Summary 2026.xlsx", None),                       # 'mar' inside a word is not March
    ("Sales June 26 vs July 26.xlsx", None),           # two months: ambiguous
])
def test_month_from_file_name(name, want):
    assert C.month_from_name(name) == want


def test_period_label():
    assert C.period_label("2026-08") == "Aug-26"


def test_cell_values():
    assert X.to_date(46235.0) == date(2026, 8, 1)                 # pyxlsb serial
    assert X.to_date(datetime(2026, 8, 1, 10, 30)) == date(2026, 8, 1)
    assert X.to_date("2026-08-01 00:00:00") == date(2026, 8, 1)
    assert X.to_date("#N/A") is None and X.to_date(None) is None
    assert X.money(1.590000033) == 1.59
    assert X.number("1,234.5") == 1234.5 and X.number("0x2a") is None and X.number("") is None
    assert X.invoice_prefix("kabhk-ip-123456") == "KABHK"
    assert X.text("  Blank ") == "Blank" and X.name("Blank ") is None and X.name("  ") is None
    assert X.text("#N/A") is None and X.code(12345.0) == "12345"
