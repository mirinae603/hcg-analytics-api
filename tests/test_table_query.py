# tests/test_table_query.py — the shared table filters (app/core/table_query.py) and their
# adoption by every family of list endpoint.
#
# The helper is tested on small hand-made frames; the endpoints on the hand-made fixtures their
# own test files build (formulary, spend, brand rules, consumables) and, for the generic KPI and
# legacy tables, on the committed KPI parquet. Every expected count below was worked out by hand
# from those rows (comments show it), not by re-running the helper.
from __future__ import annotations

import pandas as pd
import pytest
from fastapi import HTTPException
from starlette.datastructures import QueryParams

from app.core import data_access as da
from app.core import table_query as tq
from tests.test_brand_rules_api import rules_data  # noqa: F401  (fixture)
from tests.test_consumables_departments import BASE as CONSUMABLES
from tests.test_consumables_departments import data as consumables_data  # noqa: F401  (fixture)
from tests.test_consumables_departments import plants  # noqa: F401  (fixture)
from tests.test_formulary_api import AUG
from tests.test_formulary_api import data as formulary_data  # noqa: F401  (fixture)
from tests.test_spend_api import data as spend_data  # noqa: F401  (fixture)

PAYER = tq.Facet("payer", "Payer")
KIND = tq.Facet("kind", "Kind")
WHEN = tq.Facet("when", "When", kind="range")


def page(df, query: str, facets=(PAYER, KIND, WHEN), **kw) -> dict:
    return tq.table_page(df, QueryParams(query), facets=facets, **kw)


def facet(body: dict, field: str) -> dict:
    return next(f for f in body["facets"] if f["field"] == field)


def counts(f: dict) -> dict:
    return {v["value"]: v["count"] for v in f["values"]}


def get(client, path, **params):
    r = client.get(path, params=params)
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture()
def lines() -> pd.DataFrame:
    return pd.DataFrame({
        "name": ["CASH(TPA) brand", "(TPA note", "gov brand", "cash brand"],
        "payer": ["CASH", "CASH", "GOV", "CASH(TPA)"],
        "kind": ["a", "b", "a", "b"],
        "cost": [1.0, 2.0, 4.0, 8.0],
        "when": pd.to_datetime(["2026-08-01 00:00", "2026-08-31 15:00", "2026-09-01 00:00", None]),
    })


# ---------- the helper ----------

def test_search_is_literal_and_ignores_case(lines):
    # As a regular expression '(TPA' is an error (HTTP 500) and 'CASH(TPA)' matches nothing.
    assert page(lines, "global_filter=(TPA")["total"] == 3                  # names 1, 2 and payer 4
    assert page(lines, "global_filter=cash(tpa)")["total"] == 2             # name 1 and payer 4
    assert page(lines, "global_filter=(TPA&search_fields=name")["total"] == 2
    assert page(lines, "filter_field_0=name&filter_value_0=(TPA")["total"] == 2


def test_values_of_one_field_or_and_fields_and(lines):
    assert page(lines, "f.payer=CASH&f.payer=GOV")["total"] == 3            # rows 1, 2, 3
    body = page(lines, "f.payer=CASH&f.payer=GOV&f.kind=b", sum_cols=["cost"])
    assert body["total"] == 1 and body["sums"] == {"cost": 2.0}             # row 2


def test_counts_leave_out_the_facets_own_selection(lines):
    body = page(lines, "f.payer=CASH&f.payer=GOV&f.kind=b&facets=1")
    # payer counted over kind = b (rows 2, 4); kind over payer in CASH, GOV (rows 1, 2, 3)
    assert counts(facet(body, "payer")) == {"CASH": 1, "CASH(TPA)": 1, "GOV": 0}
    assert counts(facet(body, "kind")) == {"a": 2, "b": 1}
    assert facet(body, "payer")["selected"] == ["CASH", "GOV"]


def test_a_selected_value_is_listed_even_with_no_rows(lines):
    body = page(lines, "f.payer=NONE&facets=1")
    assert body["total"] == 0
    assert counts(facet(body, "payer"))["NONE"] == 0


def test_a_field_the_table_does_not_offer_is_a_400(lines):
    for q in ("f.nope=1", "fq.nope=x", "f.when=2026-08-01", "f.payer.from=1", "f.when.from=notadate",
              "facets=2", "page=two"):
        with pytest.raises(HTTPException) as e:
            page(lines, q)
        assert e.value.status_code == 400, q
    with pytest.raises(HTTPException) as e:
        page(lines, "f.nope=1")
    assert "payer, kind, when" in e.value.detail


def test_range_is_inclusive_and_a_plain_date_keeps_its_whole_day(lines):
    body = page(lines, "f.when.from=2026-08-01&f.when.to=2026-08-31&facets=1")
    assert body["total"] == 2                                               # 1 Aug and 31 Aug 15:00
    assert facet(body, "when")["range"] == {"min": "2026-08-01", "max": "2026-09-01"}
    assert facet(body, "when")["selected"] == {"from": "2026-08-01", "to": "2026-08-31"}


def test_month_filter_on_a_year_and_month_table():
    df = pd.DataFrame({"year": [2025, 2026, 2026, 2026], "month": ["December", "January", "January", "August"],
                       "v": [1, 2, 3, 4]})
    body = tq.table_page(df, QueryParams("facets=1"))
    month = facet(body, "period")
    assert month["label"] == "Month"
    assert [(v["value"], v["label"], v["count"]) for v in month["values"]] == [
        ("2026-08", "Aug-26", 1), ("2026-01", "Jan-26", 2), ("2025-12", "Dec-25", 1)]   # newest first
    body = tq.table_page(df, QueryParams("f.period=2026-01"))
    assert body["total"] == 2 and set(body["data"][0]) == {"year", "month", "v"}     # no derived column


def test_a_ticked_month_or_category_survives_a_scope_with_no_rows():
    # The table keeps its filters when the page's hospital or month changes; a scope that leaves
    # no rows must answer 0 rows, not a 400 for a Month or Category filter that went missing.
    empty = pd.DataFrame({"year": pd.Series([], dtype="int64"), "month": pd.Series([], dtype=object),
                          "category": pd.Series([], dtype=object), "v": pd.Series([], dtype="float64")})
    for q in ("f.category=Onco Drugs", "f.period=2026-08", "f.category=Onco Drugs&f.period=2026-08&facets=1"):
        body = tq.table_page(empty, QueryParams(q))
        assert body["total"] == 0 and body["data"] == [], q
    body = tq.table_page(empty, QueryParams("f.category=Onco Drugs&f.period=2026-08&facets=1"))
    assert counts(facet(body, "category")) == {"Onco Drugs": 0}                       # still listed, ticked
    assert counts(facet(body, "period")) == {"2026-08": 0}
    # With rows, a same-named column that is not the material category is still not offered.
    po = pd.DataFrame({"category": ["ANTINEOPLASTIC"], "v": [1.0]})
    assert [f["field"] for f in tq.table_page(po, QueryParams("facets=1"))["facets"]] == []


def test_fields_and_search_fields(lines):
    assert page(lines, "fields=cost,name")["data"][0] == {"cost": 1.0, "name": "CASH(TPA) brand"}
    assert page(lines, "global_filter=gov&search_fields=name")["total"] == 1
    assert page(lines, "global_filter=gov&search_fields=kind")["total"] == 0
    # renamed columns are addressed by the name the rows carry
    body = page(lines, "global_filter=gov&search_fields=label&fields=label", columns=["name"],
                rename={"name": "label"})
    assert body["data"] == [{"label": "gov brand"}]


def test_no_facets_unless_asked_and_only_skips_the_rows(lines):
    plain = page(lines, "page_size=2", sum_cols=["cost"])
    assert set(plain) == {"data", "total", "sums"}
    full = page(lines, "page_size=2&facets=1", sum_cols=["cost"])
    assert {k: full[k] for k in plain} == plain and full["scope"] == []
    only = page(lines, "facets=only")
    assert only["data"] == [] and only["total"] == 4 and only["facets"]


def test_typeahead_narrows_one_facets_options(lines):
    body = page(lines, "fq.payer=tpa&f.payer=GOV&facets=1")
    assert counts(facet(body, "payer")) == {"CASH(TPA)": 1, "GOV": 1}      # GOV stays: it is selected


def test_scope_chips_say_what_was_applied():
    assert tq.plant_chip(None) == tq.ScopeChip("Plant", "Hospital", "ALL", "All hospitals", "all-units")
    assert tq.plant_chip("HC05", split=False).value_label == "not split by hospital"
    assert tq.months_chip(["2026-07", "2026-08"], ["2026-07", "2026-08"]).token == "all-months"
    two = tq.months_chip(["2026-07", "2026-08"], ["2026-06", "2026-07", "2026-08"])
    assert (two.value_label, two.token) == ("Jul-26 + Aug-26", "2026-07+2026-08")
    assert tq.chip("Payer", "Payer", "CASH(TPA)").token == "CASH-TPA"
    assert tq.category_chip("Onco Drugs").token == "onco-drugs" and tq.category_chip(None) is None


# ---------- formulary ----------

def test_formulary_items_take_class_category_and_maker_filters(client, formulary_data):
    q = {"Plant": "HC05", "Month": AUG, "Group": "A"}
    # KR Aug formulary items: 100001 FORMULARYR 800, 200001 FORMULARYP 300 + 50, 100003 FORMULARYR-OPD 100
    either = get(client, "/formulary/items", **q, **{"f.formulary_class": ["FORMULARYR", "FORMULARYP"]})
    assert [r["material"] for r in either["data"]] == ["100001", "200001"] and either["sums"]["cost"] == 1150.0
    body = get(client, "/formulary/items", **q, facets=1,
               **{"f.formulary_class": ["FORMULARYR", "FORMULARYP"], "f.category": "Onco Drugs"})
    assert [r["material"] for r in body["data"]] == ["100001"] and body["sums"]["cost"] == 800.0
    assert body["data"][0]["share_pct"] == 64.0             # still a share of the whole selection: 800 / 1250
    assert counts(facet(body, "formulary_class")) == {"FORMULARYR": 1, "FORMULARYR-OPD": 1, "FORMULARYP": 0}
    assert counts(facet(body, "category")) == {"Onco Drugs": 1, "Other Drugs": 1}
    assert counts(facet(body, "manufacturer")) == {"Maker 1": 1}
    assert [(c["param"], c["value"], c["token"]) for c in body["scope"]] == [
        ("Plant", "HC05", "HC05"), ("Month", AUG, AUG), ("Group", "A", "A")]
    assert body["scope"][1]["value_label"] == "Aug-26" and body["scope"][2]["value_label"] == "Formulary"


def test_formulary_doctors_get_a_hospital_filter(client, formulary_data):
    body = get(client, "/formulary/doctors", Month=AUG, facets=1)
    # Aug: KR D1, D2, UNKNOWN; DR D4, D1; Rajkot D5
    assert counts(facet(body, "plant")) == {"HC05": 3, "HC06": 2, "HS01": 1}
    assert get(client, "/formulary/doctors", Month=AUG, **{"f.plant": "HC06"})["total"] == 2
    assert get(client, "/formulary/doctors", Month=AUG, global_filter="HC05", search_fields="doctor_name")["total"] == 0


def test_formulary_search_is_literal_and_unknown_filters_are_a_400(client, formulary_data):
    assert get(client, "/formulary/doctors", Month=AUG, global_filter="(TPA")["total"] == 0
    assert get(client, "/formulary/doctors", Month=AUG, global_filter="KR, B")["total"] == 3
    r = client.get("/formulary/items", params={"Month": AUG, "f.nope": "x"})
    assert r.status_code == 400 and "formulary_class" in r.json()["detail"]


def test_formulary_lists_unchanged_without_facets(client, formulary_data):
    q = {"Plant": "HC05", "Month": AUG, "Group": "A"}
    plain = get(client, "/formulary/items", **q)
    assert set(plain) == {"data", "total", "sums"}
    full = get(client, "/formulary/items", **q, facets=1)
    assert {k: full[k] for k in plain} == plain
    assert set(get(client, "/formulary/items", **q, fields="material,cost")["data"][0]) == {"material", "cost"}


# ---------- spend ----------

def test_spend_cash_items_filter_on_bifurcated(client, spend_data):
    q = {"Plant": "HC05", "Month": AUG}
    body = get(client, "/spend/cash/items", **q, facets=1)
    # KR Aug cash & TPA brands: 101001 bifurcated, 201002 not
    assert counts(facet(body, "bifurcated")) == {"True": 1, "False": 1}
    assert {v["label"] for v in facet(body, "bifurcated")["values"]} == {"Yes", "No"}
    one = get(client, "/spend/cash/items", **q, **{"f.bifurcated": "True"})
    assert [r["material"] for r in one["data"]] == ["101001"]
    assert one["sums"]["margin_loss"] == one["data"][0]["margin_loss"]      # the totals follow the filter
    tpa = get(client, "/spend/cash/items", **q, Payer="CASH(TPA)", facets=1)["scope"]
    assert [(c["param"], c["token"]) for c in tpa] == [("Plant", "HC05"), ("Month", AUG), ("Payer", "CASH-TPA")]


def test_spend_doctors_and_scheme_items(client, spend_data):
    body = get(client, "/spend/scheme/doctors", Month=AUG, facets=1)
    # Aug scheme lines: KR D1 (rows 1, 4, 5), D2 (2, 3); DR D9 (6)
    assert counts(facet(body, "plant")) == {"HC05": 2, "HC06": 1}
    items = get(client, "/spend/scheme/items", Plant="HC05", Month=AUG, facets=1)
    assert counts(facet(items, "category")) == {"Onco Drugs": 2, "Other Drugs": 1, "Consumables": 1}
    onco = get(client, "/spend/scheme/items", Plant="HC05", Month=AUG, **{"f.category": "Onco Drugs"})
    assert sorted(r["material"] for r in onco["data"]) == ["101001", "101005"]


# ---------- brand rules ----------

def test_rule_book_filters(client, rules_data):
    body = get(client, "/formulary/rules/book", Month=AUG, facets=1)
    assert counts(facet(body, "letter")) == {"R": 1, "P-SCH": 1, "R-OPD": 1, "P": 1, "SCH": 1}
    sold = get(client, "/formulary/rules/book", Month=AUG, **{"f.sold": "True"})
    assert sorted(r["material"] for r in sold["data"]) == ["100001", "100003", "200001"]
    assert [(c["param"], c["value_label"]) for c in body["scope"]] == [("Month", "Aug-26")]


# ---------- consumables ----------

def test_consumables_departments_filter_by_hospital(client, consumables_data, plants):
    A, B = plants["A"], plants["B"]
    body = get(client, f"{CONSUMABLES}/departments", facets=1)
    assert counts(facet(body, "plant")) == {A: 2, B: 1}                   # CC1, CC2 at A; CC3 at B
    assert get(client, f"{CONSUMABLES}/departments", **{"f.plant": B})["total"] == 1
    items = get(client, f"{CONSUMABLES}/items", facets=1)
    assert counts(facet(items, "uom")) == {"EA": 3}
    assert facet(items, "uom")["label"] == "Unit of measure"     # not "Unit": a unit is a hospital
    assert [c["token"] for c in items["scope"]] == ["all-units"]


# ---------- generic KPI and legacy tables (committed KPI parquet) ----------

needs_kpi = pytest.mark.skipif(not (da.KPI / "kpi_stock_out.parquet").exists(), reason="KPI parquet not present")


@needs_kpi
def test_kpi_table_offers_its_columns_and_says_what_it_holds(client):
    body = get(client, "/kpi/stock-out-rate/table", facets=1, page_size=1)
    fields = [f["field"] for f in body["facets"]]
    assert fields[:2] == ["plant", "category"] and {"material_group", "stock_status"} <= set(fields)
    out = da.load("kpi_stock_out")
    expected = int((out["stock_status"] == "Out of stock").sum())
    assert counts(facet(body, "stock_status"))["Out of stock"] == expected
    assert get(client, "/kpi/stock-out-rate/table", **{"f.stock_status": "Out of stock"})["total"] == expected
    assert body["scope"][0]["token"] == "all-units"
    lead = get(client, "/kpi/vendor-lead-time/table", Plant="HC05", facets=1)["scope"]
    assert (lead[0]["value_label"], lead[0]["token"]) == ("not split by hospital", "network")
    r = client.get("/kpi/stock-out-rate/table", params={"f.material_desc": "x"})
    assert r.status_code == 400


@needs_kpi
def test_kpi_table_month_filter(client):
    body = get(client, "/kpi/unit-sold-per-sku/table", facets=1, page_size=1)
    months = counts(facet(body, "period"))
    df = da.load("kpi_units_consumed")
    assert months["2026-05"] == int(((df["year"] == 2026) & (df["month"] == "May")).sum())
    assert get(client, "/kpi/unit-sold-per-sku/table", **{"f.period": "2026-05"})["total"] == months["2026-05"]


@needs_kpi
def test_legacy_table_filters_by_the_source_column_and_renames(client):
    body = get(client, "/kpi/inventory-valuation-table", facets=1, page_size=1,
               fields="materialId,inventoryValuation")
    assert set(body["data"][0]) == {"materialId", "inventoryValuation"}
    onco = get(client, "/kpi/inventory-valuation-table", **{"f.category": "Onco Drugs"})
    assert onco["total"] == counts(facet(body, "category"))["Onco Drugs"]
    sv = da.load("kpi_stock_value")
    assert onco["total"] == int((sv["category"] == "Onco Drugs").sum())
