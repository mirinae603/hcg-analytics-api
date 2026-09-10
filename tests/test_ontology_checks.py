# tests/test_ontology_checks.py — query checks derived from the ontology, not typed.
#
# Ontology as CONTEXT took GPT-4 from 16% to 54% on the OMG P&C schema; using it to CHECK
# generated queries and repair them took it to 72% (arXiv 2405.11706). Context tells the
# model what exists; checks stop it acting on a misunderstanding.
#
# Not one of these functions contains a table name, a column name or a business rule. Rename
# a column and they follow, because the ontology is rebuilt from the warehouse.
from __future__ import annotations

import pytest

from app.ai import ontology
from app.ai.deep import ontology_checks as oc

pytestmark = pytest.mark.skipif(not ontology.load().get("columns"),
                                reason="ontology not built")


def test_summing_a_rate_is_refused():
    # adding percentages produces a number that measures nothing — the class that once
    # printed "value share percentages as high as ₹85"
    w = oc.sums_a_non_additive_measure("SELECT SUM(fill_rate_pct) FROM kpi_fill_rate")
    assert w and "NON-ADDITIVE" in w


def test_summing_a_real_amount_is_allowed():
    assert oc.sums_a_non_additive_measure(
        "SELECT SUM(revenue) FROM sales_by_material") is None
    assert oc.sums_a_non_additive_measure("SELECT SUM(cost) FROM consumption_all") is None


def test_joining_two_disjoint_code_systems_is_refused():
    # dim_plant.plant and sales_by_hospital.hospital are both the hospital and overlap 0%
    w = oc.joins_disjoint_columns(
        "SELECT * FROM dim_plant d JOIN sales_by_hospital s ON d.plant = s.hospital")
    assert w and "CANNOT BE JOINED" in w


def test_a_real_join_is_allowed():
    assert oc.joins_disjoint_columns(
        "SELECT * FROM dim_plant d JOIN mart_procurement p ON d.plant = p.plant") is None


def test_grouping_by_an_ambiguous_name_is_flagged():
    # `category` is a 134-value material group in one table and a 5-value business segment
    # ("Onco Drugs", "Lab", "Consumables") in another, with zero overlap
    w = oc.groups_by_an_ambiguous_column(
        "SELECT category, SUM(purchase_value) FROM kpi_purchase_value GROUP BY category")
    assert w and "DIFFERENT THINGS" in w


def test_averaging_a_skewed_measure_is_refused():
    # "our overall days-on-hand is 203.34 days" — the median is 16.57. A few thousand
    # near-dead SKUs carry the mean, and it describes a warehouse nobody works in.
    w = oc.averages_a_skewed_measure("SELECT AVG(doh_days) FROM kpi_doh")
    assert w and "MISLEADING" in w and "median" in w.lower()


def test_the_median_of_the_same_column_is_fine():
    assert oc.averages_a_skewed_measure("SELECT MEDIAN(doh_days) FROM kpi_doh") is None


def test_averaging_an_unskewed_measure_is_fine():
    assert oc.averages_a_skewed_measure(
        "SELECT AVG(fill_rate_pct) FROM kpi_fill_rate") is None


def test_no_table_or_column_name_is_hardcoded_in_the_checks():
    """The whole point. These checks must survive a rename.

    Their knowledge comes from ontology.json, which is rebuilt from the warehouse — unlike
    the ~800 literals in _MEASURES / _GRAINS / _MEASURE_COLUMNS / GRAIN_COLUMNS that they
    are replacing.
    """
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "app" / "ai" / "deep"
           / "ontology_checks.py").read_text()
    tree = ast.parse(src)
    # strip docstrings properly — they NAME the traps as evidence, which is the point of
    # writing them down, and a line-based filter mistook that prose for logic
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    code = ast.unparse(ast.fix_missing_locations(tree))
    for name in ("kpi_doh", "sales_by_hospital", "dim_plant", "fill_rate_pct", "doh_days"):
        assert name not in code, f"{name} is hardcoded in the check logic"
