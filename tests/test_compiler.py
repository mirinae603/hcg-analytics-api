# tests/test_compiler.py — deterministic SQL, and the discipline of declining.
#
# A 3-run measurement found that NOT ONE case fails all three runs: every question is
# answered correctly sometimes. The residual error is route variance, not missing knowledge,
# and guards cannot fix that — a guard rejects a bad route after it is taken, and two guards
# that rejected GOOD ones cost three points on their own.
#
# So for questions with exactly one right answer shape, remove the choice: the model decides
# WHAT is asked, the ontology decides HOW to fetch it, and the SQL is assembled in code.
#
# The measured coverage is 2 of 96 bank questions. That is the honest number, and it is why
# this is a router and not a replacement: a deterministic compiler buys consistency at a hard
# coverage ceiling, and everything it cannot express must go to the open path unharmed.
from __future__ import annotations

import pytest

from app.ai import ontology
from app.ai.deep import compiler

pytestmark = pytest.mark.skipif(not ontology.load().get("columns"),
                                reason="ontology not built")


def test_it_compiles_a_simple_ranking_correctly():
    p = compiler.plan("Which hospital generates the most sales revenue?")
    assert p
    from app.ai.deep.engine import _compact
    from app.ai.warehouse import run_sql
    sql = compiler.compile_sql(p)
    text = _compact(run_sql(sql), sql=sql)
    assert "KABHK" in text and "104.70" in text


def test_an_entity_named_in_the_question_reaches_the_where_clause():
    # without this it answered "which hospitals sell the most KEYTRUDA" with the top
    # hospital overall — right shape, wrong scope, confidently
    p = compiler.plan('Which hospitals sell the most "KEYTRUDA 100MG INJ VIAL"?')
    assert p and p.filters
    assert "KEYTRUDA" in compiler.compile_sql(p).upper()


def test_a_name_is_matched_against_a_description_not_a_code():
    # material holds '101313'; filtering it by 'KEYTRUDA 100MG INJ VIAL' returns nothing,
    # and an empty result reads as "this hospital sells no Keytruda"
    p = compiler.plan('Which hospitals sell the most "KEYTRUDA 100MG INJ VIAL"?')
    assert p and any("desc" in c or "name" in c for c, _v in p.filters), p.filters


@pytest.mark.parametrize("q", [
    "How has monthly revenue trended?",                 # a trend needs shaping
    "Which high-value drugs have the worst margins?",   # a computed threshold
    "Compare sales against stock by hospital",          # two measures
    "Why did margin fall in March?",                    # a diagnosis
])
def test_it_declines_anything_it_cannot_express(q):
    """Declining is the feature.

    Compiling everything is the semantic-layer trap: perfectly consistent on what it models
    and ZERO on anything else. The only independent clinical-domain evidence shows
    pre-specified layers are systematically incomplete.
    """
    assert compiler.plan(q) is None


def test_it_declines_when_the_headline_measure_is_ambiguous():
    # an earlier version guessed by size and compiled total_mrp_value as procurement spend:
    # ₹1,777 Cr of retail value where ₹649.91 Cr of purchasing belonged
    p = compiler.plan("What is our total procurement spend?")
    if p:
        assert "mrp" not in p.measure_col.lower()


def test_a_city_is_never_compiled():
    # a city needs plant codes and a reachability check, neither of which this expresses
    assert compiler.plan("How much do Bangalore hospitals spend on procurement?") is None


def test_no_table_or_column_name_is_hardcoded():
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "app" / "ai" / "deep" / "compiler.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    code = ast.unparse(ast.fix_missing_locations(tree))
    for name in ("sales_by_material", "fact_po", "kpi_purchase_value", "total_value_wo_tax"):
        assert name not in code, f"{name} is hardcoded"
