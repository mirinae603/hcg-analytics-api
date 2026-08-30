# tests/test_fabricated_dimension.py — the guard against an INVENTED breakdown.
#
# Live failure it exists for: asked for the "sales trend of KEYTRUDA 100MG INJ VIAL", the
# assistant returned a confident six-month trend — "consistent performance across the
# 6-month period", ₹47.48 Cr and 13.0% margin in every single month. All fiction.
# `sales_by_material_hospital` has no month column, so it wrote six UNION ALL branches with
# hardcoded month literals over the identical unfiltered FROM/WHERE and dressed the same
# grand total up as a time series.
#
# Neither existing check could catch it: _unsupported_numbers passes (every figure really
# is in the evidence it was given) and the LLM auditor passed it too. It IS decidable from
# the SQL, which is what this guards.
from __future__ import annotations

import pytest

from app.ai.warehouse import _fabricated_dimension, validate, SqlError


REAL_CASE = """
SELECT '2025-12' AS month, SUM(revenue) AS revenue FROM sales_by_material_hospital WHERE material = '101313'
UNION ALL
SELECT '2026-01' AS month, SUM(revenue) AS revenue FROM sales_by_material_hospital WHERE material = '101313'
UNION ALL
SELECT '2026-02' AS month, SUM(revenue) AS revenue FROM sales_by_material_hospital WHERE material = '101313'
"""


def test_catches_the_keytruda_fabricated_trend():
    msg = _fabricated_dimension(REAL_CASE)
    assert msg is not None
    assert "made-up" in msg


def test_validate_rejects_it_rather_than_running_it():
    with pytest.raises(SqlError):
        validate(REAL_CASE)


def test_a_union_over_genuinely_different_sources_is_fine():
    # combining two real populations is a legitimate UNION ALL and must still run
    sql = """
    SELECT 'internal' AS kind, SUM(qty) AS qty FROM fact_consumption WHERE material = '101313'
    UNION ALL
    SELECT 'billed' AS kind, SUM(qty) AS qty FROM kpi_billable_consumption WHERE material = '101313'
    UNION ALL
    SELECT 'grn' AS kind, SUM(qty) AS qty FROM fact_grn WHERE material = '101313'
    """
    assert _fabricated_dimension(sql) is None


def test_different_filters_on_the_same_table_are_fine():
    # a real breakdown the table cannot GROUP BY — different WHERE per branch — is honest
    sql = """
    SELECT 'HC05' AS site, SUM(revenue) AS r FROM sales WHERE plant = 'HC05'
    UNION ALL
    SELECT 'HT12' AS site, SUM(revenue) AS r FROM sales WHERE plant = 'HT12'
    UNION ALL
    SELECT 'HM01' AS site, SUM(revenue) AS r FROM sales WHERE plant = 'HM01'
    """
    assert _fabricated_dimension(sql) is None


def test_two_identical_branches_are_left_alone():
    # two is not a "trend" and is more likely a clumsy join than an invented dimension;
    # the guard deliberately only fires from three branches up
    sql = """
    SELECT 'a' AS k, SUM(x) AS x FROM t WHERE m = '1'
    UNION ALL
    SELECT 'b' AS k, SUM(x) AS x FROM t WHERE m = '1'
    """
    assert _fabricated_dimension(sql) is None


def test_plain_queries_are_untouched():
    assert _fabricated_dimension("SELECT * FROM sales LIMIT 10") is None
    assert _fabricated_dimension("SELECT month, SUM(revenue) FROM sales GROUP BY month") is None
