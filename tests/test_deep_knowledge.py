# tests/test_deep_knowledge.py — the things the engine must KNOW about this warehouse
# before it plans anything, all of them derived from the data rather than asserted.
#
# Every failure these guard was originally reported to the user as a limit of their DATA
# when it was a limit of the agent's knowledge — which is the most damaging mistake this
# system makes, because it teaches people their warehouse is worse than it is.
from __future__ import annotations

from app.ai.deep import capability, shapes
from app.ai.deep.engine import _kpi_rows
from app.ai.deep import tools


# ── vocabulary: the user's words are not the schema's ────────────────────────
def test_hospital_is_known_to_mean_the_plant_column():
    # "Which hospital holds the most inventory value?" was answered "the schema does not
    # contain inventory value data broken down by hospital" — while fact_inventory carries
    # `plant` and `total_cost`. A missing synonym reported as missing data.
    vocab = " ".join(capability.vocabulary()).lower()
    assert "hospital" in vocab and "plant" in vocab


def test_vocabulary_covers_every_entity_family_that_has_columns():
    vocab = " ".join(capability.vocabulary()).lower()
    for word in ("vendor", "manufacturer", "material", "category"):
        assert word in vocab, word


# ── joinability: two hospital code systems that do not join ──────────────────
def test_the_disjoint_hospital_code_systems_are_reported():
    # sales rows carry KABHK/GJHCA; procurement carries HC05/AH01; they share zero rows and
    # nothing maps between them. Without this the engine invented city-filtered sales
    # figures — "KEYTRUDA ₹16.37 Cr in Bangalore hospitals", and ₹10.78 Cr the run before.
    notes = " ".join(capability.joinability()).lower()
    assert notes, "the disjoint code systems must be detected"
    assert "do not join" in notes
    assert "sales" in notes


# ── canonical metric semantics, computed not guessed ─────────────────────────
def test_within_90_days_excludes_already_expired_stock():
    # The canonical buckets are Expired / 0-30d / 31-90d / 91-180d. Asked how much expires
    # in the next 90 days the model summed all of them: 101,005 units against a true
    # 45,223, then 87,775 on the next run. The arithmetic is trivial and the judgement is
    # not, so both live here.
    res = _kpi_rows(tools.get_kpi("near-expiry"))
    cum = shapes.derive_exposure(res).get("cumulative") or {}
    assert cum["within_90_days"]["qty"] == 45223.0
    assert round(cum["within_90_days"]["value"]) == 3997173
    assert cum["already_expired"]["qty"] == 55782.0
    # and the two must never be silently added
    assert "not add them together" in cum["note"].lower()


def test_cumulative_bands_are_monotonic():
    res = _kpi_rows(tools.get_kpi("near-expiry"))
    cum = shapes.derive_exposure(res).get("cumulative") or {}
    assert cum["within_30_days"]["qty"] < cum["within_90_days"]["qty"] < cum["within_180_days"]["qty"]


def test_a_kpi_payload_flattens_to_its_breakdown_not_its_summary():
    # near-expiry holds totals/buckets/timeline/categories/ladder. "buckets" was missing
    # from the preferred keys, so it fell through to `totals` and the answer reported
    # `exposure` (₹1.98 Cr of TOTAL near-expiry) as "19,807,976 units expiring in 90 days".
    res = _kpi_rows(tools.get_kpi("near-expiry"))
    assert "bucket" in res["columns"]
    assert res["row_count"] >= 4


# ── the tools that make LOOKING possible ─────────────────────────────────────
def test_find_columns_locates_which_tables_carry_a_grain():
    hits = tools.find_columns("month")["found"]
    assert any(h["table"] == "fact_po" for h in hits)
    assert not any(h["table"].startswith("sales_by_material") for h in hits)


def test_find_value_locates_where_a_literal_actually_lives():
    # 'Bangalore' is not a column, it is a value inside dim_plant.plant_name — the fact
    # that made the engine declare the warehouse had no location data at all.
    found = tools.find_value("Bangalore")["found_in"]
    assert found and any("plant_name" in h["column"] for h in found)
