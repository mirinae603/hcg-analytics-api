# tests/test_resolve_families.py — naming a thing, and naming where its measure lives.
from __future__ import annotations

from app.ai.resolve import brief, event_of, measure_locations, resolve, source_vocabulary


def test_a_short_vendor_name_resolves_to_its_family():
    # "Vardhman" covers 1 of 3 tokens in "Vardhman Health Specialities", so the coverage
    # rule rejected it and the question was answered "there is no lead time for Vardhman"
    fams = resolve("What is the average lead time for Vardhman?")["families"]
    assert any(f["token"] == "vardhman" and f["kind"] == "vendor" for f in fams)


def test_the_family_brief_offers_a_filter_that_covers_all_of_them():
    b = brief("What is the average lead time for Vardhman?")
    assert "LIKE '%VARDHMAN%'" in b
    assert "which ones you included" in b        # not silently the biggest


def test_a_class_word_binds_to_the_class_and_not_to_a_company():
    # "tablets" matched the category M113-TABLETS and, coincidentally, a manufacturer named
    # TABLETS INDIA. Resolving to the category is right; resolving to the supplier is the
    # bug. Rarity cannot tell them apart — both words appear in exactly 5 values.
    fams = resolve("how many tablets do we stock?")["families"]
    kinds = {f["kind"] for f in fams if f["token"] == "tablets"}
    assert kinds == {"category"}, kinds


def test_a_real_company_named_after_a_class_still_resolves():
    # the escape hatch: two matching tokens make it an entity, and families only form from
    # tokens no entity claimed
    r = resolve("how much did we buy from Tablets India?")
    assert any("TABLETS INDIA" in e["text"].upper() for e in r["entities"])


def test_a_measure_is_located_not_just_named():
    assert any("lead_time" in c for c in measure_locations("lead_time"))
    assert any(c.startswith("mart_procurement.") for c in measure_locations("lead_time"))


def test_an_absent_measure_returns_nothing_rather_than_a_guess():
    assert measure_locations("nonexistent_measure") == ()


def test_marts_are_offered_before_pre_aggregated_kpis():
    locs = measure_locations("lead_time")
    assert not locs[0].startswith("kpi_")


def test_a_city_brief_gives_codes_because_names_do_not_join():
    b = brief("How much did Bangalore hospitals spend on procurement?")
    assert "HC05" in b and "plant IN (" in b
    assert "fans rows out" in b               # says WHY not to join out to the name


def test_a_table_declares_what_event_it_measures():
    assert event_of("fact_consumption")[0] == "CONSUMPTION"
    assert event_of("sales_by_material")[0] == "SALES"
    assert event_of("mart_procurement")[0] == "PROCUREMENT"


def test_consumption_evidence_forbids_the_word_sold():
    # "LEAFLET A5 ... 1,203,000 units sold" came from fact_consumption; that product has
    # zero rows in any sales table
    v = source_vocabulary(["fact_consumption"])
    assert "NOT sold" in v and 'Do NOT write "sold"' in v


def test_sales_evidence_keeps_the_word():
    v = source_vocabulary(["sales_by_material"])
    assert "Do NOT write" not in v


def test_no_tables_yields_no_instruction():
    assert source_vocabulary([]) == ""


def test_a_schema_word_never_forms_a_family():
    # family matching reintroduced the STICKER-MSDS failure: "which high-value drugs have
    # the worst margins" matched HIGH VALUE DRUG STICKERS -GEN through the word "value"
    fams = resolve("Which high-value drugs have the worst margins?")["families"]
    assert not any(f["token"] == "value" for f in fams), fams


def test_the_vocabulary_comes_from_the_schema_not_a_hand_list():
    from app.ai.resolve import _schema_vocabulary
    v = _schema_vocabulary()
    assert {"value", "margin", "material", "stock", "vendor"} <= v   # real column words
    assert "vardhman" not in v and "gloves" not in v                 # real names


def test_margin_across_entities_must_be_stated_as_a_rate():
    b = brief("Which hospitals run the thinnest sales margins?")
    assert "PERCENTAGE" in b and "smallest site" in b


def test_a_size_qualifier_carries_a_computed_threshold():
    # told only to "pick a threshold at the top of the distribution" the model chose the
    # median, then Rs 1 lakh — both admit thousands of items. So the number is computed.
    b = brief("Which high-value drugs are we making the worst margin on?")
    assert "QUALIFIER" in b and "sales_by_material.revenue >=" in b
    assert "Class-A cut" in b


def test_the_threshold_is_taken_after_aggregating_to_the_grain():
    from app.ai.resolve import size_floor
    # mart_procurement is one row per PO line; cutting it raw ranked lines, not vendors,
    # and reported 14,194 "big vendors" out of 2,251 that exist
    loc, floor, n = size_floor("purchasing", "vendor")
    assert n < 2251, n


def test_an_unknown_grain_yields_no_threshold_rather_than_a_wrong_one():
    from app.ai.resolve import size_floor
    assert size_floor("revenue", "formulary") is None


def test_ranking_a_rate_demands_a_floor():
    b = brief("Which high-value drugs are we making the worst margin on?")
    assert "revenue floor" in b


def test_a_question_without_a_qualifier_gets_no_such_line():
    assert "QUALIFIER" not in brief("Which hospitals run the thinnest sales margins?")


def test_city_reachability_is_measured_not_assumed():
    from app.ai.resolve import city_reachability
    reach, blocked = city_reachability()
    # procurement/inventory share dim_plant's codes; sales uses a different system entirely
    assert "mart_procurement.plant" in reach
    assert "sales_by_hospital.hospital" in blocked


def test_the_city_brief_names_where_the_codes_work_and_where_they_do_not():
    b = brief("What is the top-selling drug in our Bangalore hospitals?")
    assert "sales_by_hospital.hospital" in b and "NOT valid in" in b
    assert "cannot answer it rather than producing a number" in b


def test_a_city_procurement_question_still_gets_a_usable_filter():
    b = brief("How much did Bangalore hospitals spend on procurement?")
    assert "plant IN ('HC01', 'HC05', 'HC06', 'HC40')" in b


def test_a_substring_match_in_another_kind_is_flagged_as_a_different_entity():
    # RELIANCE resolves cleanly and only as a MANUFACTURER, and the engine still answered
    # "Reliance is a supplier we buy from directly" after running its own LIKE '%RELIANCE%'
    # and finding Reliance Pharmaceutical Agencies — a different company
    r = resolve("Is Reliance a supplier we buy from directly, or just a manufacturer?")
    kinds = {la["kind"] for la in r["lookalikes"]}
    assert "vendor" in kinds
    assert any("Pharmaceutical Agencies" in e
               for la in r["lookalikes"] for e in la["examples"])


def test_the_caution_says_which_kind_the_name_really_is():
    b = brief("Is Reliance a supplier we buy from directly, or just a manufacturer?")
    assert "A substring match is not identity" in b
    assert "itself is a MANUFACTURER" in b


def test_an_unambiguous_name_gets_no_caution():
    assert "CAUTION" not in brief("Which items do we buy from MSD?")


def test_the_reporting_window_is_the_transaction_window():
    from app.ai.resolve import reporting_window
    w = reporting_window()
    assert w.startswith("2025-12") and "2026-05" in w
    assert "do NOT extend the reporting period" in w


def test_a_period_question_gets_the_window_even_with_nothing_else_to_resolve():
    # answered "2020-01-31 to 2026-12-03" — a range assembled from expiry and forecast dates
    b = brief("What period does our data cover?")
    assert "REPORTING PERIOD" in b and "2026-05" in b


def test_a_trend_question_gets_it_too():
    assert "REPORTING PERIOD" in brief("How has monthly revenue moved over the period?")


def test_ordinary_words_do_not_form_families():
    for q in ("What period does our data cover?", "show me the records for last month"):
        assert not resolve(q)["families"], q


def test_evaluative_adjectives_are_qualifiers_not_names():
    # "which critical items do we buy from only one vendor" was answered with
    # CRITICAL REGISTER A4 200 PAGE at ₹2,700
    r = resolve("Which critical items do we buy from only one vendor?")
    assert not r["families"]
    assert any(q.startswith("critical") for q in r["qualifiers"])


def test_the_subject_grain_is_the_one_named_first():
    assert resolve("Which critical items do we buy from only one vendor?")["grains"][0] == "material"
    assert resolve("Which vendors supply the most items?")["grains"][0] == "vendor"


def test_an_importance_qualifier_also_gets_a_computed_floor():
    b = brief("Which critical items do we buy from only one vendor?")
    assert "QUALIFIER" in b and "Class-A cut" in b
