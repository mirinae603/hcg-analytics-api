# tests/test_no_stale_facts.py — a note that quotes a count must COUNT it.
#
# The consumption guidance told the model "fact_consumption is materials ISSUED FROM STORES
# (11,225 materials) … covering 25,153 materials". Both numbers were correct on the day they
# were written and both were LITERALS in the prompt, so the first parquet refresh would have
# had the assistant stating stale figures with full confidence — while every guard around it
# measured the data live.
#
# It is the same failure as the hand-maintained table list: a fact copied into code is a fact
# that stops being true without telling anyone.
from __future__ import annotations

from app.ai.resolve import _count_distinct, _MEASURE_NOTES, _note_text, brief


def test_the_consumption_note_is_computed_not_written():
    assert callable(_MEASURE_NOTES["consumption"]), (
        "the counts in this note must be measured at call time, not baked in")


def test_the_counts_it_quotes_match_the_warehouse():
    note = _note_text(_MEASURE_NOTES["consumption"])
    assert _count_distinct("fact_consumption", "material") in note
    assert _count_distinct("consumption_all", "material") in note


def test_a_plain_string_note_still_works():
    assert _note_text("just a string") == "just a string"
    assert _note_text(None) == ""


def test_a_note_that_raises_degrades_to_silence():
    def boom():
        raise RuntimeError("no warehouse")
    assert _note_text(boom) == ""


def test_the_counter_never_raises_on_a_missing_table():
    assert _count_distinct("no_such_table", "no_such_col") == "many"


# ── the vocabulary must not need every tense enumerated ─────────────────────────────────
def test_the_base_verb_resolves_like_its_inflections():
    # the list held "consumption" and "consumed" but not "consume", so "how much keytruda
    # did we consume?" resolved NO measure and never saw the consumption guidance at all
    from app.ai.resolve import resolve
    for q in ("how much did we consume?", "what did we consume last month",
              "how much was consumed", "total consumption"):
        assert resolve(q)["measures"] == ["consumption"], q


def test_stemming_does_not_swallow_unrelated_measures():
    from app.ai.resolve import resolve
    assert resolve("what is our margin")["measures"] == ["margin"]
    assert "purchasing" in resolve("how much did we purchase")["measures"]
    assert "stock" in resolve("stock on hand")["measures"]


def test_the_brief_shows_the_computed_note():
    b = brief("how much keytruda did we consume?")
    assert "consumption_all" in b and "ISSUED FROM STORES" in b


def test_a_missing_note_never_leaks_the_word_None_into_the_brief():
    # str(None) is "None", which is truthy — so every measure WITHOUT a note (revenue,
    # purchasing, stock, quantity: most of them) appended a bare "None" line to the brief
    for q in ("what is our total revenue by hospital?",
              "how much keytruda did we consume?",
              "what is our biggest spend category?"):
        assert not [ln for ln in brief(q).splitlines() if ln.strip() == "None"], q
