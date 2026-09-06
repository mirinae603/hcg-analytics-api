# tests/test_resolution_coverage.py — every dimension must be reachable by the words a
# clinician actually uses. A 2,580-probe audit across materials, manufacturers, vendors,
# hospitals, categories and departments found ~390 failures; these are the classes.
from __future__ import annotations

import pytest

from app.ai.resolve import _FAMILY_MAX, _MAX_DF, resolve, spelling_suggestions


def _hits(q):
    r = resolve(q)
    return ([e["text"] for e in r["entities"]]
            + [f"{f['token']}:{f['kind']}:{f['n']}" for f in r["families"]])


# ── site codes ──────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("code", ["GJHCA", "MHHIK", "APONG"])
def test_sales_side_hospital_codes_resolve(code):
    # sales_by_hospital uses its own code system, disjoint from dim_plant, and it was never
    # indexed — so all 23 sites answered "no such thing"
    assert any(code in h for h in _hits(f"How much revenue did {code} generate?")), code


# ── product families ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("word,least", [("vicryl", 20), ("pentasure", 5)])
def test_a_brand_naming_many_skus_still_resolves(word, least):
    # _FAMILY_MAX was 12, which rejected these for being too successful: "vicryl" names 77
    # suture SKUs and matched nothing at all
    hits = _hits(f"how much {word} do we have in stock")
    assert hits, word
    n = int(hits[0].rsplit(":", 1)[1])
    assert n >= least, (word, n)


def test_the_family_cap_follows_the_index_not_a_guess():
    # anything above _MAX_DF is already dropped as non-identifying, so a surviving token is
    # distinctive by construction — a second, tighter cap only discards good matches
    assert _FAMILY_MAX == _MAX_DF


# ── classes ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("word", ["sutures", "catheters", "syrups", "injections", "tablets"])
def test_a_class_word_resolves_to_its_category(word):
    hits = _hits(f"how many {word} do we stock")
    assert hits and any(":category:" in h for h in hits), (word, hits)


def test_a_dosage_form_word_is_not_silenced_in_the_question():
    # _NOISE_TOKENS strips "SYRUP" from PRODUCT NAMES; applying it to the question made
    # "how many syrups do we stock" unanswerable while M119-SYRUPS exists
    assert _hits("how many syrups do we stock")


# ── big manufacturers ───────────────────────────────────────────────────────────────────
def test_a_manufacturer_is_not_hidden_by_being_large():
    # "ROCHE" appears in 272 material descriptions (every "-ROCHE" suffix) and in exactly
    # one manufacturer value. A global document-frequency cut deleted BOTH postings, so the
    # bigger a manufacturer got, the more invisible it became.
    assert any("ROCHE" in h.upper() for h in _hits("what did we buy from Roche"))


@pytest.mark.parametrize("typo,meant", [("Rochee", "roche"), ("Relianc", "reliance"),
                                        ("Astrazenca", "astrazeneca")])
def test_manufacturer_typos_are_recognised(typo, meant):
    q = f"what were sales for {typo} last quarter"
    assert any(s["meant"] == meant for s in spelling_suggestions(q, resolve(q))), typo


# ── a verb must not masquerade as a name ────────────────────────────────────────────────
def test_a_measure_verb_does_not_bind_as_an_entity():
    # "consume" bound as a NAME, which then suppressed the misspelling check for the real
    # item, because that only runs when nothing resolved
    assert not _hits("how much did we consume")


def test_a_full_product_name_resolves_alongside_a_measure_verb():
    assert any("ATORLIP" in h.upper() for h in _hits("how much ATORLIP 20MG TAB did we consume"))


# ── the traps this resolver exists to prevent must still hold ───────────────────────────
def test_msd_still_resolves_to_msd_and_not_to_a_sticker():
    hits = _hits("which items do we buy from MSD?")
    assert any(h.upper().startswith("MSD") for h in hits)
    assert not any("STICKER" in h.upper() for h in hits)


def test_high_value_still_binds_nothing():
    assert not _hits("Which high-value drugs have the worst margins?")


def test_tablets_still_binds_the_class_not_the_company():
    hits = _hits("how many tablets do we stock?")
    assert all(":category:" in h for h in hits), hits
