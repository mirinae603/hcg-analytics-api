"""What each table DOESN'T contain, measured rather than assumed.

WHY
---
`fact_consumption` holds 11,225 of 24,931 materials — the rest are patient-billed and have
zero rows there, permanently. Asked for Keytruda's consumption the assistant queried it,
got nothing, and reported "no recorded consumption" for a drug with 2,193 units billed.

An audit of every table found that was not the only one. `fact_inventory` carries 26 of 53
hospitals: the BACC cancer centres and the Triesta labs have thousands of purchase-order
lines each and NOT ONE inventory row. So "which hospital holds the most stock" has been
answered over half the estate, with nothing saying so.

Partial coverage is not an error — it is a fact about the data. It only becomes a wrong
answer when it goes unmentioned, so this module measures it and the answer discloses it.

NOT EVERY GAP IS A DEFECT
-------------------------
`kpi_near_expiry` holds 3,997 materials because only 3,997 are near expiry; the table IS
the filter and its name says so. Verified as a clean subset of `fact_inventory` (zero
orphans), so coverage notes are suppressed for tables that are subsets of a broader one.
"""
from __future__ import annotations

import re
from functools import lru_cache

# entity kind -> (dimension table, its key column, the columns a fact table uses for it)
_UNIVERSE: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "material": ("dim_material", "material", ("material",)),
    "hospital": ("dim_plant", "plant", ("plant",)),
    # vendor is deliberately absent. dim_vendor is a MASTER LIST — 3,576 registered
    # suppliers, of which 2,251 have ever been bought from. That gap is the correct state of
    # the world, not missing data, and reporting it as a coverage hole would be a lie in the
    # opposite direction.
}

# Below this share of the universe, an aggregate over the table is materially incomplete.
_INCOMPLETE = 0.90

# Tables whose whole purpose is to be a filtered slice — their name states the filter, so
# reporting "covers only 16%" would be noise, not insight.
_BY_DESIGN = re.compile(
    r"near_expiry|stock_out|non_moving|expiry|aging|risk|forecast|outlier|variance|replenish",
    re.I)


@lru_cache(maxsize=1)
def _universe_sizes() -> dict[str, int]:
    from app.ai import warehouse
    out = {}
    for kind, (table, col, _) in _UNIVERSE.items():
        try:
            out[kind] = warehouse.con().execute(
                f'SELECT COUNT(DISTINCT "{col}") FROM "{table}"').fetchone()[0] or 0
        except Exception:
            out[kind] = 0
    return out


@lru_cache(maxsize=512)
def measure(table: str, kind: str) -> tuple[int, int, tuple[str, ...]]:
    """(present, universe, a few missing names) for one table and one entity kind."""
    from app.ai import warehouse
    if kind not in _UNIVERSE:
        return (0, 0, ())
    dim, key, candidates = _UNIVERSE[kind]
    total = _universe_sizes().get(kind, 0)
    if not total:
        return (0, 0, ())
    try:
        cols = {r[0] for r in warehouse.con().execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
            [table]).fetchall()}
    except Exception:
        return (0, 0, ())
    col = next((c for c in candidates if c in cols), None)
    if not col:
        return (0, 0, ())
    try:
        present = warehouse.con().execute(
            f'SELECT COUNT(DISTINCT "{col}") FROM "{table}" WHERE "{col}" IS NOT NULL'
        ).fetchone()[0] or 0
        # name the missing things, not their codes: "300041" tells a reader nothing
        label = {"hospital": "plant_name", "material": "material_desc"}.get(kind, key)
        missing = warehouse.con().execute(
            f'SELECT d."{label}" FROM "{dim}" d WHERE NOT EXISTS '
            f'(SELECT 1 FROM "{table}" t WHERE t."{col}" = d."{key}") LIMIT 5'
        ).fetchall()
    except Exception:
        return (0, 0, ())
    return (present, total, tuple(str(m[0]) for m in missing if m[0]))


_TABLES = re.compile(r"\b(?:FROM|JOIN)\s+\"?([A-Za-z_]\w*)\"?", re.I)


def disclosure(sql: str) -> str | None:
    """A note naming what the tables in this query do not cover, or None when they cover it.

    Appended to the evidence rather than raised as an error: unlike a subtotal exceeding its
    total, partial coverage does not make the number wrong. It makes it partial, and the
    reader has to be told which half they are looking at.
    """
    if not sql:
        return None
    notes = []
    low = sql.lower()
    for table in dict.fromkeys(t for t in _TABLES.findall(sql)):
        if _BY_DESIGN.search(table):
            continue
        for kind, (_dim, _key, cands) in _UNIVERSE.items():
            # only mention a gap in something the query actually touches — a note about
            # hospital coverage on a query that never mentions hospitals is noise
            if not any(re.search(rf"\b{c}\b", low) for c in cands):
                continue
            present, total, missing = measure(table, kind)
            if not total or present >= total * _INCOMPLETE:
                continue
            gap = total - present
            eg = (" e.g. " + ", ".join(missing[:3])) if missing else ""
            notes.append(
                f"{table} covers {present:,} of {total:,} {kind}s — {gap:,} have NO rows in "
                f"it at all{eg}. DISCLOSE this in the answer. Do NOT turn it into a filter: "
                f"the {gap:,} missing rows are still real {kind}s, and restricting the "
                f"question to the {present:,} covered ones changes what was asked. Asked how "
                f"much we spend on items never sold, an earlier answer scoped itself to the "
                f"15,171 materials that DO appear in sales — which is the exact opposite of "
                f"the question — and reported ₹6.53 Cr instead of ₹60.43 Cr.")
    if not notes:
        return None
    return "COVERAGE — state this in the answer:\n" + "\n".join(f"- {n}" for n in notes)
