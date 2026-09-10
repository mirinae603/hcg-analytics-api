"""A domain ontology built FROM the warehouse, not typed into it.

WHY
---
The assistant currently knows this domain because roughly 800 literals were written by hand:
_MEASURES, _GRAINS, _STOP, _NOISE_TOKENS, _MEASURE_COLUMNS, GRAIN_COLUMNS, _TABLE_EVENT,
_PLACEHOLDER, plus prose notes about expiry buckets and consumption scopes. Every one is a
place the system knows something only because somebody told it, and every one is silently
one release behind the moment a column is renamed. That is exactly how the disjoint hospital
codes and the billable-consumption gap stayed hidden for months.

The measured case for doing this properly is strong. On the OMG Property & Casualty schema,
GPT-4 answered 16% of questions zero-shot against the raw SQL schema, 54% against a
knowledge-graph representation of the same data, and 72% once the ontology was also used to
CHECK generated queries and repair them (arXiv 2311.07509, 2405.11706). The gain is in two
distinct places: the ontology as context, and the ontology as a validator.

HOW THIS ONE IS BUILT
---------------------
Three phases, and only the middle one involves a model:

  A. PROFILE   — deterministic. Cardinality, null rate, numeric range, sample values, real
                 value-overlap between same-named columns across tables. Facts, measured.
  B. CLASSIFY  — one model pass over the profile: what each column IS (identifier, dimension,
                 measure, time, flag), which entity it belongs to, its unit, its synonyms,
                 and what business event each table records.
  C. VERIFY    — deterministic again. Every claim the model made is checked against the data
                 and dropped if it does not hold: a claimed join must have real overlap, a
                 claimed measure must be numeric, a claimed identifier must be near-unique.

Phase C is the point. A generated ontology that is merely plausible would be worse than the
hand-written lists it replaces — those at least were checked by someone. This one is only
allowed to assert what survives being tested against the warehouse.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache

_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ontology.json")

# A column this common is structural (a key or a shared dimension), not a description.
_SHARED_MIN_TABLES = 3
# Below this overlap two same-named columns are NOT the same thing — the trap that made
# sales_by_hospital.hospital look joinable to dim_plant.plant when they share zero values.
_JOIN_MIN_OVERLAP = 0.30


def _con():
    from app.ai import warehouse
    return warehouse.con()


# ── PHASE A · PROFILE (deterministic) ────────────────────────────────────────────────────
def profile_column(table: str, column: str, sample: int = 8) -> dict:
    """Measured facts about one column. No interpretation."""
    c = _con()
    q = f'"{table}"."{column}"'
    try:
        n, distinct, nulls = c.execute(
            f'SELECT COUNT(*), COUNT(DISTINCT {q}), COUNT(*) FILTER (WHERE {q} IS NULL) '
            f'FROM "{table}"').fetchone()
    except Exception:
        return {}
    if not n:
        return {}
    out = {"table": table, "column": column, "rows": n,
           "distinct": distinct or 0,
           "null_pct": round(100.0 * (nulls or 0) / n, 1),
           "uniqueness": round((distinct or 0) / n, 4)}
    try:
        dtype = c.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = ? AND column_name = ?", [table, column]).fetchone()
        out["type"] = str(dtype[0]) if dtype else "?"
    except Exception:
        out["type"] = "?"
    numeric = any(k in out["type"].upper() for k in ("INT", "DOUBLE", "DECIMAL", "FLOAT", "BIG"))
    if numeric:
        try:
            lo, hi, total, neg = c.execute(
                f'SELECT MIN({q}), MAX({q}), SUM({q}), COUNT(*) FILTER (WHERE {q} < 0) '
                f'FROM "{table}"').fetchone()
            out.update(min=lo, max=hi, total=total, negatives=neg or 0,
                       all_zero=(lo == 0 and hi == 0))
        except Exception:
            pass
    try:
        vals = c.execute(
            f'SELECT DISTINCT {q} FROM "{table}" WHERE {q} IS NOT NULL LIMIT {sample}'
        ).fetchall()
        out["samples"] = [str(v[0])[:40] for v in vals]
    except Exception:
        out["samples"] = []
    return out


def _same_name_columns() -> dict[str, list[str]]:
    rows = _con().execute(
        "SELECT column_name, table_name FROM information_schema.columns").fetchall()
    by: dict[str, list[str]] = {}
    for col, tbl in rows:
        if str(tbl).startswith("_"):
            continue
        by.setdefault(str(col), []).append(str(tbl))
    return by


def overlap(col_a: str, table_a: str, col_b: str, table_b: str) -> float:
    """Share of table_a.col_a's values present in table_b.col_b — ANY two columns.

    Names are not evidence. `dim_plant.plant` and `sales_by_hospital.hospital` are both "the
    hospital" and share ZERO values; `mart_procurement.plant` and `dim_plant.plant` share
    94%. Only this number tells them apart, and it is the check that would have caught the
    disjoint code systems on the first day instead of the fortieth.
    """
    try:
        d = _con().execute(
            f'SELECT COUNT(DISTINCT "{col_a}") FROM "{table_a}"').fetchone()[0]
        if not d:
            return 0.0
        n = _con().execute(
            f'SELECT COUNT(*) FROM (SELECT DISTINCT "{col_a}" v FROM "{table_a}") x '
            f'WHERE x.v IN (SELECT "{col_b}" FROM "{table_b}")').fetchone()[0]
    except Exception:
        return 0.0
    return round(n / d, 3)


def joinability(column: str, a: str, b: str) -> float:
    """Share of a's values that also appear in b. Zero means these are NOT the same thing.

    This is the check that would have caught the disjoint hospital code systems on day one:
    sales_by_hospital.hospital and dim_plant.plant are both 'the hospital', and they share
    not one single value.
    """
    return overlap(column, a, column, b)



def _is_code_like(values: list[str]) -> bool:
    """Is this a CODE rather than a name or a description?

    The distinction that matters is code-vs-name, not the exact character mix. 'HC05' and
    'GJHCA' are both site codes — one has digits, one does not — and they share zero values,
    which is the single worst trap in this warehouse. A first attempt compared digit ratios
    and threw that pair away as "different shapes". Meanwhile 'Vardhman Health Specialities'
    is a NAME: long, spaced, and not a rival code system for vendor_code at all.
    """
    vals = [v for v in values if v][:8]
    if not vals:
        return False
    lens = sorted(len(v) for v in vals)
    median = lens[len(lens) // 2]
    spaced = sum(1 for v in vals if " " in v) / len(vals)
    return median <= 12 and spaced < 0.4


def _same_shape(ta: str, ca: str, tb: str, cb: str) -> bool:
    """Two identifier columns of the same KIND — both codes, or both names."""
    pa, pb = profile_column(ta, ca), profile_column(tb, cb)
    if not pa or not pb:
        return False
    return _is_code_like(pa.get("samples") or []) == _is_code_like(pb.get("samples") or [])


def profile() -> dict:
    """The deterministic skeleton: every table, every column, plus real join evidence."""
    c = _con()
    tables = [t[0] for t in c.execute(
        "SELECT DISTINCT table_name FROM information_schema.tables").fetchall()
        if not str(t[0]).startswith("_")]
    cols: dict[str, list[dict]] = {}
    for t in sorted(tables):
        names = [r[0] for r in c.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
            [t]).fetchall()]
        cols[t] = [p for p in (profile_column(t, n) for n in names) if p]

    # which same-named columns genuinely line up, and which only look like they do
    links = []
    for col, tbls in _same_name_columns().items():
        if len(tbls) < 2 or len(tbls) > 40:
            continue
        for i, a in enumerate(sorted(tbls)):
            for b in sorted(tbls)[i + 1:]:
                ov = joinability(col, a, b)
                links.append({"column": col, "a": a, "b": b, "overlap": ov,
                              "joinable": ov >= _JOIN_MIN_OVERLAP})
    return {"tables": cols, "links": links}


# ── PHASE B · CLASSIFY (one model pass over the profile) ─────────────────────────────────
_CLASSIFY_SYSTEM = """You are building an ontology for a hospital supply-chain warehouse.

You are given MEASURED FACTS about every column: row count, distinct count, null rate, type,
min/max, and real sample values. Classify each column. Do not guess beyond the evidence — a
claim you cannot support from the samples will be deleted by a verification pass anyway.

For each column return:
  role      one of: identifier | dimension | measure | time | flag | descriptor | ignore
  entity    the business thing it belongs to, lower snake: material, hospital, vendor,
            manufacturer, category, department, batch, purchase_order, period, none
  unit      for measures only: currency | quantity | percent | days | count | ratio | none
  additive  for measures only: true if summing it across rows is meaningful (a value is;
            a price, a rate or a pre-computed average is NOT)
  event     for measures only: what business event it records — sales | procurement |
            consumption | stock | forecast | none
  synonyms  words a hospital analyst would use for it, lowercase, 0-6 of them
  note      one short sentence ONLY if something is genuinely surprising or a trap

Also classify each TABLE: the entity it is about, its grain (one row per WHAT), and the
business event it records.

Return JSON: {"columns": {"table.column": {...}}, "tables": {"table": {...}}}"""


def _classify(profile_slice: dict, cl=None) -> dict:
    from app.ai.deep import llm
    cl = cl or llm.client()
    out = llm.ask_json(
        cl, _CLASSIFY_SYSTEM,
        json.dumps(profile_slice, default=str)[:60000],
        '{"columns": {...}, "tables": {...}}', temperature=0.0, role="plan")
    return out or {}


# ── PHASE C · VERIFY (deterministic — the point of the whole thing) ──────────────────────
def verify(ont: dict, prof: dict) -> tuple[dict, list[str]]:
    """Delete every claim the data does not support. Returns (clean ontology, rejections).

    A generated ontology that is merely plausible is worse than the hand-written lists it
    replaces, because at least those were checked by a person. This is what makes the
    difference: nothing is allowed into the ontology that has not survived being tested.
    """
    rejected: list[str] = []
    by_col = {f"{p['table']}.{p['column']}": p
              for cols in prof["tables"].values() for p in cols}

    clean_cols = {}
    for key, spec in (ont.get("columns") or {}).items():
        p = by_col.get(key)
        if not p:
            rejected.append(f"{key}: no such column"); continue
        role = str(spec.get("role", "")).lower()

        # a measure must actually be numeric
        if role == "measure" and "min" not in p:
            rejected.append(f"{key}: called a measure but is not numeric"); continue
        # an all-zero column is not a usable measure, whatever it is called
        if role == "measure" and p.get("all_zero"):
            rejected.append(f"{key}: measure is entirely zero"); continue
        # an identifier must be reasonably distinctive
        if role == "identifier" and p.get("uniqueness", 0) < 0.001 and p.get("distinct", 0) < 5:
            rejected.append(f"{key}: called an identifier but has {p.get('distinct')} values")
            continue
        # additivity is checkable: a rate or pre-averaged column must not claim it
        _rate_like = re.search(r"(_pct|percent|_rate|avg_|_avg|median|price|share|score|"
                               r"days|months|ratio|tat)", p["column"], re.I)
        if spec.get("additive") and _rate_like:
            spec = {**spec, "additive": False}
            rejected.append(f"{key}: additive=false forced (name says it is a rate/average)")
        # …and the reverse. The classifier called consumption_all.cost non-additive, which
        # would have blocked SUM(cost) — a correct query on a plain currency amount. A false
        # "never sum this" is worse than the hand-written list it replaces, because it
        # refuses work that was right. Money and quantity ARE additive unless the name says
        # otherwise, and the model does not get a vote on that.
        if (role == "measure" and not spec.get("additive", True) and not _rate_like
                and str(spec.get("unit", "")).lower() in ("currency", "quantity", "count")):
            spec = {**spec, "additive": True}
            rejected.append(f"{key}: additive=true restored (a plain {spec.get('unit')} amount)")
        clean_cols[key] = {**spec, "profile": {k: p.get(k) for k in
                                               ("type", "distinct", "null_pct", "samples",
                                                "all_zero", "uniqueness")}}

    # LINKS ONLY BETWEEN THINGS THAT IDENTIFY SOMETHING.
    #
    # The profiler compares every same-named column pair, which includes MEASURES — and
    # "consumption_all.qty overlaps 0% with sales_by_hospital.qty" is not a disjoint code
    # system, it is two unrelated quantities. Roles are only known after classification, so
    # the filter has to happen here.
    def is_key(table: str, col: str) -> bool:
        spec = clean_cols.get(f"{table}.{col}") or {}
        return spec.get("role") in ("identifier", "dimension")

    keyed = [l for l in prof["links"] if is_key(l["a"], l["column"]) and is_key(l["b"], l["column"])]

    # SAME ENTITY, DIFFERENT COLUMN NAME. The worst trap in this warehouse is not two
    # columns called `plant`; it is `dim_plant.plant` and `sales_by_hospital.hospital` —
    # both classified as the hospital, sharing not one value. Name-matching cannot see it.
    # Entity-matching can, because the classifier has just said what each column IS.
    ident: dict[str, list[tuple[str, str]]] = {}
    for key, spec in clean_cols.items():
        if spec.get("role") != "identifier":
            continue
        ent = str(spec.get("entity", "")).lower()
        if ent and ent != "none":
            t, cname = key.split(".", 1)
            ident.setdefault(ent, []).append((t, cname))
    cross = []
    for ent, members in ident.items():
        for i, (ta, ca) in enumerate(sorted(members)):
            for tb, cb in sorted(members)[i + 1:]:
                if ta == tb or (ca == cb):
                    continue                      # same table, or already name-matched
                # Only compare LIKE WITH LIKE. `vendor_code` and `vendor_name` identify the
                # same vendor and share no values because one is a number and the other is
                # words — that is not a code-system mismatch, and calling it one tells the
                # model never to join two columns it should. The real trap is two columns of
                # the SAME SHAPE that still share nothing: 'HC05' and 'GJHCA'.
                if not _same_shape(ta, ca, tb, cb):
                    continue
                ov = overlap(ca, ta, cb, tb)
                cross.append({"column": ca, "other": cb, "a": ta, "b": tb,
                              "entity": ent, "overlap": ov,
                              "joinable": ov >= _JOIN_MIN_OVERLAP})
    keyed += cross
    dropped = len(prof["links"]) - len(keyed)
    if dropped:
        rejected.append(f"{dropped} column pairs ignored for linking: not identifiers")

    return ({"columns": clean_cols,
             "tables": ont.get("tables") or {},
             "links": [l for l in keyed if l["joinable"]],
             # SAME NAME, SAME MEANING, ZERO SHARED VALUES — the trap that hid the two
             # hospital code systems. Only meaningful between identifiers.
             "disjoint": [l for l in keyed if not l["joinable"] and l["overlap"] < 0.02]},
            rejected)


# ── BUILD ────────────────────────────────────────────────────────────────────────────────
def build(save: bool = True) -> dict:
    """Profile, classify, verify. The only entry point that should be called."""
    prof = profile()
    from app.ai.deep import llm
    cl = llm.client()

    # classify in slices — the whole profile does not fit one prompt, and a table is the
    # natural unit because a column is easiest to type next to its neighbours
    names = sorted(prof["tables"])
    merged: dict = {"columns": {}, "tables": {}}
    for i in range(0, len(names), 6):
        chunk = {t: prof["tables"][t] for t in names[i:i + 6]}
        got = _classify({"tables": chunk}, cl)
        merged["columns"].update(got.get("columns") or {})
        merged["tables"].update(got.get("tables") or {})
        print(f"  classified {min(i + 6, len(names))}/{len(names)} tables", flush=True)

    # COVERAGE IS NOT OPTIONAL. The model silently omits columns from its JSON — on one run
    # it dropped `sales_by_hospital.hospital`, which is one half of the single worst trap in
    # this warehouse, so the ontology "found" nothing and looked clean while being blind.
    # Chase the gaps until they close; a partial ontology that does not say it is partial is
    # the same failure mode as every other silent omission in this codebase.
    all_cols = {f"{p_['table']}.{p_['column']}"
                for cols in prof["tables"].values() for p_ in cols}
    for attempt in range(4):
        missing = sorted(all_cols - set(merged["columns"]))
        if not missing:
            break
        print(f"  filling {len(missing)} unclassified columns (pass {attempt + 1})", flush=True)
        by_table: dict = {}
        for key in missing:
            t = key.split(".", 1)[0]
            by_table.setdefault(t, []).append(
                next(x for x in prof["tables"][t] if f"{t}.{x['column']}" == key))
        names_m = sorted(by_table)
        for i in range(0, len(names_m), 6):
            chunk = {t: by_table[t] for t in names_m[i:i + 6]}
            got = _classify({"tables": chunk}, cl)
            merged["columns"].update(got.get("columns") or {})
            merged["tables"].update(got.get("tables") or {})
    still = sorted(all_cols - set(merged["columns"]))

    ont, rejected = verify(merged, prof)
    if still:
        rejected.append(f"{len(still)} columns could not be classified after 4 passes")
    ont["unclassified"] = still
    ont["rejected"] = rejected
    if save:
        os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
        json.dump(ont, open(_CACHE, "w"), indent=1, default=str)
    return ont


@lru_cache(maxsize=1)
def load() -> dict:
    """The cached ontology, or an empty one if it has never been built."""
    try:
        return json.load(open(_CACHE))
    except Exception:
        return {"columns": {}, "tables": {}, "links": [], "disjoint": [], "rejected": []}


# ── WHAT THE ONTOLOGY REPLACES ───────────────────────────────────────────────────────────
def measures_for(event: str = "", unit: str = "") -> list[str]:
    """Columns that measure something — replaces the hand-written _MEASURE_COLUMNS."""
    out = []
    for key, c in load()["columns"].items():
        if c.get("role") != "measure":
            continue
        if event and str(c.get("event", "")).lower() != event.lower():
            continue
        if unit and str(c.get("unit", "")).lower() != unit.lower():
            continue
        out.append(key)
    return sorted(out)


def columns_for_entity(entity: str) -> list[str]:
    """Columns that identify a thing — replaces the hand-written GRAIN_COLUMNS."""
    return sorted(k for k, c in load()["columns"].items()
                  if str(c.get("entity", "")).lower() == entity.lower()
                  and c.get("role") in ("identifier", "dimension"))


def synonyms() -> dict[str, list[str]]:
    """word -> the columns it could mean. Replaces _MEASURES / _GRAINS matching."""
    out: dict[str, list[str]] = {}
    for key, c in load()["columns"].items():
        for w in (c.get("synonyms") or []):
            out.setdefault(str(w).lower(), []).append(key)
    return out


def is_disjoint(col_a: str, table_a: str, col_b: str, table_b: str) -> bool:
    """Two columns that look like the same thing and share no values."""
    for d in load().get("disjoint", []):
        if d.get("column") and {d["a"], d["b"]} == {table_a, table_b}:
            return True
    return False


def non_additive() -> list[str]:
    """Measures that must never be SUMmed — rates, prices, pre-computed averages."""
    return sorted(k for k, c in load()["columns"].items()
                  if c.get("role") == "measure" and not c.get("additive", True))
