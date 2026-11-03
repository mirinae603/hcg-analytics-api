# tests/test_sales_items.py — the sales_items table (revenue & margin, the sales export).
#
# The first tests run the real formulary build on the tiny made-up workbooks of
# tests/test_formulary_pipeline.py (one line's cost price left blank, consumables included) and
# check sales_items line by line: hand-worked from the invented lines, never from the build's
# own code. The last tests run on the COMMITTED tables: sales_items must add up to
# formulary_summary for every month, and its drug rows must be formulary_by_item exactly.
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from app.core.config import settings
from app.etl.formulary import config as C
from app.etl.formulary import validate as V
from app.etl.formulary.build import merge_tables, table_gaps
from app.etl.formulary.discover import BuildError
from app.etl.formulary.extract import cache_paths

from tests.test_formulary_pipeline import (DR, KR, LINES, L, _add_july, env, run,  # noqa: F401 (env: fixture)
                                           tables, write_sales)

COLUMNS = ["period", "plant", "hospital", "hospital_name", "patient_type", "payer_bucket", "material",
           "material_desc", "generic_name", "manufacturer", "category", "formulary_class", "formulary_group",
           "cost", "mrp", "qty", "lines", "cost_missing_lines", "mrp_cost_missing"]
SUMMARY_KEY = ["period", "plant", "hospital", "hospital_name", "patient_type", "category", "payer_bucket",
               "formulary_class", "formulary_group"]
BY_ITEM_KEY = ["period", "plant", "hospital", "hospital_name", "patient_type", "payer_bucket", "material",
               "material_desc", "generic_name", "manufacturer", "category", "formulary_class", "formulary_group"]
AUGUST_FILE = ("IP and OP sales", "IP & OP Consolidated File August 26.xlsx")
# The pipeline's lines, but the blank-cost line (KR IP 201005, CGHS) carries an MRP of 120, so
# the MRP behind blank costs is something to count; every other line's MRP is 1.5 x its cost.
BLANK = 8
SALES = [dict(ln, mrp=120.0) if i == BLANK else ln for i, ln in enumerate(LINES)]


def build_august(env) -> dict:
    write_sales(env["dash"].joinpath(*AUGUST_FILE), SALES)
    return run(env)


def row(si: pd.DataFrame, plant, ptype, payer, material) -> dict:
    r = si[(si["plant"] == plant) & (si["patient_type"] == ptype) & (si["payer_bucket"] == payer)
           & (si["material"] == material)]
    assert len(r) == 1, (plant, ptype, payer, material, len(r))
    return r.iloc[0].to_dict()


def test_every_line_with_mrp_consumables_and_blank_costs(env):
    assert SALES[BLANK]["cost"] is None and SALES[BLANK]["code"] == 201005
    build_august(env)
    si = tables(env)[C.SALES_ITEMS]
    assert list(si.columns) == COLUMNS                      # no doctor, bill or patient column
    assert si["lines"].dtype == "int64" and si["cost_missing_lines"].dtype == "int64"
    # all 15 lines; cost 1000 + 200 + 300 + 400 + 50 + 70 + 500 + 100 + 0 + 20 + 800 + 150 + 60 + 40
    # + 90 = 3,780 (the blank one counts 0); MRP 1.5 x that + 120 for the blank one = 5,790
    assert int(si["lines"].sum()) == 15
    assert si["cost"].sum() == pytest.approx(3780.0)
    assert si["mrp"].sum() == pytest.approx(5790.0)
    # the blank-cost line: cost 0, its MRP kept, and flagged with that MRP
    b = row(si, "HC05", "IP", "CASH(TPA)", "201005")
    assert (b["cost"], b["mrp"], b["lines"], b["cost_missing_lines"], b["mrp_cost_missing"]) == (0.0, 120.0, 1, 1, 120.0)
    assert int(si["cost_missing_lines"].sum()) == 1 and si["mrp_cost_missing"].sum() == 120.0
    # two KR OP lines of 201001, both CASH ('Blank ' and no payer): one row, 300 + 20, MRP 450 + 30
    p = row(si, "HC05", "OP", "CASH", "201001")
    assert (p["cost"], p["mrp"], p["qty"], p["lines"], p["cost_missing_lines"]) == (320.0, 480.0, 2.0, 2, 0)
    # consumables are in (formulary_by_item leaves them out): KR 70 CASH, DR 90 GOV
    c = si[si["category"] == "Consumables"]
    assert sorted(zip(c["plant"], c["payer_bucket"], c["cost"], c["mrp"])) == [
        ("HC05", "CASH", 70.0, 105.0), ("HC06", "GOV", 90.0, 135.0)]
    assert set(c["formulary_class"]) == {"NOT IN MASTER"} and set(c["formulary_group"]) == {"X"}
    assert "300001" not in set(tables(env)[C.BY_ITEM]["material"])
    # names come from the month's Item Master, as in formulary_by_item
    m = row(si, "HC05", "IP", "GOV", "101001")
    assert (m["material_desc"], m["generic_name"], m["category"]) == ("MASTER DESC 101001", "MINOR 101001",
                                                                       "Onco Drugs")


def test_it_adds_up_to_the_summary_and_holds_by_item_exactly(env):
    build_august(env)
    t = tables(env)
    si, s = t[C.SALES_ITEMS], t[C.SUMMARY]
    g = si.groupby(SUMMARY_KEY, as_index=False)[["cost", "mrp", "qty", "lines"]].sum()
    pd.testing.assert_frame_equal(g.sort_values(SUMMARY_KEY).reset_index(drop=True),
                                  s[SUMMARY_KEY + ["cost", "mrp", "qty", "lines"]]
                                  .sort_values(SUMMARY_KEY).reset_index(drop=True))
    drugs = si[si["category"].isin(["Onco Drugs", "Other Drugs"])]
    pd.testing.assert_frame_equal(drugs[BY_ITEM_KEY + ["cost", "qty", "lines"]].sort_values(BY_ITEM_KEY)
                                  .reset_index(drop=True),
                                  t[C.BY_ITEM].sort_values(BY_ITEM_KEY).reset_index(drop=True))


def test_the_validation_report_counts_it(env):
    rep = build_august(env)
    v = json.loads((env["dirs"]["data"] / C.VALIDATION_FILE).read_text())
    si = v["months"]["2026-08"]["sales_items"]
    assert si["lines"] == 15 and si["rows"] == v["tables"][C.SALES_ITEMS]["rows"]
    assert (si["cost_missing_lines"], si["mrp_cost_missing"]) == (1, 120.0)
    assert (si["cost_diff_vs_summary"], si["mrp_diff_vs_summary"], si["qty_diff_vs_summary"]) == (0.0, 0.0, 0.0)
    assert si["cost_missing_lines"] == v["months"]["2026-08"]["source"]["cost_empty_lines"]
    assert any("1 lines have a blank cost price (MRP 120.00)" in w for w in v["months"]["2026-08"]["warnings"])
    # a sales_items that lost a line, or a blank-cost count that disagrees with the read, aborts
    m = json.loads(json.dumps(rep["months"]["2026-08"]))
    assert V.abort_reasons(m) == []
    m["sales_items"]["lines"] -= 1
    m["sales_items"]["cost_missing_lines"] = 0
    m["sales_items"]["mrp_diff_vs_summary"] = m["sales_items"]["rounding_allowance"] + 1
    problems = " | ".join(V.abort_reasons(m))
    assert "sales_items holds 14 lines" in problems and "blank-cost lines" in problems
    assert "sales_items mrp differs" in problems


def test_a_partial_rebuild_cannot_publish_it_with_months_missing(env):
    # the tables were published before sales_items existed (only August)
    run(env)
    (env["dirs"]["data"] / f"{C.SALES_ITEMS}.parquet").unlink()
    before = tables_without_sales_items(env)
    _add_july(env)
    with pytest.raises(BuildError, match=r"sales_items has not been published before.*2026-08.*--from-cache"):
        run(env, months=["2026-07"])
    after = tables_without_sales_items(env)
    for name in before:                                       # nothing was published
        pd.testing.assert_frame_equal(before[name], after[name])
    assert not (env["dirs"]["data"] / f"{C.SALES_ITEMS}.parquet").exists()
    # every month rebuilt: every table covers both months
    run(env)
    t = tables(env)
    assert {name: sorted(set(t[name]["period"])) for name in (C.SUMMARY, C.SALES_ITEMS)} == {
        C.SUMMARY: ["2026-07", "2026-08"], C.SALES_ITEMS: ["2026-07", "2026-08"]}


JUL, AUG = "2026-07", "2026-08"
JULY_FILE = ("IP and OP sales", "IP and OP Sales Consolidated file July 2026.xlsx")
FROM_CACHE = "python -m app.etl.formulary --from-cache"


def published_files(env) -> dict:
    d = env["dirs"]["data"]
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())}


def test_the_guard_stops_before_any_workbook_is_read_a_dry_run_too(env):
    run(env)
    (env["dirs"]["data"] / f"{C.SALES_ITEMS}.parquet").unlink()
    _add_july(env)
    before = published_files(env)
    for dry_run in (True, False):
        with pytest.raises(BuildError, match=re.escape(
                f"sales_items has not been published before, and this run keeps {AUG} from the published "
                f"tables: every table must cover the same months, so rebuild every month: {FROM_CACHE}, "
                f"then run this build again")):
            run(env, months=[JUL], dry_run=dry_run)
        assert not cache_paths(env["dirs"]["cache"], JUL)[0].exists()     # July's workbook was never read
    assert published_files(env) == before
    assert not (env["dirs"]["cache"] / "formulary_validation.dry-run.json").exists()


def test_the_advice_names_the_months_the_line_cache_has_lost(env):
    # Jul and Aug published before sales_items existed; Jul's cached lines are gone since. A
    # --from-cache rebuild would keep Jul again, so the advice is the build that reads the workbooks
    # (and, with Jul's workbook gone too, to put it back first).
    _add_july(env)
    run(env)
    (env["dirs"]["data"] / f"{C.SALES_ITEMS}.parquet").unlink()
    for path in cache_paths(env["dirs"]["cache"], JUL):
        path.unlink()
    with pytest.raises(BuildError, match=re.escape(
            f"this run keeps {JUL} from the published tables: every table must cover the same months, so "
            f"rebuild every month: python -m app.etl.formulary (it reads every month's sales workbook again; "
            f"not in the line cache: {JUL})")):
        run(env, from_cache=True)
    july = env["dash"].joinpath(*JULY_FILE)
    moved = july.rename(env["dirs"]["raw"] / july.name)
    with pytest.raises(BuildError, match=re.escape(
            f"rebuild every month: put the sales workbook of {JUL} back in the 'IP and OP sales' folder, then "
            f"python -m app.etl.formulary (not in the line cache: {JUL})")):
        run(env, from_cache=True)
    moved.rename(july)
    run(env)                                                  # the advice followed: both months, every table
    t = tables(env)
    assert all(sorted(set(t[name]["period"])) == [JUL, AUG] for name in (C.SUMMARY, C.SALES_ITEMS))


def test_a_table_restored_from_an_older_copy_cannot_be_published_short(env):
    # sales_items is there but holds August only (an older copy put back): a July-only rebuild would
    # publish it without July. The guard compares the months, not only that the file exists.
    _add_july(env)
    run(env)
    path = env["dirs"]["data"] / f"{C.SALES_ITEMS}.parquet"
    si = pd.read_parquet(path)
    si[si["period"] == AUG].to_parquet(path, index=False)
    before = published_files(env)
    with pytest.raises(BuildError, match=re.escape(
            f"sales_items holds no rows for {JUL}, and this run keeps {JUL} from the published tables: every "
            f"table must cover the same months, so rebuild every month: {FROM_CACHE}, then run this build again")):
        run(env, months=[AUG])
    assert published_files(env) == before
    run(env, from_cache=True)
    assert sorted(set(tables(env)[C.SALES_ITEMS]["period"])) == [JUL, AUG]
    # merge_tables checks the files it merges too, for any caller
    path.unlink()
    with pytest.raises(BuildError, match="sales_items has not been published before: rebuild every month"):
        merge_tables([], data=env["dirs"]["data"])


def test_only_the_tables_that_hold_every_line_must_hold_every_month():
    every = {JUL, AUG}
    published = {name: set(every) for name in C.TABLES}
    # the brand-rule and out-of-formulary item tables can lack a month (no breach, no B or C item)
    published[C.RULE_BREACH] = {AUG}
    published[C.DOCTOR_ITEM] = {AUG}
    assert table_gaps(published, [AUG]) == ([JUL], [])
    published[C.SALES_ITEMS] = {AUG}
    assert table_gaps(published, [AUG]) == ([JUL], [f"sales_items holds no rows for {JUL}"])
    assert table_gaps(published, [JUL, AUG]) == ([], [])      # nothing kept: every month is rebuilt
    published[C.SALES_ITEMS] = None
    assert table_gaps(published, [AUG]) == ([JUL], ["sales_items has not been published before"])
    assert table_gaps({name: None for name in C.TABLES}, [AUG]) == ([], [])     # a first build


def tables_without_sales_items(env) -> dict:
    d = env["dirs"]["data"]
    return {t: pd.read_parquet(d / f"{t}.parquet") for t in C.TABLES if t != C.SALES_ITEMS}


# ---------- the committed tables ----------

_DATA = Path(settings.FORMULARY_DATA_DIR)
_real = pytest.mark.skipif(not (_DATA / f"{C.SALES_ITEMS}.parquet").exists()
                           or not (_DATA / f"{C.SUMMARY}.parquet").exists(),
                           reason="formulary tables not built — run python -m app.etl.formulary")


@_real
def test_real_sales_items_equals_the_summary_every_month():
    si = pd.read_parquet(_DATA / f"{C.SALES_ITEMS}.parquet")
    s = pd.read_parquet(_DATA / f"{C.SUMMARY}.parquet")
    assert list(si.columns) == COLUMNS
    assert sorted(set(si["period"])) == sorted(set(s["period"]))
    g = si.groupby(SUMMARY_KEY, as_index=False)[["cost", "mrp", "qty", "lines"]].sum()
    both = g.merge(s, on=SUMMARY_KEY, how="outer", suffixes=("_items", "_summary"), indicator=True)
    assert set(both["_merge"]) == {"both"}, "a summary row with no items, or items with no summary row"
    rows = si.groupby(SUMMARY_KEY).size().reindex(pd.MultiIndex.from_frame(both[SUMMARY_KEY])).to_numpy()
    for period, part in both.groupby("period"):
        # each table rounds its own rows to the paisa: allow half a paisa per row behind a cell
        allow = V.ROUNDING_PER_ROW * (rows[part.index] + 1)
        for m in ("cost", "mrp"):
            gap = (part[f"{m}_items"] - part[f"{m}_summary"]).abs().to_numpy()
            assert (gap <= allow).all(), (period, m, float(gap.max()))
        assert (part["qty_items"] == part["qty_summary"]).all(), period
        assert (part["lines_items"] == part["lines_summary"]).all(), period


@_real
def test_real_drug_rows_are_formulary_by_item_and_blank_costs_match_the_read():
    si = pd.read_parquet(_DATA / f"{C.SALES_ITEMS}.parquet")
    drugs = si[si["category"].isin(["Onco Drugs", "Other Drugs"])]
    by_item = pd.read_parquet(_DATA / f"{C.BY_ITEM}.parquet")
    pd.testing.assert_frame_equal(
        drugs[BY_ITEM_KEY + ["cost", "qty", "lines"]].sort_values(BY_ITEM_KEY).reset_index(drop=True),
        by_item[BY_ITEM_KEY + ["cost", "qty", "lines"]].sort_values(BY_ITEM_KEY).reset_index(drop=True))
    months = json.loads((_DATA / C.VALIDATION_FILE).read_text())["months"]
    blank = si.groupby("period")["cost_missing_lines"].sum()
    for period, n in blank.items():
        assert int(n) == months[period]["source"]["cost_empty_lines"], period
    assert (si["mrp_cost_missing"] <= si["mrp"] + 1e-9).all()
    assert (si["cost_missing_lines"] <= si["lines"]).all()
