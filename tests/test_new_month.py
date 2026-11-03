# tests/test_new_month.py — what the builds do when HCG sends a new month without everything:
# a file of the wrong type in a source folder, a GRN month whose Item Master has not come, a
# month built without HCG's own out-of-formulary working. Each one WARNS — in the build log and
# in the validation report's `warnings` — and the build carries on: prices and generic vs branded
# are built for every GRN month, out of formulary for every month it can be. The purchase build
# runs on the made-up inputs of tests/test_purchase_oof.py (Aug with HCG's worksheet, Sep
# without) and, end to end, on tiny workbooks written here; the spend build on
# tests/test_spend_pipeline.py's. Every directory is a tmp_path: nothing touches app/data, the
# committed reference files or the real caches.
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import Workbook

from app.etl.formulary import config as FC
from app.etl.formulary.discover import BuildError, SkippedFile
from app.etl.purchase import build as B
from app.etl.purchase import config as C
from app.etl.purchase import generic as Gn
from app.etl.purchase import grn as G
from app.etl.purchase import masters as M
from app.etl.purchase import oof as O
from app.etl.spend import build as SB
from tests.test_formulary_pipeline import DASH, write_master
from tests.test_purchase_oof import AUG, SEP, make_inputs, read_table
from tests.test_purchase_readers import (GRN_DIR, HASH, JULY, JUNE, MASTER_ROWS, OOF_DIR, OOF_MASTER_DIR,
                                         OOF_MASTER_FILE, write_grn)
from tests.test_spend_pipeline import env, write_scheme_list  # noqa: F401  (env is a fixture)

JUN, JUL = "2026-06", "2026-07"
WAIT_SEP = "Out of formulary for Sep-26 waits for its Item Master; prices and generic are built"
EST_SEP = ("Estimated: HCG's working for Sep-26 was not received; suggestions come from HCG's other months "
           "and lists, then the app's match (Estimated)")
XLS = "an Excel 97-2003 workbook (.xls), which the build does not read: open it and save it as .xlsx"
LOST_SEP = (f"{SEP}: out of formulary for Sep-26 is published, built on the Item Master 'IM Sep.xlsx', "
            "which is no longer in the Item Master folder; put it back (a published month is rebuilt only "
            "on its own Item Master), or rebuild only the other months with --months")


def report(data: Path) -> dict:
    return json.loads((data / C.VALIDATION_FILE).read_text())


def without_item_master(inputs: B.Inputs, period: str) -> B.Inputs:
    """The inputs as if `period`'s Item Master had not come (its GRN has)."""
    inputs.item_masters = {p: f for p, f in inputs.item_masters.items() if p != period}
    inputs.item_master_files = {p: f for p, f in inputs.item_master_files.items() if p != period}
    return inputs


# --- which months wait for an Item Master -------------------------------------------------------
def test_months_from_the_first_item_master_on_wait_for_their_own():
    grn = ["2025-12", "2026-05", JUN, JUL, AUG, SEP]
    # Dec-May come before the first Item Master: HCG's purchase history, prices and generic only
    assert O.waiting_for_item_master(grn, {JUN, AUG}) == [JUL, SEP]          # a gap waits too
    assert O.waiting_for_item_master(grn, {JUN, JUL, AUG, SEP}) == []
    assert O.waiting_for_item_master(grn, {}) == grn                          # no Item Master at all: every month
    assert O.waiting_for_item_master([], {AUG}) == []
    assert O.waiting_note(SEP) == WAIT_SEP and O.estimated_note(SEP) == EST_SEP


# --- the purchase build ------------------------------------------------------------------------
def test_a_grn_month_without_its_item_master_waits_and_prices_and_generic_are_built(tmp_path):
    logs: list = []
    inputs = without_item_master(make_inputs(), SEP)
    inputs.log = logs.append
    data = tmp_path / "data"
    B.run(inputs, data=data, cache=tmp_path / "cache")                       # a warning, not an abort
    assert set(read_table(data)["period"]) == {AUG}                           # no Sep out-of-formulary row
    for name in (C.PRICES, C.GENERIC):                                        # Sep's purchases are built
        assert set(read_table(data, name)["period"]) == {AUG, SEP}, name
    v = report(data)
    assert v["months"][SEP]["oof"] == {"item_master": None, "waiting_for_item_master": True, "warnings": [WAIT_SEP]}
    assert "prices" in v["months"][SEP] and "generic" in v["months"][SEP]
    assert v["oof"]["months"] == [AUG] and v["warnings"] == [WAIT_SEP]
    assert f"[purchase] warning: {WAIT_SEP}" in logs


def test_a_month_built_without_hcgs_working_is_estimated(tmp_path):
    logs: list = []
    inputs = make_inputs()                       # Aug from HCG's worksheet, Sep from sources 2-5
    inputs.log = logs.append
    data = tmp_path / "data"
    B.run(inputs, data=data, cache=tmp_path / "cache")
    v = report(data)
    aug, sep = v["months"][AUG]["oof"], v["months"][SEP]["oof"]
    assert (aug["hcg_working"], aug["estimated"], aug["warnings"]) == (True, False, [])
    assert (sep["hcg_working"], sep["estimated"], sep["warnings"]) == (False, True, [EST_SEP])
    assert v["warnings"] == [EST_SEP]
    assert f"[purchase] warning: {EST_SEP}" in logs
    # the API reads it off the published table: a month is HCG's when any row is HCG's sheet
    assert O.hcg_working_months(read_table(data)) == {AUG}


def test_an_emailed_month_published_without_its_workbook_says_both(tmp_path):
    rep = B.run(make_inputs(sheet=False, without_sheet={AUG}), data=tmp_path / "data", cache=tmp_path)
    aug = rep["months"][AUG]["oof"]
    assert aug["estimated"] and aug["warnings"][0] == O.estimated_note(AUG)
    assert "worksheet is missing" in aug["warnings"][1]
    assert rep["warnings"] == aug["warnings"] + [EST_SEP]


def test_a_month_before_the_first_item_master_is_history_not_waiting(tmp_path):
    # Aug without an Item Master while Sep has one: Aug is before the first, so it neither waits
    # nor warns (the real Dec-25..May-26 GRN months); its prices and generic are built
    data = tmp_path / "data"
    B.run(without_item_master(make_inputs(), AUG), data=data, cache=tmp_path)
    v = report(data)
    assert "oof" not in v["months"][AUG] and v["warnings"] == [EST_SEP]
    assert set(read_table(data)["period"]) == {SEP}
    assert set(read_table(data, C.PRICES)["period"]) == {AUG, SEP}


def test_rebuilding_one_month_keeps_the_others_warnings(tmp_path):
    data = tmp_path / "data"
    B.run(without_item_master(make_inputs(), SEP), data=data, cache=tmp_path)
    B.run(without_item_master(make_inputs(periods=(AUG,)), SEP), data=data, cache=tmp_path)
    v = report(data)                             # Sep was not rebuilt: it still waits, and says so
    assert v["months"][SEP]["oof"]["waiting_for_item_master"] and v["warnings"] == [WAIT_SEP]


# --- a published month whose Item Master has left the folder -----------------------------------
def published(data: Path) -> dict:
    """Every purchase table as published, and the validation file's text."""
    return {**{name: read_table(data, name) for name in C.TABLES},
            "report": (data / C.VALIDATION_FILE).read_text()}


def in_month(df: pd.DataFrame, period: str) -> pd.DataFrame:
    return df[df["period"] == period].reset_index(drop=True)


def test_a_published_month_whose_item_master_has_gone_stops_the_build(tmp_path):
    # Aug and Sep are published, each on its own Item Master; then Sep's master leaves the folder
    # (deleted, renamed, moved into a sub-folder). Sep is not a new month waiting for one: built
    # on, its published rows would stay while its report said it waits, and its prices and generic
    # would be rebuilt on Aug's master. The build stops before building anything, a dry run too.
    data = tmp_path / "data"
    B.run(make_inputs(), data=data, cache=tmp_path)
    before = published(data)
    for dry_run in (True, False):
        with pytest.raises(BuildError) as e:
            B.run(without_item_master(make_inputs(), SEP), data=data, cache=tmp_path, dry_run=dry_run)
        assert str(e.value).splitlines() == ["an Item Master is missing, nothing published:", f"  {LOST_SEP}"]
    after = published(data)
    assert after.pop("report") == before["report"]
    for name, df in after.items():
        pd.testing.assert_frame_equal(df, before[name])
    # the months that have their master can still be rebuilt: Sep's rows and report stay as published
    B.run(without_item_master(make_inputs(periods=(AUG,)), SEP), data=data, cache=tmp_path)
    v = report(data)
    assert v["months"][SEP]["oof"]["item_master"]["file"] == "IM Sep.xlsx" and v["warnings"] == [EST_SEP]
    for name in (C.OOF, C.PRICES, C.GENERIC):
        pd.testing.assert_frame_equal(in_month(read_table(data, name), SEP), in_month(before[name], SEP))
    # the master put back, Sep builds as before
    B.run(make_inputs(), data=data, cache=tmp_path)
    for name, df in published(data).items():
        if name != "report":
            pd.testing.assert_frame_equal(df, before[name])


def test_a_lost_first_item_master_stops_the_build_too(tmp_path):
    # Aug's master gone while Sep's stays: Aug now comes before the first Item Master, which would
    # make it history, but it was published on its own master, so it stops all the same. With no
    # report to name the file, the month is still named.
    data = tmp_path / "data"
    B.run(make_inputs(), data=data, cache=tmp_path)
    with pytest.raises(BuildError, match=re.escape(f"{AUG}: out of formulary for Aug-26 is published, built "
                                                   f"on the Item Master 'IM Aug.xlsx', which is no longer")):
        B.run(without_item_master(make_inputs(), AUG), data=data, cache=tmp_path)
    (data / C.VALIDATION_FILE).unlink()
    with pytest.raises(BuildError, match=re.escape(f"{AUG}: out of formulary for Aug-26 is published, built "
                                                   f"on its own Item Master, which is no longer")):
        B.run(without_item_master(make_inputs(), AUG), data=data, cache=tmp_path)
    # a month never published that has no master still only waits
    assert O.lost_item_masters([AUG, SEP], {AUG}, {AUG: "IM Aug.xlsx"}) == []


def test_files_not_read_are_in_the_report(tmp_path):
    inputs = make_inputs()
    inputs.skipped = [SkippedFile("Item Master Sheet", "Item Master Sep 26.xls", XLS)]
    rep = B.run(inputs, data=tmp_path / "data", cache=tmp_path)
    assert rep["warnings"] == [f"Item Master Sheet/Item Master Sep 26.xls is not read: {XLS}", EST_SEP]


# --- the purchase build's folders --------------------------------------------------------------
def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a workbook")         # discovery never opens a file it does not read
    return path


def test_the_purchase_folders_name_the_files_they_do_not_read(tmp_path):
    raw = tmp_path / "raw"
    grn = raw / HASH / GRN_DIR
    touch(grn / "05.Consolidated GRN Aug 2026.xlsx")
    for name in ("06.Consolidated GRN Sep 2026.xls", "GRN Sep 2026.csv", "notes.pdf", "README",
                 ".~lock.05.Consolidated GRN Aug 2026.xlsx#", "~$06.Consolidated GRN Sep 2026.xlsx", ".DS_Store"):
        touch(grn / name)
    skipped: list = []
    assert [p.name for p in G.grn_files(raw, skipped)] == ["05.Consolidated GRN Aug 2026.xlsx"]
    assert [(s.folder, s.name) for s in skipped] == [(GRN_DIR, "06.Consolidated GRN Sep 2026.xls"),
                                                    (GRN_DIR, "GRN Sep 2026.csv"), (GRN_DIR, "README"),
                                                    (GRN_DIR, "notes.pdf")]      # lock and system files: no word
    assert [s.reason for s in skipped] == [
        XLS, FC.SKIPPED_SUFFIX_REASONS[".csv"], "not an Excel workbook (no file type): only .xlsx and .xlsb files are read",
        "not an Excel workbook (.pdf): only .xlsx and .xlsb files are read"]
    # a GRN folder holding only an .xls stops with the file named
    (grn / "05.Consolidated GRN Aug 2026.xlsx").unlink()
    with pytest.raises(BuildError, match=r"no GRN workbooks \(.xlsx or .xlsb\) in .*not read: "
                                         r"'06.Consolidated GRN Sep 2026.xls' \(an Excel 97-2003"):
        G.read_grn_files(raw=raw, cache=tmp_path / "cache", log=lambda *_: None)


def test_the_oof_folders_name_the_files_they_do_not_read(tmp_path):
    raw = tmp_path / "raw"
    oof = raw / HASH / OOF_DIR
    touch(oof / "Out of Formulary Brand Margin loss Spend anlayiss August 26.xlsx")
    touch(oof / "Out of Formulary Spend Analysis September 26 WS.xls")
    touch(oof / OOF_MASTER_DIR / "Copy of Out of formualry vs suggested formualry brand master-2026-27.xls")
    skipped: list = []
    assert [p.name for p in M.find_hcg_workbooks(raw, skipped)] == [
        "Out of Formulary Brand Margin loss Spend anlayiss August 26.xlsx"]
    assert [str(s) for s in skipped] == [f"{OOF_DIR}/Out of Formulary Spend Analysis September 26 WS.xls is not read: {XLS}"]
    with pytest.raises(BuildError, match=r"expected one OOF master workbook \(.xlsx or .xlsb\) .*found 0; not "
                                         r"read: 'Copy of Out of formualry .*2026-27.xls' \(an Excel 97-2003"):
        M.find_oof_master(raw, skipped)
    assert [s.folder for s in skipped] == [OOF_DIR, OOF_MASTER_DIR]


def sub_folder(parent: str) -> str:
    return (f"a sub-folder (the build reads only the files directly in '{parent}'): "
            f"move its files up into '{parent}'")


def test_sub_folders_of_the_purchase_folders_are_named(tmp_path):
    # a GRN or an OOF working put in a sub-folder of its own is not read: named, never passed over.
    # The OOF folder's master sub-folder is read (no word); Office temp and macOS zip files are not HCG's.
    raw = tmp_path / "raw"
    grn = raw / HASH / GRN_DIR
    touch(grn / "05.Consolidated GRN Aug 2026.xlsx")
    touch(grn / "Sep 26" / "06.Consolidated GRN Sep 2026.xlsx")
    for temp in ("~WRL0001.tmp", "05.Consolidated GRN Aug 2026.xlsx.tmp", "__MACOSX/._06.Consolidated GRN.xlsx"):
        touch(grn / temp)
    skipped: list = []
    assert [p.name for p in G.grn_files(raw, skipped)] == ["05.Consolidated GRN Aug 2026.xlsx"]
    assert [str(s) for s in skipped] == [f"{GRN_DIR}/Sep 26 is not read: {sub_folder(GRN_DIR)}"]
    oof = raw / HASH / OOF_DIR
    touch(oof / "Out of Formulary Brand Margin loss Spend anlayiss August 26.xlsx")
    touch(oof / "Sep 26" / "Out of Formulary Spend Analysis September 26 WS.xlsx")
    touch(oof / OOF_MASTER_DIR / OOF_MASTER_FILE)
    touch(oof / OOF_MASTER_DIR / "old" / "brand master-2025-26.xlsx")
    skipped = []
    assert [p.name for p in M.find_hcg_workbooks(raw, skipped)] == [
        "Out of Formulary Brand Margin loss Spend anlayiss August 26.xlsx"]
    assert M.find_oof_master(raw, skipped).name == OOF_MASTER_FILE
    assert [str(s) for s in skipped] == [f"{OOF_DIR}/Sep 26 is not read: {sub_folder(OOF_DIR)}",
                                         f"{OOF_MASTER_DIR}/old is not read: {sub_folder(OOF_MASTER_DIR)}"]


# --- end to end: September's GRN arrives with an .xls Item Master ---------------------------------
@pytest.fixture()
def new_month(tmp_path):
    """Jun and Jul GRN workbooks; Jun's Item Master (100001 out of formulary) and Jul's sent as .xls;
    HCG's OOF master but no OOF working (Jul's sent as .xls); the Main Sheet; a reference folder
    with the scope and the generic definition and no email figures."""
    raw = tmp_path / "raw" / HASH
    (raw / GRN_DIR).mkdir(parents=True)
    write_grn(raw / GRN_DIR / "03.Consolidated GRN June 2026.xlsx", JUNE)
    write_grn(raw / GRN_DIR / "04.Consolidated GRN July 2026.xlsx", JULY)
    touch(raw / GRN_DIR / "GRN notes.pdf")
    im = raw / DASH / "Item Master Sheet"
    im.mkdir(parents=True)
    write_master(im / "Item master for formulary complience June 26.xlsx",
                 {100001: ("OUT OF FORMULARY", None), 200005: ("FORMULARY", "P"), 200006: ("FORMULARY", "P")})
    touch(im / "Item master for formulary complience July 26.xls")
    write_scheme_list(raw / DASH / "Copy of Onco Formulary -Spend Analysis Main Sheet_11.08.26.xlsx")
    touch(raw / DASH / "Payer Master.csv")
    oof = raw / OOF_DIR
    (oof / OOF_MASTER_DIR).mkdir(parents=True)
    wb = Workbook()
    for r in MASTER_ROWS:
        wb.active.append(r)
    wb.save(oof / OOF_MASTER_DIR / OOF_MASTER_FILE)
    touch(oof / "Out of Formulary Spend Analysis July 26 WS.xls")
    ref = tmp_path / "reference"
    ref.mkdir()
    for name in (C.SCOPE_FILE, Gn.ORIGINATOR_FILE, Gn.OVERRIDES_FILE):
        shutil.copy(C.reference_dir() / name, ref / name)
    return dict(raw=tmp_path / "raw", cache=tmp_path / "cache", reference=ref, data=tmp_path / "data")


def test_a_new_month_with_an_xls_item_master_is_built_with_warnings(new_month):
    logs: list = []
    e = new_month
    inputs = B.load_inputs(raw=e["raw"], cache=e["cache"], reference=e["reference"], log=logs.append)
    not_read = [str(s) for s in inputs.skipped]
    assert not_read == [
        f"{GRN_DIR}/GRN notes.pdf is not read: not an Excel workbook (.pdf): only .xlsx and .xlsb files are read",
        f"Item Master Sheet/Item master for formulary complience July 26.xls is not read: {XLS}",
        f"{OOF_DIR}/Out of Formulary Spend Analysis July 26 WS.xls is not read: {XLS}",
        f"{DASH}/Payer Master.csv is not read: {FC.SKIPPED_SUFFIX_REASONS['.csv']}"]
    assert all(f"[purchase] warning: {s}" in logs for s in not_read)
    assert (inputs.periods, list(inputs.item_masters), inputs.hcg) == ([JUN, JUL], [JUN], {})
    rep = B.run(inputs, data=e["data"], cache=e["cache"])
    # Jun: out of formulary from sources 2-5 (no HCG working); Jul waits for its Item Master
    assert rep["warnings"] == not_read + [O.estimated_note(JUN), O.waiting_note(JUL)]
    assert report(e["data"])["warnings"] == rep["warnings"]
    assert set(read_table(e["data"])["period"]) == {JUN}
    assert set(read_table(e["data"], C.PRICES)["period"]) == {JUN, JUL}
    assert set(read_table(e["data"], C.GENERIC)["period"]) == {JUN, JUL}


def test_the_list_command_names_the_waiting_months_and_the_files_not_read(new_month):
    e = new_month
    B.load_inputs(raw=e["raw"], cache=e["cache"], reference=e["reference"], log=lambda *_: None)   # fills the cache
    logs: list = []
    B.list_files(raw=e["raw"], cache=e["cache"], data=e["data"], reference=e["reference"], log=logs.append)
    text = "\n".join(logs)
    assert f"  waiting:     {O.waiting_note(JUL)}" in logs
    assert "Item master for formulary complience July 26.xls is not read" in text
    assert "GRN notes.pdf is not read" in text and "Payer Master.csv is not read" in text


def test_the_list_command_names_a_published_month_whose_item_master_has_gone(new_month):
    e = new_month
    B.load_inputs(raw=e["raw"], cache=e["cache"], reference=e["reference"], log=lambda *_: None)   # fills the cache
    # Jul published from an Item Master that has since left the folder (only an .xls copy is there)
    e["data"].mkdir()
    pd.DataFrame({"period": [JUL]}).to_parquet(e["data"] / f"{C.OOF}.parquet")
    (e["data"] / C.VALIDATION_FILE).write_text(json.dumps(
        {"months": {JUL: {"oof": {"item_master": {"file": "Item master July 26.xlsx"}}}}}))
    logs: list = []
    B.list_files(raw=e["raw"], cache=e["cache"], data=e["data"], reference=e["reference"], log=logs.append)
    lost = (f"{JUL}: out of formulary for Jul-26 is published, built on the Item Master 'Item master July "
            "26.xlsx', which is no longer in the Item Master folder; put it back (a published month is "
            "rebuilt only on its own Item Master), or rebuild only the other months with --months")
    assert f"  missing:     {lost}" in logs
    assert not [ln for ln in logs if ln.startswith("  waiting:")]          # it does not wait
    with pytest.raises(BuildError, match=re.escape(lost)):
        B.build(raw=e["raw"], cache=e["cache"], data=e["data"], reference=e["reference"], log=lambda *_: None)


# --- the spend build ---------------------------------------------------------------------------
def test_the_spend_build_names_the_files_it_does_not_read(env):    # noqa: F811
    touch(env["dash"] / "Spend Analysis Main Sheet Sep 26.xls")
    touch(env["dash"] / ".~lock.Copy of Onco Formulary -Spend Analysis Main Sheet_11.08.26.xlsx#")
    logs: list = []
    rep = SB.build(log=logs.append, data=env["data"], **env["dirs"])
    want = f"{DASH}/Spend Analysis Main Sheet Sep 26.xls is not read: {XLS}"
    assert rep["warnings"] == [want] and f"[spend] warning: {want}" in logs
    assert json.loads((env["data"] / "spend_validation.json").read_text())["warnings"] == [want]
    # the build went on: the tables are published
    assert (env["data"] / "scheme_switch.parquet").exists()


def test_a_main_sheet_in_the_wrong_format_is_named_in_the_abort(env):    # noqa: F811
    main = next(env["dash"].glob("*Spend Analysis*"))
    main.rename(main.with_suffix(".xls"))
    with pytest.raises(BuildError, match=r"no '\*Spend Analysis\*' workbook \(.xlsx or .xlsb\) .*not read: "
                                         r"'Copy of Onco Formulary -Spend Analysis Main Sheet_11.08.26.xls'"):
        SB.build(log=lambda *_: None, data=env["data"], **env["dirs"])
