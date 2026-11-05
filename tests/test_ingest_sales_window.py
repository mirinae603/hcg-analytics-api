# tests/test_ingest_sales_window.py — what the legacy sales ingest leaves out, and how it says so.
#
# app/etl/ingest_sales.py keeps the billing months of LEGACY_SALES_WINDOW (Dec-25..May-26). A row
# outside the window is left out on purpose, and the run log says which side it fell on: after the
# window (served by the formulary build's sales_items), before it, or with no readable SALESDATE
# at all, which is no month anyone serves. A file whose dates mostly cannot be read stops the run.
# The ingest runs on tiny made-up workbooks in tmp_path: nothing touches app/data or HCG's files.
from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import Workbook

from app.etl import ingest_sales as S

HEADER = ["SALESDATE", "MATERIALCODE", "MATERIALLINEITEM", "QTY", "TOTALMRP", "TOTALCOSTPRICE",
          "MANUFACTURERNAME", "LOCATIONNAME"]


def line(date, mrp, cost, code=101001):
    return [date, code, f"ITEM {code}", 1, mrp, cost, "ACME PHARMA", "HCG KR, Bangalore"]


# IP: two rows in the window (31-May-26 and 1-Dec-25: MRP 100 + 50, cost 60 + 30), two after it
# (a date and a text date), one before it, one blank date and one written day-first with slashes,
# which parse_ym does not read. Every row left out carries MRP 999, so a leak would show.
IP_ROWS = [line(datetime(2026, 5, 31), 100.0, 60.0), line(datetime(2025, 12, 1), 50.0, 30.0),
           line(datetime(2026, 6, 2), 999.0, 1.0), line("2026-07-01", 999.0, 1.0),
           line(datetime(2025, 11, 30), 999.0, 1.0), line(None, 999.0, 1.0), line("31/05/2026", 999.0, 1.0)]
OP_ROWS = [line(datetime(2026, 3, 15), 40.0, 25.0, code=201001)]


def write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    wb.active.append(HEADER)
    for r in rows:
        wb.active.append(r)
    wb.save(path)


@pytest.fixture()
def ingest(tmp_path, monkeypatch):
    """run() pointed at tmp folders: SALES_RAW, the output folder and the backend root it looks
    for dim_material under (absent here: no category is looked up)."""
    monkeypatch.setattr(S, "RAW", str(tmp_path / "raw"))
    monkeypatch.setattr(S, "KPI", str(tmp_path / "kpi"))
    monkeypatch.setattr(S, "HERE", str(tmp_path))
    monkeypatch.setattr(S, "LIMIT", 0)
    assert S.LEGACY_SALES_MONTHS[0] == "2025-12" and S.LEGACY_SALES_MONTHS[-1] == "2026-05"
    return tmp_path


def test_each_row_left_out_is_counted_for_what_it_is(ingest, capsys):
    write(ingest / "raw" / "IP Sales" / "IP May 2026.xlsx", IP_ROWS)
    write(ingest / "raw" / "OP Sales" / "OP Mar 2026.xlsx", OP_ROWS)
    S.run()
    out = capsys.readouterr().out.splitlines()
    ip = next(ln for ln in out if ln.startswith("[sales] IP"))
    assert "seen=       7 kept=       2 outside_window=       3 no_date=       2" in ip
    assert [ln for ln in out if "left out" in ln or "no readable" in ln] == [
        "[sales] 1 rows dated before 2025-12 were left out (SALES_MONTHS moves the window).",
        "[sales] 2 rows dated after 2026-05 were left out; months after the window come from the formulary "
        "build (sales_items).",
        "[sales] 2 rows have no readable SALESDATE (blank, or a date format the reader does not know) and were "
        "not counted."]
    totals = pd.read_parquet(ingest / "kpi" / "sales_totals.parquet").set_index("patient")
    assert totals.loc["IP", ["revenue", "cost", "lines"]].tolist() == [150.0, 90.0, 2.0]
    assert totals.loc["OP", ["revenue", "cost", "lines"]].tolist() == [40.0, 25.0, 1.0]
    monthly = pd.read_parquet(ingest / "kpi" / "sales_monthly.parquet")
    assert sorted(monthly["month"]) == ["2025-12", "2026-03", "2026-05"]


def test_only_the_reasons_that_occur_are_said():
    assert S.left_out_lines(Counter()) == []
    assert S.left_out_lines(Counter(after=3)) == [
        "[sales] 3 rows dated after 2026-05 were left out; months after the window come from the formulary "
        "build (sales_items)."]
    assert [S.outside_window(ym) for ym in (None, (2025, 11), (2026, 6), (2024, 12))] == [
        "no_date", "before", "after", "before"]


def test_a_file_whose_dates_mostly_cannot_be_read_stops_the_run(ingest):
    # one row in the window, two written day-first with slashes: 2 of 3 rows unreadable, over half
    write(ingest / "raw" / "IP Sales" / "IP May 2026.xlsx",
          [IP_ROWS[0], line("31/05/2026", 70.0, 40.0), line("30/05/2026", 70.0, 40.0)])
    write(ingest / "raw" / "OP Sales" / "OP Mar 2026.xlsx", OP_ROWS)
    with pytest.raises(SystemExit, match=r"no readable SALESDATE on most rows.*\n  IP May 2026.xlsx: 2 of 3 rows"):
        S.run()
    assert not (ingest / "kpi").exists()                    # nothing was written
