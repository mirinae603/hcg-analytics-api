"""HCG's purchase-side masters and workbooks, read into typed objects and small frames.

OOF master   'Out Of formulary Master Sheet (...)/Copy of Out of formualry vs suggested
             formualry brand master-2026-27.xlsb', Sheet1: a title row, then Material,
             Material Description, Generic name, (blank), MATERIAL, DESCRIPTION, PRICE, MRP,
             GST. One row per out-of-formulary brand: its suggested formulary brand with that
             brand's FY price (excl GST), MRP (incl GST) and GST %. Codes are stored as
             decimals. Four suggested codes carry two different prices on different OOF rows,
             so a price belongs to the OOF row, never to the suggested code. No plant column:
             the list is chain-wide.
OOF workbooks
             one a month in 'Out of Formulary Spend Analysis @6-27' (Jun, Jul .xlsb; Aug
             .xlsx). The month is read from the worksheet's GR dates, never from a title (the
             Aug-26 Summary's title says July). Read from each:
               worksheet  HCG's final OOF lines (Jun-26 'WS Final Sheet', which drops the 4
                          lines the WS could not price; else 'Out of formulary <Month> 26
                          WS'), aggregated to plant x material x base price -> HCG's
                          suggested brand with HCG's typed price / MRP / GST. No GR number.
               parked     Jul-26 'Need To check flaging later': OOF lines HCG doubts the flag
                          of; shown by HCG but left out of its loss.
               Summary    per centre: used PP, used MRP, margin, suggested PP, suggested MRP,
                          margin, and L 'Margin Value Differnce' = the loss HCG reports.
Item Master  the month's own master through app/etl/formulary: its 'Formulary' (Aug-26:
             'Formulary flagging') column decides OUT OF FORMULARY, as in HCG's worksheets —
             not the GRN's own Formulary column (Jun-26: 1,134 lines instead of HCG's 765).
Main Sheet   the CASH&TPA list, every priced CASH&TPA brand and the brand blocks with their
             P-ranks, through app/etl/spend/masters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

import pandas as pd

from app.etl.formulary import classify as K
from app.etl.formulary import config as FC
from app.etl.formulary import excel as X
from app.etl.formulary.discover import BuildError, _excel_files, discover
from app.etl.formulary.masters import ItemMaster, load_item_master
from app.etl.purchase import config as C
from app.etl.purchase.grn import clean_code, parse_gr_date, single_period
from app.etl.spend.masters import BrandList, _find, load_brand_blocks, load_cash_list, load_price_list


def _get(r, i):
    return r[i] if i is not None and r is not None and i < len(r) else None


def _eq(name: str):
    return lambda h: h == name


def _scan(rows: Iterator, preds: Iterable, where: str, scan: int = X.HEADER_SCAN_ROWS) -> tuple:
    """(rows above the header, normalised header) of the first row within `scan` rows in
    which every predicate matches some cell. The iterator is left just after the header."""
    preds = list(preds)
    above = []
    for i, row in enumerate(rows):
        if i >= scan:
            break
        hdr = [X.header_key(v) for v in (row or ())]
        if all(any(p(h) for h in hdr) for p in preds):
            return above, hdr
        above.append(row)
    raise BuildError(f"{where}: no header row in the first {scan} rows")


# --------------------------------------------------------------------------- #
# Folders
# --------------------------------------------------------------------------- #
def find_oof_dir(raw=None) -> Path:
    """The one 'Out of Formulary Spend Analysis ...' folder under FORMULARY_RAW_DIR."""
    root = C.raw_dir(raw)
    if not root.is_dir():
        raise BuildError(f"FORMULARY_RAW_DIR does not exist: {root}")
    found = sorted(d for d in root.rglob("*") if d.is_dir() and C.OOF_DIR_RE.match(d.name.strip()))
    if len(found) != 1:
        raise BuildError(f"expected one 'Out of Formulary Spend Analysis' folder under {root}, "
                         f"found {len(found)}")
    return found[0]


def find_oof_master(raw=None) -> Path:
    """The one workbook in the OOF folder's 'Out Of formulary Master Sheet' sub-folder."""
    d = find_oof_dir(raw)
    found = [p for sub in sorted(d.iterdir())
             if sub.is_dir() and C.OOF_MASTER_DIR_RE.match(sub.name.strip())
             for p in _excel_files(sub)]
    if len(found) != 1:
        raise BuildError(f"expected one OOF master workbook in {d.name}/'Out Of formulary Master "
                         f"Sheet ...', found {len(found)}")
    return found[0]


def find_hcg_workbooks(raw=None) -> list:
    """HCG's monthly OOF workbooks: every Excel file directly in the OOF folder."""
    return _excel_files(find_oof_dir(raw))


# --------------------------------------------------------------------------- #
# OOF master
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OofMasterRow:
    row: int                        # sheet row (1-based)
    material: str                   # the out-of-formulary brand
    material_desc: Optional[str]
    generic_name: Optional[str]
    suggested: str                  # the formulary brand HCG suggests instead
    suggested_desc: Optional[str]
    price: float                    # HCG Pharma selling price of the suggested brand, excl GST
    mrp: float                      # its MRP, incl GST
    gst: float                      # %


@dataclass
class OofMaster:
    name: str                       # file name
    sheet: str
    fy: Optional[tuple]             # (2026, 2027) from the file name, None when it has none
    rows: int = 0                   # OOF rows read (with a material code)
    entries: dict = field(default_factory=dict)       # OOF material -> OofMasterRow
    skipped: list = field(default_factory=list)       # rows that cannot be used, with a reason

    def covers(self, period: str) -> bool:
        """True when 'YYYY-MM' is in the master's financial year (April to March); True for
        every month when the file name carries no year."""
        if self.fy is None:
            return True
        y, m = map(int, period.split("-"))
        return (y == self.fy[0] and m >= 4) or (y == self.fy[1] and m <= 3)

    def frame(self) -> pd.DataFrame:
        """One row per OOF material: row, material, material_desc, generic_name, suggested,
        suggested_desc, price, mrp, gst."""
        cols = list(OofMasterRow.__dataclass_fields__)
        return pd.DataFrame([e.__dict__ for e in self.entries.values()], columns=cols)


def _fy_from_name(name: str) -> Optional[tuple]:
    m = C.FY_IN_NAME_RE.search(name)
    if not m:
        return None
    start = int(m.group(1))
    if int(m.group(2)) != (start + 1) % 100:
        raise BuildError(f"{name}: financial year {m.group(0)!r} is not two consecutive years")
    return start, start + 1


def parse_oof_master_rows(rows: Iterable, name: str = "OOF master", sheet: str = "Sheet1") -> OofMaster:
    """HCG's OOF master rows (as cell-value lists, title row first) -> OofMaster.

    An OOF code listed twice must carry the same suggestion and prices (else BuildError);
    a row without a suggested code, price, MRP or GST is skipped with the reason."""
    where = f"{name} {sheet!r}"
    it = iter(rows)
    above, hdr = _scan(it, [_eq(h) for h in C.OOF_MASTER_REQUIRED], where)
    col = _find(hdr, Path(name), sheet, {
        "code": (_eq("MATERIAL"), None, True),
        "name": (_eq("MATERIAL DESCRIPTION"), None, True),
        "generic": (_eq("GENERIC NAME"), None, True),
        "sug_code": (_eq("MATERIAL"), "generic", True),
        "sug_name": (_eq("DESCRIPTION"), "sug_code", True),
        "price": (_eq("PRICE"), "sug_code", True),
        "mrp": (_eq("MRP"), "sug_code", True),
        "gst": (_eq("GST"), "sug_code", True),
    })
    out = OofMaster(name=name, sheet=sheet, fy=_fy_from_name(name))
    for n, r in enumerate(it, start=len(above) + 2):
        code = clean_code(_get(r, col["code"]))
        if not code:
            continue
        out.rows += 1
        desc = X.text(_get(r, col["name"]))
        sug = clean_code(_get(r, col["sug_code"]))
        v = {k: X.number(_get(r, col[k])) for k in ("price", "mrp", "gst")}
        gaps = (["suggested code"] if not sug else []) + [k for k, x in v.items() if x is None]
        if gaps:
            out.skipped.append({"row": n, "material": code, "material_desc": desc,
                                "reason": "no " + ", ".join(gaps)})
            continue
        e = OofMasterRow(row=n, material=code, material_desc=desc,
                         generic_name=X.text(_get(r, col["generic"])), suggested=sug,
                         suggested_desc=X.text(_get(r, col["sug_name"])), **v)
        old = out.entries.get(code)
        if old is not None:
            if (old.suggested, old.price, old.mrp, old.gst) != (e.suggested, e.price, e.mrp, e.gst):
                raise BuildError(f"{where}: material {code} listed twice (rows {old.row}, {n}) with "
                                 f"different suggestions or prices")
            continue
        out.entries[code] = e
    if not out.entries:
        raise BuildError(f"{where}: no usable OOF rows")
    return out


def load_oof_master(path=None, raw=None) -> OofMaster:
    """HCG's OOF master (default: find_oof_master(raw)), first sheet."""
    path = Path(path) if path else find_oof_master(raw)
    with X.Workbook(path) as wb:
        sheet = wb.sheet_names[0]
        return parse_oof_master_rows(wb.rows(sheet), name=path.name, sheet=sheet)


# --------------------------------------------------------------------------- #
# HCG's monthly OOF workbooks
# --------------------------------------------------------------------------- #
WS_LINE_COLUMNS = ["plant", "material", "material_desc", "gr_date", "gr_qty", "base_price",
                   "suggested", "suggested_desc", "sug_price", "sug_mrp", "sug_gst"]
PAIR_KEYS = ["period", "plant", "material", "base_price", "suggested", "suggested_desc",
             "sug_price", "sug_mrp", "sug_gst"]
PAIR_COLUMNS = PAIR_KEYS[:4] + ["material_desc"] + PAIR_KEYS[4:] + ["lines", "qty"]
PARKED_COLUMNS = ["period", "plant", "material", "material_desc", "base_price", "lines", "qty"]
SUMMARY_COLUMNS = ["period", "plant", "centre_name", "used_pp", "used_mrp", "used_margin",
                   "sug_pp", "sug_mrp", "sug_margin", "loss"]


def parse_ws_rows(rows: Iterable, name: str = "OOF workbook", sheet: str = "WS") -> pd.DataFrame:
    """An HCG OOF worksheet (header in the first rows, HCG's layout A..AG) -> one row per
    line: plant, material, material_desc, gr_date (date), gr_qty, base_price, suggested,
    suggested_desc, sug_price, sug_mrp, sug_gst. The GR number column is never read.
    A line HCG could not price (#N/A in the suggested block) has suggested None. Lines stay
    in memory: callers aggregate them (ws_pairs, parked_lines)."""
    it = iter(rows)
    _, hdr = _scan(it, [_eq(h) for h in C.WS_REQUIRED], f"{name} {sheet!r}")
    col = _find(hdr, Path(name), sheet, {
        "gr_date": (_eq("GR DATE"), None, True),
        "plant": (_eq("PLANT"), None, True),
        "material": (_eq("MATERIAL"), None, True),
        "material_desc": (_eq("MATERIAL DESCRIPTION"), None, False),
        "gr_qty": (_eq("GR QTY"), None, True),
        "base_price": (_eq("BASE PRICE"), None, True),
        "suggested": (_eq("SAP CODE"), "base_price", True),
        "suggested_desc": (_eq("SAP NAME"), "suggested", True),
        "sug_price": (lambda h: h.startswith(C.WS_SUG_PRICE_PREFIX), "suggested", True),
        "sug_mrp": (lambda h: h.startswith("MRP"), "suggested", True),
        "sug_gst": (lambda h: h.startswith("GST"), "suggested", True),
    })
    recs = []
    for r in it:
        material = clean_code(_get(r, col["material"]))
        if not material:
            continue
        recs.append({
            "plant": X.upper(_get(r, col["plant"])), "material": material,
            "material_desc": X.text(_get(r, col["material_desc"])),
            "gr_date": parse_gr_date(_get(r, col["gr_date"])),
            "gr_qty": X.number(_get(r, col["gr_qty"])),
            "base_price": X.number(_get(r, col["base_price"])),
            "suggested": clean_code(_get(r, col["suggested"])),
            "suggested_desc": X.text(_get(r, col["suggested_desc"])),
            "sug_price": X.number(_get(r, col["sug_price"])),
            "sug_mrp": X.number(_get(r, col["sug_mrp"])),
            "sug_gst": X.number(_get(r, col["sug_gst"])),
        })
    return pd.DataFrame(recs, columns=WS_LINE_COLUMNS)


def ws_pairs(lines: pd.DataFrame, period: str) -> pd.DataFrame:
    """Worksheet lines -> HCG's pairs: one row per period x plant x material x base price x
    suggestion (code, name, price, MRP, GST as typed), with lines and qty. A key with two
    suggestions keeps both rows; which one a GRN line gets is the OOF build's rule."""
    df = lines.assign(period=period)
    g = df.groupby(PAIR_KEYS, dropna=False, sort=True)
    out = g.agg(material_desc=("material_desc", "first"), lines=("material", "size"),
                qty=("gr_qty", "sum")).reset_index()
    return out[PAIR_COLUMNS]


def parked_lines(lines: pd.DataFrame, period: str) -> pd.DataFrame:
    """Parked worksheet lines -> period x plant x material x base price: lines, qty."""
    df = lines.assign(period=period)
    g = df.groupby(["period", "plant", "material", "base_price"], dropna=False, sort=True)
    out = g.agg(material_desc=("material_desc", "first"), lines=("material", "size"),
                qty=("gr_qty", "sum")).reset_index()
    return out[PARKED_COLUMNS]


@dataclass
class SummaryRead:
    frame: pd.DataFrame             # SUMMARY_COLUMNS without period
    titles: list                    # the text above the header (HCG's titles, never trusted)
    computed: int                   # centre rows whose loss cell had no value: J - E used
    warnings: list


def parse_summary_rows(rows: Iterable, name: str = "OOF workbook", sheet: str = "Summary") -> SummaryRead:
    """HCG's OOF Summary sheet -> one row per centre: plant (A), centre_name (B), used_pp (C),
    used_mrp (D), used_margin (E), sug_pp (H), sug_mrp (I), sug_margin (J), loss (L 'Margin
    Value Differnce' = J - E). A formula cell without a cached value (an .xlsx saved by a
    tool that does not compute) is recomputed the way HCG's formula does: E = D - C,
    J = I - H, L = J - E. Stops at 'Grand Total' and checks its C, D, H, I against the sum
    of the centres (a warning when they differ by more than a rupee)."""
    where = f"{name} {sheet!r}"
    it = iter(rows)
    above, hdr = _scan(it, [C.SUMMARY_CENTRE_RE.match, C.SUMMARY_LOSS_RE.match], where)
    col = _find(hdr, Path(name), sheet, {
        "plant": (C.SUMMARY_CENTRE_RE.match, None, True),
        "name": (C.SUMMARY_NAME_RE.match, None, False),
        "used_pp": (C.SUMMARY_USED_PP_RE.search, None, True),
        "used_mrp": (C.SUMMARY_USED_MRP_RE.search, "used_pp", True),
        "used_margin": (C.SUMMARY_MARGIN_RE.match, "used_mrp", True),
        "sug_pp": (C.SUMMARY_SUG_PP_RE.match, "used_margin", True),
        "sug_mrp": (C.SUMMARY_SUG_MRP_RE.match, "sug_pp", True),
        "sug_margin": (C.SUMMARY_MARGIN_RE.match, "sug_mrp", True),
        "loss": (C.SUMMARY_LOSS_RE.match, "sug_margin", True),
    })
    titles = [t for r in above for t in (X.text(v) for v in (r or ())) if t]
    recs, computed, warnings, total = [], 0, [], None
    for r in it:
        plant = X.upper(_get(r, col["plant"]))
        if not plant:
            continue
        if plant == C.SUMMARY_TOTAL:
            total = r
            break
        v = {k: X.number(_get(r, col[k])) for k in ("used_pp", "used_mrp", "used_margin",
                                                     "sug_pp", "sug_mrp", "sug_margin", "loss")}
        base = [v[k] for k in ("used_pp", "used_mrp", "sug_pp", "sug_mrp")]
        if any(x is None for x in base):
            raise BuildError(f"{where}: centre {plant} has no PP / MRP figures")
        if v["used_margin"] is None:
            v["used_margin"] = v["used_mrp"] - v["used_pp"]
        if v["sug_margin"] is None:
            v["sug_margin"] = v["sug_mrp"] - v["sug_pp"]
        if v["loss"] is None:
            v["loss"] = v["sug_margin"] - v["used_margin"]
            computed += 1
        recs.append({"plant": plant, "centre_name": X.text(_get(r, col["name"])), **v})
    df = pd.DataFrame(recs, columns=SUMMARY_COLUMNS[1:])
    if df.empty:
        raise BuildError(f"{where}: no centre rows")
    dup = sorted(df.loc[df["plant"].duplicated(), "plant"])
    if dup:
        raise BuildError(f"{where}: centres listed twice: {dup}")
    if total is None:
        warnings.append(f"{where}: no 'Grand Total' row to check the centres against")
    else:
        for k in ("used_pp", "used_mrp", "sug_pp", "sug_mrp"):
            t = X.number(_get(total, col[k]))
            if t is not None and abs(t - df[k].sum()) > 1.0:
                warnings.append(f"{where}: Grand Total {k} {t:,.2f} is not the sum of the centres "
                                f"{df[k].sum():,.2f}")
    return SummaryRead(frame=df, titles=titles, computed=computed, warnings=warnings)


@dataclass
class HcgWorkbook:
    """One month of HCG's OOF workings. Aggregates only; no GR number, no line."""
    name: str                       # file name
    period: str                     # from the worksheet's GR dates
    ws_sheet: str                   # the worksheet read as HCG's final answer
    ws_lines: int                   # its lines
    ws_unpriced: int                # of which HCG left without a suggested brand
    pairs: pd.DataFrame             # PAIR_COLUMNS
    parked_sheet: Optional[str]
    parked: pd.DataFrame            # PARKED_COLUMNS (empty when the month parks nothing)
    summary_sheet: str
    summary: pd.DataFrame           # SUMMARY_COLUMNS
    summary_computed: int           # centre losses recomputed (J - E): no cached cell value
    titles: list                    # the Summary's own titles, kept for the report only
    warnings: list = field(default_factory=list)

    def report(self) -> dict:
        return {"file": self.name, "period": self.period, "ws_sheet": self.ws_sheet,
                "ws_lines": self.ws_lines, "ws_unpriced": self.ws_unpriced,
                "pairs": len(self.pairs), "parked_sheet": self.parked_sheet,
                "parked_lines": int(self.parked["lines"].sum()) if len(self.parked) else 0,
                "summary_sheet": self.summary_sheet, "centres": len(self.summary),
                "loss": round(float(self.summary["loss"].sum()), 2),
                "summary_computed": self.summary_computed, "warnings": list(self.warnings)}


def _named_months(texts: Iterable) -> set:
    """'YYYY-MM' of every month the texts spell ('July 2026', 'August 26'): HCG's titles and
    file names, reported when they disagree with the GR dates, never used for the month."""
    out = set()
    for t in texts:
        for m, y in FC.MONTH_IN_NAME_RE.findall(str(t)):
            out.add(f"{int(y) + (2000 if len(y) == 2 else 0):04d}-{FC.MONTHS[m.lower()[:3]]:02d}")
    return out


def _pick(sheets: list, rx, what: str, name: str) -> Optional[str]:
    hits = [s for s in sheets if rx.search(K.norm_text(s))]
    if len(hits) > 1:
        raise BuildError(f"{name}: more than one {what} sheet: {hits}")
    return hits[0] if hits else None


def choose_sheets(sheet_names: list, name: str) -> tuple:
    """(worksheet, parked sheet or None, Summary sheet) of one HCG OOF workbook. The
    worksheet is 'WS Final Sheet' when present, else the 'Out of formulary ... WS' sheet."""
    ws = _pick(sheet_names, C.WS_FINAL_SHEET_RE, "'WS Final'", name) or \
        _pick(sheet_names, C.WS_SHEET_RE, "'Out of formulary ... WS'", name)
    summary = [s for s in sheet_names if K.norm_text(s) == C.SUMMARY_SHEET]
    if ws is None or len(summary) != 1:
        raise BuildError(f"{name}: needs one OOF worksheet and one Summary sheet (sheets: "
                         f"{sheet_names})")
    return ws, _pick(sheet_names, C.PARKED_SHEET_RE, "parked", name), summary[0]


def load_hcg_workbook(path) -> HcgWorkbook:
    """One of HCG's monthly OOF workbooks -> HcgWorkbook (month from the GR dates)."""
    path = Path(path)
    warnings = []
    with X.Workbook(path) as wb:
        ws, parked_sheet, summary_sheet = choose_sheets(wb.sheet_names, path.name)
        lines = parse_ws_rows(wb.rows(ws), path.name, ws)
        parked = (parse_ws_rows(wb.rows(parked_sheet), path.name, parked_sheet)
                  if parked_sheet else pd.DataFrame(columns=WS_LINE_COLUMNS))
        summary = parse_summary_rows(wb.rows(summary_sheet), path.name, summary_sheet)
    if lines.empty:
        raise BuildError(f"{path.name} {ws!r}: no OOF lines")
    period = single_period(lines["gr_date"], f"{path.name} {ws!r}")
    if len(parked):
        p = single_period(parked["gr_date"], f"{path.name} {parked_sheet!r}")
        if p != period:
            raise BuildError(f"{path.name}: parked lines are {p}, the worksheet {period}")
    named = _named_months(summary.titles + [path.stem])
    if named - {period}:
        warnings.append(f"{path.name}: titles / file name say {sorted(named)}, the worksheet's GR "
                        f"dates {period}; {period} used")
    return HcgWorkbook(
        name=path.name, period=period, ws_sheet=ws, ws_lines=len(lines),
        ws_unpriced=int(lines["suggested"].isna().sum()), pairs=ws_pairs(lines, period),
        parked_sheet=parked_sheet, parked=parked_lines(parked, period), summary_sheet=summary_sheet,
        summary=summary.frame.assign(period=period)[SUMMARY_COLUMNS],
        summary_computed=summary.computed, titles=summary.titles,
        warnings=warnings + summary.warnings)


def load_hcg_workbooks(raw=None, files=None) -> dict:
    """Every HCG OOF workbook -> {period: HcgWorkbook}; two workbooks for one month abort."""
    out: dict = {}
    for p in (files if files is not None else find_hcg_workbooks(raw)):
        wb = load_hcg_workbook(p)
        if wb.period in out:
            raise BuildError(f"two OOF workbooks hold {wb.period}: {out[wb.period].name!r} and "
                             f"{wb.name!r}")
        out[wb.period] = wb
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- #
# The month's Item Master (app/etl/formulary)
# --------------------------------------------------------------------------- #
ITEM_MASTER_COLUMNS = ["material", "formulary_class", "formulary_flag", "tier", "material_desc",
                       "minor_group", "manufacturer"]


def item_master_paths(raw=None) -> dict:
    """{period: Item Master path} as the formulary build discovers them (month from the
    file name; one master per month)."""
    return dict(discover(raw).item_masters)


def load_month_item_master(period: str, raw=None, paths: Optional[dict] = None) -> ItemMaster:
    """That month's own Item Master. A month without one aborts: another month's master
    gives wrong flags."""
    paths = item_master_paths(raw) if paths is None else paths
    if period not in paths:
        raise BuildError(f"no Item Master for {period} (found: {', '.join(sorted(paths)) or 'none'})")
    return load_item_master(paths[period])


def item_master_frame(im: ItemMaster) -> pd.DataFrame:
    """ItemMaster -> one row per material: formulary_class ('FORMULARYP', 'NON FORMULARY',
    'OUT OF FORMULARY', ...), formulary_flag (FORMULARY / NON FORMULARY / OUT OF FORMULARY),
    tier (the Category letter of a formulary item: 'P', 'R', 'R-OPD', ...; None otherwise),
    material_desc, minor_group, manufacturer."""
    rows = []
    for code, it in im.items.items():
        formulary = it.formulary_class.startswith(K.FORMULARY)
        rows.append({"material": code, "formulary_class": it.formulary_class,
                     "formulary_flag": K.FORMULARY if formulary else it.formulary_class,
                     "tier": K.rule_letter(it.formulary_class) if formulary else None,
                     "material_desc": it.material_desc, "minor_group": it.minor_group,
                     "manufacturer": it.manufacturer})
    return pd.DataFrame(rows, columns=ITEM_MASTER_COLUMNS)


# --------------------------------------------------------------------------- #
# The Main Sheet's CASH&TPA list (app/etl/spend)
# --------------------------------------------------------------------------- #
@dataclass
class MainSheet:
    path: Path
    cash: BrandList                 # CASH&TPA suggestions: used material -> CashSwitch
    prices: dict                    # every priced CASH&TPA brand: material -> PricedBrand
    blocks: pd.DataFrame            # brand blocks: material, block_id, sharing, p_rank, tie, ...


def load_main_sheet(raw=None, path=None) -> MainSheet:
    """HCG's Spend Analysis Main Sheet (default: the one in the dashboard-data folder): the
    CASH&TPA list, its price list and its brand blocks with P-ranks (p_rank 1 = the highest
    2026-27 SHARING in the block, HCG's P1)."""
    if path is None:
        from app.etl.spend.build import find_main_sheet   # the spend build owns the lookup
        path = find_main_sheet(raw)
    path = Path(path)
    cash = load_cash_list(path)
    return MainSheet(path=path, cash=cash, prices=load_price_list(path),
                     blocks=load_brand_blocks(path, cash=cash))
