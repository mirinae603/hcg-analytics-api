"""HCG's suggested-brand lists (two sheets of the Spend Analysis Main Sheet) and our
bifurcation reference file.

SCHEME     for a government-scheme patient, use the suggested brand instead of this one.
           Both brands carry an FY price-list price (HCG Pharma selling price, excl GST),
           the row says whether the suggested brand is an Attis brand, and CATEGORY names the
           consumable group (SUTURES, Endo, ...) of HCG's J&J & Covidien -> Healthium view.
CASH&TPA   for a cash or insurance patient, use the suggested brand: both brands carry the FY
           price, MRP (incl GST) and GST %, so each has a unit margin = MRP excl GST - price.
           SHARING is the share of the molecule's volume a brand may keep when HCG bifurcates
           it (cash_tpa_bifurcation.csv says which brands, and their molecule group).
           Every priced row, with or without a suggestion, is also read as a price list
           (load_price_list) for the purchase build.

A row whose suggested-brand columns are empty is not a switch and is dropped. HCG fills the
quantity and value columns by hand each month for one hospital; the build computes them for
every hospital and month instead, so those columns are never read.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from app.etl.formulary import config as FC
from app.etl.formulary import excel as X
from app.etl.formulary.classify import canonical_category, norm_code
from app.etl.formulary.discover import BuildError
from app.etl.spend import config as C


@dataclass(frozen=True)
class Switch:
    material_desc: str | None           # used brand, as the list names it
    used_price: float                   # FY price-list price of the used brand, excl GST
    hcg_category: str | None            # the used brand's FY category: P, R, P-SCH, ...
    suggested: str                      # suggested brand's material code
    suggested_desc: str | None
    suggested_price: float
    attis: bool                         # the suggested brand is an Attis brand
    scheme_group: str | None = None     # SCHEME 'CATEGORY': SUTURES, Endo, ... (consumables)


@dataclass(frozen=True)
class CashSwitch:
    material_desc: str | None
    used_price: float                   # FY HCG Pharma selling price, excl GST
    used_mrp: float                     # FY MRP, incl GST
    used_gst: float                     # %
    hcg_category: str | None
    sharing: float | None               # share of the molecule's volume allowed when bifurcated
    suggested: str
    suggested_desc: str | None
    suggested_price: float
    suggested_mrp: float
    suggested_gst: float

    @property
    def used_margin(self) -> float:
        """Unit margin of the used brand: MRP excl GST - price (HCG's 'Unit Margin Value')."""
        return self.used_mrp / (1 + self.used_gst / 100) - self.used_price

    @property
    def suggested_margin(self) -> float:
        return self.suggested_mrp / (1 + self.suggested_gst / 100) - self.suggested_price


@dataclass(frozen=True)
class PricedBrand:
    """One brand row of the CASH&TPA sheet with its FY prices, whether or not it has a suggestion."""
    material: str
    material_desc: str | None
    price: float                  # HCG Pharma selling price, excl GST
    mrp: float                    # MRP incl GST
    gst: float                    # %
    category: str | None          # the sheet's FY category (P, R, P-SCH, ...)

    @property
    def unit_margin(self) -> float:
        return self.mrp / (1 + self.gst / 100) - self.price


@dataclass
class BrandList:
    path: Path
    sheet: str
    fy: tuple                           # (2026, 2027): April 2026 .. March 2027
    rows: int = 0                       # brand rows read
    switches: dict = field(default_factory=dict)       # used material code -> (Cash)Switch
    skipped: list = field(default_factory=list)        # suggestions the build cannot price

    def covers(self, period: str) -> bool:
        """True when 'YYYY-MM' falls in the list's financial year (April to March)."""
        y, m = map(int, period.split("-"))
        start = self.fy[0]
        return (y == start and m >= 4) or (y == start + 1 and m <= 3)



def _fy(title_rows) -> tuple:
    for row in title_rows:
        for v in row or ():
            m = C.FY_RE.search(str(v or "").upper())
            if m:
                start = int(m.group(1))
                if int(m.group(2)) != (start + 1) % 100:
                    raise BuildError(f"financial year {m.group(0)!r} is not two consecutive years")
                return start, start + 1
    return None


def _open(wb, path: Path, sheet: str, marker: str):
    """(row iterator positioned after the header, header, FY) of one list sheet."""
    if sheet not in wb.sheet_names:
        raise BuildError(f"{path.name}: no sheet {sheet!r} (found: {', '.join(wb.sheet_names)})")
    rows = wb.rows(sheet)
    above, hdr = [], None
    for i, r in enumerate(rows):
        h = [X.header_key(v) for v in (r or ())]
        if C.H_CODE in h and marker in h:
            hdr = h
            break
        above.append(r)
        if i >= X.HEADER_SCAN_ROWS:
            break
    if hdr is None:
        raise BuildError(f"{path.name}: sheet {sheet!r} has no header row with 'SAP CODE' and "
                         f"{marker!r}")
    fy = _fy(above)
    if fy is None:
        raise BuildError(f"{path.name}: no 'FORMULARY FOR THE YEAR 20xx-yy' title above the "
                         f"{sheet} header, so the months it applies to are unknown")
    return rows, hdr, fy


def _find(hdr: list, path: Path, sheet: str, spec: dict) -> dict:
    """Column positions by header. spec: key -> (predicate, after-key or None, required)."""
    out, missing = {}, []
    for key, (pred, after, required) in spec.items():
        lo = out.get(after, -1) if after else -1
        hits = [i for i, h in enumerate(hdr) if i > lo and pred(h)]
        if hits:
            out[key] = hits[0]
        elif required:
            missing.append(key)
        else:
            out[key] = None
    if missing:
        raise BuildError(f"{path.name}: sheet {sheet!r} has no column for {', '.join(missing)}")
    return out


def _get(r, i):
    return r[i] if i is not None and i < len(r) else None


def _keep(out: BrandList, used: str, sw, path: Path) -> None:
    old = out.switches.get(used)
    if old is not None and old != sw:
        raise BuildError(f"{path.name} {out.sheet}: material {used} listed twice with different "
                         f"suggestions or prices ({old.suggested}, {sw.suggested})")
    out.switches.setdefault(used, sw)


def load_scheme_list(path) -> BrandList:
    path = Path(path)
    sheet = C.SCHEME_SHEET
    with X.Workbook(path) as wb:
        rows, hdr, fy = _open(wb, path, sheet, C.H_ATTIS)
        col = _find(hdr, path, sheet, {
            "code": (lambda h: h == C.H_CODE, None, True),
            "name": (lambda h: h == C.H_NAME, None, True),
            "group": (lambda h: h == C.H_GROUP, None, False),
            "used_price": (C.H_USED_PRICE_RE.match, None, True),
            "category": (C.H_USED_CATEGORY_RE.match, None, False),
            "sug_code": (lambda h: h == C.H_CODE, "code", True),
            "sug_name": (lambda h: h == C.H_NAME, "sug_code", True),
            "attis": (lambda h: h == C.H_ATTIS, None, True),
            "sug_price": (lambda h: h.startswith(C.H_SUGGESTED_PRICE), "sug_code", True),
        })
        out = BrandList(path=path, sheet=sheet, fy=fy)
        for r in rows:
            used = X.code(_get(r, col["code"])) if r else None
            if not used:
                continue
            out.rows += 1
            sug = X.code(_get(r, col["sug_code"]))
            if not sug or sug == "0":
                continue
            up, sp = X.number(_get(r, col["used_price"])), X.number(_get(r, col["sug_price"]))
            if up is None or sp is None:
                out.skipped.append({"material": used, "material_desc": X.text(_get(r, col["name"])),
                                    "suggested": sug, "reason": "no price-list price for the "
                                    + ("used" if up is None else "suggested") + " brand"})
                continue
            cat = X.text(_get(r, col["category"]))
            _keep(out, used, Switch(
                material_desc=X.text(_get(r, col["name"])), used_price=up,
                hcg_category=canonical_category(cat) if cat else None, suggested=sug,
                suggested_desc=X.text(_get(r, col["sug_name"])), suggested_price=sp,
                attis=X.header_key(_get(r, col["attis"])) == C.ATTIS_BRAND,
                scheme_group=X.text(_get(r, col["group"]))), path)
    if not out.switches:
        raise BuildError(f"{path.name}: sheet {sheet!r} has no suggested brands")
    return out


def load_cash_list(path) -> BrandList:
    path = Path(path)
    sheet = C.CASH_SHEET
    with X.Workbook(path) as wb:
        rows, hdr, fy = _open(wb, path, sheet, C.H_CASH_MARKER)
        col = _find(hdr, path, sheet, {
            "code": (lambda h: h == C.H_CODE, None, True),
            "name": (lambda h: h == C.H_NAME, None, True),
            "used_price": (C.H_USED_PRICE_RE.match, None, True),
            "used_mrp": (C.H_USED_MRP_RE.match, None, True),
            "used_gst": (C.H_USED_GST_RE.match, None, True),
            "category": (C.H_USED_CATEGORY_RE.match, None, False),
            "sharing": (C.H_SHARING_RE.match, None, False),
            "sug_code": (lambda h: h == C.H_CODE, "code", True),
            "sug_name": (lambda h: h == C.H_NAME, "sug_code", True),
            "sug_price": (lambda h: h.startswith(C.H_SUGGESTED_PRICE), "sug_code", True),
            "sug_mrp": (lambda h: h.startswith("MRP"), "sug_code", True),
            "sug_gst": (lambda h: h.startswith("GST"), "sug_code", True),
        })
        out = BrandList(path=path, sheet=sheet, fy=fy)
        for r in rows:
            used = X.code(_get(r, col["code"])) if r else None
            if not used:
                continue
            out.rows += 1
            sug = X.code(_get(r, col["sug_code"]))
            if not sug or sug == "0":
                continue
            v = {k: X.number(_get(r, col[k])) for k in ("used_price", "used_mrp", "used_gst",
                                                         "sug_price", "sug_mrp", "sug_gst")}
            gaps = [k for k, x in v.items() if x is None]
            if gaps:
                out.skipped.append({"material": used, "material_desc": X.text(_get(r, col["name"])),
                                    "suggested": sug, "reason": "no " + ", ".join(gaps)})
                continue
            cat = X.text(_get(r, col["category"]))
            _keep(out, used, CashSwitch(
                material_desc=X.text(_get(r, col["name"])), used_price=v["used_price"],
                used_mrp=v["used_mrp"], used_gst=v["used_gst"],
                hcg_category=canonical_category(cat) if cat else None,
                sharing=X.number(_get(r, col["sharing"])), suggested=sug,
                suggested_desc=X.text(_get(r, col["sug_name"])), suggested_price=v["sug_price"],
                suggested_mrp=v["sug_mrp"], suggested_gst=v["sug_gst"]), path)
    if not out.switches:
        raise BuildError(f"{path.name}: sheet {sheet!r} has no suggested brands")
    return out


def load_brand_blocks(path, scheme: BrandList = None, cash: BrandList = None):
    """The CASH&TPA sheet's blocks: runs of brand rows (same molecule and strength) separated by
    rows without a SAP CODE. Each brand's '2026-27 SHARING' is its target share of the block;
    p_rank 1 = the highest share (HCG's preferred brand, P1), ties share a rank. Returned as rows
    for the brand_blocks table, with each brand's suggested brands from both lists."""
    import pandas as pd
    path = Path(path)
    rows = []
    with X.Workbook(path) as wb:
        it, hdr, _ = _open(wb, path, C.CASH_SHEET, C.H_CASH_MARKER)
        col = _find(hdr, path, C.CASH_SHEET, {
            "code": (lambda h: h == C.H_CODE, None, True),
            "name": (lambda h: h == C.H_NAME, None, True),
            "company": (lambda h: h == "COMPANY NAME", None, False),
            "category": (C.H_USED_CATEGORY_RE.match, None, True),
            "sharing": (C.H_SHARING_RE.match, None, True),
        })
        block, seen = 0, set()
        open_block = False
        for r in it:
            code = X.code(_get(r, col["code"])) if r else None
            if not code:
                open_block = False
                continue
            if not open_block:
                block += 1
                open_block = True
            if code in seen:                        # a code listed twice keeps its first block
                continue
            seen.add(code)
            cat = X.text(_get(r, col["category"]))
            rows.append({"material": code, "material_desc": X.text(_get(r, col["name"])),
                         "company": X.text(_get(r, col["company"])),
                         "ms_category": canonical_category(cat) if cat else None,
                         "sharing": X.number(_get(r, col["sharing"])), "block_id": block})
    df = pd.DataFrame(rows, columns=["material", "material_desc", "company", "ms_category", "sharing",
                                     "block_id"])
    if df.empty:
        raise BuildError(f"{path.name}: sheet {C.CASH_SHEET!r} has no brand rows")
    g = df.groupby("block_id")
    df["block_size"] = g["material"].transform("size").astype("int64")
    df["block_sharing_sum"] = g["sharing"].transform("sum").round(4)
    df["p_rank"] = g["sharing"].rank(method="min", ascending=False).fillna(0).astype("int64")
    df["tie"] = df.groupby(["block_id", "p_rank"])["material"].transform("size").gt(1) & df["block_size"].gt(1)
    for prefix, lst in (("cash", cash), ("scheme", scheme)):
        sw = lst.switches if lst else {}
        df[f"{prefix}_suggested"] = df["material"].map(lambda m: sw[m].suggested if m in sw else None)
        df[f"{prefix}_suggested_desc"] = df["material"].map(lambda m: sw[m].suggested_desc if m in sw else None)
    sw = scheme.switches if scheme else {}
    df["scheme_attis"] = df["material"].map(lambda m: bool(sw[m].attis) if m in sw else False)
    return df


def load_price_list(path) -> dict:
    """Every brand row of the CASH&TPA sheet that carries its FY price, MRP and GST %, with
    or without a suggestion: {material: PricedBrand}. The purchase build's out-of-formulary
    fallback rule (app/etl/purchase) picks and prices its formulary brands from these. A code
    listed twice keeps its first row (as the blocks do)."""
    path = Path(path)
    out: dict = {}
    with X.Workbook(path) as wb:
        it, hdr, _ = _open(wb, path, C.CASH_SHEET, C.H_CASH_MARKER)
        col = _find(hdr, path, C.CASH_SHEET, {
            "code": (lambda h: h == C.H_CODE, None, True),
            "name": (lambda h: h == C.H_NAME, None, True),
            "price": (C.H_USED_PRICE_RE.match, None, True),
            "mrp": (C.H_USED_MRP_RE.match, None, True),
            "gst": (C.H_USED_GST_RE.match, None, True),
            "category": (C.H_USED_CATEGORY_RE.match, None, False),
        })
        for r in it:
            code = X.code(_get(r, col["code"])) if r else None
            if not code or code in out:
                continue
            price, mrp, gst = (X.number(_get(r, col[k])) for k in ("price", "mrp", "gst"))
            if price is None or mrp is None or gst is None or mrp <= 0:
                continue
            cat = X.text(_get(r, col["category"]))
            out[code] = PricedBrand(material=code, material_desc=X.text(_get(r, col["name"])), price=price,
                                    mrp=mrp, gst=gst, category=canonical_category(cat) if cat else None)
    return out


def load_cash_exclusions(ref_dir=None) -> dict:
    """cash_tpa_excluded.csv -> {material: note}: cash/TPA brands the build does not count.

    Seeded from HCG's KR Aug-26 report (brands in HCG's cash list that HCG's sheet leaves out);
    HCG to confirm why. A missing file excludes nothing."""
    path = FC.reference_dir(ref_dir) / C.EXCLUDED_FILE
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for n, row in enumerate(csv.DictReader(f), start=2):
            m = norm_code(row.get("material"))
            if not m:
                raise BuildError(f"{path.name} line {n}: needs a material code")
            if m in out:
                raise BuildError(f"{path.name} line {n}: {m} listed twice")
            out[m] = (row.get("note") or "").strip()
    return out


def load_bifurcation(ref_dir=None) -> dict:
    """cash_tpa_bifurcation.csv -> {material: (flag, frozenset of the molecule group)}.

    The brands HCG bifurcates and their molecule groups are not in HCG's master: the file is
    seeded from HCG's KR Aug-26 Spend Analysis and replaced when HCG sends the full rules.
    A missing file means no brand is bifurcated."""
    path = FC.reference_dir(ref_dir) / C.BIFURCATION_FILE
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for n, row in enumerate(csv.DictReader(f), start=2):
            m = norm_code(row.get("material"))
            flag = (row.get("flag") or "").strip().upper()
            group = frozenset(norm_code(g) for g in (row.get("group") or "").split() if norm_code(g))
            if not m or flag not in ("Y", "N"):
                raise BuildError(f"{path.name} line {n}: needs a material and a flag Y or N")
            if m not in group:
                raise BuildError(f"{path.name} line {n}: the group of {m} must include {m} itself")
            if m in out:
                raise BuildError(f"{path.name} line {n}: {m} listed twice")
            out[m] = (flag, group)
    return out
