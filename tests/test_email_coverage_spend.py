# tests/test_email_coverage_spend.py — what HCG's spend email shows, through /spend/*.
#
# Two halves:
# - On test_spend_api's hand-made tables plus a few rows added here (a Covidien Endo row at KR,
#   eleven extra brands at DR so a top 10 has something to cut): the drugs-only segment, the
#   top 10 with its amounts and share, the trend's month-table fields, the Sutures / Endo split,
#   the margin %s, the cash brands' bifurcation columns and the tables' Total-row sums. Every
#   expected number is worked out by hand in the comments.
# - On the real built tables (skipped when they are absent): HCG's KR figures — Aug Sutures and
#   Endo, Jun drugs only, the 48.56% used margin and the 41 bifurcated brands.
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.api import spend
from app.core.config import settings
from tests import test_spend_api as base
from tests.test_formulary_api import write_hospital_map

AUG, JUL, JUN = "2026-08", "2026-07", "2026-06"
KR, DR = "HC05", "HC06"

# One Covidien stapler at KR (HCG's 'Endo' group): 100 used vs 60 Healthium -> 40 missed.
ENDO = {"material": "300002", "material_desc": "STAPLER COVIDIEN", "category": "Consumables",
        "scheme_group": "ENDO", "suggested_material": "300902", "suggested_desc": "STAPLER HEALTHIUM",
        "used_price": 100.0, "suggested_price": 60.0, "qty": 1.0, "patient_type": "IP", "doctor_id": "D2",
        "plant": KR}
# Eleven brands at DR, Aug: used 200 each, suggested 200 - 10i -> saving 10, 20, ... 110.
EXTRA = [{"material": f"4000{i:02d}", "material_desc": f"BRAND {i}", "category": "Other Drugs",
          "scheme_group": None, "suggested_material": f"4009{i:02d}", "suggested_desc": f"CHEAPER {i}",
          "used_price": 200.0, "suggested_price": 200.0 - 10 * i, "qty": 1.0, "patient_type": "OP",
          "doctor_id": "D9", "plant": DR} for i in range(1, 12)]

def _add_switch_rows(folder: Path) -> None:
    p = folder / "scheme_switch.parquet"
    df = pd.read_parquet(p)
    rows = []
    for r in [ENDO, *EXTRA]:
        plant = r["plant"]
        rows.append({**r, "period": AUG, "hospital": base.HIS[plant][0], "hospital_name": base.HIS[plant][1],
                     "doctor_name": base.DOCTORS[r["doctor_id"]], "hcg_category": "R", "attis": False,
                     "actual_cost": r["used_price"] * r["qty"], "lines": 1,
                     "used_value": r["used_price"] * r["qty"], "suggested_value": r["suggested_price"] * r["qty"]})
    pd.concat([df, pd.DataFrame(rows)[df.columns]], ignore_index=True).to_parquet(p, index=False)


@pytest.fixture()
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SPEND_DATA_DIR", str(tmp_path / "spend"))
    monkeypatch.setattr(settings, "FORMULARY_DATA_DIR", str(tmp_path / "formulary-not-built"))
    # the cash overview's with_oof reads the purchase build: none here, so no real purchase row
    # is added to these hand-made cash figures
    monkeypatch.setattr(settings, "PURCHASE_DATA_DIR", str(tmp_path / "purchase-not-built"))
    monkeypatch.setattr(settings, "REFERENCE_DIR", str(tmp_path / "reference"))
    base.write_tables(tmp_path / "spend")
    _add_switch_rows(tmp_path / "spend")
    write_hospital_map(tmp_path / "reference")
    spend.clear_caches()
    yield tmp_path
    spend.clear_caches()


def get(client, path, **params):
    r = client.get(path, params=params)
    assert r.status_code == 200, r.text
    return r.json()


def money(b) -> tuple:
    return b["used_value"], b["suggested_value"], b["saving_missed"]


def margins(b) -> tuple:
    return b["actual_margin"], b["potential_margin"], b["margin_loss"]


# ---------- Material type: the three categories add up to All ----------

def test_the_three_material_types_add_up_to_all(client, data):
    for path, keys in (("/spend/scheme/overview", ("used_value", "suggested_value", "saving_missed")),
                       ("/spend/attis/overview", ("used_value", "suggested_value", "saving_missed")),
                       ("/spend/cash/overview", ("actual_margin", "potential_margin", "margin_loss"))):
        every = get(client, path, Plant=KR, Month=AUG)["headline"]
        parts = [get(client, path, Plant=KR, Month=AUG, Segment=s)["headline"] for s in ("onco", "non-onco", "consumables")]
        for k in keys:
            assert sum(p[k] for p in parts) == pytest.approx(every[k], abs=0.01), (path, k)


def test_there_is_no_drugs_only_option(client, data):
    assert [c["key"] for c in get(client, "/spend/meta")["categories"]] == ["onco", "non-onco", "consumables"]
    r = client.get("/spend/scheme/overview", params={"Segment": "drugs"})
    assert r.status_code == 400


# ---------- top 10, trend fields, Sutures / Endo ----------

def test_top_10_with_amounts_and_share_of_the_whole_loss(client, data):
    top = get(client, "/spend/scheme/overview", Plant=DR, Month=AUG)["top_items"]
    # DR Aug: 101001 (600) and the eleven extras (10 .. 110) -> 12 brands, 1260 missed in all
    assert len(top) == 10
    assert [t["saving_missed"] for t in top] == [600.0, 110.0, 100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0]
    a = top[0]
    assert (a["material"], a["used_value"], a["suggested_value"]) == ("101001", 1000.0, 400.0)
    assert a["share_of_total_pct"] == pytest.approx(47.619, abs=1e-3)          # 600 / 1260
    assert top[-1]["share_of_total_pct"] == pytest.approx(2.381, abs=1e-3)     # 30 / 1260
    kr = get(client, "/spend/attis/overview", Plant=KR, Month=AUG)["top_items"]
    assert [(t["material"], t["used_value"], t["suggested_value"], t["share_of_total_pct"]) for t in kr] == \
        [("101001", 3000.0, 1200.0, 90.0), ("201002", 500.0, 300.0, 10.0)]


def test_trend_rows_carry_hcgs_month_table(client, data):
    t = get(client, "/spend/attis/overview", Plant=KR, Month=AUG)["trend"]
    jul = t[0]
    assert (jul["label"], jul["used_value"], jul["suggested_value"], jul["saving_missed"], jul["pct"]) == \
        ("Jul-26", 3000.0, 1200.0, 1800.0, 60.0)
    assert all(x["pct"] == x["saving_pct"] for x in t)
    ct = get(client, "/spend/cash/overview", Plant=KR, Month=AUG)["trend"]
    # Jul c5: 4 x 201002 -> 200 actual, 480 potential, 280 lost, 58.33% of potential
    assert (ct[0]["actual"], ct[0]["potential"], ct[0]["loss"]) == (200.0, 480.0, 280.0)
    assert ct[0]["pct"] == pytest.approx(58.3333, abs=1e-4)
    assert (ct[1]["actual"], ct[1]["potential"], ct[1]["loss"]) == pytest.approx((1680.0, 4860.0, 3180.0))
    assert all(x["pct"] == x["loss_pct"] for x in ct)


def test_healthium_splits_into_sutures_and_endo(client, data):
    h = get(client, "/spend/scheme/overview", Plant=KR, Month=AUG)["groups"]["healthium"]
    assert money(h) == (128.0, 80.0, 48.0)                              # 28 / 20 sutures + 100 / 60 endo
    s, e = h["sutures"], h["endo"]
    assert (s["label"], *money(s)) == ("Sutures", 28.0, 20.0, 8.0)
    assert s["pct"] == pytest.approx(28.5714, abs=1e-4)
    assert (e["label"], *money(e), e["pct"]) == ("Endo", 100.0, 60.0, 40.0, 40.0)
    none = get(client, "/spend/scheme/overview", Plant=DR, Month=AUG)["groups"]["healthium"]
    assert money(none["endo"]) == (0.0, 0.0, 0.0) and none["endo"]["pct"] is None


# ---------- cash & TPA: margin %s, top 10, bifurcation columns ----------

def test_cash_margin_percentages_beside_the_mrp_values(client, data):
    d = get(client, "/spend/cash/overview", Plant=KR, Month=AUG)
    assert d["with_oof"] is None                                            # no purchase build here
    h = d["headline"]
    assert (h["used_mrp_value"], h["suggested_mrp_value"]) == pytest.approx((8880.0, 11250.0))
    assert h["used_margin_pct"] == pytest.approx(18.9189, abs=1e-4)           # 1680 / 8880
    assert h["suggested_margin_pct"] == pytest.approx(43.2, abs=1e-4)         # 4860 / 11250
    assert d["trend"][0]["used_margin_pct"] == pytest.approx(50.0)            # Jul: 200 / 400
    top = d["top_items"]
    assert [(t["material"], t["actual_margin"], t["potential_margin"], t["margin_loss"]) for t in top] == \
        [("101001", 1380.0, 4140.0, 2760.0), ("201002", 300.0, 720.0, 420.0)]
    assert [t["share_of_total_pct"] for t in top] == pytest.approx([86.7925, 13.2075], abs=1e-4)


def test_cash_items_carry_the_bifurcation_and_unit_margin_columns(client, data):
    d = get(client, "/spend/cash/items", Plant=KR, Month=AUG)
    a, b = d["data"]
    # 101001: unit margin 200 of 1200 MRP excl GST, suggested 600 of 1500; HCG's KR Aug allowance
    assert (a["material"], a["bifurcated"], a["sharing"], a["group_qty"], a["allowed_qty"], a["excess_qty"]) == \
        ("101001", True, 0.1, 11.0, 1.1, 6.9)
    assert (a["used_margin_pct"], a["suggested_margin_pct"]) == pytest.approx((16.6667, 40.0), abs=1e-4)
    assert (b["material"], b["bifurcated"], b["sharing"], b["group_qty"], b["allowed_qty"], b["excess_qty"]) == \
        ("201002", False, None, None, None, None)
    assert (b["used_margin_pct"], b["suggested_margin_pct"]) == (50.0, 80.0)
    # no Total for group_qty: brands of one molecule group each carry the whole group's qty
    assert d["sums"] == pytest.approx({"cash_qty": 14.0, "qty": 12.9, "allowed_qty": 1.1,
                                       "excess_qty": 6.9, "actual_margin": 1680.0, "potential_margin": 4860.0,
                                       "margin_loss": 3180.0, "topline_increase": 2370.0, "lines": 3})
    # the allowance is the whole brand's at the hospital and month, whatever the doctor
    one = get(client, "/spend/cash/items", Plant=KR, Month=AUG, Doctor="D2")["data"][0]
    assert (one["material"], one["excess_qty"]) == ("101001", 6.9)
    assert one["qty"] == pytest.approx(2.5875, abs=1e-3)                      # D2's own counted qty
    every = get(client, "/spend/cash/items", Month=AUG)["data"][0]               # KR + DR
    assert (every["group_qty"], every["allowed_qty"], every["excess_qty"]) == (14.0, 1.4, 8.6)


# ---------- tables: Total-row sums over every filtered row ----------

def test_every_list_returns_sums_over_the_filtered_rows(client, data):
    d = get(client, "/spend/attis/items", Plant=KR, Month=AUG, page_size=1)
    assert len(d["data"]) == 1 and d["total"] == 2
    assert d["sums"] == {"qty": 13.0, "used_value": 3500.0, "suggested_value": 1500.0, "saving_missed": 2000.0,
                         "lines": 3}
    assert get(client, "/spend/attis/items", Plant=KR, Month=AUG, global_filter="para")["sums"]["saving_missed"] \
        == 200.0
    assert get(client, "/spend/attis/doctors", Plant=KR, Month=AUG)["sums"]["used_value"] == 3500.0
    s = get(client, "/spend/scheme/items", Plant=KR, Month=AUG, Group="healthium")["sums"]
    assert (s["used_value"], s["suggested_value"], s["saving_missed"]) == (128.0, 80.0, 48.0)
    assert get(client, "/spend/scheme/doctors", Plant=KR, Month=AUG)["sums"]["saving_missed"] == 2098.0
    assert get(client, "/spend/cash/doctors", Plant=KR, Month=AUG)["sums"] == pytest.approx(
        {"cash_qty": 14.0, "qty": 12.9, "actual_margin": 1680.0, "potential_margin": 4860.0, "margin_loss": 3180.0,
         "lines": 3})
    assert get(client, "/spend/attis/consumption", Plant=KR, Month=AUG)["sums"] == \
        {"qty": 5.0, "cost": 1110.0, "gov_cost": 700.0, "ip_cost": 760.0, "op_cost": 350.0, "lines": 3}


# ---------- only computed figures ----------

def test_no_typed_figures_ride_along(client, data):
    for path in ("/spend/scheme/overview", "/spend/cash/overview", "/spend/attis/overview"):
        assert "hcg_reported" not in get(client, path, Plant=KR, Month=AUG)


# ---------- HCG's real KR figures (only when the tables are built) ----------

_SWITCH = Path(settings.SPEND_DATA_DIR) / "scheme_switch.parquet"


def _built(month: str) -> bool:
    return _SWITCH.exists() and month in set(pd.read_parquet(_SWITCH, columns=["period"])["period"].astype(str))


real = pytest.mark.skipif(not _SWITCH.exists(), reason="spend tables not built — run python -m app.etl.spend")


@pytest.fixture()
def real_aug():
    if not _built(AUG):
        pytest.skip(f"not built: {AUG}")
    spend.clear_caches()


@real
def test_real_kr_aug_sutures_and_endo_match_hcgs_summary(client, real_aug):
    h = get(client, "/spend/scheme/overview", Plant=KR, Month=AUG)["groups"]["healthium"]
    # HCG's S6 Summary rows 10-11
    assert money(h["sutures"]) == pytest.approx((50_199.59, 17_233.66, 32_965.93), abs=0.01)
    assert money(h["endo"]) == pytest.approx((28_875.00, 13_377.61, 15_497.39), abs=0.01)
    assert h["sutures"]["pct"] == pytest.approx(65.67, abs=0.01)
    assert h["endo"]["pct"] == pytest.approx(53.67, abs=0.01)
    assert money(h) == pytest.approx((79_074.59, 30_611.27, 48_463.32), abs=0.01)


@real
def test_real_kr_aug_top_10(client, real_aug):
    for path, loss in (("/spend/scheme/overview", "saving_missed"), ("/spend/cash/overview", "margin_loss")):
        d = get(client, path, Plant=KR, Month=AUG)
        top = d["top_items"]
        assert len(top) == 10, path
        assert [t[loss] for t in top] == sorted((t[loss] for t in top), reverse=True), path
        whole = d["headline"][loss]
        for t in top:
            assert t["share_of_total_pct"] == pytest.approx(t[loss] / whole * 100, abs=0.01), (path, t["material"])
    s = get(client, "/spend/scheme/overview", Plant=KR, Month=AUG)["top_items"][0]
    assert s["used_value"] - s["suggested_value"] == pytest.approx(s["saving_missed"], abs=0.02)
    c = get(client, "/spend/cash/overview", Plant=KR, Month=AUG)["top_items"][0]
    assert c["potential_margin"] - c["actual_margin"] == pytest.approx(c["margin_loss"], abs=0.02)


@real
def test_real_kr_jun_onco_plus_non_onco_matches_hcgs_june(client):
    if not _built(JUN):
        pytest.skip(f"not built: {JUN}")
    spend.clear_caches()
    # HCG's Jun-26 email counted drugs only: our Onco + Non-Onco for June, added here
    def drugs(path, fields):
        parts = [get(client, path, Plant=KR, Month=JUN, Segment=x)["headline"] for x in ("onco", "non-onco")]
        return tuple(sum(p[f] for p in parts) for f in fields)
    # HCG's Jun-26 scheme figures to the paisa
    assert drugs("/spend/scheme/overview", ("used_value", "suggested_value", "saving_missed")) == \
        pytest.approx((811_575.18, 489_055.33, 322_519.85), abs=0.01)
    # HCG printed 5,519,879 / 9,898,288 / 4,378,409: a known, unexplained residual (+36,312 actual)
    assert drugs("/spend/cash/overview", ("actual_margin", "potential_margin", "margin_loss")) == \
        pytest.approx((5_556_191.04, 9_936_285.43, 4_380_094.39), abs=0.01)


@real
def test_real_kr_aug_margin_percentages(client, real_aug):
    h = get(client, "/spend/cash/overview", Plant=KR, Month=AUG)["headline"]
    # HCG's S6 CASH&TPA summary: used 1,04,46,684.32 of 2,15,11,386.61 MRP; suggested 1,82,55,008.42 of 2,93,58,412.45
    assert (h["used_mrp_value"], h["suggested_mrp_value"]) == pytest.approx((21_511_386.61, 29_358_412.45), abs=0.01)
    assert h["used_margin_pct"] == pytest.approx(48.56, abs=0.005)
    assert h["suggested_margin_pct"] == pytest.approx(62.18, abs=0.005)


@real
def test_real_kr_aug_cash_items_bifurcation_columns(client, real_aug):
    d = get(client, "/spend/cash/items", Plant=KR, Month=AUG, page_size=5000)
    rows = d["data"]
    assert d["total"] == len(rows)
    bif = [r for r in rows if r["bifurcated"]]
    assert len(bif) == 41                                               # HCG's KR Aug bifurcated brands
    for r in bif:
        assert r["sharing"] is not None and 0 < r["sharing"] <= 1, r["material"]
        assert r["allowed_qty"] == pytest.approx(r["sharing"] * r["group_qty"], abs=0.002), r["material"]
        assert r["excess_qty"] == pytest.approx(r["qty"], abs=0.002), r["material"]   # all of IP/OP/cash/TPA
    for r in rows:
        if not r["bifurcated"]:
            assert (r["sharing"], r["group_qty"], r["allowed_qty"], r["excess_qty"]) == (None,) * 4, r["material"]
    # the Total row adds the rows' 2-dp amounts, so it may sit a few paise off the unrounded headline
    assert d["sums"]["margin_loss"] == pytest.approx(7_808_324.11, abs=1.0)
    assert d["sums"]["excess_qty"] == pytest.approx(sum(r["qty"] for r in bif), abs=0.01)
    # MABTAS and REDITUX 100MG share one molecule group, so both carry its qty (83); a column
    # total would count the group twice, so the table gets none
    shared = {r["material"]: r["group_qty"] for r in bif if r["material"] in ("100308", "100603")}
    assert shared == {"100308": 83.0, "100603": 83.0}
    assert "group_qty" not in d["sums"]
