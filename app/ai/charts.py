"""
Beautiful, deterministic Plotly builder. The agent chooses a chart SPEC (type +
which result columns map to which encodings); this module turns that + the real
rows into a polished Plotly {data, layout} figure. No LLM-written plot code.

Quality rules baked in:
  • value axes show Indian units (₹Cr / ₹L) with nice ticks + ~15% headroom so
    top bars & their labels are never clipped — never raw "300,000,000".
  • a value + a percentage in the same result → a combo (bars + % line on y2).
  • premium HCG theme, rounded bars, spline lines, data labels, clean legends.
"""
from __future__ import annotations
import math

PALETTE = ["#3b5bdb", "#12b886", "#e0992f", "#e8604a", "#7048e8", "#0ea5e9",
           "#0ca678", "#d9663e", "#748ffc", "#f06595", "#22b8cf", "#82c91e"]
SEQ = [[0, "#eef3ff"], [0.4, "#9db8f0"], [0.75, "#3b5bdb"], [1, "#1e3a8a"]]
FONT = "Inter, 'Segoe UI', -apple-system, sans-serif"
INK = "#1a1f36"
MUT = "#8a91a3"
GRID = "rgba(150,160,180,0.13)"


def _num(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _rows_get(rows, key):
    return [r.get(key) for r in rows]


def _nice_ceil(x: float) -> float:
    if x <= 0:
        return 1.0
    exp = math.floor(math.log10(x))
    base = 10 ** exp
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * base >= x:
            return m * base
    return 10 * base


def _inr_unit(m: float):
    if m >= 1e7:
        return 1e7, " Cr"
    if m >= 1e5:
        return 1e5, " L"
    if m >= 1e3:
        return 1e3, " K"
    return 1.0, ""


def _inr_label(v):
    if v is None:
        return ""
    div, suf = _inr_unit(abs(v))
    d = 2 if div >= 1e5 else (1 if div == 1e3 else 0)
    return f"₹{v/div:.{d}f}{suf}"


def _label(v, kind):
    if v is None:
        return ""
    if kind == "inr":
        return _inr_label(v)
    if kind == "pct":
        return f"{v:.1f}%"
    if kind == "days":
        return f"{v:,.0f} d"
    return f"{v:,.0f}"


def _value_axis(vals, kind, title=None):
    nums = [abs(v) for v in vals if v is not None]
    m = max(nums) if nums else 1.0
    ax = {"gridcolor": GRID, "zerolinecolor": GRID, "automargin": True, "tickfont": {"size": 10.5, "color": MUT},
          "showline": False, "rangemode": "tozero"}
    if title:
        ax["title"] = {"text": title, "font": {"size": 10.5, "color": MUT}}
    if kind == "inr":
        top = _nice_ceil(m)
        div, suf = _inr_unit(top)
        step = top / 5
        ticks = [round(step * i, 4) for i in range(6)]
        d = 2 if div >= 1e5 else (1 if div == 1e3 else 0)
        ax["tickvals"] = ticks
        ax["ticktext"] = [f"₹{t/div:.{0 if t/div == int(t/div) else d}f}{suf}" for t in ticks]
        ax["range"] = [0, top * 1.16]  # headroom for outside labels
    elif kind == "pct":
        top = min(100.0, _nice_ceil(m))
        ax["ticksuffix"] = "%"
        ax["range"] = [0, top * 1.16]
    elif kind == "days":
        ax["ticksuffix"] = " d"
        ax["range"] = [0, _nice_ceil(m) * 1.14]
    else:
        ax["range"] = [0, _nice_ceil(m) * 1.14]
    return ax


def _base_layout(title):
    return {
        "title": {"text": title or "", "font": {"size": 15, "color": INK, "family": FONT}, "x": 0.015, "xanchor": "left", "y": 0.96, "yanchor": "top"},
        "font": {"family": FONT, "size": 11.5, "color": MUT},
        "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
        "margin": {"l": 70, "r": 24, "t": 56, "b": 62, "pad": 4},
        "hoverlabel": {"bgcolor": "#ffffff", "bordercolor": "#e6e9f0", "font": {"family": FONT, "size": 12, "color": INK}},
        "showlegend": False,
        "xaxis": {"gridcolor": "rgba(0,0,0,0)", "zerolinecolor": GRID, "automargin": True, "tickfont": {"size": 10.5, "color": MUT}, "showline": False},
        "yaxis": {"gridcolor": GRID, "zerolinecolor": GRID, "automargin": True, "tickfont": {"size": 10.5, "color": MUT}},
    }


def build(rows: list[dict], spec: dict) -> dict | None:
    if not rows or not spec:
        return None
    t = (spec.get("type") or "bar").lower().strip()
    title = spec.get("title") or ""
    kind = (spec.get("value_format") or "num").lower()
    x = spec.get("x")
    y = spec.get("y")
    ys = y if isinstance(y, list) else ([y] if y else [])
    horizontal = (spec.get("orientation") or ("h" if t == "bar" and len(rows) > 8 else "v")) == "h"
    try:
        if t == "indicator":
            return _indicator(rows, spec, kind, title)
        if t in ("pie", "donut"):
            return _pie(rows, x, ys[0] if ys else None, title, hole=0.58 if t == "donut" else 0.0, kind=kind)
        if t == "treemap":
            return _treemap(rows, x, ys[0] if ys else None, title, kind)
        if t == "sunburst":
            return _sunburst(rows, spec, title, kind)
        if t == "heatmap":
            return _heatmap(rows, spec, title)
        if t == "funnel":
            return _funnel(rows, x, ys[0] if ys else None, title, kind)
        if t == "waterfall":
            return _waterfall(rows, x, ys[0] if ys else None, title, kind)
        if t in ("scatter", "bubble"):
            return _scatter(rows, spec, title, kind, bubble=(t == "bubble"))
        if t == "histogram":
            return _histogram(rows, x or (ys[0] if ys else None), title, kind)
        if t == "box":
            return _box(rows, spec, title, kind)
        if t in ("line", "area"):
            return _line(rows, x, ys, title, kind, area=(t == "area"))
        if t == "combo":
            return _combo(rows, x, ys, spec.get("y2"), title, kind, spec.get("y2_format"))
        return _bar(rows, x, ys, title, kind, horizontal, stack=(t == "stacked_bar"))
    except Exception:
        try:
            vy = ys[0] if ys else next((c for c in rows[0].keys() if isinstance(_num(rows[0][c]), float)), None)
            return _bar(rows, x or list(rows[0].keys())[0], [vy], title, kind, horizontal, stack=False)
        except Exception:
            return None


def _bar(rows, x, ys, title, kind, horizontal, stack):
    xs = _rows_get(rows, x)
    allvals = []
    data = []
    single = len(ys) == 1
    for i, yc in enumerate(ys):
        vals = [_num(v) for v in _rows_get(rows, yc)]
        allvals += [v for v in vals if v is not None]
        labels = [_label(v, kind) for v in vals]
        trace = {
            "type": "bar", "name": yc, "orientation": "h" if horizontal else "v",
            "marker": {"color": PALETTE[i % len(PALETTE)], "line": {"width": 0}, "cornerradius": 7},
            "customdata": labels,
            "text": labels if single else None,
            "textposition": "outside" if single else "none",
            "textfont": {"size": 10.5, "color": INK, "family": FONT},
            "cliponaxis": False,
            "hovertemplate": (f"%{{y}}<br><b>%{{customdata}}</b><extra>{yc}</extra>" if horizontal else f"%{{x}}<br><b>%{{customdata}}</b><extra>{yc}</extra>"),
        }
        if horizontal:
            trace["x"] = vals; trace["y"] = xs
        else:
            trace["x"] = xs; trace["y"] = vals
        data.append(trace)
    lay = _base_layout(title)
    lay["bargap"] = 0.36
    vax = _value_axis(allvals, kind)
    if horizontal:
        lay["xaxis"] = {**lay["xaxis"], **vax, "gridcolor": GRID}
        lay["yaxis"]["autorange"] = "reversed"
    else:
        lay["yaxis"] = {**lay["yaxis"], **vax}
    if len(ys) > 1:
        lay["showlegend"] = True
        lay["barmode"] = "stack" if stack else "group"
        lay["legend"] = {"orientation": "h", "y": -0.18, "x": 0.5, "xanchor": "center", "font": {"size": 10.5}}
    return {"data": data, "layout": lay}


def _line(rows, x, ys, title, kind, area):
    xs = _rows_get(rows, x)
    allvals = []
    data = []
    for i, yc in enumerate(ys):
        vals = [_num(v) for v in _rows_get(rows, yc)]
        allvals += [v for v in vals if v is not None]
        col = PALETTE[i % len(PALETTE)]
        rgb = tuple(int(col[j:j + 2], 16) for j in (1, 3, 5))
        data.append({
            "type": "scatter", "mode": "lines+markers", "name": yc, "x": xs, "y": vals,
            "line": {"color": col, "width": 3, "shape": "spline"},
            "marker": {"color": col, "size": 7, "line": {"color": "#fff", "width": 1.5}},
            "fill": "tozeroy" if area else None,
            "fillcolor": f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.10)" if area else None,
            "customdata": [_label(v, kind) for v in vals],
            "hovertemplate": f"%{{x}}<br><b>%{{customdata}}</b><extra>{yc}</extra>",
        })
    lay = _base_layout(title)
    lay["yaxis"] = {**lay["yaxis"], **_value_axis(allvals, kind)}
    lay["xaxis"]["gridcolor"] = "rgba(0,0,0,0)"
    if len(ys) > 1:
        lay["showlegend"] = True
        lay["legend"] = {"orientation": "h", "y": -0.18, "x": 0.5, "xanchor": "center", "font": {"size": 10.5}}
    return {"data": data, "layout": lay}


def _combo(rows, x, ys, y2, title, kind, y2_format=None):
    xs = _rows_get(rows, x)
    primvals = []
    data = []
    for i, yc in enumerate(ys):
        vals = [_num(v) for v in _rows_get(rows, yc)]
        primvals += [v for v in vals if v is not None]
        data.append({"type": "bar", "name": yc, "x": xs, "y": vals, "marker": {"color": PALETTE[i % len(PALETTE)], "cornerradius": 6},
                     "customdata": [_label(v, kind) for v in vals], "hovertemplate": f"%{{x}}<br><b>%{{customdata}}</b><extra>{yc}</extra>"})
    y2vals = []
    if y2:
        y2vals = [_num(v) for v in _rows_get(rows, y2)]
        y2k = (y2_format or ("pct" if "pct" in str(y2).lower() or "rate" in str(y2).lower() or "%" in str(y2) else "num"))
        data.append({"type": "scatter", "mode": "lines+markers+text", "name": y2, "x": xs, "y": y2vals, "yaxis": "y2",
                     "line": {"color": INK, "width": 2.5, "shape": "spline"}, "marker": {"color": INK, "size": 6},
                     "text": [_label(v, y2k) for v in y2vals], "textposition": "top center", "textfont": {"size": 9.5, "color": INK},
                     "hovertemplate": f"%{{x}}<br><b>%{{text}}</b><extra>{y2}</extra>"})
    lay = _base_layout(title)
    lay["yaxis"] = {**lay["yaxis"], **_value_axis(primvals, kind)}
    lay["xaxis"]["gridcolor"] = "rgba(0,0,0,0)"
    lay["showlegend"] = True
    lay["legend"] = {"orientation": "h", "y": -0.18, "x": 0.5, "xanchor": "center", "font": {"size": 10.5}}
    if y2:
        y2ax = _value_axis(y2vals, y2k)
        y2ax.update({"overlaying": "y", "side": "right", "showgrid": False, "rangemode": "tozero"})
        lay["yaxis2"] = y2ax
        lay["margin"]["r"] = 58
    return {"data": data, "layout": lay}


def _pie(rows, x, y, title, hole, kind):
    labels = [str(v) for v in _rows_get(rows, x)]
    vals = [_num(v) for v in _rows_get(rows, y)]
    many = len(labels) > 6
    data = [{"type": "pie", "labels": labels, "values": vals, "hole": hole, "sort": True, "direction": "clockwise",
             "marker": {"colors": (PALETTE * (len(labels) // len(PALETTE) + 1))[:len(labels)], "line": {"color": "#fff", "width": 2}},
             "textinfo": "percent" if many else "label+percent", "textposition": "inside" if many else "outside",
             "insidetextorientation": "horizontal", "textfont": {"size": 11, "family": FONT}, "automargin": True,
             "customdata": [_label(v, kind) for v in vals],
             "hovertemplate": "<b>%{label}</b><br>%{customdata} · %{percent}<extra></extra>"}]
    lay = _base_layout(title)
    lay["height"] = 430
    lay["margin"] = {"l": 24, "r": 24, "t": 54, "b": 24}
    lay["uniformtext"] = {"minsize": 9, "mode": "hide"}
    if many:
        lay["showlegend"] = True
        lay["legend"] = {"orientation": "h", "y": -0.02, "x": 0.5, "xanchor": "center", "font": {"size": 10}}
    return {"data": data, "layout": lay}


def _treemap(rows, x, y, title, kind):
    labels = [str(v) for v in _rows_get(rows, x)]
    vals = [_num(v) for v in _rows_get(rows, y)]
    data = [{"type": "treemap", "labels": labels, "parents": [""] * len(labels), "values": vals,
             "marker": {"colors": (PALETTE * (len(labels) // len(PALETTE) + 1))[:len(labels)], "line": {"width": 2, "color": "#fff"}},
             "customdata": [_label(v, kind) for v in vals], "texttemplate": "%{label}<br>%{customdata}",
             "hovertemplate": "<b>%{label}</b><br>%{customdata}<extra></extra>"}]
    lay = _base_layout(title); lay["height"] = 420; lay["margin"] = {"l": 8, "r": 8, "t": 52, "b": 8}
    return {"data": data, "layout": lay}


def _sunburst(rows, spec, title, kind):
    parent = spec.get("color"); leaf = spec.get("x")
    y = spec.get("y") if isinstance(spec.get("y"), str) else (spec.get("y") or [None])[0]
    labels, pars, vals = [], [], []
    for p in sorted({str(r.get(parent)) for r in rows}) if parent else []:
        labels.append(p); pars.append(""); vals.append(0)
    for r in rows:
        labels.append(str(r.get(leaf))); pars.append(str(r.get(parent)) if parent else ""); vals.append(_num(r.get(y)))
    data = [{"type": "sunburst", "labels": labels, "parents": pars, "values": vals, "branchvalues": "total",
             "marker": {"line": {"width": 1.5, "color": "#fff"}}, "hovertemplate": "<b>%{label}</b><br>%{value:,.0f}<extra></extra>"}]
    lay = _base_layout(title); lay["height"] = 420; lay["margin"] = {"l": 8, "r": 8, "t": 52, "b": 8}
    return {"data": data, "layout": lay}


def _heatmap(rows, spec, title):
    x = spec.get("x"); yk = spec.get("color") or spec.get("series")
    z = spec.get("y") if isinstance(spec.get("y"), str) else (spec.get("y") or [None])[0]
    xs = sorted({str(r.get(x)) for r in rows}); ys = sorted({str(r.get(yk)) for r in rows})
    zmap = {(str(r.get(yk)), str(r.get(x))): _num(r.get(z)) for r in rows}
    zmat = [[zmap.get((yy, xx)) for xx in xs] for yy in ys]
    data = [{"type": "heatmap", "x": xs, "y": ys, "z": zmat, "colorscale": SEQ, "hoverongaps": False,
             "colorbar": {"thickness": 12, "outlinewidth": 0, "len": 0.8},
             "hovertemplate": "%{x} · %{y}<br><b>%{z:,.1f}</b><extra></extra>"}]
    return {"data": data, "layout": _base_layout(title)}


def _funnel(rows, x, y, title, kind):
    labels = [str(v) for v in _rows_get(rows, x)]; vals = [_num(v) for v in _rows_get(rows, y)]
    data = [{"type": "funnel", "y": labels, "x": vals, "marker": {"color": PALETTE[:len(labels)]},
             "textinfo": "value+percent initial", "customdata": [_label(v, kind) for v in vals],
             "hovertemplate": "%{y}<br><b>%{customdata}</b><extra></extra>"}]
    lay = _base_layout(title); lay["margin"]["l"] = 120
    return {"data": data, "layout": lay}


def _waterfall(rows, x, y, title, kind):
    labels = [str(v) for v in _rows_get(rows, x)]; vals = [_num(v) for v in _rows_get(rows, y)]
    data = [{"type": "waterfall", "x": labels, "y": vals, "connector": {"line": {"color": "rgba(150,160,180,0.4)"}},
             "increasing": {"marker": {"color": PALETTE[1]}}, "decreasing": {"marker": {"color": PALETTE[3]}},
             "totals": {"marker": {"color": PALETTE[0]}}, "text": [_label(v, kind) for v in vals], "textposition": "outside"}]
    lay = _base_layout(title); lay["yaxis"] = {**lay["yaxis"], **_value_axis(vals, kind)}
    return {"data": data, "layout": lay}


def _scatter(rows, spec, title, kind, bubble):
    x = spec.get("x"); y = spec.get("y") if isinstance(spec.get("y"), str) else (spec.get("y") or [None])[0]
    color = spec.get("color"); size = spec.get("size")
    labelcol = next((c for c in rows[0].keys() if c not in (x, y, color, size)), None)
    xs = [_num(r.get(x)) for r in rows]; yvals = [_num(r.get(y)) for r in rows]
    sizes = [_num(r.get(size)) for r in rows] if size else None
    if sizes:
        mx = max([s for s in sizes if s] or [1]); sizes = [9 + 34 * (s / mx if s else 0) for s in sizes]
    marker = {"color": PALETTE[0], "size": sizes or 11, "opacity": 0.82, "line": {"color": "#fff", "width": 1}}
    if color:
        cmap = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(sorted({str(r.get(color)) for r in rows}))}
        marker["color"] = [cmap[str(r.get(color))] for r in rows]
    data = [{"type": "scatter", "mode": "markers", "x": xs, "y": yvals, "marker": marker,
             "text": [str(r.get(labelcol)) for r in rows] if labelcol else None,
             "hovertemplate": (f"<b>%{{text}}</b><br>{x}: %{{x:,.1f}}<br>{y}: %{{y:,.1f}}<extra></extra>" if labelcol else f"{x}: %{{x:,.1f}}<br>{y}: %{{y:,.1f}}<extra></extra>")}]
    lay = _base_layout(title)
    lay["xaxis"]["title"] = {"text": x, "font": {"size": 11, "color": MUT}}
    lay["yaxis"]["title"] = {"text": y, "font": {"size": 11, "color": MUT}}
    lay["xaxis"]["gridcolor"] = GRID
    return {"data": data, "layout": lay}


def _histogram(rows, col, title, kind):
    vals = [_num(r.get(col)) for r in rows]
    data = [{"type": "histogram", "x": vals, "marker": {"color": PALETTE[0], "line": {"color": "#fff", "width": 1}}, "nbinsx": 24, "opacity": 0.9}]
    lay = _base_layout(title); lay["xaxis"]["title"] = {"text": col, "font": {"size": 11}}; lay["bargap"] = 0.04
    return {"data": data, "layout": lay}


def _box(rows, spec, title, kind):
    y = spec.get("y") if isinstance(spec.get("y"), str) else (spec.get("y") or [None])[0]
    group = spec.get("x") or spec.get("color")
    data = []
    if group:
        for i, g in enumerate(sorted({str(r.get(group)) for r in rows})):
            data.append({"type": "box", "name": g, "y": [_num(r.get(y)) for r in rows if str(r.get(group)) == g], "marker": {"color": PALETTE[i % len(PALETTE)]}, "boxmean": True})
    else:
        data.append({"type": "box", "y": [_num(r.get(y)) for r in rows], "marker": {"color": PALETTE[0]}, "boxmean": True})
    lay = _base_layout(title); lay["showlegend"] = len(data) > 1
    return {"data": data, "layout": lay}


def _indicator(rows, spec, kind, title):
    y = spec.get("y") if isinstance(spec.get("y"), str) else (spec.get("y") or [None])[0]
    val = _num(rows[0].get(y)) if rows else 0
    if kind == "inr":
        div, suf = _inr_unit(abs(val or 0)); num = {"value": (val or 0) / div, "font": {"size": 46, "color": INK, "family": FONT}, "prefix": "₹", "suffix": suf, "valueformat": ".2f"}
    else:
        num = {"value": val or 0, "font": {"size": 46, "color": INK, "family": FONT}, "suffix": "%" if kind == "pct" else (" d" if kind == "days" else ""), "valueformat": ",.0f"}
    data = [{"type": "indicator", "mode": "number", "number": num, "title": {"text": title, "font": {"size": 14, "color": MUT}}}]
    lay = _base_layout(""); lay["height"] = 200; lay["margin"] = {"l": 20, "r": 20, "t": 30, "b": 20}
    return {"data": data, "layout": lay}
