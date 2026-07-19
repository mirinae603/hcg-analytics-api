"""
Beautiful, deterministic Plotly builder. The agent chooses a chart SPEC (type +
which result columns map to which encodings); this module turns that + the real
rows into a polished Plotly {data, layout} figure. No LLM-written plot code — so
charts are always valid AND consistently gorgeous.

Supported types: bar, grouped_bar, stacked_bar, line, area, combo, pie, donut,
scatter, bubble, heatmap, treemap, sunburst, funnel, waterfall, histogram, box,
indicator (KPI). Premium HCG theme, value-aware hovers/ticks, data labels,
peak annotations, rounded bars, spline lines, gradient fills.
"""
from __future__ import annotations

# ── Theme ───────────────────────────────────────────────────────────────────
PALETTE = ["#4b7bd4", "#16a37f", "#e0992f", "#e8604a", "#7c6cd4", "#0ea5e9",
           "#0e9f6e", "#d9663e", "#8a9a5b", "#b5524a", "#5b8def", "#2bb3a3"]
SEQ = [[0, "#eaf1fb"], [0.35, "#9dc0ef"], [0.7, "#4b7bd4"], [1, "#1f4e9e"]]
FONT = "Outfit, Inter, -apple-system, BlinkMacSystemFont, sans-serif"
INK = "#1f2333"
GRID = "rgba(150,160,175,0.14)"


def _num(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _tickfmt(kind):
    return {"inr": ",.0f", "pct": ".1f", "days": ",.0f", "num": ",.0f"}.get(kind, "")


def _hover(kind, name="%{y}"):
    if kind == "inr":
        return "%{x}<br><b>%{customdata}</b><extra></extra>"
    if kind == "pct":
        return "%{x}<br><b>%{y:.1f}%</b><extra></extra>"
    if kind == "days":
        return "%{x}<br><b>%{y:,.0f} d</b><extra></extra>"
    return "%{x}<br><b>%{y:,.0f}</b><extra></extra>"


def _inr_label(v):
    if v is None:
        return ""
    a = abs(v)
    if a >= 1e7:
        return f"₹{v/1e7:.2f} Cr"
    if a >= 1e5:
        return f"₹{v/1e5:.2f} L"
    if a >= 1e3:
        return f"₹{v/1e3:.1f} K"
    return f"₹{v:.0f}"


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


def _base_layout(title, kind=None, horizontal=False):
    lay = {
        "title": {"text": title or "", "font": {"size": 15, "color": INK, "family": FONT}, "x": 0.02, "xanchor": "left", "y": 0.96, "yanchor": "top"},
        "font": {"family": FONT, "size": 11.5, "color": "#5a6072"},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "margin": {"l": 66, "r": 26, "t": 58, "b": 66, "pad": 4},
        "hoverlabel": {"bgcolor": "#ffffff", "bordercolor": "#e6e9f0", "font": {"family": FONT, "size": 12, "color": INK}},
        "showlegend": False,
        "xaxis": {"gridcolor": GRID, "zerolinecolor": GRID, "automargin": True, "tickfont": {"size": 10.5}},
        "yaxis": {"gridcolor": GRID, "zerolinecolor": GRID, "automargin": True, "tickfont": {"size": 10.5}},
    }
    valax = "xaxis" if horizontal else "yaxis"
    if kind:
        lay[valax]["tickformat"] = _tickfmt(kind)
    return lay


def _rows_get(rows, key):
    return [r.get(key) for r in rows]


def build(rows: list[dict], spec: dict) -> dict | None:
    """rows: query result rows. spec: {type,x,y,color,size,title,value_format,orientation,y2}."""
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
            return _pie(rows, x, ys[0] if ys else None, title, hole=0.55 if t == "donut" else 0.0, kind=kind)
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
            return _combo(rows, x, ys, spec.get("y2"), title, kind)
        if t in ("grouped_bar", "stacked_bar", "bar"):
            return _bar(rows, x, ys, title, kind, horizontal, stack=(t == "stacked_bar"))
    except Exception:
        # any spec/data mismatch → fall back to a simple bar of the first value col
        try:
            vy = ys[0] if ys else next((c for c in rows[0].keys() if isinstance(_num(rows[0][c]), float)), None)
            return _bar(rows, x or list(rows[0].keys())[0], [vy], title, kind, horizontal, stack=False)
        except Exception:
            return None
    return None


# ── builders ────────────────────────────────────────────────────────────────
def _bar(rows, x, ys, title, kind, horizontal, stack):
    xs = _rows_get(rows, x)
    data = []
    for i, yc in enumerate(ys):
        vals = [_num(v) for v in _rows_get(rows, yc)]
        labels = [_label(v, kind) for v in vals]
        col = PALETTE[i % len(PALETTE)]
        single = len(ys) == 1
        trace = {
            "type": "bar", "name": yc,
            "orientation": "h" if horizontal else "v",
            "marker": {"color": col, "line": {"width": 0}, "cornerradius": 6},
            "customdata": labels,
            "text": labels if single else None,
            "textposition": "outside" if (single and not horizontal) else ("auto" if single else "none"),
            "textfont": {"size": 10.5, "color": INK, "family": FONT},
            "hovertemplate": (f"%{{y}}<br><b>%{{customdata}}</b><extra>{yc}</extra>" if horizontal else f"%{{x}}<br><b>%{{customdata}}</b><extra>{yc}</extra>"),
            "cliponaxis": False,
        }
        if horizontal:
            trace["x"] = vals
            trace["y"] = xs
        else:
            trace["x"] = xs
            trace["y"] = vals
        data.append(trace)
    lay = _base_layout(title, kind, horizontal)
    lay["bargap"] = 0.34
    if len(ys) > 1:
        lay["showlegend"] = True
        lay["barmode"] = "stack" if stack else "group"
        lay["legend"] = {"orientation": "h", "y": -0.16, "x": 0.5, "xanchor": "center"}
    if horizontal:
        lay["yaxis"]["autorange"] = "reversed"
    return {"data": data, "layout": lay}


def _line(rows, x, ys, title, kind, area):
    xs = _rows_get(rows, x)
    data = []
    for i, yc in enumerate(ys):
        vals = [_num(v) for v in _rows_get(rows, yc)]
        col = PALETTE[i % len(PALETTE)]
        data.append({
            "type": "scatter", "mode": "lines+markers", "name": yc, "x": xs, "y": vals,
            "line": {"color": col, "width": 3, "shape": "spline"},
            "marker": {"color": col, "size": 7, "line": {"color": "#fff", "width": 1.5}},
            "fill": "tozeroy" if area else None,
            "fillcolor": (col.replace(")", "") and f"rgba({int(col[1:3],16)},{int(col[3:5],16)},{int(col[5:7],16)},0.12)") if area else None,
            "customdata": [_label(v, kind) for v in vals],
            "hovertemplate": f"%{{x}}<br><b>%{{customdata}}</b><extra>{yc}</extra>",
        })
    lay = _base_layout(title, kind, False)
    if len(ys) > 1:
        lay["showlegend"] = True
        lay["legend"] = {"orientation": "h", "y": -0.16, "x": 0.5, "xanchor": "center"}
    return {"data": data, "layout": lay}


def _combo(rows, x, ys, y2, title, kind):
    xs = _rows_get(rows, x)
    data = []
    for i, yc in enumerate(ys):
        vals = [_num(v) for v in _rows_get(rows, yc)]
        data.append({"type": "bar", "name": yc, "x": xs, "y": vals, "marker": {"color": PALETTE[i % len(PALETTE)], "cornerradius": 5},
                     "customdata": [_label(v, kind) for v in vals], "hovertemplate": f"%{{x}}<br><b>%{{customdata}}</b><extra>{yc}</extra>"})
    if y2:
        v2 = [_num(v) for v in _rows_get(rows, y2)]
        data.append({"type": "scatter", "mode": "lines+markers", "name": y2, "x": xs, "y": v2, "yaxis": "y2",
                     "line": {"color": PALETTE[3], "width": 3, "shape": "spline"}, "marker": {"color": PALETTE[3], "size": 7},
                     "hovertemplate": f"%{{x}}<br><b>%{{y:,.1f}}</b><extra>{y2}</extra>"})
    lay = _base_layout(title, kind, False)
    lay["showlegend"] = True
    lay["legend"] = {"orientation": "h", "y": -0.16, "x": 0.5, "xanchor": "center"}
    if y2:
        lay["yaxis2"] = {"overlaying": "y", "side": "right", "showgrid": False, "tickfont": {"size": 10.5}}
    return {"data": data, "layout": lay}


def _pie(rows, x, y, title, hole, kind):
    labels = [str(v) for v in _rows_get(rows, x)]
    vals = [_num(v) for v in _rows_get(rows, y)]
    data = [{"type": "pie", "labels": labels, "values": vals, "hole": hole, "sort": True, "direction": "clockwise",
             "marker": {"colors": PALETTE[:len(labels)], "line": {"color": "#fff", "width": 2}},
             "textinfo": "label+percent", "textposition": "outside", "textfont": {"size": 11, "family": FONT},
             "customdata": [_label(v, kind) for v in vals],
             "hovertemplate": "<b>%{label}</b><br>%{customdata} · %{percent}<extra></extra>"}]
    lay = _base_layout(title)
    lay["margin"] = {"l": 20, "r": 20, "t": 58, "b": 30}
    return {"data": data, "layout": lay}


def _treemap(rows, x, y, title, kind):
    labels = [str(v) for v in _rows_get(rows, x)]
    vals = [_num(v) for v in _rows_get(rows, y)]
    data = [{"type": "treemap", "labels": labels, "parents": [""] * len(labels), "values": vals,
             "marker": {"colors": PALETTE * (len(labels) // len(PALETTE) + 1), "line": {"width": 2, "color": "#fff"}},
             "textinfo": "label+value", "customdata": [_label(v, kind) for v in vals],
             "hovertemplate": "<b>%{label}</b><br>%{customdata}<extra></extra>", "texttemplate": "%{label}<br>%{customdata}"}]
    lay = _base_layout(title)
    lay["margin"] = {"l": 10, "r": 10, "t": 52, "b": 10}
    return {"data": data, "layout": lay}


def _sunburst(rows, spec, title, kind):
    # expects x = leaf label, color = parent group, y = value
    parent = spec.get("color")
    leaf = spec.get("x")
    y = (spec.get("y") if isinstance(spec.get("y"), str) else (spec.get("y") or [None])[0])
    parents = sorted({str(r.get(parent)) for r in rows}) if parent else []
    labels, pars, vals = [], [], []
    for p in parents:
        labels.append(p); pars.append(""); vals.append(0)
    for r in rows:
        labels.append(str(r.get(leaf))); pars.append(str(r.get(parent)) if parent else ""); vals.append(_num(r.get(y)))
    data = [{"type": "sunburst", "labels": labels, "parents": pars, "values": vals, "branchvalues": "total",
             "marker": {"line": {"width": 1.5, "color": "#fff"}}, "hovertemplate": "<b>%{label}</b><br>%{value:,.0f}<extra></extra>"}]
    lay = _base_layout(title); lay["margin"] = {"l": 10, "r": 10, "t": 52, "b": 10}
    return {"data": data, "layout": lay}


def _heatmap(rows, spec, title):
    x = spec.get("x"); yk = spec.get("color") or spec.get("series")
    z = (spec.get("y") if isinstance(spec.get("y"), str) else (spec.get("y") or [None])[0])
    xs = sorted({str(r.get(x)) for r in rows})
    ys = sorted({str(r.get(yk)) for r in rows})
    zmap = {(str(r.get(yk)), str(r.get(x))): _num(r.get(z)) for r in rows}
    zmat = [[zmap.get((yy, xx)) for xx in xs] for yy in ys]
    data = [{"type": "heatmap", "x": xs, "y": ys, "z": zmat, "colorscale": SEQ, "hoverongaps": False,
             "colorbar": {"thickness": 12, "outlinewidth": 0, "len": 0.8},
             "hovertemplate": "%{x} · %{y}<br><b>%{z:,.1f}</b><extra></extra>"}]
    lay = _base_layout(title)
    return {"data": data, "layout": lay}


def _funnel(rows, x, y, title, kind):
    labels = [str(v) for v in _rows_get(rows, x)]
    vals = [_num(v) for v in _rows_get(rows, y)]
    data = [{"type": "funnel", "y": labels, "x": vals, "marker": {"color": PALETTE[:len(labels)]},
             "textinfo": "value+percent initial", "customdata": [_label(v, kind) for v in vals],
             "hovertemplate": "%{y}<br><b>%{customdata}</b><extra></extra>"}]
    lay = _base_layout(title, kind, True)
    return {"data": data, "layout": lay}


def _waterfall(rows, x, y, title, kind):
    labels = [str(v) for v in _rows_get(rows, x)]
    vals = [_num(v) for v in _rows_get(rows, y)]
    data = [{"type": "waterfall", "x": labels, "y": vals, "connector": {"line": {"color": "rgba(150,160,175,0.4)"}},
             "increasing": {"marker": {"color": PALETTE[1]}}, "decreasing": {"marker": {"color": PALETTE[3]}},
             "totals": {"marker": {"color": PALETTE[0]}}, "text": [_label(v, kind) for v in vals], "textposition": "outside"}]
    lay = _base_layout(title, kind, False)
    return {"data": data, "layout": lay}


def _scatter(rows, spec, title, kind, bubble):
    x = spec.get("x"); y = (spec.get("y") if isinstance(spec.get("y"), str) else (spec.get("y") or [None])[0])
    color = spec.get("color"); size = spec.get("size")
    labelcol = next((c for c in rows[0].keys() if c not in (x, y, color, size)), None)
    xs = [_num(r.get(x)) for r in rows]
    yvals = [_num(r.get(y)) for r in rows]
    sizes = [_num(r.get(size)) for r in rows] if size else None
    if sizes:
        mx = max([s for s in sizes if s] or [1])
        sizes = [8 + 34 * (s / mx if s else 0) for s in sizes]
    marker = {"color": PALETTE[0], "size": sizes or 11, "opacity": 0.8, "line": {"color": "#fff", "width": 1}}
    if color:
        groups = sorted({str(r.get(color)) for r in rows})
        cmap = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}
        marker["color"] = [cmap[str(r.get(color))] for r in rows]
    data = [{"type": "scatter", "mode": "markers", "x": xs, "y": yvals, "marker": marker,
             "text": [str(r.get(labelcol)) for r in rows] if labelcol else None,
             "hovertemplate": (f"<b>%{{text}}</b><br>{x}: %{{x:,.1f}}<br>{y}: %{{y:,.1f}}<extra></extra>" if labelcol else f"{x}: %{{x:,.1f}}<br>{y}: %{{y:,.1f}}<extra></extra>")}]
    lay = _base_layout(title)
    lay["xaxis"]["title"] = {"text": x, "font": {"size": 11}}
    lay["yaxis"]["title"] = {"text": y, "font": {"size": 11}}
    return {"data": data, "layout": lay}


def _histogram(rows, col, title, kind):
    vals = [_num(r.get(col)) for r in rows]
    data = [{"type": "histogram", "x": vals, "marker": {"color": PALETTE[0], "line": {"color": "#fff", "width": 1}}, "nbinsx": 24, "opacity": 0.9}]
    lay = _base_layout(title)
    lay["xaxis"]["title"] = {"text": col, "font": {"size": 11}}
    lay["bargap"] = 0.04
    return {"data": data, "layout": lay}


def _box(rows, spec, title, kind):
    y = (spec.get("y") if isinstance(spec.get("y"), str) else (spec.get("y") or [None])[0])
    group = spec.get("x") or spec.get("color")
    data = []
    if group:
        for i, g in enumerate(sorted({str(r.get(group)) for r in rows})):
            data.append({"type": "box", "name": g, "y": [_num(r.get(y)) for r in rows if str(r.get(group)) == g],
                         "marker": {"color": PALETTE[i % len(PALETTE)]}, "boxmean": True})
    else:
        data.append({"type": "box", "y": [_num(r.get(y)) for r in rows], "marker": {"color": PALETTE[0]}, "boxmean": True})
    lay = _base_layout(title, kind, False)
    lay["showlegend"] = len(data) > 1
    return {"data": data, "layout": lay}


def _indicator(rows, spec, kind, title):
    y = (spec.get("y") if isinstance(spec.get("y"), str) else (spec.get("y") or [None])[0])
    val = _num(rows[0].get(y)) if rows else 0
    data = [{"type": "indicator", "mode": "number", "value": val,
             "number": {"font": {"size": 46, "color": INK, "family": FONT},
                        "prefix": "₹" if kind == "inr" else "", "suffix": "%" if kind == "pct" else (" d" if kind == "days" else "")},
             "title": {"text": title, "font": {"size": 14, "color": "#5a6072"}}}]
    lay = _base_layout("")
    lay["margin"] = {"l": 20, "r": 20, "t": 30, "b": 20}
    return {"data": data, "layout": lay}
