"""
Deterministic Plotly builder. Turns a catalog Result (rows + chart spec) into a
Plotly {data, layout} dict — the exact transport the reused ChatBubble expects
(it JSON-parses the string and calls Plotly.newPlot). NO LLM-written plot code:
the trace type and fields come from the tool's suggested_chart (or an LLM override
constrained to that schema), and the values are the real rows. The frontend layers
on premium styling; here we produce correct, branded traces.
"""
from __future__ import annotations

# HCG editorial palette (kept in sync with the dashboard accents)
_PALETTE = ["#4b7bd4", "#16a37f", "#e0992f", "#e8604a", "#7c6cd4", "#0ea5e9",
            "#0e9f6e", "#d9663e", "#8a9a5b", "#b5524a"]
_KIND_HOVER = {
    "inr": "%{y:,.0f}",       # value shown; frontend/locale handles ₹ context
    "pct": "%{y:.1f}%",
    "days": "%{y:.0f} d",
    "num": "%{y:,.0f}",
}


def _col_kind(result, key):
    for c in result.get("columns", []):
        if c.get("key") == key:
            return c.get("kind", "num")
    return "num"


def _axis_label(result, key):
    for c in result.get("columns", []):
        if c.get("key") == key:
            return c.get("label", key)
    return key


def build(result: dict, override: dict | None = None) -> dict | None:
    """Return {data, layout} or None when a chart doesn't add value."""
    spec = dict(result.get("suggested_chart") or {})
    if override:
        # LLM may only steer type / y / title — never invent data
        for k in ("type", "x", "y", "title"):
            if override.get(k):
                spec[k] = override[k]
    if not spec:
        return None
    rows = result.get("rows") or []
    if len(rows) < 1:
        return None

    ctype = (spec.get("type") or "bar").lower()
    x = spec.get("x") or "name"
    y = spec.get("y") or next((c["key"] for c in result.get("columns", []) if c.get("kind") in ("inr", "num", "pct", "days") and c["key"] != x), "value")
    title = spec.get("title") or result.get("title") or ""
    xs = [r.get(x) for r in rows]
    ys = [_num(r.get(y)) for r in rows]
    ykind = _col_kind(result, y)
    hov = _KIND_HOVER.get(ykind, "%{y:,.2f}")

    if ctype == "pie":
        data = [{
            "type": "pie", "labels": [str(v) for v in xs], "values": ys, "hole": 0.5,
            "marker": {"colors": _PALETTE[:len(xs)]},
            "textinfo": "label+percent", "textposition": "outside",
            "hovertemplate": "%{label}: %{value:,.0f} (%{percent})<extra></extra>",
        }]
        layout = {"title": {"text": title}, "showlegend": True}
        return {"data": data, "layout": layout}

    if ctype in ("line", "area"):
        data = [{
            "type": "scatter", "mode": "lines+markers", "x": xs, "y": ys, "name": _axis_label(result, y),
            "line": {"color": _PALETTE[0], "width": 3, "shape": "spline"},
            "marker": {"color": _PALETTE[0], "size": 7},
            "fill": "tozeroy" if ctype == "area" else None,
            "fillcolor": "rgba(75,123,212,0.12)" if ctype == "area" else None,
            "hovertemplate": f"%{{x}}<br>{_axis_label(result, y)}: {hov}<extra></extra>",
        }]
    else:  # bar (default)
        data = [{
            "type": "bar", "x": xs, "y": ys, "name": _axis_label(result, y),
            "marker": {"color": _PALETTE[0]},
            "hovertemplate": f"%{{x}}<br>{_axis_label(result, y)}: {hov}<extra></extra>",
        }]

    layout = {
        "title": {"text": title},
        "xaxis": {"title": {"text": _axis_label(result, x)}, "automargin": True},
        "yaxis": {"title": {"text": _axis_label(result, y)}, "automargin": True},
        "showlegend": False,
        "bargap": 0.35,
    }
    return {"data": data, "layout": layout}


def _num(v):
    try:
        if v is None:
            return 0
        return float(v)
    except (TypeError, ValueError):
        return 0
