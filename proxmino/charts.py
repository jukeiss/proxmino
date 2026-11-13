"""Simple charts drawn by the app itself, so a model only has to send the numbers (a local 8B model writing a whole
SVG by hand took close to a minute)."""
from __future__ import annotations

import math
from html import escape

COLORS = ["#7aa2ff", "#ff7a59", "#5cc8a0", "#f2c94c", "#bb86fc", "#56ccf2", "#eb5757", "#a0a0a8"]


def _num(v) -> float:
    try:
        return float(str(v).replace(",", "").replace("$", "").replace("%", ""))
    except ValueError:
        return 0.0


def svg_chart(title: str, labels: list, values: list, kind: str = "bar") -> str:
    labels = [str(x)[:24] for x in labels][:24]
    vals = [_num(v) for v in values][: len(labels)]
    labels = labels[: len(vals)]
    if not vals:
        raise ValueError("A chart needs at least one label and value")
    W, H, L, R, T, B = 640, 400, 56, 24, 48, 56
    pw, ph = W - L - R, H - T - B
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui, sans-serif">',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
           f'<text x="{W/2}" y="28" text-anchor="middle" font-size="18" font-weight="600" fill="#222">{escape(title)}</text>']
    if kind == "pie":
        total = sum(v for v in vals if v > 0) or 1
        cx, cy, r, a = W / 2 - 80, T + ph / 2 + 8, min(pw, ph) / 2 - 4, -math.pi / 2
        for i, v in enumerate(vals):
            if v <= 0:
                continue
            b = a + 2 * math.pi * v / total
            x1, y1, x2, y2 = cx + r * math.cos(a), cy + r * math.sin(a), cx + r * math.cos(b), cy + r * math.sin(b)
            big = 1 if b - a > math.pi else 0
            out.append(f'<path d="M{cx},{cy} L{x1:.1f},{y1:.1f} A{r},{r} 0 {big} 1 {x2:.1f},{y2:.1f} Z" fill="{COLORS[i % 8]}"/>')
            ly = T + 20 + i * 22
            out.append(f'<rect x="{W-200}" y="{ly-11}" width="12" height="12" fill="{COLORS[i % 8]}"/>'
                       f'<text x="{W-182}" y="{ly}" font-size="13" fill="#333">{escape(labels[i])}: {v:g} ({100*v/total:.0f}%)</text>')
            a = b
        return "".join(out) + "</svg>"
    top = max(max(vals), 0) or 1
    step = 10 ** math.floor(math.log10(top))
    top = math.ceil(top / step) * step
    for k in range(6):  # gridlines and axis numbers
        y = T + ph - ph * k / 5
        out.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="#e6e6e6"/>'
                   f'<text x="{L-8}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="#666">{top*k/5:g}</text>')
    n = len(vals)
    slot = pw / n
    pts = []
    for i, v in enumerate(vals):
        x = L + slot * i + slot / 2
        h = ph * max(v, 0) / top
        if kind == "line":
            pts.append((x, T + ph - h))
        else:
            bw = slot * 0.6
            out.append(f'<rect x="{x-bw/2:.1f}" y="{T+ph-h:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="3" fill="{COLORS[0]}"/>')
            out.append(f'<text x="{x:.1f}" y="{T+ph-h-6:.1f}" text-anchor="middle" font-size="12" fill="#333">{v:g}</text>')
        out.append(f'<text x="{x:.1f}" y="{T+ph+20}" text-anchor="middle" font-size="12" fill="#444">{escape(labels[i])}</text>')
    if pts:
        if kind == "line":
            out.append('<polyline fill="none" stroke="#7aa2ff" stroke-width="3" points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '"/>')
            out += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#7aa2ff"/>' for x, y in pts]
    out.append(f'<line x1="{L}" y1="{T+ph}" x2="{W-R}" y2="{T+ph}" stroke="#999"/>')
    return "".join(out) + "</svg>"
