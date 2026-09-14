"""Redraw Figure 2 without overlapping labels (Pillow)."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures" / "fig2-example-graph.png"
FONT_REG = ROOT / "fonts" / "DejaVuSerif.ttf"
FONT_BOLD = ROOT / "fonts" / "DejaVuSerif-Bold.ttf"
FONT_ITA = ROOT / "fonts" / "DejaVuSerif-Italic.ttf"

W, H = 1780, 1180
img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)

title_f = ImageFont.truetype(str(FONT_BOLD), 42)
node_f = ImageFont.truetype(str(FONT_REG), 28)
sub_f = ImageFont.truetype(str(FONT_ITA), 26)
edge_f = ImageFont.truetype(str(FONT_REG), 24)
foot_f = ImageFont.truetype(str(FONT_REG), 24)


def text_size(font, text):
    b = draw.textbbox((0, 0), text, font=font)
    return b[2] - b[0], b[3] - b[1]


def center_text(xy, text, font, fill="black"):
    tw, th = text_size(font, text)
    draw.text((xy[0] - tw / 2, xy[1] - th / 2), text, font=font, fill=fill)


nodes = {
    "event": (890, 310, 390, 170, "Event 1", ""),
    "subj": (280, 560, 400, 180, "Subject", "person\u2081"),
    "act": (890, 560, 380, 170, "Action", "buy\u2081"),
    "obj": (1500, 560, 380, 170, "Object", "item\u2081"),
    "time": (280, 860, 380, 170, "Time", "t\u2081"),
    "loc": (890, 860, 380, 170, "Location", "loc\u2081"),
    "s2": (1500, 860, 420, 180, "Sentence 2", "pronoun \u2192 person\u2081"),
}


def ellipse_edge_point(src, dst):
    x1, y1, w1, h1, *_ = nodes[src]
    x2, y2, w2, h2, *_ = nodes[dst]
    dx, dy = x2 - x1, y2 - y1
    n = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / n, dy / n
    rx, ry = w1 / 2 * 0.98, h1 / 2 * 0.98
    t = 1.0 / (((ux / rx) ** 2 + (uy / ry) ** 2) ** 0.5)
    p0 = (x1 + ux * t, y1 + uy * t)
    rx, ry = w2 / 2 * 0.98, h2 / 2 * 0.98
    t = 1.0 / (((ux / rx) ** 2 + (uy / ry) ** 2) ** 0.5)
    p1 = (x2 - ux * t, y2 - uy * t)
    return p0, p1


def edge_from_node(src, toward):
    x1, y1, w1, h1, *_ = nodes[src]
    dx, dy = toward[0] - x1, toward[1] - y1
    n = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / n, dy / n
    rx, ry = w1 / 2 * 0.98, h1 / 2 * 0.98
    t = 1.0 / (((ux / rx) ** 2 + (uy / ry) ** 2) ** 0.5)
    return (x1 + ux * t, y1 + uy * t)


def chip(pos, label):
    mx, my = pos
    tw, th = text_size(edge_f, label)
    pad = 6
    draw.rectangle(
        [mx - tw / 2 - pad, my - th / 2 - pad, mx + tw / 2 + pad, my + th / 2 + pad],
        fill="white",
    )
    center_text((mx, my), label, edge_f, fill="#222222")


straight = [
    ("event", "subj", "subject", (-28, -6)),
    ("event", "act", "predicate", (78, 0)),
    ("event", "obj", "object", (28, -6)),
    ("act", "loc", "location", (108, -62)),
]
for src, dst, _, _ in straight:
    p0, p1 = ellipse_edge_point(src, dst)
    draw.line([p0, p1], fill="black", width=3)

p_event_left = edge_from_node("event", (70, 310))
p_time_left = edge_from_node("time", (70, 860))
draw.line([p_event_left, (70, 310), (70, 860), p_time_left], fill="black", width=3)

p_subj_right = edge_from_node("subj", (1500, 710))
p_s2_left = edge_from_node("s2", (280, 710))
draw.line([p_subj_right, (520, 710), (1260, 710), p_s2_left], fill="black", width=3)

for x, y, w, h, title, sub in nodes.values():
    box = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
    draw.ellipse(box, fill="white", outline="black", width=3)
    if sub:
        center_text((x, y - 22), title, node_f)
        center_text((x, y + 22), sub, sub_f)
    else:
        center_text((x, y), title, node_f)

for src, dst, label, off in straight:
    p0, p1 = ellipse_edge_point(src, dst)
    chip(((p0[0] + p1[0]) / 2 + off[0], (p0[1] + p1[1]) / 2 + off[1]), label)

chip((70, 430), "time")
chip((1180, 710), "coreference")

center_text((W / 2, 70), "Example Semantic Dependency Graph", title_f)
center_text(
    (W / 2, H - 48),
    "A change to person\u2081 invalidates only dependent nodes/edges rather than the entire sequence.",
    foot_f,
    fill="#333333",
)

img.save(OUT, "PNG")
print("wrote", OUT, OUT.stat().st_size)
