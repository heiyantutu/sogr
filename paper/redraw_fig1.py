"""Redraw Figure 1 without crossing arrows or text inside boxes."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures" / "fig1-runtime.png"
FONT_REG = ROOT / "fonts" / "DejaVuSerif.ttf"
FONT_BOLD = ROOT / "fonts" / "DejaVuSerif-Bold.ttf"

W, H = 1780, 980
img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)

title_f = ImageFont.truetype(str(FONT_BOLD), 40)
box_f = ImageFont.truetype(str(FONT_REG), 26)
sub_f = ImageFont.truetype(str(FONT_REG), 20)
lab_f = ImageFont.truetype(str(FONT_REG), 20)
foot_f = ImageFont.truetype(str(FONT_REG), 22)


def text_size(font, text):
    b = draw.textbbox((0, 0), text, font=font)
    return b[2] - b[0], b[3] - b[1]


def center_text(xy, lines, font, fill="black", gap=4):
    heights = [text_size(font, ln)[1] for ln in lines]
    total = sum(heights) + gap * (len(lines) - 1)
    y = xy[1] - total / 2
    for ln, h in zip(lines, heights):
        tw, _ = text_size(font, ln)
        draw.text((xy[0] - tw / 2, y), ln, font=font, fill=fill)
        y += h + gap


def chip(pos, text):
    tw, th = text_size(lab_f, text)
    pad = 5
    x, y = pos
    draw.rectangle(
        [x - tw / 2 - pad, y - th / 2 - pad, x + tw / 2 + pad, y + th / 2 + pad],
        fill="white",
    )
    draw.text((x - tw / 2, y - th / 2), text, font=lab_f, fill="#222222")


def arrow(p0, p1, width=3):
    draw.line([p0, p1], fill="black", width=width)
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    n = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    size = 14
    a = (x1, y1)
    b = (x1 - ux * size + px * 6, y1 - uy * size + py * 6)
    c = (x1 - ux * size - px * 6, y1 - uy * size - py * 6)
    draw.polygon([a, b, c], fill="black")


boxes = {
    "tok": (220, 290, 300, 130, ["Token sequence"]),
    "ggc": (620, 290, 340, 130, ["Graph Constructor", "(full / global attention)"]),
    "sog": (1060, 290, 340, 130, ["Semantic Object Graph"]),
    "cache": (1500, 290, 300, 130, ["Structured Cache"]),
    "gle": (620, 640, 340, 130, ["Graph-conditioned", "Linear Executor"]),
    "hbr": (1060, 640, 340, 130, ["Block Router / Diff"]),
    "loc": (1500, 640, 300, 130, ["Local Recompute"]),
}


def edge(name, side):
    x, y, w, h, _ = boxes[name]
    return {
        "l": (x - w / 2, y),
        "r": (x + w / 2, y),
        "t": (x, y - h / 2),
        "b": (x, y + h / 2),
    }[side]


# arrows first
arrow(edge("tok", "r"), edge("ggc", "l"))
arrow(edge("ggc", "r"), edge("sog", "l"))
arrow(edge("sog", "r"), edge("cache", "l"))

# SOG down then left into GLE (orthogonal, not diagonal through empty space)
p_sog_b = edge("sog", "b")
mid_y = 470
arrow(p_sog_b, (p_sog_b[0], mid_y))
draw.line([(p_sog_b[0], mid_y), (boxes["gle"][0], mid_y)], fill="black", width=3)
arrow((boxes["gle"][0], mid_y), edge("gle", "t"))

arrow(edge("gle", "r"), edge("hbr", "l"))

# HBR → Local and Local → HBR, offset so they do not sit on box text
y_fwd = boxes["hbr"][1] - 18
y_back = boxes["hbr"][1] + 18
hbr_r = (boxes["hbr"][0] + boxes["hbr"][2] / 2, y_fwd)
loc_l = (boxes["loc"][0] - boxes["loc"][2] / 2, y_fwd)
arrow(hbr_r, loc_l)
hbr_r2 = (boxes["hbr"][0] + boxes["hbr"][2] / 2, y_back)
loc_l2 = (boxes["loc"][0] - boxes["loc"][2] / 2, y_back)
arrow(loc_l2, hbr_r2)

# skip from HBR downward, not through Local
arrow(edge("hbr", "b"), (boxes["hbr"][0], 820))

# Local writes back up to cache
arrow(edge("loc", "t"), edge("cache", "b"))

# boxes on top
for x, y, w, h, lines in boxes.values():
    box = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
    draw.rounded_rectangle(box, radius=8, fill="white", outline="black", width=3)
    if len(lines) == 1:
        center_text((x, y), lines, box_f)
    else:
        center_text((x, y - 12), [lines[0]], box_f)
        center_text((x, y + 18), [lines[1]], sub_f)

chip((410, 250), "input")
chip((840, 250), "low-frequency")
chip((1280, 250), "persist")
chip((840, 448), "execute on G")
chip((870, 640), "high-frequency")
chip((1280, 598), "dirty")
chip((1280, 682), "patch")
chip((1500, 465), "write-back")
chip((1060, 790), "skip stable blocks")

center_text((W / 2, 58), ["Semantic Object Graph Runtime (SOGR)"], title_f)
center_text(
    (W / 2, H - 40),
    ["Global construction is low-frequency; graph propagation and local recomputation are high-frequency."],
    foot_f,
)

img.save(OUT, "PNG")
print("wrote", OUT, OUT.stat().st_size)
