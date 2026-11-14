"""Shared canvas, palette, drawing helpers and layout checker for the BRIDGE workflow figures.

The canvas is a 16:9 slide, 13.333 x 7.5 in. One data unit is one inch, so point sizes
match PowerPoint when the exported image fills the slide.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Wedge

W, H = 40 / 3, 7.5
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
ASSETS = HERE / "assets"


def _pick_font() -> str:
    have = {f.name for f in fm.fontManager.ttflist}
    return next((n for n in ("Arial", "Helvetica Neue", "Helvetica") if n in have), "DejaVu Sans")


FONT = _pick_font()
plt.rcParams.update({"font.family": FONT, "svg.fonttype": "none"})  # none = SVG text stays editable


@dataclass(frozen=True)
class Tone:
    fill: str  # card background tint
    edge: str  # borders and arrows
    text: str  # accent text and filled pills; >= 4.5:1 on `fill` and on white


SIM = Tone("#EEF1F4", "#7F8FA0", "#3E4C5B")
OBS = Tone("#E7F1F8", "#2B7BAE", "#175A85")
TRUTH = Tone("#E4F4EF", "#1B8E7B", "#0F6657")
ML = Tone("#F0EBF8", "#7656B0", "#58398F")
EVAL = Tone("#FCF1D8", "#B87A0A", "#855700")
INK, MUTED, RULE, WHITE = "#14263A", "#556575", "#D3DAE1", "#FFFFFF"


class Canvas:
    def __init__(self, name: str):
        self.name = name
        self.fig = plt.figure(figsize=(W, H), facecolor=WHITE)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(0, H)
        self.ax.axis("off")
        self.texts: list[dict] = []
        self.nodes: dict[str, tuple[float, float, float, float]] = {}
        self.arrows: list[dict] = []
        self.images: dict[str, tuple[float, float, float, float]] = {}

    # ---- drawing ---------------------------------------------------------
    def card(self, name, x, y, w, h, tone, *, fill=None, lw=1.8, r=0.14, ls="-", z=1, node=True):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                           fc=fill or tone.fill, ec=tone.edge, lw=lw, ls=ls, zorder=z)
        p.set_gid(name)
        self.ax.add_patch(p)
        if node:
            self.nodes[name] = (x, y, x + w, y + h)
        return p

    def text(self, x, y, s, *, size=14, weight="normal", color=INK, ha="left", va="center",
             style="normal", within=None, name=None, z=6, bbox=None, spacing=1.2,
             on_arrow=None, role="body"):
        t = self.ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va,
                         fontstyle=style, zorder=z, linespacing=spacing, bbox=bbox)
        if name:
            t.set_gid(name)
        self.texts.append(dict(artist=t, within=within, name=name, on_arrow=on_arrow, role=role))
        return t

    def block(self, x, y_top, lines, *, size=14, step=1.32, color=INK, weight="normal", within=None,
              name=None, bullet=None):
        """Left-aligned lines on an even baseline grid (multi-line text spacing varies with descenders).

        Returns the baseline of the last line.
        """
        em = size / 72
        y = y_top - 0.78 * em
        for i, s in enumerate(lines):
            tx = x
            if bullet:
                self.ax.add_patch(Circle((x + 0.07, y + 0.3 * em), 0.034, fc=bullet, ec="none", zorder=5))
                tx = x + 0.22
            self.text(tx, y, s, size=size, color=color, weight=weight, va="baseline", within=within,
                      name=f"{name}_{i}" if name else None)
            y -= step * em
        return y + step * em

    def chip(self, x, y, s, tone, *, size=12, on_arrow=None, name=None):
        """Small outlined label, centred on (x, y); sits on an arrow or beside it."""
        box = dict(boxstyle="round,pad=0.3,rounding_size=0.6", fc=WHITE, ec=tone.edge, lw=1.3)
        return self.text(x, y, s, size=size, weight="bold", color=tone.text, ha="center",
                         bbox=box, on_arrow=on_arrow, name=name, spacing=1.1)

    def pill(self, name, x, y, w, h, label, tone, *, size=12):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={h / 2}",
                           fc=tone.text, ec=tone.text, lw=0, zorder=4)
        p.set_gid(name)
        self.ax.add_patch(p)
        self.nodes[name] = (x, y, x + w, y + h)
        self.text(x + w / 2, y + h / 2, label, size=size, weight="bold", color=WHITE,
                  ha="center", within=name, z=6)

    def badge(self, x, y, r, letter, tone, *, size=14):
        self.ax.add_patch(Circle((x, y), r, fc=tone.text, ec=tone.text, lw=0, zorder=4))
        self.text(x, y, letter, size=size, weight="bold", color=WHITE, ha="center", z=6)

    def arrow(self, name, pts, color, *, src=None, dst=None, lw=2.6, dashed=False,
              head=0.22, half=0.13, z=3):
        """Orthogonal polyline with a filled head; the head stays solid when the shaft is dashed."""
        P = [np.asarray(p, float) for p in pts]
        d = P[-1] - P[-2]
        u = d / np.hypot(*d)
        base = P[-1] - u * head
        xs, ys = zip(*([tuple(p) for p in P[:-1]] + [tuple(base)]))
        (line,) = self.ax.plot(xs, ys, color=color, lw=lw, ls=(0, (1.5, 1.0)) if dashed else "-",
                               solid_capstyle="butt", dash_capstyle="butt",
                               solid_joinstyle="miter", dash_joinstyle="miter", zorder=z)
        n = np.array([-u[1], u[0]])
        tip = Polygon([P[-1], base + n * half, base - n * half], closed=True, fc=color, ec=color,
                      lw=0.6, joinstyle="miter", zorder=z)
        self.ax.add_patch(tip)
        line.set_gid(f"{name}_line")
        tip.set_gid(f"{name}_head")
        self.arrows.append(dict(name=name, pts=P, src=src, dst=dst))

    def pie(self, x, y, r, fracs, colors, *, ec=INK, lw=1.0, start=90, z=4):
        ang = start
        for f, col in zip(fracs, colors):
            self.ax.add_patch(Wedge((x, y), r, ang, ang + 360 * f, fc=col, ec=ec, lw=lw, zorder=z))
            ang += 360 * f
        self.ax.add_patch(Circle((x, y), r, fc="none", ec=ec, lw=lw, zorder=z + 0.1))

    def image(self, name, path, x, y, w, h, *, tone, label):
        """Fit an image inside (x, y, w, h) keeping its aspect; draw a labelled placeholder if missing."""
        if Path(path).exists():
            img = plt.imread(path)
            ih, iw = img.shape[:2]
            s = min(w / iw, h / ih)
            iw_in, ih_in = iw * s, ih * s
            x0, y0 = x + (w - iw_in) / 2, y + (h - ih_in) / 2
            im = self.ax.imshow(img, extent=(x0, x0 + iw_in, y0, y0 + ih_in), aspect="auto",
                                zorder=2, interpolation="lanczos")
            im.set_gid(name)
            self.ax.set_xlim(0, W)
            self.ax.set_ylim(0, H)
            self.images[name] = (x0, y0, x0 + iw_in, y0 + ih_in)
            self.nodes[name] = self.images[name]
            return True
        self.card(name, x, y, w, h, tone, fill=WHITE, lw=1.4, ls=(0, (4, 3)), z=2)
        self.text(x + w / 2, y + h / 2, f"PLACEHOLDER\n{label}", size=12, color=tone.text, ha="center",
                  weight="bold", within=name)
        self.images[name] = (x, y, x + w, y + h)
        return False

    # ---- layout checks ---------------------------------------------------
    def reserve(self, x, y, w, h):
        """Mark the region of a drawn glyph so layout checks treat it like an image."""
        name = f"glyph_{len(self.images)}"
        self.images[name] = self.nodes[name] = (x, y, x + w, y + h)

    def check(self, min_font=12.0, max_words=140, pad=0.04, margin=0.25):
        self.fig.canvas.draw()
        rend = self.fig.canvas.get_renderer()
        inv = self.ax.transData.inverted()

        def ext(artist):
            bb = artist.get_window_extent(rend)
            (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
            return x0, y0, x1, y1

        for d in self.texts:
            t = d["artist"]
            patch = t.get_bbox_patch()
            d["ext"] = ext(patch if patch is not None else t)
            d["label"] = d["name"] or repr(t.get_text().replace("\n", " ")[:30])

        issues = []
        for d in self.texts:
            x0, y0, x1, y1 = d["ext"]
            if x0 < margin or y0 < margin or x1 > W - margin or y1 > H - margin:
                issues.append(f"text near/outside slide edge: {d['label']} ({x0:.2f},{y0:.2f},{x1:.2f},{y1:.2f})")
            if d["artist"].get_fontsize() < min_font:
                issues.append(f"font below {min_font}pt: {d['label']}")
            if d["within"]:
                nx0, ny0, nx1, ny1 = self.nodes[d["within"]]
                if x0 < nx0 + pad or y0 < ny0 + pad or x1 > nx1 - pad or y1 > ny1 - pad:
                    issues.append(f"text exceeds '{d['within']}': {d['label']} "
                                  f"text=({x0:.2f},{y0:.2f},{x1:.2f},{y1:.2f}) box=({nx0:.2f},{ny0:.2f},{nx1:.2f},{ny1:.2f})")
        for a, b in combinations(self.texts, 2):
            ax0, ay0, ax1, ay1 = a["ext"]
            bx0, by0, bx1, by1 = b["ext"]
            if min(ax1, bx1) - max(ax0, bx0) > 0.01 and min(ay1, by1) - max(ay0, by0) > 0.01:
                issues.append(f"text overlap: {a['label']}  <->  {b['label']}")

        def pts(P, step=0.03):
            for p, q in zip(P[:-1], P[1:]):
                for s in np.linspace(0, 1, max(2, int(np.hypot(*(q - p)) / step))):
                    yield p + (q - p) * s

        for arr in self.arrows:
            for d in self.texts:
                if d["on_arrow"] == arr["name"]:
                    continue
                x0, y0, x1, y1 = d["ext"]
                if any(x0 - 0.03 <= p[0] <= x1 + 0.03 and y0 - 0.03 <= p[1] <= y1 + 0.03 for p in pts(arr["pts"])):
                    issues.append(f"arrow '{arr['name']}' runs through text: {d['label']}")
            for nm, (nx0, ny0, nx1, ny1) in self.nodes.items():
                if nm in (arr["src"], arr["dst"]):
                    continue
                if any(nx0 + 0.03 < p[0] < nx1 - 0.03 and ny0 + 0.03 < p[1] < ny1 - 0.03 for p in pts(arr["pts"])):
                    issues.append(f"arrow '{arr['name']}' passes through node '{nm}'")
            for end, nm in ((arr["pts"][0], arr["src"]), (arr["pts"][-1], arr["dst"])):
                if nm is None:
                    continue
                nx0, ny0, nx1, ny1 = self.nodes[nm]
                dist = min(abs(end[0] - nx0), abs(end[0] - nx1), abs(end[1] - ny0), abs(end[1] - ny1))
                inside = nx0 - 0.05 <= end[0] <= nx1 + 0.05 and ny0 - 0.05 <= end[1] <= ny1 + 0.05
                if not (inside and dist <= 0.05):
                    issues.append(f"arrow '{arr['name']}' end {tuple(np.round(end, 2))} not on boundary of '{nm}'")
        for a, b in combinations(self.arrows, 2):
            for p, q in zip(a["pts"][:-1], a["pts"][1:]):
                for r_, s in zip(b["pts"][:-1], b["pts"][1:]):
                    if _segments_cross(p, q, r_, s):
                        issues.append(f"arrows cross: {a['name']} x {b['name']}")
        for nm, (x0, y0, x1, y1) in self.images.items():
            if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
                issues.append(f"image outside slide: {nm}")
            for d in self.texts:
                if d["within"] == nm:
                    continue
                tx0, ty0, tx1, ty1 = d["ext"]
                if min(tx1, x1) - max(tx0, x0) > 0.01 and min(ty1, y1) - max(ty0, y0) > 0.01:
                    issues.append(f"text overlaps image/glyph '{nm}': {d['label']}")

        words = sum(len(d["artist"].get_text().split()) for d in self.texts if d["role"] != "footer")
        if words > max_words:
            issues.append(f"dense: {words} words (limit {max_words})")
        sizes = sorted({d["artist"].get_fontsize() for d in self.texts})
        print(f"[{self.name}] {len(self.texts)} texts, {len(self.arrows)} arrows, {words} words, "
              f"font sizes {sizes}, font '{FONT}'")
        print("  layout OK" if not issues else "  ISSUES:\n    " + "\n    ".join(issues))
        return issues

    def save(self, dpi=300):
        OUT.mkdir(parents=True, exist_ok=True)
        stem = OUT / f"workflow_{self.name}"
        self.fig.savefig(f"{stem}.png", dpi=dpi, facecolor=WHITE)
        self.fig.savefig(f"{stem}.svg", facecolor=WHITE, metadata={"Date": None})
        plt.close(self.fig)
        return stem


def _segments_cross(p, q, r, s) -> bool:
    d1, d2 = q - p, s - r
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9:
        return False
    t = ((r[0] - p[0]) * d2[1] - (r[1] - p[1]) * d2[0]) / den
    u = ((r[0] - p[0]) * d1[1] - (r[1] - p[1]) * d1[0]) / den
    return 1e-6 < t < 1 - 1e-6 and 1e-6 < u < 1 - 1e-6


def header(c: Canvas, tag: str, title: str, subtitle: str | None = None, *, title_size=28,
           title_y=6.7, rule_y=5.98):
    c.text(0.5, 7.12, tag, size=12, weight="bold", color=OBS.text, name="tag")
    c.text(0.5, title_y, title, size=title_size, weight="bold", name="title")
    if subtitle:
        c.text(0.5, 6.25, subtitle, size=16, color=MUTED, name="subtitle")
    c.ax.plot([0.5, W - 0.5], [rule_y, rule_y], color=RULE, lw=1.2, solid_capstyle="butt")


def footer(c: Canvas, lines: list[str], y0=0.7, step=0.26):
    for i, s in enumerate(lines):
        c.text(0.5, y0 - i * step, s, size=12, color=MUTED, name=f"footnote_{i + 1}", role="footer")


def titled_card(c: Canvas, name, x, y, w, h, tone: Tone, eyebrow: str, title: str, *, title_size=16, **kw):
    """Card with a small-caps eyebrow and a bold title (\\n splits lines); returns the last title baseline."""
    c.card(name, x, y, w, h, tone, **kw)
    c.text(x + 0.2, y + h - 0.26, eyebrow, size=12, weight="bold", color=tone.text, within=name,
           name=f"{name}_eyebrow")
    return c.block(x + 0.2, y + h - 0.42, title.split("\n"), size=title_size, step=1.25, weight="bold",
                   within=name, name=f"{name}_title")


def rule(c: Canvas, x0, x1, y, color=RULE, lw=1.0):
    c.ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=2)
