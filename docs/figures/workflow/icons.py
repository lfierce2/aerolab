"""Schematic glyphs drawn with Matplotlib primitives; shapes are illustrative, not data."""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle, Rectangle

from common import EVAL, INK, ML, OBS, WHITE, Canvas

# Approximates the species colours of the supplied particle-resolved and sectional schematics.
SPECIES = ["#3D3A8E", "#8DC27B", "#F2C14E", "#3FA9D8"]
_SMALL = np.array([0.50, 0.10, 0.20, 0.20])
_LARGE = np.array([0.08, 0.55, 0.17, 0.20])
OBS_BAR = "#9CC3DE"


def _finish(c: Canvas, x, y, w, h):
    c.ax.plot([x, x + w], [y, y], color=INK, lw=1.4, solid_capstyle="butt", zorder=5)
    c.reserve(x, y, w, h)


def population(c: Canvas, x, y, w, h, *, rows=4, cols=4, seed=3):
    """Grid of pie 'particles' with varied size and mixing state (cf. the particle-resolved schematic)."""
    rng = np.random.default_rng(seed)
    cw, ch = w / cols, h / rows
    conc = np.array([1.2, 2.4, 1.2, 0.8])
    for i in range(rows):
        for j in range(cols):
            r = min(cw, ch) * rng.uniform(0.22, 0.47)
            frac = rng.dirichlet(conc * rng.choice([0.25, 1.0, 4.0]))
            c.pie(x + (j + 0.5) * cw, y + (i + 0.5) * ch, r, frac, SPECIES, lw=0.9)
    c.reserve(x, y, w, h)


def histogram(c: Canvas, x, y, w, h, *, n=9):
    """Unimodal size distribution with no composition information."""
    t = np.arange(n)
    hts = np.exp(-(((t - 0.46 * (n - 1) - 0.4) / (0.26 * n)) ** 2))
    bw = w / n
    for i, hh in enumerate(hts):
        c.ax.add_patch(Rectangle((x + i * bw + 0.07 * bw, y), 0.86 * bw, 0.95 * h * hh, fc=OBS_BAR,
                                 ec=OBS.text, lw=1.1, zorder=4))
    _finish(c, x, y, w, h)


def bulk_bar(c: Canvas, x, y, w, h, frac=(0.35, 0.30, 0.20, 0.15)):
    """One stacked bar: composition with no size information."""
    bw, base = 0.42 * w, y
    for fk, col in zip(frac, SPECIES):
        c.ax.add_patch(Rectangle((x + (w - bw) / 2, base), bw, fk * h, fc=col, ec=INK, lw=0.8, zorder=4))
        base += fk * h
    _finish(c, x, y, w, h)


def optics(c: Canvas, x, y, w, h):
    lam = np.linspace(0, 1, 40)
    for curve, col in ((np.exp(-1.6 * lam), OBS.edge), (0.55 * np.exp(-2.4 * lam), "#7FB1D3")):
        c.ax.plot(x + 0.08 * w + 0.88 * w * lam, y + 0.06 * h + 0.88 * h * curve, color=col, lw=2.4,
                  solid_capstyle="round", zorder=5)
    c.ax.plot([x, x, x + w], [y + h, y, y], color=INK, lw=1.4, solid_joinstyle="miter", zorder=5)
    c.reserve(x, y, w, h)


def stack(c: Canvas, x, y, w, h, *, bins=8, whiskers=False, seed=0):
    """100%-stacked composition bars whose mix drifts with size."""
    rng = np.random.default_rng(seed)
    bw = w / bins
    for i in range(bins):
        t = i / max(bins - 1, 1)
        f = np.clip((1 - t) * _SMALL + t * _LARGE + rng.normal(0, 0.03, 4), 0.02, None)
        f /= f.sum()
        base, xc = y, x + (i + 0.5) * bw
        for k, fk in enumerate(f):
            c.ax.add_patch(Rectangle((x + i * bw + 0.06 * bw, base), 0.88 * bw, fk * h, fc=SPECIES[k],
                                     ec=INK, lw=0.7, zorder=4))
            base += fk * h
            if whiskers and k < 3:
                e = 0.07 * h
                c.ax.plot([xc, xc], [base - e, base + e], color=INK, lw=1.2, zorder=6)
                for yy in (base - e, base + e):
                    c.ax.plot([xc - 0.1 * bw, xc + 0.1 * bw], [yy, yy], color=INK, lw=1.2, zorder=6)
    _finish(c, x, y, w, h)


def network(c: Canvas, x, y, w, h, *, layers=(3, 4, 3)):
    xs = np.linspace(x + 0.08 * w, x + 0.92 * w, len(layers))
    pos = [[(xs[k], y + h * (0.5 + (j - (n - 1) / 2) / max(layers))) for j in range(n)]
           for k, n in enumerate(layers)]
    for a, b in zip(pos[:-1], pos[1:]):
        for (x0, y0) in a:
            for (x1, y1) in b:
                c.ax.plot([x0, x1], [y0, y1], color=ML.edge, lw=0.9, alpha=0.55, zorder=4)
    for layer in pos:
        for (px, py) in layer:
            c.ax.add_patch(Circle((px, py), 0.075 * min(w, h) / 0.75, fc=ML.text, ec=WHITE, lw=1.0, zorder=5))
    c.reserve(x, y, w, h)


def compare(c: Canvas, x, y, w, h):
    """Prediction (with uncertainty whiskers) next to truth."""
    stack(c, x, y, 0.35 * w, h, bins=4, whiskers=True, seed=1)
    c.text(x + w / 2, y + h / 2, "vs.", size=14, weight="bold", color=EVAL.text, ha="center")
    stack(c, x + 0.65 * w, y, 0.35 * w, h, bins=4, seed=1)
