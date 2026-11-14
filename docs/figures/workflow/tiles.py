"""AI/ML and evaluation tiles shared by the icon-based options (B and C)."""
import icons
from common import EVAL, ML, Canvas, titled_card


def ml_tile(c: Canvas, x, y, w, h):
    top = y + h
    titled_card(c, "ml", x, y, w, h, ML, "3 · RECONSTRUCT", "AI/ML model", title_size=17)
    icons.network(c, x + 0.25, top - 1.725, w - 0.5, 0.95)
    c.block(x + 0.2, top - 1.925, ["composition", "± uncertainty"], size=14, within="ml", name="ml_sub")


def eval_tile(c: Canvas, x, y, w, h):
    top = y + h
    titled_card(c, "eval", x, y, w, h, EVAL, "4 · EVALUATE", "Compare", title_size=17)
    icons.compare(c, x + 0.2, top - 1.675, w - 0.4, 0.8)
    c.block(x + 0.2, top - 1.925, ["with held-out", "truth"], size=14, within="eval", name="eval_sub")
