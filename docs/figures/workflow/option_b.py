"""Option B: visual schematic that reuses the supplied particle-resolved and sectional images.

Drop the two exported schematics into assets/ as particle_resolved.png and sectional.png (PNG or JPG,
>= 1200 px wide). Until they exist, labelled placeholders mark the slots.
"""
import icons
import tiles
from common import ASSETS, INK, ML, OBS, SIM, TRUTH, Canvas, footer, header, titled_card

TOP, BOT = 6.05, 0.9
MID = (TOP + BOT) / 2
X1, W1 = 0.5, 2.4
X2, W2 = 3.6, 4.6
X3, W3 = 8.75, 1.85
X4, W4 = 11.05, 1.78
R1, R2, R3 = X1 + W1, X2 + W2, X3 + W3
OBS_H, TRU_H = 2.45, 2.4
OBS_Y = TOP - OBS_H
TILE_Y, TILE_H = MID - 1.4, 2.8
JX = R1 + 0.2  # x of the fork between the population and the two pathways
IMG_W, IMG_H = 2.0, 1.74


def _asset(stem):
    for ext in (".png", ".jpg", ".jpeg"):
        if (ASSETS / f"{stem}{ext}").exists():
            return ASSETS / f"{stem}{ext}"
    return ASSETS / f"{stem}.png"


def build() -> Canvas:
    c = Canvas("B_schematic")
    header(c, "BRIDGE  ·  SYNTHETIC-OBSERVATION WORKFLOW", "Particle-resolved populations as a synthetic testbed",
           title_y=6.72, rule_y=6.3)

    # 1 · population (supplied particle-resolved schematic)
    pop_y, pop_h = MID - 1.8, 3.6
    titled_card(c, "pop", X1, pop_y, W1, pop_h, SIM, "1 · SIMULATE", "PartMC–MOSAIC")
    c.image("img_pop", _asset("particle_resolved"), X1 + 0.2, pop_y + pop_h - 0.8 - IMG_H, IMG_W, IMG_H,
            tone=SIM, label="particle_resolved.png\nin assets/")
    c.block(X1 + 0.2, pop_y + 0.91, ["~10,000 computational", "particles, each with its", "own size and composition"],
            size=13, within="pop", name="pop_cap")

    # 2 · model inputs
    titled_card(c, "obs", X2, OBS_Y, W2, OBS_H, OBS, "2 · MODEL INPUTS", "Synthetic observations", title_size=17)
    cw = (W2 - 0.4) / 3
    for k, (fn, name, what) in enumerate([(icons.histogram, "SMPS", ("size", "distribution")),
                                          (icons.bulk_bar, "ACSM", ("bulk", "composition")),
                                          (icons.optics, "Optics", ("optical", "properties"))]):
        cx = X2 + 0.2 + (k + 0.5) * cw
        fn(c, cx - 0.475, OBS_Y + 0.95, 0.95, 0.7)
        c.text(cx, OBS_Y + 0.69, name, size=15, weight="bold", color=OBS.text, ha="center", va="baseline", within="obs")
        for i, line in enumerate(what):
            c.text(cx, OBS_Y + 0.47 - 0.24 * i, line, size=13, color=INK, ha="center", va="baseline", within="obs")

    # 2 · training targets (supplied sectional schematic)
    titled_card(c, "truth", X2, BOT, W2, TRU_H, TRUTH, "2 · TRAINING TARGETS", "Known\ncomposition", title_size=17)
    c.block(X2 + 0.2, BOT + TRU_H - 1.3, ["Species fractions", "averaged within", "size bins"], size=13,
            within="truth", name="truth_cap")
    c.image("img_truth", _asset("sectional"), X2 + W2 - 0.2 - IMG_W, BOT + (TRU_H - IMG_H) / 2, IMG_W, IMG_H,
            tone=TRUTH, label="sectional.png\nin assets/")

    # 3 · AI/ML and 4 · evaluation
    tiles.ml_tile(c, X3, TILE_Y, W3, TILE_H)
    tiles.eval_tile(c, X4, TILE_Y, W4, TILE_H)

    # arrows: the fork shows both pathways start from the same populations
    c.ax.plot([R1, JX], [MID, MID], color=SIM.edge, lw=2.6, solid_capstyle="butt", zorder=3)
    c.arrow("fork_obs", [(JX, MID), (JX, 4.9), (X2, 4.9)], OBS.edge, dst="obs")
    c.arrow("fork_truth", [(JX, MID), (JX, 2.0), (X2, 2.0)], TRUTH.edge, dst="truth")
    c.arrow("obs_ml", [(R2, 4.4), (X3, 4.4)], OBS.edge, src="obs", dst="ml")
    c.arrow("truth_ml", [(R2, 2.6), (X3, 2.6)], TRUTH.edge, src="truth", dst="ml")
    c.arrow("ml_eval", [(R3, MID), (X4, MID)], ML.edge, src="ml", dst="eval")
    c.arrow("truth_eval", [(R2, 1.3), (X4 + W4 / 2, 1.3), (X4 + W4 / 2, TILE_Y)], TRUTH.edge, src="truth",
            dst="eval", dashed=True)
    c.text((R2 + X3) / 2, 2.85, "train", size=12, weight="bold", color=TRUTH.text, ha="center", name="lbl_train")
    c.chip(9.55, 1.3, "held-out truth", TRUTH, on_arrow="truth_eval")

    footer(c, ["Truth is exact only within the simulated aerosol system; observations may not uniquely determine composition.",
               "Held-out scenarios are independent simulations, not later time steps of the training runs."],
           y0=0.6, step=0.24)
    return c
