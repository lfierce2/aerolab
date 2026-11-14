"""Option C: simplified presentation figure; icons carry the message and text is minimal."""
import icons
import tiles
from common import INK, ML, OBS, SIM, TRUTH, Canvas, footer, header, titled_card

TOP, BOT = 6.05, 0.9
MID = (TOP + BOT) / 2
X1, W1 = 0.5, 2.2
X2, W2 = 3.45, 4.55
X3, W3 = 8.65, 1.85
X4, W4 = 10.98, 1.85
R1, R2, R3 = X1 + W1, X2 + W2, X3 + W3
OBS_H, TRU_H = 2.6, 2.2
OBS_Y = TOP - OBS_H
TILE_Y, TILE_H = MID - 1.4, 2.8
JX = 2.95  # x of the fork between the population and the two pathways


def build() -> Canvas:
    c = Canvas("C_simplified")
    header(c, "BRIDGE  ·  SYNTHETIC-OBSERVATION WORKFLOW", "Testing composition reconstruction against known truth",
           title_y=6.72, rule_y=6.3)

    # 1 · population
    titled_card(c, "pop", X1, MID - 1.55, W1, 3.1, SIM, "1 · SIMULATE", "PartMC–MOSAIC")
    c.text(X1 + 0.2, 4.2, "simulated populations", size=13, color=INK, within="pop", name="pop_sub")
    icons.population(c, X1 + 0.2, 2.18, W1 - 0.4, W1 - 0.4)

    # 2 · model inputs: three instruments, each blind to part of the picture
    titled_card(c, "obs", X2, OBS_Y, W2, OBS_H, OBS, "2 · MODEL INPUTS", "Synthetic observations", title_size=17)
    cw = (W2 - 0.4) / 3
    for k, (fn, name, what) in enumerate([(icons.histogram, "SMPS", ("size", "distribution")),
                                          (icons.bulk_bar, "ACSM", ("bulk", "composition")),
                                          (icons.optics, "Optics", ("optical", "properties"))]):
        cx = X2 + 0.2 + (k + 0.5) * cw
        fn(c, cx - 0.5, 4.5, 1.0, 0.75)
        c.text(cx, 4.24, name, size=15, weight="bold", color=OBS.text, ha="center", va="baseline", within="obs")
        for i, line in enumerate(what):
            c.text(cx, 4.02 - 0.24 * i, line, size=13, color=INK, ha="center", va="baseline", within="obs")

    # 2 · training targets
    titled_card(c, "truth", X2, BOT, W2, TRU_H, TRUTH, "2 · TRAINING TARGETS", "Known composition", title_size=17)
    icons.stack(c, X2 + (W2 - 3.0) / 2, 1.52, 3.0, 0.8)
    c.text(X2 + W2 / 2, 1.25, "Species fractions vs. particle diameter", size=14, color=INK, ha="center",
           within="truth")

    # 3 · AI/ML and 4 · evaluation
    tiles.ml_tile(c, X3, TILE_Y, W3, TILE_H)
    tiles.eval_tile(c, X4, TILE_Y, W4, TILE_H)

    # arrows: the fork shows both pathways start from the same populations
    c.ax.plot([R1, JX], [MID, MID], color=SIM.edge, lw=2.6, solid_capstyle="butt", zorder=3)
    c.arrow("fork_obs", [(JX, MID), (JX, 4.8), (X2, 4.8)], OBS.edge, dst="obs")
    c.arrow("fork_truth", [(JX, MID), (JX, 2.1), (X2, 2.1)], TRUTH.edge, dst="truth")
    c.arrow("obs_ml", [(R2, 4.3), (X3, 4.3)], OBS.edge, src="obs", dst="ml")
    c.arrow("truth_ml", [(R2, 2.6), (X3, 2.6)], TRUTH.edge, src="truth", dst="ml")
    c.arrow("ml_eval", [(R3, MID), (X4, MID)], ML.edge, src="ml", dst="eval")
    c.arrow("truth_eval", [(R2, 1.3), (X4 + W4 / 2, 1.3), (X4 + W4 / 2, TILE_Y)], TRUTH.edge, src="truth",
            dst="eval", dashed=True)
    c.text((R2 + X3) / 2, 2.85, "train", size=12, weight="bold", color=TRUTH.text, ha="center", name="lbl_train")
    c.chip(9.6, 1.3, "held-out truth", TRUTH, on_arrow="truth_eval")

    footer(c, ["Schematic only: icons are illustrative and no results are shown. "
               "Truth is exact only within the simulated aerosol system.",
               "Observations may not uniquely determine composition, so predictions carry uncertainty."],
           y0=0.6, step=0.24)
    return c
