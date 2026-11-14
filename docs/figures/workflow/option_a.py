"""Option A: clean scientific flowchart, closest to the draft slide 1."""
from common import EVAL, INK, ML, OBS, SIM, TRUTH, WHITE, Canvas, footer, header, rule, titled_card

TOP, BOT = 5.7, 0.95
X1, W1 = 0.5, 2.4
X2, W2 = 3.65, 3.25
X3, W3 = 7.35, 2.75
X4, W4 = 10.55, 2.28
R1, R2, R3 = X1 + W1, X2 + W2, X3 + W3
OBS_Y = 2.95
TRU_H = 1.7
ML_Y, ML_H = 2.4, 2.6
JX = 3.17  # x of the fork between the ensemble and the two pathways


def build() -> Canvas:
    c = Canvas("A_flowchart")
    header(c, "BRIDGE  ·  SYNTHETIC-OBSERVATION WORKFLOW", "A synthetic testbed for composition reconstruction",
           "Synthetic observations and known composition come from the same PartMC–MOSAIC populations")

    # 1 · ensemble
    titled_card(c, "ens", X1, BOT, W1, TOP - BOT, SIM, "1 · SIMULATE", "PartMC–MOSAIC")
    c.text(X1 + 0.2, 4.78, "Scenarios vary in", size=13, weight="bold", color=SIM.text, within="ens", name="ens_h1")
    c.block(X1 + 0.2, 4.6, ["conditions", "composition", "size distribution", "mixing state"], size=13,
            bullet=SIM.edge, within="ens", name="ens_vary")
    c.text(X1 + 0.2, 3.45, "Each population has", size=13, weight="bold", color=SIM.text, within="ens", name="ens_h2")
    c.block(X1 + 0.2, 3.27, ["particle sizes", "species masses", "number weights"], size=13,
            bullet=SIM.edge, within="ens", name="ens_has")
    c.text(X1 + 0.2, 2.32, "Split by scenario", size=13, weight="bold", color=SIM.text, within="ens", name="ens_h3")
    c.card("split_train", X1 + 0.2, 1.8, W1 - 0.4, 0.36, SIM, fill=WHITE, lw=1.4, r=0.1)
    c.text(X1 + W1 / 2, 1.98, "Training scenarios", size=12, weight="bold", color=SIM.text, ha="center",
           within="split_train")
    c.card("split_held", X1 + 0.2, 1.33, W1 - 0.4, 0.36, SIM, fill=WHITE, lw=1.4, r=0.1, ls=(0, (4, 3)))
    c.text(X1 + W1 / 2, 1.51, "Held-out scenarios", size=12, weight="bold", color=SIM.text, ha="center",
           within="split_held")

    # 2 · model inputs
    titled_card(c, "obs", X2, OBS_Y, W2, TOP - OBS_Y, OBS, "2 · MODEL INPUTS", "Synthetic observations")
    for k, (name, what) in enumerate([("SMPS simulator", "number size distribution"),
                                      ("ACSM simulator", "bulk non-refractory species"),
                                      ("Optics simulator", "scattering, absorption")]):
        y0 = 4.28 - 0.6 * k
        c.card(name.lower(), X2 + 0.2, y0, W2 - 0.4, 0.54, OBS, fill=WHITE, lw=1.2, r=0.1)
        c.text(X2 + 0.35, y0 + 0.37, name, size=14, weight="bold", color=OBS.text, within=name.lower())
        c.text(X2 + 0.35, y0 + 0.15, what, size=13, color=INK, within=name.lower())

    # 2 · training targets
    titled_card(c, "truth", X2, BOT, W2, TRU_H, TRUTH, "2 · TRAINING TARGETS", "Ground-truth composition")
    c.block(X2 + 0.2, BOT + TRU_H - 0.78, ["Species mass fractions vs.", "particle diameter, calculated",
                                          "directly from each population"], size=14, within="truth", name="truth_body")

    # 3 · AI/ML
    titled_card(c, "ml", X3, ML_Y, W3, ML_H, ML, "3 · RECONSTRUCT", "AI/ML model")
    c.block(X3 + 0.2, 4.3, ["Predicts composition", "with uncertainty"], size=14, within="ml", name="ml_sub")
    rule(c, X3 + 0.2, R3 - 0.2, 3.7, ML.edge, 0.9)
    c.text(X3 + 0.2, 3.5, "TRAIN", size=12, weight="bold", color=ML.text, within="ml", name="ml_train_l")
    c.text(X3 + 0.2, 3.26, "inputs + known targets", size=13, within="ml", name="ml_train")
    c.text(X3 + 0.2, 2.95, "PREDICT", size=12, weight="bold", color=ML.text, within="ml", name="ml_pred_l")
    c.text(X3 + 0.2, 2.71, "inputs only; truth withheld", size=13, within="ml", name="ml_pred")

    # 4 · evaluation
    titled_card(c, "eval", X4, BOT, W4, TOP - BOT, EVAL, "4 · EVALUATE", "Compare with\nheld-out truth")
    for k, (name, what) in enumerate([("Accuracy", "by species and size"),
                                      ("Consistency", "with observations"),
                                      ("Ambiguity", "non-unique cases"),
                                      ("Uncertainty", "calibrated intervals"),
                                      ("Generalization", "to new scenarios")]):
        y = 4.45 - 0.72 * k
        c.text(X4 + 0.2, y, name, size=14, weight="bold", color=EVAL.text, within="eval")
        c.text(X4 + 0.2, y - 0.26, what, size=13, color=INK, within="eval")
        if k < 4:
            rule(c, X4 + 0.2, X4 + W4 - 0.2, y - 0.5, EVAL.edge, 0.8)

    # arrows
    c.ax.plot([R1, JX], [3.375, 3.375], color=SIM.edge, lw=2.6, solid_capstyle="butt", zorder=3)
    c.arrow("fork_obs", [(JX, 3.375), (JX, 4.4), (X2, 4.4)], OBS.edge, dst="obs")
    c.arrow("fork_truth", [(JX, 3.375), (JX, 1.9), (X2, 1.9)], TRUTH.edge, dst="truth")
    c.arrow("obs_ml", [(R2, 4.4), (X3, 4.4)], OBS.edge, src="obs", dst="ml")
    c.arrow("truth_ml", [(R2, 1.9), (9.2, 1.9), (9.2, ML_Y)], TRUTH.edge, src="truth", dst="ml")
    c.arrow("ml_eval", [(R3, 3.7), (X4, 3.7)], ML.edge, src="ml", dst="eval")
    c.arrow("truth_eval", [(R2, 1.4), (X4, 1.4)], TRUTH.edge, src="truth", dst="eval", dashed=True)
    c.chip((R2 + 9.2) / 2, 1.9, "training targets", TRUTH, on_arrow="truth_ml")
    c.chip((R2 + X4) / 2, 1.4, "held-out truth", TRUTH, on_arrow="truth_eval")

    footer(c, ["Truth is exact only within the simulated aerosol system; observations may not uniquely determine composition.",
               "Held-out scenarios are independent simulations, not later time steps of the training runs."],
           y0=0.62, step=0.25)
    return c
