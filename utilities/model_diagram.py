"""Diagram of the model: one week in the full city (left) and the six build steps (right)."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from source.presets import stage

from .plotting import CLASS_COLOURS, INK, INK_2

GIVER, TAKER = CLASS_COLOURS["giver"], CLASS_COLOURS["taker"]
BOX, STEP = "#eef3fb", "#f7f6f2"


def _box(ax, x, y, w, h, title, body):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                facecolor=BOX, edgecolor=GIVER, linewidth=1.1))
    ax.text(x + 0.012, y + h - 0.03, title, fontsize=9.5, fontweight="bold", color=INK, va="top")
    ax.text(x + 0.012, y + h - 0.095, body, fontsize=8.2, color=INK_2, va="top", linespacing=1.35)


def _arrow(ax, a, b, colour=INK):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, color=colour, linewidth=1.1))


def model_diagram():
    """Draw the weekly cycle and the build steps. The numbers in the boxes are read from the S5
    configuration, so the diagram follows the defaults if they change."""
    cfg = stage("M4")
    T, R, P, S = (int(v) for v in cfg.game.payoff)
    enc, ties, pop = cfg.encounter, cfg.ties, cfg.popularity
    with plt.rc_context({"font.family": "sans-serif", "font.size": 9}):
        fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6), width_ratios=[0.56, 0.42])
        for ax in (left, right):
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

        left.text(0.02, 0.97, "One week in the city (S5; S6 adds refusal)", fontsize=11, fontweight="bold",
                  va="top")
        w, h = 0.30, 0.24
        cells = [
            (0.02, 0.62, "1. Choose a partner",
             "chance proportional to\n$(w_{ij}+\\varepsilon)^{\\beta} R_j^{\\gamma} d_{ij}^{-\\lambda}$, "
             f"within {enc.d_max} steps;\nat most {enc.max_received} meetings received;\n"
             "S6: never a refused partner"),
            (0.35, 0.62, "2. Play one game",
             f"Prisoner's Dilemma,\n$(T,R,P,S) = ({T},{R},{P},{S})$;\neach strategy uses its own\n"
             "history with this partner"),
            (0.68, 0.62, "3. Add to wealth",
             f"wealth is created, not moved:\na game creates {2 * R}, {T + S} or {2 * P}\nunits; nothing is spent"),
            (0.68, 0.24, "4. Update ties",
             f"+{ties.reinforcement:g} strength per meeting;\nunused ties fade, $w\\,e^{{-\\alpha \\Delta t}}$;\n"
             f"break below $\\theta = {ties.severance_threshold:g}$;\nS6: refused pairs lose their tie"),
            (0.35, 0.24, "5. Refill spare capacity",
             "new ties mostly with\nfriends of friends,\nweighted by rating\nand distance"),
            (0.02, 0.24, "6. Update ratings",
             f"$R_j = {pop.r_min:g} + {pop.r_max - pop.r_min:g}\\,\\kappa_j \\sigma_j$:\n"
             "recent cooperation $\\times$\nshare of tie strength"),
        ]
        for x, y, title, body in cells:
            _box(left, x, y, w, h, title, body)
        # The six updates run in a loop, clockwise from the top left.
        _arrow(left, (0.32, 0.74), (0.35, 0.74))
        _arrow(left, (0.65, 0.74), (0.68, 0.74))
        _arrow(left, (0.83, 0.62), (0.83, 0.48))
        _arrow(left, (0.68, 0.36), (0.65, 0.36))
        _arrow(left, (0.35, 0.36), (0.32, 0.36))
        _arrow(left, (0.17, 0.48), (0.17, 0.62))
        left.text(0.02, 0.16, "Every citizen keeps one strategy for the whole run.\n"
                  "Givers (AllC, TFT, GTFT, TF2T, Grudger, Pavlov) never defect first;\n"
                  "takers (AllD, Random; Joss in one experiment) defect unprovoked.",
                  fontsize=8.2, color=INK_2, va="top", linespacing=1.4)

        right.text(0.02, 0.97, "Six steps, each adding one rule", fontsize=11, fontweight="bold", va="top")
        steps = [
            ("S1", "everyone can meet everyone", "complete network, partners at random"),
            ("S2", "a fixed network", f"random (Erdős–Rényi) network, mean degree {cfg.topology.mean_degree}"),
            ("S3", "+ ties fade and break", "decay $\\alpha$ and threshold $\\theta$"),
            ("S4", "+ citizens choose partners", "distance, past meetings and rating; cap"),
            ("S5", "+ lost ties are replaced", "the full city; S5b starts scale-free"),
            ("S6", "+ citizens refuse those who cheated them", "a pair stops meeting after a defection"),
        ]
        top, gap = 0.86, 0.135
        for k, (name, rule, detail) in enumerate(steps):
            y = top - k * gap
            edge = TAKER if name == "S6" else GIVER
            right.add_patch(FancyBboxPatch((0.03, y - 0.1), 0.94, 0.1,
                                           boxstyle="round,pad=0.006,rounding_size=0.012",
                                           facecolor=STEP, edgecolor=edge, linewidth=1.0))
            right.text(0.06, y - 0.05, name, fontsize=10.5, fontweight="bold", color=edge, va="center")
            right.text(0.17, y - 0.032, rule, fontsize=9, color=INK, va="center")
            right.text(0.17, y - 0.072, detail, fontsize=8, color=INK_2, va="center")
            if k < len(steps) - 1:
                _arrow(right, (0.5, y - 0.1), (0.5, y - gap), colour=INK_2)
        right.text(0.03, 0.025, "With its new rule switched off, each step reproduces the one before exactly.",
                   fontsize=8, color=INK_2, va="bottom")
    return fig
