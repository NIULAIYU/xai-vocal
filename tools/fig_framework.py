"""
Research-framework figure (Fig. 1): four phases from data to explanation,
with the validation layer and the pedagogical output.

Run from the package root:  python -m tools.fig_framework
"""
from __future__ import annotations
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

C.apply_style()
plt.rcParams["axes.grid"] = False

INK = "#222222"
FAM = {"Pitch stability": C.PALETTE[0], "Vibrato": C.PALETTE[1],
       "Resonance": C.PALETTE[2], "Breath control": C.PALETTE[3]}


def box(ax, x, y, w, h, fc, ec=INK, lw=0.8, ls="-", r=0.6, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0,rounding_size={r}",
                 fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def arrow(ax, p, q, lw=1.3, z=5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11,
                 lw=lw, color=INK, zorder=z, shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.set_xlim(0, 100); ax.set_ylim(2.5, 62.5); ax.axis("off")

    cols = [
        ("1  Data", C.PALETTE[7], [
            ("VocalSet corpus", "20 singers, 10 techniques\n1,681 clips"),
            ("Pre-processing", "22.05 kHz, mono\npeak-normalised"),
            ("Quality filter", "clips < 0.5 s removed"),
        ]),
        ("2  Features (31)", C.PALETTE[5], None),
        ("3  Classification", C.PALETTE[4], [
            ("XGBoost", "fixed hyperparameters\nLR, SVM, RF baselines"),
            ("Singer-grouped CV", "5-fold GroupKFold\nnested CV, LOSO"),
            ("Statistics", "singer bootstrap CI\nMcNemar test"),
        ]),
        ("4  Explanation", C.PALETTE[1], [
            ("TreeSHAP", "family and cluster\ngrouped attribution"),
            ("Three levels", "global, per technique\nper clip (held-out)"),
            ("Agreement", "permutation and\ngain importance"),
        ]),
    ]
    W, gap, x0 = 22.0, 3.0, 1.0
    top, bot = 57.0, 17.0
    xs = [x0 + i * (W + gap) for i in range(4)]

    for (title, col, items), x in zip(cols, xs):
        box(ax, x, bot, W, top - bot, fc="#F4F5F7", ec="#9AA0A6", lw=0.7, r=1.0, z=1)
        box(ax, x, top - 0.2, W, 5.0, fc=col, ec=col, r=1.0, z=2)
        ax.text(x + W / 2, top + 2.3, title, ha="center", va="center",
                color="white", fontsize=9, fontweight="bold", zorder=3)
        if items:
            h, g = 11.0, 1.6
            for k, (head, body) in enumerate(items):
                y = top - 2.0 - (k + 1) * h - k * g
                box(ax, x + 1.2, y, W - 2.4, h, fc="white", ec="#555555", lw=0.6, z=2)
                ax.text(x + W / 2, y + h - 2.4, head, ha="center", va="center",
                        fontsize=8, fontweight="bold", color=INK, zorder=3)
                ax.text(x + W / 2, y + 4.0, body, ha="center", va="center",
                        fontsize=7.2, color="#333333", linespacing=1.15, zorder=3)
                if k < len(items) - 1:
                    arrow(ax, (x + W / 2, y), (x + W / 2, y - g), lw=0.9)

    # phase 2: the four indicator families
    x = xs[1]
    fams = [("Pitch stability (9)", "σ cents, jitter, HNR"),
            ("Vibrato (5)", "rate, extent, strength"),
            ("Resonance (10)", "formants, spectrum"),
            ("Breath control (7)", "RMS, sustain, shimmer")]
    h, g = 7.4, 1.0
    for k, ((head, body), fc) in enumerate(zip(fams, FAM.values())):
        y = top - 2.0 - (k + 1) * h - k * g
        box(ax, x + 1.2, y, W - 2.4, h, fc=fc, ec=fc, z=2)
        ax.text(x + W / 2, y + h - 2.3, head, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=3)
        ax.text(x + W / 2, y + 2.1, body, ha="center", va="center",
                fontsize=7, color="white", zorder=3)
    ax.text(x + W / 2, bot + 1.8, "Praat / parselmouth, librosa",
            ha="center", va="center", fontsize=7.2, color="#444444", zorder=3)

    # arrows between phases
    ymid = (top + bot) / 2 + 2
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + W + 0.2, ymid), (b - 0.2, ymid), lw=1.6)

    # validation layer under phases 2 and 3
    vx, vw, vy, vh = xs[1], xs[2] + W - xs[1], 3.0, 10.0
    box(ax, vx, vy, vw, vh, fc="white", ec="#555555", lw=0.8, ls="--", r=0.8)
    ax.text(vx + vw / 2, vy + vh / 2 + 2.6, "Validation and stress tests",
            ha="center", va="center", fontsize=8, fontweight="bold", color=INK)
    ax.text(vx + vw / 2, vy + vh / 2 - 1.4,
            "vibrato estimator: synthetic tones, cycle-based, pYIN\nnoise, reverberation, cross-gender transfer",
            ha="center", va="center", fontsize=7.2, color="#333333", linespacing=1.2)
    for xc in (xs[1] + W / 2, xs[2] + W / 2):
        arrow(ax, (xc, vy + vh), (xc, bot), lw=0.9)

    # output
    ox, ow, oy, oh = xs[3], W, 3.0, 10.0
    box(ax, ox, oy, ow, oh, fc=C.PALETTE[1], ec=C.PALETTE[1], r=0.8)
    ax.text(ox + ow / 2, oy + oh / 2 + 2.6, "Pedagogical output", ha="center",
            va="center", fontsize=8, fontweight="bold", color="white")
    ax.text(ox + ow / 2, oy + oh / 2 - 1.4, "teacher-legible feedback\nfor expert validation",
            ha="center", va="center", fontsize=7.2, color="white", linespacing=1.2)
    arrow(ax, (ox + ow / 2, bot), (ox + ow / 2, oy + oh), lw=1.3)

    # data source note under phase 1
    box(ax, xs[0], 3.0, W, 10.0, fc="white", ec="#555555", lw=0.8, r=0.8)
    ax.text(xs[0] + W / 2, 3.0 + 6.5, "Unit of analysis", ha="center", va="center",
            fontsize=8, fontweight="bold", color=INK)
    ax.text(xs[0] + W / 2, 3.0 + 3.4, "whole clip\nsinger held out", ha="center",
            va="center", fontsize=7.2, color="#333333", linespacing=1.2)
    arrow(ax, (xs[0] + W / 2, bot), (xs[0] + W / 2, 3.0 + 10.0), lw=0.9)

    out = os.path.join(C.FIG_DIR, "fig1_framework")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}")
    plt.close(fig)
    print("[fig] fig1_framework")


if __name__ == "__main__":
    main()
