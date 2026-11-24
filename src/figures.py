"""
All manuscript figures. Every figure uses the shared style
(Times New Roman 11 pt, consistent palette) and is saved as both PDF
(vector, for LaTeX) and PNG (preview).

Figures that need the dataset take the feature table / results as arguments;
the schematic and the synthetic-validation figure stand alone.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import config as C
from src.features.extract import FEATURE_NAMES, FEATURE_GROUPS

C.apply_style()


def _save(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(C.FIG_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"[fig] {name}")


# --------------------------------------------------------------------------- #
# 1. Pipeline schematic (mechanistic) -- no data needed
# --------------------------------------------------------------------------- #
def fig_pipeline(name="fig1_pipeline"):
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    stages = [
        ("Singing\naudio", 0.6, C.PALETTE[7]),
        ("Pre-processing\n(resample, normalise,\nvoiced detection)", 2.3, C.PALETTE[5]),
        ("Interpretable features\nPitch · Vibrato ·\nResonance · Breath", 4.6, C.PALETTE[0]),
        ("Classifier\n(XGBoost, speaker-\nindependent CV)", 6.9, C.PALETTE[2]),
        ("SHAP\nexplanations", 8.7, C.PALETTE[1]),
    ]
    w, h = 1.7, 1.9
    centers = []
    for txt, x, col in stages:
        box = FancyBboxPatch((x - w / 2, 2 - h / 2), w, h,
                             boxstyle="round,pad=0.05,rounding_size=0.12",
                             linewidth=1.1, edgecolor="#333333",
                             facecolor=col, alpha=0.90)
        ax.add_patch(box)
        ax.text(x, 2, txt, ha="center", va="center", color="white",
                fontsize=9.5, linespacing=1.15, fontweight="bold")
        centers.append(x)
    for a, b in zip(centers[:-1], centers[1:]):
        ax.add_patch(FancyArrowPatch((a + w / 2, 2), (b - w / 2, 2),
                     arrowstyle="-|>", mutation_scale=16,
                     linewidth=1.3, color="#333333"))
    ax.text(5.0, 0.35,
            "Personalised, explainable acoustic feedback for vocal training",
            ha="center", fontsize=10, style="italic", color="#333333")
    _save(fig, name)


# --------------------------------------------------------------------------- #
# 2. Feature distributions by technique -- 3x3 grid (complex multi-panel)
# --------------------------------------------------------------------------- #
def fig_feature_grid(data: pd.DataFrame, name="fig_feature_grid",
                     feats=None):
    feats = feats or ["f0_std_cents", "jitter_local", "hnr_mean",
                      "vibrato_rate", "vibrato_extent", "vibrato_strength",
                      "singers_formant_ratio", "spectral_tilt",
                      "amplitude_stability"]
    techs = [t for t in C.TECHNIQUES if t in data["technique"].unique()]
    colors = {t: C.PALETTE[i % len(C.PALETTE)] for i, t in enumerate(techs)}
    fig, axes = plt.subplots(3, 3, figsize=(9.5, 8.2))
    for ax, feat in zip(axes.ravel(), feats):
        groups = [data.loc[data.technique == t, feat].dropna().values
                  for t in techs]
        bp = ax.boxplot(groups, patch_artist=True, widths=0.6,
                        showfliers=False, medianprops=dict(color="#222222"))
        for patch, t in zip(bp["boxes"], techs):
            patch.set_facecolor(colors[t]); patch.set_alpha(0.85)
            patch.set_edgecolor("#333333")
        ax.set_title(feat, fontsize=10)
        ax.set_xticks(range(1, len(techs) + 1))
        ax.set_xticklabels(techs, rotation=45, ha="right", fontsize=7)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Distribution of interpretable acoustic features by technique",
                 y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    _save(fig, name)


# --------------------------------------------------------------------------- #
# 3. Model comparison bar
# --------------------------------------------------------------------------- #
def fig_model_compare(results: dict, name="fig_model_compare"):
    models = list(results["models"].keys())
    acc = [results["models"][m]["accuracy"] for m in models]
    f1 = [results["models"][m]["macro_f1"] for m in models]
    x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x - w / 2, acc, w, label="Accuracy",
           color=C.PALETTE[0], edgecolor="#333")
    ax.bar(x + w / 2, f1, w, label="Macro-F1",
           color=C.PALETTE[3], edgecolor="#333")
    for xi, (a, f) in enumerate(zip(acc, f1)):
        ax.text(xi - w / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, f + 0.01, f"{f:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.0)
    ax.set_title("Speaker-independent technique classification")
    ax.legend(frameon=False)
    fig.tight_layout(); _save(fig, name)


# --------------------------------------------------------------------------- #
# 4. Confusion matrix
# --------------------------------------------------------------------------- #
def fig_confusion(results: dict, model_name="XGBoost", name="fig_confusion"):
    classes = results["classes"]
    cm = np.array(results["models"][model_name]["confusion"], dtype=float)
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cmn, cmap=C.SEQ_CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix ({model_name})")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "#222", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalised")
    fig.tight_layout(); _save(fig, name)


# --------------------------------------------------------------------------- #
# 5. SHAP global importance bar
# --------------------------------------------------------------------------- #
def fig_shap_bar(importance: pd.DataFrame, name="fig_shap_bar", top=18):
    imp = importance.sort_values("overall", ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(6.6, 6.4))
    ax.barh(imp.index, imp["overall"], color=C.PALETTE[0], edgecolor="#333")
    ax.set_xlabel("Mean |SHAP value| (mean impact on model output)")
    ax.set_title("Global feature importance (SHAP)")
    fig.tight_layout(); _save(fig, name)


# --------------------------------------------------------------------------- #
# 6. SHAP per-class heatmap (feature x technique)
# --------------------------------------------------------------------------- #
def fig_shap_heatmap(importance: pd.DataFrame, classes, name="fig_shap_heatmap",
                     top=18):
    imp = importance.sort_values("overall", ascending=False).head(top)
    mat = imp[classes].values
    mat = mat / mat.max(axis=0, keepdims=True).clip(min=1e-9)  # per-class norm
    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    im = ax.imshow(mat, aspect="auto", cmap=C.SEQ_CMAP)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes,
                  rotation=45, ha="right")
    ax.set_yticks(range(len(imp.index))); ax.set_yticklabels(imp.index)
    ax.set_title("Which acoustic indicators drive each technique\n"
                 "(per-class normalised mean |SHAP|)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); _save(fig, name)


# --------------------------------------------------------------------------- #
# 7. Vibrato rate vs extent scatter
# --------------------------------------------------------------------------- #
def fig_vibrato_scatter(data: pd.DataFrame, name="fig_vibrato_scatter"):
    techs = [t for t in C.TECHNIQUES if t in data["technique"].unique()]
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for i, t in enumerate(techs):
        sub = data[data.technique == t]
        ax.scatter(sub["vibrato_rate"], sub["vibrato_extent"],
                   s=22, alpha=0.6, label=t,
                   color=C.PALETTE[i % len(C.PALETTE)], edgecolor="none")
    ax.axvspan(5.5, 7.5, color="#999999", alpha=0.12,
               label="typical vibrato 5.5-7.5 Hz")
    ax.set_xlabel("Vibrato rate (Hz)")
    ax.set_ylabel("Vibrato extent (cents, peak-to-peak)")
    ax.set_title("Vibrato rate vs extent by technique")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout(); _save(fig, name)


# --------------------------------------------------------------------------- #
# 8. Feature correlation matrix
# --------------------------------------------------------------------------- #
def fig_corr(data: pd.DataFrame, name="fig_corr"):
    corr = data[FEATURE_NAMES].corr()
    fig, ax = plt.subplots(figsize=(8.8, 8.0))
    im = ax.imshow(corr, cmap=C.DIV_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=6)
    ax.set_yticks(range(len(FEATURE_NAMES)))
    ax.set_yticklabels(FEATURE_NAMES, fontsize=6)
    ax.set_title("Pearson correlation of acoustic features")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); _save(fig, name)


# --------------------------------------------------------------------------- #
# 9. Synthetic validation -- stands alone (no dataset)
# --------------------------------------------------------------------------- #
def fig_synthetic_validation(gt, meas, name="fig_synthetic_validation"):
    """gt, meas: dicts with keys 'rate' and 'extent' (lists)."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    for ax, key, unit in zip(axes, ["rate", "extent"],
                             ["Hz", "cents"]):
        g = np.array(gt[key]); m = np.array(meas[key])
        ax.scatter(g, m, s=40, color=C.PALETTE[0], edgecolor="#333", zorder=3)
        lim = [min(g.min(), m.min()) * 0.9, max(g.max(), m.max()) * 1.1]
        ax.plot(lim, lim, "--", color=C.PALETTE[1], label="ideal y=x")
        ax.set_xlabel(f"Ground-truth vibrato {key} ({unit})")
        ax.set_ylabel(f"Measured vibrato {key} ({unit})")
        ax.set_title(f"Vibrato {key} recovery")
        ax.legend(frameon=False)
    fig.suptitle("Synthetic-signal validation of the vibrato estimator")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); _save(fig, name)


if __name__ == "__main__":
    fig_pipeline()
    print("schematic ok")
