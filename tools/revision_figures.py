"""
Figures added in the revision. Reads the JSON/CSV written by
tools/revision_stats.py and tools/revision_audio.py (outputs/revision/)
and writes PDF+PNG to outputs/figures/ with the shared manuscript style.

Run from the package root:  python -m tools.revision_figures
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from src.features.extract import FEATURE_NAMES, FEATURE_GROUPS

C.apply_style()
REV = os.path.join(C.OUT_DIR, "revision")
FAMS = list(FEATURE_GROUPS)
FAM_COL = {f: C.PALETTE[i] for i, f in enumerate(FAMS)}
MODELS = ["Logistic Regression", "SVM (RBF)", "Random Forest", "XGBoost"]


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(C.FIG_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"[fig] {name}")


def _panel(ax, letter):
    ax.text(-0.14, 1.04, f"({letter})", transform=ax.transAxes,
            fontweight="bold", fontsize=11, va="bottom")


# --------------------------------------------------------------------------- #
def fig_statistics(S):
    rc = {"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
          "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8}
    with plt.rc_context(rc):
        _fig_statistics(S)


def _fig_statistics(S):
    # one row, ~40% of a page at \textwidth; text prints at ~7.5-8 pt
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0),
                             gridspec_kw={"width_ratios": [1.05, 1.0, 1.15]})
    # (a) fold-level macro-F1, fixed vs nested
    ax = axes[0]
    names = MODELS + ["XGBoost (class-weighted)"]
    short = ["LR", "SVM", "RF", "XGB", "XGB-w"]
    for i, n in enumerate(names):
        f = [d["f1"] for d in S["fixed"][n]["folds"]]
        ax.scatter(np.full(len(f), i) - 0.15, f, s=9, color=C.PALETTE[0],
                   alpha=0.6, zorder=3, lw=0)
        ax.errorbar(i - 0.15, np.mean(f), yerr=np.std(f, ddof=1), fmt="o",
                    color=C.PALETTE[0], ms=4, capsize=2, zorder=4, lw=1,
                    label="fixed" if i == 0 else None)
        if n in S["nested"]:
            fn = [d["f1"] for d in S["nested"][n]["folds"]]
            ax.scatter(np.full(len(fn), i) + 0.15, fn, s=9,
                       color=C.PALETTE[1], alpha=0.6, zorder=3, lw=0)
            ax.errorbar(i + 0.15, np.mean(fn), yerr=np.std(fn, ddof=1),
                        fmt="s", color=C.PALETTE[1], ms=4, capsize=2, lw=1,
                        zorder=4, label="nested" if i == 0 else None)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(short)
    ax.set_ylabel("Macro-F1 per singer fold"); ax.set_ylim(0.55, 0.9)
    ax.set_title("Fold variability")
    ax.legend(frameon=False, loc="lower right", handletextpad=0.2)
    _panel(ax, "a")

    # (b) paired differences with singer-bootstrap CI
    ax = axes[1]
    pairs = [("XGBoost", "Random Forest"), ("XGBoost", "Logistic Regression"),
             ("XGBoost", "SVM (RBF)"), ("XGBoost (class-weighted)", "XGBoost")]
    lab = ["XGB − RF", "XGB − LR", "XGB − SVM", "XGB-w − XGB"]
    for i, (a, bb) in enumerate(pairs):
        d = S["bootstrap"][f"{a} - {bb}"]
        lo, hi = d["d_f1_ci"]
        pt = S["fixed"][a]["f1_pooled"] - S["fixed"][bb]["f1_pooled"]
        col = C.PALETTE[2] if lo > 0 else C.PALETTE[7]
        ax.errorbar(pt, i, xerr=[[pt - lo], [hi - pt]], fmt="o", color=col,
                    capsize=3, ms=4, lw=1.1)
        p = d["mcnemar_p"]
        ax.text(hi + 0.003, i, "p<0.001" if p < 0.001 else f"p={p:.2f}",
                va="center", fontsize=8)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab)
    ax.set_ylim(len(lab) - 0.5, -0.5)
    ax.set_xlabel("Δ macro-F1 (95% CI)")
    ax.set_title("Paired comparisons")
    xmax = max(S["bootstrap"][f"{a} - {bb}"]["d_f1_ci"][1] for a, bb in pairs)
    ax.set_xlim(None, xmax + 0.042)
    _panel(ax, "b")

    # (c) per-class F1 with CI
    ax = axes[2]
    pc = S["per_class"]["XGBoost"]["f1"]
    ci = S["per_class_f1_ci"]
    order = sorted(pc, key=lambda c: pc[c])
    for i, c in enumerate(order):
        lo, hi = ci[c]
        ax.errorbar(pc[c], i, xerr=[[pc[c] - lo], [hi - pc[c]]], fmt="o",
                    color=C.PALETTE[0], capsize=2, ms=4, lw=1,
                    label="XGBoost" if i == 0 else None)
        pw = S["per_class"]["XGBoost (class-weighted)"]["f1"][c]
        ax.scatter(pw, i + 0.25, marker="x", s=18, color=C.PALETTE[1], zorder=4,
                   lw=1, label="weighted" if i == 0 else None)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([c.replace("_", " ") for c in order])
    ax.set_xlabel("F1 (95% CI)")
    ax.set_xlim(0.55, 1.02)
    ax.set_title("Per-technique F1")
    ax.legend(frameon=False, loc="lower right", handletextpad=0.1,
              borderaxespad=0.2)
    _panel(ax, "c")
    fig.tight_layout(w_pad=0.6); _save(fig, "fig_statistics")


# --------------------------------------------------------------------------- #
def fig_attribution(S):
    imp = pd.read_csv(os.path.join(REV, "importance_methods.csv"), index_col=0)
    fig = plt.figure(figsize=(13.0, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 0.9, 1.0])
    # (a) grouped (family) SHAP per class
    ax = fig.add_subplot(gs[0])
    classes = S["classes"]
    M = np.array([[S["family_shap"][f][c] for c in classes] for f in FAMS])
    Mn = M / M.sum(axis=0, keepdims=True)
    ax.grid(False)
    im = ax.imshow(Mn, aspect="auto", cmap=C.SEQ_CMAP, vmin=0, vmax=Mn.max())
    for i in range(len(FAMS)):
        for j in range(len(classes)):
            ax.text(j, i, f"{Mn[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if Mn[i, j] > 0.45 else "#222")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(FAMS))); ax.set_yticklabels(FAMS)
    ax.set_title("Grouped SHAP: share of |φ| per family")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    _panel(ax, "a")
    # (b) family-level permutation importance vs grouped SHAP
    ax = fig.add_subplot(gs[1])
    y = np.arange(len(FAMS))
    gsh = np.array([S["family_shap"][f]["overall"] for f in FAMS])
    pdrop = np.array([S["family_permutation"][f]["mean_drop"] for f in FAMS])
    psd = np.array([S["family_permutation"][f]["sd"] for f in FAMS])
    ax.barh(y - 0.2, gsh / gsh.max(), 0.38, color=C.PALETTE[0],
            label="grouped SHAP")
    ax.barh(y + 0.2, pdrop / pdrop.max(), 0.38, color=C.PALETTE[3],
            xerr=psd / pdrop.max(), capsize=2, label="joint permutation")
    ax.set_yticks(y); ax.set_yticklabels(FAMS); ax.invert_yaxis()
    ax.set_xlabel("Relative importance (max = 1)")
    ax.set_title("Family importance, two methods")
    ax.set_xlim(0, 1.6)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    _panel(ax, "b")
    # (c) feature-level rank agreement
    ax = fig.add_subplot(gs[2])
    fam_of = {f: k for k, fs in FEATURE_GROUPS.items() for f in fs}
    rs, rp = imp.shap.rank(ascending=False), imp.permutation.rank(ascending=False)
    for f in imp.index:
        ax.scatter(rs[f], rp[f], color=FAM_COL[fam_of[f]], s=26,
                   edgecolor="#333", lw=0.4)
    for f in imp.shap.sort_values(ascending=False).index[:5]:
        ax.annotate(f, (rs[f], rp[f]), fontsize=6.5, xytext=(3, 2),
                    textcoords="offset points")
    ax.plot([1, 31], [1, 31], "--", color="#888", lw=0.8)
    ax.set_xlabel("SHAP rank"); ax.set_ylabel("Permutation-importance rank")
    a = S["importance_agreement"]
    ax.set_title(f"Feature ranks (Spearman ρ = {a['spearman_shap_perm']:.2f})")
    for f in FAMS:
        ax.scatter([], [], color=FAM_COL[f], label=f, s=20)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    _panel(ax, "c")
    fig.tight_layout(); _save(fig, "fig_attribution")


# --------------------------------------------------------------------------- #
def fig_vibrato(A):
    syn = pd.read_csv(os.path.join(REV, "synthetic_nonsinusoidal.csv"))
    real = pd.read_csv(os.path.join(REV, "real_vibrato_estimators.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.0))
    ax = axes[0, 0]
    ax.scatter(syn.extent_true, syn.extent_spec, s=18, color=C.PALETTE[0],
               label="spectral (Eq. 5)")
    ax.scatter(syn.extent_true, syn.extent_cycle, s=18, marker="^",
               color=C.PALETTE[1], label="cycle-based")
    lim = [20, 180]; ax.plot(lim, lim, "--", color="#777")
    ax.set_xlabel("True extent (cents)"); ax.set_ylabel("Estimated extent (cents)")
    ax.set_title("Non-sinusoidal synthetic vibrato: extent")
    ax.legend(frameon=False, fontsize=8); _panel(ax, "a")
    ax = axes[0, 1]
    ax.scatter(syn.rate_true, syn.rate_spec, s=18, color=C.PALETTE[0],
               label="spectral (Eq. 4)")
    ax.scatter(syn.rate_true, syn.rate_cycle, s=18, marker="^",
               color=C.PALETTE[1], label="cycle-based")
    lim = [4.2, 7.4]; ax.plot(lim, lim, "--", color="#777")
    ax.set_xlabel("True mean rate (Hz)"); ax.set_ylabel("Estimated rate (Hz)")
    ax.set_title("Non-sinusoidal synthetic vibrato: rate")
    ax.legend(frameon=False, fontsize=8); _panel(ax, "b")
    ctxs = ["arpeggios", "scales", "excerpts"]
    ax = axes[1, 0]
    for i, c in enumerate(ctxs):
        s = real[real.context == c]
        ax.scatter(s.extent_cycle, s.extent_spec, s=16, alpha=0.75,
                   color=C.PALETTE[i + 2], label=c)
    lo, hi = np.nanmin(real[["extent_spec", "extent_cycle"]].values), \
        np.nanmax(real[["extent_spec", "extent_cycle"]].values)
    ax.plot([lo, hi], [lo, hi], "--", color="#777")
    r = A["real_vibrato"]["all"]["extent"]
    ax.set_xlabel("Cycle-based extent (cents)")
    ax.set_ylabel("Spectral extent (cents)")
    ax.set_title(f"Real VocalSet vibrato (n={r['n']}): extent, r={r['pearson_r']:.2f}")
    ax.legend(frameon=False, fontsize=8); _panel(ax, "c")
    ax = axes[1, 1]
    m = (real.extent_spec + real.extent_cycle) / 2
    d = real.extent_spec - real.extent_cycle
    ax.scatter(m, d, s=14, color=C.PALETTE[0], alpha=0.7)
    ax.axhline(r["bias"], color=C.PALETTE[1])
    for l in r["loa"]:
        ax.axhline(l, ls="--", color=C.PALETTE[1], lw=0.9)
    ax.text(0.98, 0.95, f"bias {r['bias']:+.1f} c\nLoA [{r['loa'][0]:.0f}, {r['loa'][1]:.0f}] c",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)
    ax.set_xlabel("Mean of the two estimators (cents)")
    ax.set_ylabel("Spectral − cycle-based (cents)")
    ax.set_title("Bland–Altman agreement on real vibrato")
    _panel(ax, "d")
    fig.tight_layout(); _save(fig, "fig_vibrato_validation")


# --------------------------------------------------------------------------- #
def fig_robustness(A, S):
    rob = A["robustness"]
    conds = ["gain_-12dB", "noise_30dB", "noise_20dB", "reverb_0.5s"]
    clab = ["gain −12 dB", "noise 30 dB", "noise 20 dB", "reverb 0.5 s"]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2),
                             gridspec_kw={"width_ratios": [1.2, 0.85, 1.1]})
    ax = axes[0]
    x = np.arange(len(conds)); w = 0.2
    for i, f in enumerate(FAMS):
        ax.bar(x + (i - 1.5) * w, [rob[c]["family_mean_rho"][f] for c in conds],
               w, color=FAM_COL[f], label=f, edgecolor="#333", lw=0.4)
    ax.set_xticks(x); ax.set_xticklabels(clab)
    ax.set_ylabel("Mean Spearman ρ with clean features")
    ax.set_ylim(0, 1.25); ax.set_title("Feature rank stability under perturbation")
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=4); _panel(ax, "a")
    ax = axes[1]
    allc = ["clean"] + conds
    for k, mname in enumerate(["XGBoost", "Random Forest"]):
        vals = [rob["clean_classification"][mname]["f1"]] + \
            [rob[c]["classification"][mname]["f1"] for c in conds]
        ax.plot(range(len(allc)), vals, "-o", color=C.PALETTE[k], label=mname)
    ax.set_xticks(range(len(allc)))
    ax.set_xticklabels(["clean"] + clab, rotation=30, ha="right")
    ax.set_ylabel("Macro-F1 (trained on clean audio)")
    ax.set_ylim(0.4, 0.85); ax.set_title(f"Accuracy on perturbed clips (n={A['robustness_n_clips']})")
    ax.legend(frameon=False, fontsize=8); _panel(ax, "b")
    ax = axes[2]
    var = pd.read_csv(os.path.join(REV, "singer_vs_technique_variance.csv"))
    for f in FAMS:
        s = var[var.family == f]
        ax.scatter(s.eta2_technique, s.eta2_singer, color=FAM_COL[f], s=28,
                   edgecolor="#333", lw=0.4, label=f)
    for _, r in var.nlargest(4, "eta2_singer").iterrows():
        ax.annotate(r.feature, (r.eta2_technique, r.eta2_singer), fontsize=6.5,
                    xytext=(3, 2), textcoords="offset points")
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("η² explained by technique")
    ax.set_ylabel("η² explained by singer")
    ax.set_title("Technique- vs singer-driven variance")
    ax.legend(frameon=False, fontsize=7, loc="upper right"); _panel(ax, "c")
    fig.tight_layout(); _save(fig, "fig_robustness")


# --------------------------------------------------------------------------- #
def fig_instance(S):
    ex = S["instance_examples"][:2]
    fig, axes = plt.subplots(1, len(ex), figsize=(6.2 * len(ex), 3.8))
    axes = np.atleast_1d(axes)
    fam_of = {f: k for k, fs in FEATURE_GROUPS.items() for f in fs}
    for k, (ax, e) in enumerate(zip(axes, ex)):
        top = e["top"][::-1]
        vals = [t["shap"] for t in top]
        cols = [FAM_COL[fam_of[t["feature"]]] for t in top]
        ax.barh(range(len(top)), vals, color=cols, edgecolor="#333", lw=0.4)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([f"{t['feature']} = {t['value']:.3g}" for t in top],
                           fontsize=8)
        ax.axvline(0, color="#333", lw=0.8)
        ax.set_xlabel(f"SHAP contribution to class '{e['pred']}' (log-odds)")
        ok = "correct" if e["true"] == e["pred"] else "error"
        ax.set_title(f"{e['clip']}  (true {e['true']}, pred {e['pred']}, "
                     f"p={e['p_pred']:.2f}, {ok})", fontsize=9)
        _panel(ax, "abcd"[k])
    from matplotlib.patches import Patch
    axes[-1].legend(handles=[Patch(color=FAM_COL[f], label=f) for f in FAMS],
                    frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(); _save(fig, "fig_instance")


def main():
    A = json.load(open(os.path.join(REV, "revision_audio.json")))
    fig_vibrato(A)
    sp = os.path.join(REV, "revision_stats.json")
    if os.path.exists(sp):
        S = json.load(open(sp))
        fig_statistics(S); fig_attribution(S); fig_robustness(A, S)
        fig_instance(S)
    else:
        print("revision_stats.json not found; skipped stats figures")



def fig_loso():
    """Per-singer leave-one-singer-out macro-F1 (run after revision_extra)."""
    p = os.path.join(REV, "loso_per_singer.csv")
    if not os.path.exists(p):
        print("loso_per_singer.csv not found; skipped"); return
    d = pd.read_csv(p).sort_values(["gender", "f1"])
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0),
                             gridspec_kw={"width_ratios": [1.6, 1.0]})
    ax = axes[0]
    col = [C.PALETTE[1] if g == "female" else C.PALETTE[0] for g in d.gender]
    ax.bar(range(len(d)), d.f1, color=col, edgecolor="#333", lw=0.4)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d.singer.str.replace("female", "F").str.replace("male", "M"),
                       rotation=60, fontsize=7)
    ax.axhline(d.f1.mean(), ls="--", color="#555", lw=0.9)
    ax.set_ylabel("Macro-F1 on the held-out singer"); ax.set_ylim(0, 1)
    ax.set_title("Leave-one-singer-out: per-singer macro-F1")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C.PALETTE[1], label="female"),
                       Patch(color=C.PALETTE[0], label="male")],
              frameon=False, fontsize=8, loc="upper left")
    _panel(ax, "a")
    ax = axes[1]
    for g, c in (("female", C.PALETTE[1]), ("male", C.PALETTE[0])):
        s = d[d.gender == g]
        ax.scatter(s.median_f0, s.f1, color=c, s=30, edgecolor="#333", lw=0.4, label=g)
    for _, r in d.nsmallest(2, "f1").iterrows():
        ax.annotate(r.singer, (r.median_f0, r.f1), fontsize=7, xytext=(4, 2),
                    textcoords="offset points")
    ax.set_xlabel("Singer median $f_0$ (Hz)"); ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1); ax.set_title("Accuracy versus register")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _panel(ax, "b")
    fig.tight_layout(); _save(fig, "fig_loso")


def fig_corr_compact():
    """Compact replacement for the 31x31 correlation matrix."""
    import itertools
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform
    data = pd.read_csv(C.FEATURE_CSV)
    R = data[FEATURE_NAMES].rank().corr().fillna(0).values
    A = np.abs(R)
    fam_of = {f: k for k, fs in FEATURE_GROUPS.items() for f in fs}
    fig = plt.figure(figsize=(13.0, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.62, 2.0])
    # (a) family x family mean |rho|
    ax = fig.add_subplot(gs[0])
    ax.grid(False)
    M = np.zeros((len(FAMS), len(FAMS)))
    for i, fa in enumerate(FAMS):
        for j, fb in enumerate(FAMS):
            ia = [FEATURE_NAMES.index(f) for f in FEATURE_GROUPS[fa]]
            ib = [FEATURE_NAMES.index(f) for f in FEATURE_GROUPS[fb]]
            vals = [A[a, b] for a in ia for b in ib if a != b]
            M[i, j] = np.mean(vals)
    im = ax.imshow(M, cmap=C.SEQ_CMAP, vmin=0, vmax=0.5)
    for i in range(len(FAMS)):
        for j in range(len(FAMS)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if M[i, j] > 0.3 else "#222")
    short = ["Pitch", "Vibrato", "Resonance", "Breath"]
    ax.set_xticks(range(4)); ax.set_xticklabels(short, rotation=30, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(short)
    ax.set_title("Mean |Spearman ρ| between families")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _panel(ax, "a")
    # (b) dendrogram
    ax = fig.add_subplot(gs[1])
    D = 1 - A
    np.fill_diagonal(D, 0)
    Z = hierarchy.linkage(squareform(D, checks=False), method="average")
    hierarchy.set_link_color_palette(["#222222"])
    dn = hierarchy.dendrogram(Z, labels=FEATURE_NAMES, ax=ax,
                              color_threshold=0.3, above_threshold_color="#999999",
                              leaf_rotation=90, leaf_font_size=8)
    for lbl in ax.get_xticklabels():
        lbl.set_color(FAM_COL[fam_of[lbl.get_text()]])
    ax.axhline(0.3, ls="--", color=C.PALETTE[1], lw=1)
    ax.text(ax.get_xlim()[0] + 2, 0.31, "|ρ| = 0.7 cut", ha="left", va="bottom",
            fontsize=8, color=C.PALETTE[1])
    ax.set_ylabel("1 − |Spearman ρ|")
    ax.set_title("Average-linkage clustering of the 31 features (labels coloured by family)")
    ax.grid(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=FAM_COL[f], label=f) for f in FAMS],
              frameon=False, fontsize=8, loc="upper right", ncol=4)
    ax.set_ylim(0, 1.08)
    _panel(ax, "b")
    fig.tight_layout(); _save(fig, "fig_corr_compact")


if __name__ == "__main__":
    main()
    fig_loso()
    fig_corr_compact()
