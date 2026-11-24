"""
Revision analyses that need only the cached feature matrix
(outputs/features.csv) and SHAP values (outputs/shap_values.npz).

Addresses the reviewer's statistical requests:
  1. per-fold mean +/- SD for every model, plus majority / stratified-chance
     baselines
  2. singer-level cluster bootstrap CIs and a paired XGBoost-vs-RF test
  3. nested (inner GroupKFold) hyper-parameter tuning vs the fixed a-priori
     configuration
  4. class-weighted XGBoost (imbalance handling in the objective)
  5. repeated random 15/5 singer splits (protocol of the published benchmark)
  6. cross-gender transfer (train on one gender, test on the other)
  7. permutation / gain importance vs SHAP (rank agreement)
  8. family-level and correlation-cluster-level grouped SHAP
  9. singer- vs technique-driven variance of every feature
 10. instance-level (out-of-fold) SHAP worked examples
 11. per-technique mean +/- SD table, and accuracy by exercise context

Run from the package root:  python -m tools.revision_stats
Writes JSON/CSV to outputs/revision/.
"""
from __future__ import annotations
import os
import sys
import json
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import shap

import config as C
from src.model import make_models, _pipeline
from src.features.extract import FEATURE_NAMES, FEATURE_GROUPS

REV_DIR = os.path.join(C.OUT_DIR, "revision")
os.makedirs(REV_DIR, exist_ok=True)
N_BOOT = 2000
RNG = np.random.default_rng(0)


def load():
    data = pd.read_csv(C.FEATURE_CSV)
    classes = sorted(data["technique"].unique().tolist())
    y = np.array([classes.index(t) for t in data["technique"]])
    X = data[FEATURE_NAMES].values
    g = data["singer"].values
    return data, X, y, g, classes


def context_of(path: str) -> str:
    b = os.path.basename(str(path)).lower()
    for k in ("arpeggios", "scales", "long_tones", "long"):
        if k in b:
            return "long_tones" if k.startswith("long") else k
    return "excerpts"


# --------------------------------------------------------------------------- #
# Cross-validation helpers
# --------------------------------------------------------------------------- #
def oof_predict(model, X, y, g, splits, sample_weight_fn=None):
    """Out-of-fold predictions + per-fold metrics for a fixed set of splits."""
    pred = np.zeros_like(y)
    proba = np.zeros((len(y), len(np.unique(y))))
    folds = []
    fitted = []
    for tr, te in splits:
        m = clone(model)
        kw = {}
        if sample_weight_fn is not None:
            kw["clf__sample_weight"] = sample_weight_fn(y[tr])
        m.fit(X[tr], y[tr], **kw)
        pred[te] = m.predict(X[te])
        proba[te] = m.predict_proba(X[te]) if hasattr(m, "predict_proba") else 0
        folds.append({"acc": accuracy_score(y[te], pred[te]),
                      "f1": f1_score(y[te], pred[te], average="macro")})
        fitted.append(m)
    return pred, proba, folds, fitted


def singer_bootstrap(y, preds: dict, g, n=N_BOOT):
    """Cluster bootstrap over singers: resample singers with replacement."""
    singers = np.unique(g)
    idx_by = {s: np.where(g == s)[0] for s in singers}
    out = {k: {"acc": [], "f1": []} for k in preds}
    for _ in range(n):
        pick = RNG.choice(singers, size=len(singers), replace=True)
        idx = np.concatenate([idx_by[s] for s in pick])
        for k, p in preds.items():
            out[k]["acc"].append(accuracy_score(y[idx], p[idx]))
            out[k]["f1"].append(f1_score(y[idx], p[idx], average="macro"))
    return {k: {m: np.array(v) for m, v in d.items()} for k, d in out.items()}


def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


# --------------------------------------------------------------------------- #
# Nested CV
# --------------------------------------------------------------------------- #
GRIDS = {
    "XGBoost": {"clf__max_depth": [3, 5, 7], "clf__learning_rate": [0.05, 0.1],
                "clf__n_estimators": [200, 400]},
    "Random Forest": {"clf__max_features": ["sqrt", 0.5],
                      "clf__min_samples_leaf": [1, 3]},
    "SVM (RBF)": {"clf__C": [1, 10, 100], "clf__gamma": ["scale", 0.01]},
    "Logistic Regression": {"clf__C": [0.1, 1.0, 10.0]},
}


def nested_cv(name, model, X, y, g, outer):
    pred = np.zeros_like(y)
    chosen, folds = [], []
    keys = list(GRIDS[name])
    combos = list(itertools.product(*GRIDS[name].values()))
    for tr, te in outer:
        inner = list(GroupKFold(n_splits=4).split(X[tr], y[tr], g[tr]))
        best, best_s = None, -1
        for combo in combos:
            params = dict(zip(keys, combo))
            s = []
            for itr, ite in inner:
                m = clone(model).set_params(**params)
                m.fit(X[tr][itr], y[tr][itr])
                s.append(f1_score(y[tr][ite], m.predict(X[tr][ite]),
                                  average="macro"))
            if np.mean(s) > best_s:
                best_s, best = float(np.mean(s)), params
        m = clone(model).set_params(**best).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        chosen.append({k: (v if isinstance(v, str) else float(v))
                       for k, v in best.items()})
        folds.append({"acc": accuracy_score(y[te], pred[te]),
                      "f1": f1_score(y[te], pred[te], average="macro")})
    return pred, folds, chosen


def summarize(folds, pred, y):
    a = [f["acc"] for f in folds]
    f = [f["f1"] for f in folds]
    return {"acc_pooled": float(accuracy_score(y, pred)),
            "f1_pooled": float(f1_score(y, pred, average="macro")),
            "acc_fold_mean": float(np.mean(a)), "acc_fold_sd": float(np.std(a, ddof=1)),
            "f1_fold_mean": float(np.mean(f)), "f1_fold_sd": float(np.std(f, ddof=1)),
            "folds": folds}


# --------------------------------------------------------------------------- #
def main():
    data, X, y, g, classes = load()
    R = {"classes": classes, "n": int(len(y)), "n_singers": int(len(np.unique(g)))}
    outer = list(GroupKFold(n_splits=5).split(X, y, g))
    R["fold_test_singers"] = [sorted(np.unique(g[te]).tolist()) for _, te in outer]

    # ---- 1. fixed models, per-fold variance -------------------------------
    models = make_models(len(classes))
    preds, probas, fitted_all = {}, {}, {}
    R["fixed"] = {}
    for name, mdl in models.items():
        p, pr, folds, fitted = oof_predict(mdl, X, y, g, outer)
        preds[name], probas[name], fitted_all[name] = p, pr, fitted
        R["fixed"][name] = summarize(folds, p, y)
        print(f"[fixed] {name:20s} {R['fixed'][name]['acc_pooled']:.3f} "
              f"{R['fixed'][name]['f1_pooled']:.3f}  "
              f"fold F1 {R['fixed'][name]['f1_fold_mean']:.3f}"
              f"+/-{R['fixed'][name]['f1_fold_sd']:.3f}")

    # class-weighted XGBoost (balanced sample weights in the objective)
    p, _, folds, _ = oof_predict(models["XGBoost"], X, y, g, outer,
                                 lambda yy: compute_sample_weight("balanced", yy))
    preds["XGBoost (class-weighted)"] = p
    R["fixed"]["XGBoost (class-weighted)"] = summarize(folds, p, y)
    print("[fixed] XGB weighted", R["fixed"]["XGBoost (class-weighted)"]["f1_pooled"])

    # baselines
    maj = np.bincount(y).argmax()
    R["baselines"] = {
        "majority_class": classes[maj],
        "majority_acc": float(np.mean(y == maj)),
        "majority_macro_f1": float(f1_score(y, np.full_like(y, maj), average="macro")),
        "uniform_chance_acc": 1.0 / len(classes),
        # stratified random guessing: E[acc] = sum p_c^2, E[macroF1] = mean p_c
        "stratified_chance_acc": float(np.sum((np.bincount(y) / len(y)) ** 2)),
        "stratified_chance_macro_f1": float(np.mean(np.bincount(y) / len(y))),
    }

    # per-class recall and F1 for XGB variants
    R["per_class"] = {}
    for name in ("XGBoost", "XGBoost (class-weighted)"):
        R["per_class"][name] = {
            "recall": dict(zip(classes, recall_score(y, preds[name], average=None).round(4).tolist())),
            "f1": dict(zip(classes, f1_score(y, preds[name], average=None).round(4).tolist()))}

    # ---- 2. singer bootstrap ---------------------------------------------
    B = singer_bootstrap(y, preds, g)
    R["bootstrap"] = {k: {"acc_ci": ci(v["acc"]), "f1_ci": ci(v["f1"])}
                      for k, v in B.items()}
    for a, b in [("XGBoost", "Random Forest"), ("XGBoost", "Logistic Regression"),
                 ("XGBoost", "SVM (RBF)"), ("XGBoost (class-weighted)", "XGBoost")]:
        dacc = B[a]["acc"] - B[b]["acc"]
        df1 = B[a]["f1"] - B[b]["f1"]
        # McNemar exact test on paired clip-level correctness
        ca, cb = preds[a] == y, preds[b] == y
        n01, n10 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
        pmc = float(stats.binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0
        R["bootstrap"][f"{a} - {b}"] = {
            "d_acc": float(accuracy_score(y, preds[a]) - accuracy_score(y, preds[b])),
            "d_acc_ci": ci(dacc), "d_f1_ci": ci(df1),
            "p_boot_f1_le0": float(np.mean(df1 <= 0)),
            "mcnemar_a_only": n01, "mcnemar_b_only": n10, "mcnemar_p": pmc}
    # per-class F1 CI for XGBoost
    singers = np.unique(g)
    idx_by = {s: np.where(g == s)[0] for s in singers}
    pcs = []
    for _ in range(N_BOOT):
        idx = np.concatenate([idx_by[s] for s in RNG.choice(singers, len(singers))])
        pcs.append(f1_score(y[idx], preds["XGBoost"][idx], average=None,
                            labels=range(len(classes)), zero_division=0))
    pcs = np.array(pcs)
    R["per_class_f1_ci"] = {c: ci(pcs[:, i]) for i, c in enumerate(classes)}

    # ---- 3. nested CV ----------------------------------------------------
    R["nested"] = {}
    for name, mdl in models.items():
        p, folds, chosen = nested_cv(name, mdl, X, y, g, outer)
        R["nested"][name] = summarize(folds, p, y) | {"chosen": chosen}
        print(f"[nested] {name:20s} {R['nested'][name]['acc_pooled']:.3f} "
              f"{R['nested'][name]['f1_pooled']:.3f}")

    # ---- 5. repeated random 15/5 singer splits ----------------------------
    rep = []
    for r in range(30):
        test = RNG.choice(singers, 5, replace=False)
        te = np.isin(g, test)
        m = clone(models["XGBoost"]).fit(X[~te], y[~te])
        pp = m.predict(X[te])
        rep.append({"acc": accuracy_score(y[te], pp),
                    "f1": f1_score(y[te], pp, average="macro")})
    R["random_15_5"] = {
        "acc_mean": float(np.mean([r["acc"] for r in rep])),
        "acc_sd": float(np.std([r["acc"] for r in rep], ddof=1)),
        "f1_mean": float(np.mean([r["f1"] for r in rep])),
        "f1_sd": float(np.std([r["f1"] for r in rep], ddof=1)),
        "f1_min": float(np.min([r["f1"] for r in rep])),
        "f1_max": float(np.max([r["f1"] for r in rep]))}

    # ---- 6. cross-gender transfer ----------------------------------------
    gen = data["gender"].values
    R["cross_gender"] = {}
    for tr_g, te_g in [("male", "female"), ("female", "male")]:
        tr, te = gen == tr_g, gen == te_g
        m = clone(models["XGBoost"]).fit(X[tr], y[tr])
        pp = m.predict(X[te])
        R["cross_gender"][f"{tr_g}->{te_g}"] = {
            "acc": float(accuracy_score(y[te], pp)),
            "f1": float(f1_score(y[te], pp, average="macro"))}
    for gg in ("male", "female"):   # within-gender OOF score of the main model
        mask = gen == gg
        R["cross_gender"][f"oof_{gg}"] = {
            "acc": float(accuracy_score(y[mask], preds["XGBoost"][mask])),
            "f1": float(f1_score(y[mask], preds["XGBoost"][mask], average="macro"))}

    # ---- 7. attribution-method agreement ----------------------------------
    perm = np.zeros(len(FEATURE_NAMES))
    for (tr, te), m in zip(outer, fitted_all["XGBoost"]):
        pi = permutation_importance(m, X[te], y[te], scoring="f1_macro",
                                    n_repeats=10, random_state=0, n_jobs=-1)
        perm += pi.importances_mean / len(outer)
    sh = pd.read_csv(os.path.join(C.TAB_DIR, "shap_importance.csv"), index_col=0)
    shap_g = sh.loc[FEATURE_NAMES, "overall"].values
    # gain importance from a model fitted on all data
    full = clone(models["XGBoost"]).fit(X, y)
    booster = full[-1].get_booster()
    gain_d = booster.get_score(importance_type="gain")
    gain = np.array([gain_d.get(f"f{i}", 0.0) for i in range(len(FEATURE_NAMES))])
    imp = pd.DataFrame({"shap": shap_g, "permutation": perm, "gain": gain},
                       index=FEATURE_NAMES)
    imp.to_csv(os.path.join(REV_DIR, "importance_methods.csv"))
    top10 = lambda s: set(imp[s].sort_values(ascending=False).index[:10])
    R["importance_agreement"] = {
        "spearman_shap_perm": float(stats.spearmanr(imp.shap, imp.permutation)[0]),
        "spearman_shap_gain": float(stats.spearmanr(imp.shap, imp.gain)[0]),
        "spearman_perm_gain": float(stats.spearmanr(imp.permutation, imp.gain)[0]),
        "kendall_shap_perm": float(stats.kendalltau(imp.shap, imp.permutation)[0]),
        "top10_overlap_shap_perm": len(top10("shap") & top10("permutation")),
        "top10_overlap_shap_gain": len(top10("shap") & top10("gain")),
        "perm_top10": imp.permutation.sort_values(ascending=False).index[:10].tolist(),
        "gain_top10": imp.gain.sort_values(ascending=False).index[:10].tolist()}

    # family-level permutation importance (permute a whole family jointly)
    fam_perm = {}
    for fam, feats in FEATURE_GROUPS.items():
        cols = [FEATURE_NAMES.index(f) for f in feats]
        drops = []
        for (tr, te), m in zip(outer, fitted_all["XGBoost"]):
            base = f1_score(y[te], m.predict(X[te]), average="macro")
            for r in range(10):
                Xp = X[te].copy()
                perm_idx = np.random.default_rng(r).permutation(len(te))
                Xp[:, cols] = Xp[perm_idx][:, cols]
                drops.append(base - f1_score(y[te], m.predict(Xp), average="macro"))
        fam_perm[fam] = {"mean_drop": float(np.mean(drops)), "sd": float(np.std(drops))}
    R["family_permutation"] = fam_perm

    # ---- 8. grouped SHAP --------------------------------------------------
    npz = np.load(os.path.join(C.OUT_DIR, "shap_values.npz"), allow_pickle=True)
    arr = npz["shap"]                                   # (n, f, c)
    fam_sh = {}
    for fam, feats in FEATURE_GROUPS.items():
        cols = [FEATURE_NAMES.index(f) for f in feats]
        s = arr[:, cols, :].sum(axis=1)                  # (n, c)
        fam_sh[fam] = {"overall": float(np.abs(s).mean()),
                       **{c: float(np.abs(s[:, i]).mean()) for i, c in enumerate(classes)}}
    R["family_shap"] = fam_sh

    # correlation structure: Spearman, clusters at |rho| > 0.7
    D = data[FEATURE_NAMES].rank().corr(method="pearson").fillna(0).values
    absr = np.abs(D)
    fam_of = {f: fam for fam, fs in FEATURE_GROUPS.items() for f in fs}
    within, across = [], []
    for i, j in itertools.combinations(range(len(FEATURE_NAMES)), 2):
        (within if fam_of[FEATURE_NAMES[i]] == fam_of[FEATURE_NAMES[j]] else across).append(absr[i, j])
    dist = 1 - absr
    np.fill_diagonal(dist, 0)
    Z = hierarchy.linkage(squareform(dist, checks=False), method="average")
    lab = hierarchy.fcluster(Z, t=0.3, criterion="distance")
    clusters = {}
    for f, l in zip(FEATURE_NAMES, lab):
        clusters.setdefault(int(l), []).append(f)
    multi = {k: v for k, v in clusters.items() if len(v) > 1}
    cl_sh = []
    for k, feats in multi.items():
        cols = [FEATURE_NAMES.index(f) for f in feats]
        s = arr[:, cols, :].sum(axis=1)
        indiv = [float(np.abs(arr[:, c, :]).mean()) for c in cols]
        cl_sh.append({"features": feats, "grouped_shap": float(np.abs(s).mean()),
                      "member_shap": dict(zip(feats, indiv))})
    pairs = sorted([(FEATURE_NAMES[i], FEATURE_NAMES[j], float(D[i, j]))
                    for i, j in itertools.combinations(range(len(FEATURE_NAMES)), 2)],
                   key=lambda t: -abs(t[2]))[:15]
    R["correlation"] = {"mean_abs_within": float(np.mean(within)),
                        "mean_abs_across": float(np.mean(across)),
                        "n_pairs_gt07": int(np.sum(np.array(within + across) > 0.7)),
                        "n_pairs_gt07_across": int(np.sum(np.array(across) > 0.7)),
                        "top_pairs": pairs, "clusters_gt07": cl_sh}

    # ---- 9. singer vs technique variance ----------------------------------
    rows = []
    for f in FEATURE_NAMES:
        d = data[[f, "technique", "singer"]].dropna()
        v = d[f].rank().values                        # rank-based, robust
        tot = np.var(v)
        dv = pd.DataFrame({"v": v, "t": d.technique.values, "s": d.singer.values})
        r2_t = 1 - dv.groupby("t").v.transform(lambda s: s - s.mean()).var(ddof=0) / tot
        r2_ts = 1 - dv.groupby(["t", "s"]).v.transform(lambda s: s - s.mean()).var(ddof=0) / tot
        r2_s = 1 - dv.groupby("s").v.transform(lambda s: s - s.mean()).var(ddof=0) / tot
        rows.append({"feature": f, "family": fam_of[f], "eta2_technique": r2_t,
                     "eta2_singer": r2_s, "eta2_singer_given_tech": r2_ts - r2_t})
    var = pd.DataFrame(rows)
    var.to_csv(os.path.join(REV_DIR, "singer_vs_technique_variance.csv"), index=False)
    R["variance"] = {
        "by_family": var.groupby("family")[["eta2_technique", "eta2_singer",
                                            "eta2_singer_given_tech"]].mean().round(3).to_dict("index"),
        "singer_dominated": var.loc[var.eta2_singer > var.eta2_technique, "feature"].tolist()}

    # ---- 10. instance-level OOF SHAP examples -----------------------------
    examples = []
    xgb_proba = probas["XGBoost"]
    med = data.groupby("technique")[FEATURE_NAMES].median()
    for target, want_correct in [("breathy", True), ("breathy", False),
                                 ("vibrato", True), ("straight", False)]:
        ti = classes.index(target)
        cand = np.where((y == ti) & ((preds["XGBoost"] == ti) == want_correct))[0]
        if cand.size == 0:
            continue
        k = cand[np.argmax(xgb_proba[cand, preds["XGBoost"][cand]])]
        fold = next(i for i, (_, te) in enumerate(outer) if k in te)
        m = fitted_all["XGBoost"][fold]
        xt = m[:-1].transform(X[k:k + 1])
        sv = shap.TreeExplainer(m[-1]).shap_values(xt)
        sv = np.stack(sv, -1) if isinstance(sv, list) else sv   # (1, f, c)
        pc = int(preds["XGBoost"][k])
        contrib = sv[0, :, pc]
        order = np.argsort(-np.abs(contrib))[:6]
        fam_c = {fam: float(contrib[[FEATURE_NAMES.index(f) for f in fs]].sum())
                 for fam, fs in FEATURE_GROUPS.items()}
        examples.append({
            "clip": os.path.basename(str(data.path[k])), "singer": data.singer[k],
            "true": classes[y[k]], "pred": classes[pc],
            "p_pred": float(xgb_proba[k, pc]),
            "p_true": float(xgb_proba[k, y[k]]),
            "family_contrib_pred": fam_c,
            "top": [{"feature": FEATURE_NAMES[i], "value": float(X[k, i]),
                     "shap": float(contrib[i]),
                     "median_true": float(med.loc[classes[y[k]], FEATURE_NAMES[i]]),
                     "median_pred": float(med.loc[classes[pc], FEATURE_NAMES[i]]),
                     "median_straight": float(med.loc["straight", FEATURE_NAMES[i]]),
                     "median_vibrato": float(med.loc["vibrato", FEATURE_NAMES[i]])}
                    for i in order]})
    R["instance_examples"] = examples

    # ---- 11. mean +/- SD table, context breakdown -------------------------
    tab_cols = ["f0_std_cents", "vibrato_rate", "vibrato_extent", "vibrato_strength",
                "singers_formant_ratio", "sustain_max_sec", "hnr_mean"]
    msd = data.groupby("technique")[tab_cols].agg(["mean", "std"])
    msd.to_csv(os.path.join(REV_DIR, "feature_mean_sd_by_technique.csv"))
    ctx = data["path"].map(context_of)
    R["context"] = {
        "counts": pd.crosstab(data.technique, ctx).to_dict("index"),
        "acc_by_context": {c: float(accuracy_score(y[ctx == c], preds["XGBoost"][ctx == c]))
                           for c in ctx.unique()},
        "sigma_cents_median_by_context": data.groupby(ctx)["f0_std_cents"].median().to_dict()}

    with open(os.path.join(REV_DIR, "revision_stats.json"), "w") as f:
        json.dump(R, f, indent=2, default=float)
    print("wrote", os.path.join(REV_DIR, "revision_stats.json"))


if __name__ == "__main__":
    main()
