"""
Second round of revision analyses.

  A. Focal-loss XGBoost (gamma = 1, 2) versus the standard softmax objective,
     same singer folds, same hyper-parameters (reviewer 5.2).
  B. Leave-one-singer-out (LOSO) evaluation: per-singer accuracy / macro-F1,
     and per-singer feature deviation from the other singers, by gender and
     register (reviewer 3.1).
  C. Vibrato on real recordings with an established, independent pitch
     tracker (pYIN, librosa) in place of Praat, scored with the same spectral
     estimator and with the cycle-based estimator (reviewer 3.5).
  D. Per-fold nested-CV selections and test singers, as a table (reviewer 5.1).

Run from the package root:  python -m tools.revision_extra
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import signal, stats
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import librosa
import xgboost as xgb
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, recall_score

import config as C
from src.model import make_models
from src.features.extract import FEATURE_NAMES, FEATURE_GROUPS
from src.features.vibrato import _longest_voiced, FS, VIB_LO, VIB_HI
from tools.revision_audio import cycle_vibrato, agreement, local_path

REV = os.path.join(C.OUT_DIR, "revision")
RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# A. Focal loss
# --------------------------------------------------------------------------- #
def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _focal_grad(z, y, gamma):
    p = np.clip(_softmax(z), 1e-12, 1 - 1e-12)
    n = len(y)
    pt = p[np.arange(n), y]
    # L = -(1-pt)^g log pt ;  dL/dpt
    dl_dpt = gamma * (1 - pt) ** (gamma - 1) * np.log(pt) - (1 - pt) ** gamma / pt
    onehot = np.zeros_like(p)
    onehot[np.arange(n), y] = 1.0
    # dpt/dz_j = pt (1[j=t] - p_j)
    return (dl_dpt * pt)[:, None] * (onehot - p)


def focal_objective(gamma, k):
    def obj(preds, dtrain):
        y = dtrain.get_label().astype(int)
        z = preds.reshape(len(y), k)
        g = _focal_grad(z, y, gamma)
        # diagonal Hessian by central finite differences of the gradient
        h = np.zeros_like(z)
        eps = 1e-3
        for j in range(k):
            zp, zm = z.copy(), z.copy()
            zp[:, j] += eps
            zm[:, j] -= eps
            h[:, j] = (_focal_grad(zp, y, gamma)[:, j] -
                       _focal_grad(zm, y, gamma)[:, j]) / (2 * eps)
        h = np.maximum(h, 1e-6)
        return g.reshape(-1), h.reshape(-1)
    return obj


PARAMS = dict(max_depth=5, eta=0.05, subsample=0.9, colsample_bytree=0.8,
              tree_method="hist", seed=0)


def native_oof(X, y, splits, k, gamma=None):
    pred = np.zeros_like(y)
    for tr, te in splits:
        imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X[tr])
        dtr = xgb.DMatrix(imp.transform(X[tr]), label=y[tr])
        dte = xgb.DMatrix(imp.transform(X[te]))
        if gamma is None:
            b = xgb.train({**PARAMS, "objective": "multi:softprob",
                           "num_class": k}, dtr, 400)
            pr = b.predict(dte).reshape(len(te), k)
        else:
            b = xgb.train({**PARAMS, "num_class": k, "disable_default_eval_metric": 1},
                          dtr, 400, obj=focal_objective(gamma, k))
            pr = b.predict(dte, output_margin=True).reshape(len(te), k)
        pred[te] = pr.argmax(1)
    return pred


def singer_boot_diff(y, pa, pb, g, n=2000):
    singers = np.unique(g)
    idx_by = {s: np.where(g == s)[0] for s in singers}
    d = []
    for _ in range(n):
        idx = np.concatenate([idx_by[s] for s in RNG.choice(singers, len(singers))])
        d.append(f1_score(y[idx], pa[idx], average="macro") -
                 f1_score(y[idx], pb[idx], average="macro"))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def mcnemar(y, pa, pb):
    ca, cb = pa == y, pb == y
    a, b = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    return a, b, float(stats.binomtest(a, a + b, 0.5).pvalue) if a + b else 1.0


# --------------------------------------------------------------------------- #
# C. pYIN contour + spectral estimator (same maths as src.features.vibrato)
# --------------------------------------------------------------------------- #
def spectral_from_cents(cents):
    sos_hp = signal.butter(2, 2.5, btype="high", fs=FS, output="sos")
    resid = signal.sosfiltfilt(sos_hp, cents - float(np.mean(cents)))
    n = resid.size
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    psd = np.abs(np.fft.rfft(resid * np.hanning(n))) ** 2
    band = (freqs >= VIB_LO) & (freqs <= VIB_HI)
    if not band.any() or psd[band].sum() <= 0:
        return np.nan, np.nan
    rate = float(freqs[np.argmax(np.where(band, psd, 0.0))])
    lo, hi = max(VIB_LO, rate - 2.0), min(VIB_HI, rate + 2.0)
    sos = signal.butter(4, [lo, hi], btype="band", fs=FS, output="sos")
    bp = signal.sosfiltfilt(sos, resid)
    return rate, float(2.0 * np.sqrt(2.0) * np.std(bp))


def pyin_vibrato(path):
    y, sr = librosa.load(path, sr=C.SR, mono=True)
    y = librosa.util.normalize(y)
    hop = int(round(C.HOP_SEC * sr))
    f0, vflag, _ = librosa.pyin(y, fmin=C.F0_MIN, fmax=C.F0_MAX, sr=sr,
                                frame_length=2048, hop_length=hop)
    f0 = np.where(vflag & np.isfinite(f0), f0, 0.0)
    seg = _longest_voiced(f0)
    if seg.size < int(0.4 * FS):
        return np.nan, np.nan, np.nan, np.nan
    cents = 1200 * np.log2(seg / np.median(seg))
    r, e = spectral_from_cents(cents)
    rc, ec = cycle_vibrato(cents)
    return r, e, rc, ec


# --------------------------------------------------------------------------- #
def main():
    data = pd.read_csv(C.FEATURE_CSV)
    classes = sorted(data.technique.unique().tolist())
    k = len(classes)
    y = np.array([classes.index(t) for t in data.technique])
    X = data[FEATURE_NAMES].values
    g = data.singer.values
    gen = data.gender.values
    outer = list(GroupKFold(n_splits=5).split(X, y, g))
    R = {}

    # ---------------- A. focal loss ---------------------------------------
    base = native_oof(X, y, outer, k)
    R["focal"] = {"softmax_native": {
        "acc": float(accuracy_score(y, base)),
        "f1": float(f1_score(y, base, average="macro")),
        "recall": dict(zip(classes, recall_score(y, base, average=None).round(3).tolist()))}}
    for gm in (1.0, 2.0):
        p = native_oof(X, y, outer, k, gamma=gm)
        a, b, pv = mcnemar(y, p, base)
        R["focal"][f"gamma_{gm:g}"] = {
            "acc": float(accuracy_score(y, p)),
            "f1": float(f1_score(y, p, average="macro")),
            "recall": dict(zip(classes, recall_score(y, p, average=None).round(3).tolist())),
            "d_f1_ci_vs_softmax": singer_boot_diff(y, p, base, g),
            "mcnemar": [a, b, pv]}
        print("[focal]", gm, R["focal"][f"gamma_{gm:g}"]["f1"])
    print("[focal] softmax", R["focal"]["softmax_native"]["f1"])

    # ---------------- B. LOSO ----------------------------------------------
    logo = list(LeaveOneGroupOut().split(X, y, g))
    model = make_models(k)["XGBoost"]
    pred = np.zeros_like(y)
    for tr, te in logo:
        pred[te] = clone(model).fit(X[tr], y[tr]).predict(X[te])
    rows = []
    for s in np.unique(g):
        m = g == s
        rows.append({"singer": s, "gender": gen[m][0],
                     "median_f0": float(np.nanmedian(data.f0_median[m])),
                     "n": int(m.sum()),
                     "acc": float(accuracy_score(y[m], pred[m])),
                     "f1": float(f1_score(y[m], pred[m], average="macro",
                                          labels=np.unique(y[m])))})
    loso = pd.DataFrame(rows)
    # per-singer feature deviation: technique-wise median of the singer vs the
    # median and IQR of the other singers' technique medians, |robust z|
    med = data.groupby(["singer", "technique"])[FEATURE_NAMES].median()
    dev = []
    for s in np.unique(g):
        own = med.loc[s]
        others = med.drop(index=s, level=0)
        z = []
        for t in own.index:
            o = others.xs(t, level=1)
            iqr = (o.quantile(0.75) - o.quantile(0.25)).replace(0, np.nan)
            z.append(((own.loc[t] - o.median()) / iqr).abs())
        z = pd.DataFrame(z).median()
        dev.append({"singer": s, **{fam: float(z[fs].median())
                                    for fam, fs in FEATURE_GROUPS.items()}})
    loso = loso.merge(pd.DataFrame(dev), on="singer")
    loso.to_csv(os.path.join(REV, "loso_per_singer.csv"), index=False)
    fam = list(FEATURE_GROUPS)
    R["loso"] = {
        "acc": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, average="macro")),
        "singer_f1_mean": float(loso.f1.mean()), "singer_f1_sd": float(loso.f1.std()),
        "singer_f1_min": float(loso.f1.min()), "singer_f1_max": float(loso.f1.max()),
        "worst3": loso.nsmallest(3, "f1")[["singer", "f1", "median_f0"]].to_dict("records"),
        "by_gender": loso.groupby("gender")[["acc", "f1"] + fam].mean().round(3).to_dict("index"),
        "spearman_f1_vs_dev": {f: float(stats.spearmanr(loso.f1, loso[f])[0]) for f in fam},
        "spearman_f1_vs_medianf0_within_gender": {
            gg: float(stats.spearmanr(loso[loso.gender == gg].f1,
                                      loso[loso.gender == gg].median_f0)[0])
            for gg in ("male", "female")}}
    print("[loso]", R["loso"]["f1"], R["loso"]["by_gender"])

    # ---------------- D. per-fold table -----------------------------------
    S = json.load(open(os.path.join(REV, "revision_stats.json")))
    R["folds"] = [{"fold": i + 1, "test_singers": S["fold_test_singers"][i],
                   "xgb_fixed_f1": S["fixed"]["XGBoost"]["folds"][i]["f1"],
                   "xgb_nested_f1": S["nested"]["XGBoost"]["folds"][i]["f1"],
                   "xgb_chosen": S["nested"]["XGBoost"]["chosen"][i],
                   "rf_chosen": S["nested"]["Random Forest"]["chosen"][i],
                   "svm_chosen": S["nested"]["SVM (RBF)"]["chosen"][i],
                   "lr_chosen": S["nested"]["Logistic Regression"]["chosen"][i]}
                  for i in range(5)]

    # ---------------- C. pYIN vibrato --------------------------------------
    vib = data[data.technique == "vibrato"].reset_index(drop=True)
    res = Parallel(n_jobs=-1, verbose=2)(delayed(pyin_vibrato)(local_path(p))
                                         for p in vib.path)
    real = pd.read_csv(os.path.join(REV, "real_vibrato_estimators.csv"))
    real["rate_pyin_spec"] = [r[0] for r in res]
    real["extent_pyin_spec"] = [r[1] for r in res]
    real["rate_pyin_cycle"] = [r[2] for r in res]
    real["extent_pyin_cycle"] = [r[3] for r in res]
    real.to_csv(os.path.join(REV, "real_vibrato_estimators_pyin.csv"), index=False)
    R["pyin"] = {
        "tracker_only": {"rate": agreement(real.rate_spec, real.rate_pyin_spec),
                         "extent": agreement(real.extent_spec, real.extent_pyin_spec)},
        "fully_independent": {"rate": agreement(real.rate_spec, real.rate_pyin_cycle),
                              "extent": agreement(real.extent_spec, real.extent_pyin_cycle)},
        "by_context": {}}
    for ctx in real.context.unique():
        s = real[real.context == ctx]
        R["pyin"]["by_context"][ctx] = {
            "rate_tracker_only": agreement(s.rate_spec, s.rate_pyin_spec),
            "extent_tracker_only": agreement(s.extent_spec, s.extent_pyin_spec)}
    print("[pyin]", json.dumps(R["pyin"]["tracker_only"]))

    with open(os.path.join(REV, "revision_extra.json"), "w") as f:
        json.dump(R, f, indent=2, default=float)
    print("wrote revision_extra.json")


if __name__ == "__main__":
    main()
