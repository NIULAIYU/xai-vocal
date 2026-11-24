"""
Revision analyses that need the audio.

  A. Recording-condition robustness: re-extract the 31 features from a
     stratified subset after controlled perturbations (gain change, additive
     noise at 30/20 dB SNR, synthetic room reverberation) and measure
     (i) per-feature rank agreement with the clean features and
     (ii) the accuracy of the speaker-independent models on perturbed clips.
  B. Vibrato estimator on REAL vibrato: compare the spectral estimator with an
     independent cycle-based (peak-picking) estimator on every VocalSet
     vibrato clip.
  C. Vibrato estimator on NON-SINUSOIDAL synthetic vibrato: rate jitter,
     extent drift, coupled amplitude modulation, waveform asymmetry, drift.

Run from the package root:  python -m tools.revision_audio
Audio root defaults to ../vocalset_subset (override with VOCALSET_DIR).
"""
from __future__ import annotations
import os
import sys
import json
import re
import numpy as np
import pandas as pd
from scipy import signal, stats
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import librosa
import parselmouth
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score

import config as C
from src.model import make_models
from src.features.extract import FEATURE_NAMES, FEATURE_GROUPS
from src.features.pitch import pitch_features
from src.features.vibrato import vibrato_features, _longest_voiced, FS
from src.features.resonance import resonance_features
from src.features.breath import breath_features

REV_DIR = os.path.join(C.OUT_DIR, "revision")
os.makedirs(REV_DIR, exist_ok=True)
AUDIO_ROOT = os.environ.get(
    "VOCALSET_DIR",
    os.path.join(os.path.dirname(C.ROOT), "vocalset_subset"))


def local_path(p: str) -> str:
    """Resolve a clip path from features.csv under AUDIO_ROOT.

    Paths are stored relative to the VocalSet root ("female1/belt/x.flac");
    absolute paths from another machine are also accepted and re-rooted at
    the singer folder.
    """
    parts = p.replace("\\", "/").split("/")
    i = next(k for k, s in enumerate(parts) if re.fullmatch(r"(fe)?male\d+", s))
    return os.path.join(AUDIO_ROOT, *parts[i:])


# --------------------------------------------------------------------------- #
# Perturbations
# --------------------------------------------------------------------------- #
def _reverb_ir(sr, rt60=0.5, seed=0):
    rng = np.random.default_rng(seed)
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-6.9078 * t / rt60)   # -60 dB at rt60
    ir[0] = 0.0
    ir /= np.sqrt(np.sum(ir ** 2))
    return ir


def perturb(y, sr, cond, seed=0):
    rng = np.random.default_rng(seed)
    if cond == "clean":
        return y
    if cond == "gain_-12dB":
        return y * 10 ** (-12 / 20)
    if cond.startswith("noise_"):
        snr = float(cond.split("_")[1].replace("dB", ""))
        p = np.mean(y ** 2)
        return y + rng.standard_normal(y.size) * np.sqrt(p / 10 ** (snr / 10))
    if cond == "reverb_0.5s":
        # direct-to-reverberant ratio 0 dB: a singer a few metres from the mic
        wet = signal.fftconvolve(y, _reverb_ir(sr, 0.5, seed))[: y.size]
        wet *= np.sqrt(np.mean(y ** 2) / (np.mean(wet ** 2) + 1e-12))
        return y + wet
    raise ValueError(cond)


def features_from_signal(y, sr):
    y = librosa.util.normalize(y)
    snd = parselmouth.Sound(values=y.astype(np.float64), sampling_frequency=sr)
    f = {}
    f.update(pitch_features(snd))
    f.update(vibrato_features(snd))
    f.update(resonance_features(snd, y, sr))
    f.update(breath_features(snd, y, sr))
    return [f.get(k, np.nan) for k in FEATURE_NAMES]


def extract_conditions(path, conds):
    y, sr = librosa.load(path, sr=C.SR, mono=True)
    return {c: features_from_signal(perturb(y, sr, c), sr) for c in conds}


# --------------------------------------------------------------------------- #
# Cycle-based (peak-picking) vibrato estimator, independent of the spectral one
# --------------------------------------------------------------------------- #
def cycle_vibrato(cents: np.ndarray, fs: float = FS):
    """Rate/extent from alternating local extrema of the cents contour."""
    c = signal.savgol_filter(cents, 5, 2) if cents.size > 7 else cents
    # remove slow drift with a centred moving average of ~0.5 s
    w = int(0.5 * fs) | 1
    if c.size > w:
        trend = np.convolve(np.pad(c, w // 2, mode="edge"),
                            np.ones(w) / w, mode="valid")
        c = c - trend
    dmin = int(fs / (2 * 9.0))                    # half period at 9 Hz
    pk, _ = signal.find_peaks(c, distance=max(dmin, 1), prominence=10)
    tr, _ = signal.find_peaks(-c, distance=max(dmin, 1), prominence=10)
    ext = sorted([(i, 1) for i in pk] + [(i, -1) for i in tr])
    alt = []
    for e in ext:                                 # enforce alternation
        if alt and alt[-1][1] == e[1]:
            if (e[1] == 1 and c[e[0]] > c[alt[-1][0]]) or \
               (e[1] == -1 and c[e[0]] < c[alt[-1][0]]):
                alt[-1] = e
        else:
            alt.append(e)
    if len(alt) < 5:
        return np.nan, np.nan
    idx = np.array([a[0] for a in alt])
    half = np.diff(idx) / fs
    ok = (half >= 1 / (2 * 9.0)) & (half <= 1 / (2 * 3.0))
    if ok.sum() < 3:
        return np.nan, np.nan
    rate = 1.0 / (2.0 * np.median(half[ok]))
    depth = np.abs(np.diff(c[idx]))[ok]
    return float(rate), float(np.median(depth))


def contour_of(path):
    y, sr = librosa.load(path, sr=C.SR, mono=True)
    y = librosa.util.normalize(y)
    snd = parselmouth.Sound(values=y.astype(np.float64), sampling_frequency=sr)
    vf = vibrato_features(snd)
    pitch = snd.to_pitch_ac(pitch_floor=C.F0_MIN, pitch_ceiling=C.F0_MAX,
                            time_step=C.HOP_SEC)
    seg = _longest_voiced(pitch.selected_array["frequency"])
    if seg.size < int(0.4 * FS):
        return vf, (np.nan, np.nan), 0.0
    cents = 1200 * np.log2(seg / np.median(seg))
    return vf, cycle_vibrato(cents), seg.size / FS


def agreement(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    d = a - b
    return {"n": int(m.sum()), "pearson_r": float(stats.pearsonr(a, b)[0]),
            "spearman_rho": float(stats.spearmanr(a, b)[0]),
            "bias": float(np.mean(d)), "mae": float(np.mean(np.abs(d))),
            "loa": [float(np.mean(d) - 1.96 * np.std(d, ddof=1)),
                    float(np.mean(d) + 1.96 * np.std(d, ddof=1))]}


# --------------------------------------------------------------------------- #
# Non-sinusoidal synthetic vibrato
# --------------------------------------------------------------------------- #
def synth_realistic(rng, sr=C.SR, dur=3.0):
    f0 = rng.uniform(150, 500)
    rate0 = rng.uniform(4.5, 7.0)
    ext0 = rng.uniform(40, 160)                    # mean peak-to-peak (cents)
    n = int(dur * sr)
    t = np.arange(n) / sr
    # rate jitter: smooth random walk, ~ +/-8 % around rate0
    walk = np.cumsum(rng.standard_normal(n)) / np.sqrt(sr)
    walk = signal.sosfiltfilt(signal.butter(2, 1.0, fs=sr, output="sos"), walk)
    walk = walk / (np.std(walk) + 1e-12) * 0.08
    inst_rate = rate0 * (1 + walk)
    ph = 2 * np.pi * np.cumsum(inst_rate) / sr
    # extent drift: linear ramp +/-30 % across the note
    ramp = 1 + rng.uniform(-0.3, 0.3) * (t / dur - 0.5) * 2
    asym = rng.uniform(0, 0.25)                    # waveform asymmetry (2nd harm.)
    shape = np.sin(ph) + asym * np.sin(2 * ph)
    shape /= (np.max(shape) - np.min(shape)) / 2
    mod_cents = (ext0 / 2) * ramp * shape
    drift = rng.uniform(-20, 20) * t               # cents/s pitch drift
    ratio = 2 ** ((mod_cents + drift) / 1200)
    phase = 2 * np.pi * np.cumsum(f0 * ratio) / sr
    am = 1 + 0.15 * np.sin(ph + rng.uniform(0, np.pi))   # coupled AM (~1.3 dB)
    y = am * sum((1.0 / k) * np.sin(k * phase) for k in range(1, 8))
    y += 0.01 * rng.standard_normal(n)
    # ground truth on the true modulation contour
    pk, _ = signal.find_peaks(mod_cents)
    tr, _ = signal.find_peaks(-mod_cents)
    ext_true = float(np.mean(mod_cents[pk]) - np.mean(mod_cents[tr]))
    return (y / np.max(np.abs(y))).astype(np.float64), \
        float(np.mean(inst_rate)), ext_true, {"f0": f0, "asym": asym}


def synth_case(seed):
    rng = np.random.default_rng(1000 + seed)
    y, r_true, e_true, meta = synth_realistic(rng)
    snd = parselmouth.Sound(values=y, sampling_frequency=C.SR)
    vf = vibrato_features(snd)
    pitch = snd.to_pitch_ac(pitch_floor=C.F0_MIN, pitch_ceiling=C.F0_MAX,
                            time_step=C.HOP_SEC)
    seg = _longest_voiced(pitch.selected_array["frequency"])
    cyc = cycle_vibrato(1200 * np.log2(seg / np.median(seg)))
    return {"rate_true": r_true, "extent_true": e_true,
            "rate_spec": vf["vibrato_rate"], "extent_spec": vf["vibrato_extent"],
            "rate_cycle": cyc[0], "extent_cycle": cyc[1], **meta}


# --------------------------------------------------------------------------- #
def main():
    data = pd.read_csv(C.FEATURE_CSV)
    data["local"] = data["path"].map(local_path)
    classes = sorted(data.technique.unique().tolist())
    y = np.array([classes.index(t) for t in data.technique])
    X = data[FEATURE_NAMES].values
    g = data.singer.values
    R = {}

    # ---------------- C. synthetic non-sinusoidal -------------------------
    syn = Parallel(n_jobs=-1)(delayed(synth_case)(s) for s in range(60))
    syn = pd.DataFrame(syn)
    syn.to_csv(os.path.join(REV_DIR, "synthetic_nonsinusoidal.csv"), index=False)
    R["synthetic_nonsinusoidal"] = {
        "rate_spec": agreement(syn.rate_spec, syn.rate_true),
        "extent_spec": agreement(syn.extent_spec, syn.extent_true),
        "extent_spec_rel_err_median": float(np.nanmedian(
            np.abs(syn.extent_spec - syn.extent_true) / syn.extent_true)),
        "rate_cycle": agreement(syn.rate_cycle, syn.rate_true),
        "extent_cycle": agreement(syn.extent_cycle, syn.extent_true),
        "extent_cycle_rel_err_median": float(np.nanmedian(
            np.abs(syn.extent_cycle - syn.extent_true) / syn.extent_true))}
    print("[C]", json.dumps(R["synthetic_nonsinusoidal"], indent=1)[:600])

    # ---------------- B. real vibrato: spectral vs cycle estimator --------
    vib = data[data.technique == "vibrato"].reset_index(drop=True)
    res = Parallel(n_jobs=-1)(delayed(contour_of)(p) for p in vib.local)
    real = pd.DataFrame({
        "clip": vib.path.map(os.path.basename),
        "rate_spec": [r[0]["vibrato_rate"] for r in res],
        "extent_spec": [r[0]["vibrato_extent"] for r in res],
        "rate_cycle": [r[1][0] for r in res],
        "extent_cycle": [r[1][1] for r in res],
        "seg_sec": [r[2] for r in res]})
    real["context"] = real["clip"].str.lower().map(
        lambda b: "long_tones" if "long" in b else
        ("arpeggios" if "arpeggio" in b else ("scales" if "scale" in b else "excerpts")))
    real.to_csv(os.path.join(REV_DIR, "real_vibrato_estimators.csv"), index=False)
    R["real_vibrato"] = {
        "all": {"rate": agreement(real.rate_spec, real.rate_cycle),
                "extent": agreement(real.extent_spec, real.extent_cycle)},
        "by_context_n": real.context.value_counts().to_dict()}
    for ctx in real.context.unique():
        s = real[real.context == ctx]
        if s.rate_cycle.notna().sum() >= 8:
            R["real_vibrato"][ctx] = {"rate": agreement(s.rate_spec, s.rate_cycle),
                                      "extent": agreement(s.extent_spec, s.extent_cycle)}
    print("[B]", json.dumps(R["real_vibrato"]["all"], indent=1))

    # ---------------- A. robustness to recording conditions ---------------
    conds = ["clean", "gain_-12dB", "noise_30dB", "noise_20dB", "reverb_0.5s"]
    rng = np.random.default_rng(0)
    sub = np.concatenate([rng.choice(np.where(y == c)[0],
                                     min(25, np.sum(y == c)), replace=False)
                          for c in range(len(classes))])
    feats = Parallel(n_jobs=-1, verbose=2)(
        delayed(extract_conditions)(data.local[i], conds) for i in sub)
    Fc = {c: np.array([f[c] for f in feats], float) for c in conds}
    np.savez_compressed(os.path.join(REV_DIR, "robustness_features.npz"),
                        sub=sub, **{c: Fc[c] for c in conds})

    # rank agreement per feature with the clean re-extraction
    rob = {}
    for c in conds[1:]:
        rho = {}
        for j, f in enumerate(FEATURE_NAMES):
            a, b = Fc["clean"][:, j], Fc[c][:, j]
            m = np.isfinite(a) & np.isfinite(b)
            rho[f] = float(stats.spearmanr(a[m], b[m])[0]) if m.sum() > 10 else np.nan
        rob[c] = {"rho": rho,
                  "family_mean_rho": {fam: float(np.nanmean([rho[f] for f in fs]))
                                      for fam, fs in FEATURE_GROUPS.items()}}
    # sanity: re-extraction of clean equals cached features
    cached = X[sub]
    R["reextraction_max_abs_diff"] = float(np.nanmax(np.abs(cached - Fc["clean"])))

    # classification on perturbed clips with models trained on clean data
    outer = list(GroupKFold(n_splits=5).split(X, y, g))
    pos = {k: i for i, k in enumerate(sub)}
    models = make_models(len(classes))
    acc = {c: {} for c in conds}
    for name in ("XGBoost", "Random Forest"):
        pred = {c: np.zeros(len(sub), int) for c in conds}
        for tr, te in outer:
            m = clone(models[name]).fit(X[tr], y[tr])
            te_sub = [pos[k] for k in te if k in pos]
            for c in conds:
                pred[c][te_sub] = m.predict(Fc[c][te_sub])
        for c in conds:
            acc[c][name] = {"acc": float(accuracy_score(y[sub], pred[c])),
                            "f1": float(f1_score(y[sub], pred[c], average="macro"))}
    for c in conds[1:]:
        rob[c]["classification"] = acc[c]
    rob["clean_classification"] = acc["clean"]
    R["robustness"] = rob
    R["robustness_n_clips"] = int(len(sub))

    with open(os.path.join(REV_DIR, "revision_audio.json"), "w") as f:
        json.dump(R, f, indent=2, default=float)
    print("wrote", os.path.join(REV_DIR, "revision_audio.json"))


if __name__ == "__main__":
    main()
