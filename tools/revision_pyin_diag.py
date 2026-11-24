"""
Diagnostic for the Praat-vs-pYIN vibrato comparison: separates the effect
of the pitch TRACKER from the effect of SEGMENT selection.

For every VocalSet vibrato clip, the spectral estimator is applied to
  (a) the Praat contour on Praat's longest voiced segment (as in the paper),
  (b) the pYIN contour on pYIN's own longest voiced segment,
  (c) the pYIN contour restricted to the SAME time span as (a).
(a) vs (c) isolates the tracker; (b) vs (c) isolates segmentation.

Run from the package root:  python -m tools.revision_pyin_diag
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import librosa
import parselmouth

import config as C
from src.features.vibrato import FS
from tools.revision_audio import agreement, local_path
from tools.revision_extra import spectral_from_cents

REV = os.path.join(C.OUT_DIR, "revision")


def _longest_span(f0):
    voiced = f0 > 0
    best = (0, 0); s = None
    for i, v in enumerate(np.append(voiced, False)):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if i - s > best[1] - best[0]:
                best = (s, i)
            s = None
    return best


def one(path):
    y, sr = librosa.load(path, sr=C.SR, mono=True)
    y = librosa.util.normalize(y)
    snd = parselmouth.Sound(values=y.astype(np.float64), sampling_frequency=sr)
    pp = snd.to_pitch_ac(pitch_floor=C.F0_MIN, pitch_ceiling=C.F0_MAX,
                         time_step=C.HOP_SEC)
    f_pr = pp.selected_array["frequency"]
    t_pr = pp.xs()
    hop = int(round(C.HOP_SEC * sr))
    f_py, vf, _ = librosa.pyin(y, fmin=C.F0_MIN, fmax=C.F0_MAX, sr=sr,
                               frame_length=2048, hop_length=hop)
    f_py = np.where(vf & np.isfinite(f_py), f_py, 0.0)
    t_py = librosa.times_like(f_py, sr=sr, hop_length=hop)
    out = {"seg_praat": np.nan, "seg_pyin": np.nan, "overlap": np.nan,
           "rate_a": np.nan, "ext_a": np.nan, "rate_b": np.nan, "ext_b": np.nan,
           "rate_c": np.nan, "ext_c": np.nan}
    a0, a1 = _longest_span(f_pr)
    b0, b1 = _longest_span(f_py)
    out["seg_praat"] = (a1 - a0) / FS
    out["seg_pyin"] = (b1 - b0) / FS
    if a1 - a0 >= int(0.4 * FS):
        seg = f_pr[a0:a1]
        out["rate_a"], out["ext_a"] = spectral_from_cents(1200 * np.log2(seg / np.median(seg)))
        # same time span on the pYIN contour, interpolating unvoiced gaps
        m = (t_py >= t_pr[a0]) & (t_py <= t_pr[a1 - 1])
        segc = f_py[m]
        if segc.size >= int(0.4 * FS) and np.mean(segc > 0) > 0.9:
            idx = np.arange(segc.size)
            ok = segc > 0
            segc = np.interp(idx, idx[ok], segc[ok])
            out["rate_c"], out["ext_c"] = spectral_from_cents(1200 * np.log2(segc / np.median(segc)))
    if b1 - b0 >= int(0.4 * FS):
        seg = f_py[b0:b1]
        out["rate_b"], out["ext_b"] = spectral_from_cents(1200 * np.log2(seg / np.median(seg)))
    # overlap fraction of the two longest segments (in time)
    lo = max(t_pr[a0] if a1 > a0 else 0, t_py[b0] if b1 > b0 else 0)
    hi = min(t_pr[a1 - 1] if a1 > a0 else 0, t_py[b1 - 1] if b1 > b0 else 0)
    out["overlap"] = max(0.0, hi - lo) / max(out["seg_praat"], out["seg_pyin"], 1e-9)
    return out


def main():
    data = pd.read_csv(C.FEATURE_CSV)
    vib = data[data.technique == "vibrato"].reset_index(drop=True)
    res = pd.DataFrame(Parallel(n_jobs=-1)(delayed(one)(local_path(p)) for p in vib.path))
    res["clip"] = vib.path.map(os.path.basename)
    res.to_csv(os.path.join(REV, "pyin_diagnostic.csv"), index=False)
    R = {"tracker_same_span": {"rate": agreement(res.rate_a, res.rate_c),
                               "extent": agreement(res.ext_a, res.ext_c)},
         "own_segments": {"rate": agreement(res.rate_a, res.rate_b),
                          "extent": agreement(res.ext_a, res.ext_b)},
         "segmentation_only_pyin": {"rate": agreement(res.rate_b, res.rate_c),
                                    "extent": agreement(res.ext_b, res.ext_c)},
         "seg_praat_median": float(res.seg_praat.median()),
         "seg_pyin_median": float(res.seg_pyin.median()),
         "overlap_median": float(res.overlap.median()),
         "n_same_span": int(res.ext_c.notna().sum())}
    json.dump(R, open(os.path.join(REV, "pyin_diagnostic.json"), "w"), indent=2, default=float)
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
