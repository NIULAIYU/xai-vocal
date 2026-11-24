"""
Pitch-stability features.

Fundamental frequency and its perturbation are extracted with Praat
(via parselmouth), the de-facto standard in voice science. Pitch stability
is expressed in cents so it is comparable across registers:

    c(t) = 1200 * log2( f0(t) / f0_median )

Jitter (cycle-to-cycle period perturbation) and HNR (harmonics-to-noise
ratio) quantify short-term pitch/periodicity control; a steadier, more
periodic phonation yields lower jitter and higher HNR.
"""
from __future__ import annotations
import numpy as np
import parselmouth
from parselmouth.praat import call

import config as C


def _safe(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def pitch_features(snd: parselmouth.Sound) -> dict:
    """Return pitch-stability features for a parselmouth Sound."""
    out = {}
    pitch = snd.to_pitch_ac(pitch_floor=C.F0_MIN, pitch_ceiling=C.F0_MAX,
                            time_step=C.HOP_SEC)
    f0 = pitch.selected_array["frequency"]      # 0 where unvoiced
    voiced = f0[f0 > 0]

    if voiced.size < 3:
        # Degenerate (essentially unvoiced) clip
        keys = ["f0_mean", "f0_median", "f0_std_hz", "f0_std_cents",
                "f0_range_cents", "voiced_fraction", "jitter_local",
                "jitter_rap", "hnr_mean"]
        return {k: np.nan for k in keys} | {"voiced_fraction": 0.0}

    f0_med = float(np.median(voiced))
    cents = 1200.0 * np.log2(voiced / f0_med)

    out["f0_mean"] = float(np.mean(voiced))
    out["f0_median"] = f0_med
    out["f0_std_hz"] = float(np.std(voiced))
    out["f0_std_cents"] = float(np.std(cents))                 # pitch stability
    out["f0_range_cents"] = float(np.percentile(cents, 95) -
                                  np.percentile(cents, 5))
    out["voiced_fraction"] = float(voiced.size / max(f0.size, 1))

    # Jitter / HNR via Praat point process (needs a periodic signal)
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", C.F0_MIN, C.F0_MAX)
        out["jitter_local"] = _safe(
            call(pp, "Get jitter (local)", 0, 0, 1e-4, 0.02, 1.3))
        out["jitter_rap"] = _safe(
            call(pp, "Get jitter (rap)", 0, 0, 1e-4, 0.02, 1.3))
    except Exception:
        out["jitter_local"] = np.nan
        out["jitter_rap"] = np.nan

    try:
        harm = call(snd, "To Harmonicity (cc)", C.HOP_SEC, C.F0_MIN, 0.1, 1.0)
        out["hnr_mean"] = _safe(call(harm, "Get mean", 0, 0))
    except Exception:
        out["hnr_mean"] = np.nan

    return out
