"""
Breath-control / dynamic-stability features.

Breath support shows up in the amplitude envelope: a well-supported tone has
a steady loudness, a long sustainable phrase, and controlled dynamics.

    * rms_mean / rms_std        loudness level and its steadiness
    * dynamic_range_db          max-min loudness within the clip (dB)
    * amplitude_stability       1 - CV(rms): higher = steadier support
    * sustain_max_sec           longest continuously-voiced span (phrase proxy)
    * shimmer_local             cycle-to-cycle amplitude perturbation (Praat)
    * spectral_flux             rate of spectral change (steadiness of timbre)
"""
from __future__ import annotations
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call

import config as C


def _safe(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def breath_features(snd: parselmouth.Sound, y: np.ndarray, sr: int) -> dict:
    out = {}
    hop = int(C.HOP_SEC * sr)
    frame = int(C.FRAME_SEC * sr)

    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    rms = rms[rms > 0]
    if rms.size == 0:
        rms = np.array([1e-6])

    out["rms_mean"] = float(np.mean(rms))
    out["rms_std"] = float(np.std(rms))
    db = 20.0 * np.log10(rms / (rms.max() + 1e-12) + 1e-12)
    out["dynamic_range_db"] = float(np.percentile(db, 95) - np.percentile(db, 5))
    cv = np.std(rms) / (np.mean(rms) + 1e-12)
    out["amplitude_stability"] = float(1.0 / (1.0 + cv))      # 0..1, higher steadier

    # Sustain: longest continuously voiced span from the pitch track
    pitch = snd.to_pitch_ac(pitch_floor=C.F0_MIN, pitch_ceiling=C.F0_MAX,
                            time_step=C.HOP_SEC)
    voiced = pitch.selected_array["frequency"] > 0
    best = cur = 0
    for v in voiced:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    out["sustain_max_sec"] = float(best * C.HOP_SEC)

    # Shimmer (Praat)
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", C.F0_MIN, C.F0_MAX)
        out["shimmer_local"] = _safe(
            call([snd, pp], "Get shimmer (local)", 0, 0, 1e-4, 0.02, 1.3, 1.6))
    except Exception:
        out["shimmer_local"] = np.nan

    # Spectral flux (mean L2 change of successive magnitude spectra)
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=hop))
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    out["spectral_flux"] = float(np.mean(flux)) if flux.size else np.nan

    return out
