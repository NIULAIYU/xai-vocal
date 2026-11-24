"""
Resonance / timbre features.

Formants (vocal-tract resonances) are estimated with Praat's Burg method.
The singer's-formant region (~2.3-3.5 kHz) is quantified from the long-term
average spectrum (LTAS) as a band-energy ratio, following the "quality
ratio" idea (energy in the high resonance cluster vs the low formants):

    QR = E(2.3-3.5 kHz) / E(0-2 kHz)

Spectral shape descriptors (centroid, bandwidth, roll-off, flatness, tilt)
summarise brightness and the balance of harmonic energy.
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


def resonance_features(snd: parselmouth.Sound, y: np.ndarray, sr: int) -> dict:
    out = {}

    # ---- Formants (Praat Burg) ----
    try:
        formant = call(snd, "To Formant (burg)", C.HOP_SEC, 5, 5500, 0.025, 50)
        for i in (1, 2, 3, 4):
            out[f"F{i}_mean"] = _safe(
                call(formant, "Get mean", i, 0, 0, "hertz"))
    except Exception:
        for i in (1, 2, 3, 4):
            out[f"F{i}_mean"] = np.nan

    # ---- Singer's-formant / quality ratio from LTAS ----
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=int(C.HOP_SEC * sr))) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    ltas = S.mean(axis=1)                         # long-term average spectrum
    e_low = ltas[(freqs >= 0) & (freqs < 2000)].sum()
    e_sf = ltas[(freqs >= 2300) & (freqs <= 3500)].sum()
    out["singers_formant_ratio"] = float(e_sf / (e_low + 1e-12))

    # ---- Spectral shape (librosa) ----
    hop = int(C.HOP_SEC * sr)
    out["spectral_centroid"] = float(
        np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)))
    out["spectral_bandwidth"] = float(
        np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)))
    out["spectral_rolloff"] = float(
        np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop,
                                                 roll_percent=0.85)))
    out["spectral_flatness"] = float(
        np.mean(librosa.feature.spectral_flatness(y=y, hop_length=hop)))

    # Spectral tilt: slope of log-power vs log-frequency (dB/decade)
    mask = (freqs > 50) & (ltas > 0)
    if mask.sum() > 5:
        lf = np.log10(freqs[mask])
        lp = 10.0 * np.log10(ltas[mask] + 1e-12)
        slope = np.polyfit(lf, lp, 1)[0]
        out["spectral_tilt"] = float(slope)
    else:
        out["spectral_tilt"] = np.nan

    return out
