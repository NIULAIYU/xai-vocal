"""
Per-file feature extraction: glue the four interpretable feature families
(pitch, vibrato, resonance, breath) into a single flat record.
"""
from __future__ import annotations
import numpy as np
import librosa
import parselmouth

import config as C
from src.features.pitch import pitch_features
from src.features.vibrato import vibrato_features
from src.features.resonance import resonance_features
from src.features.breath import breath_features

# Ordered feature list (keeps column order stable across runs)
FEATURE_NAMES = [
    # pitch stability
    "f0_mean", "f0_median", "f0_std_hz", "f0_std_cents", "f0_range_cents",
    "voiced_fraction", "jitter_local", "jitter_rap", "hnr_mean",
    # vibrato
    "vibrato_rate", "vibrato_extent", "vibrato_strength",
    "vibrato_regularity", "vibrato_jitter",
    # resonance
    "F1_mean", "F2_mean", "F3_mean", "F4_mean", "singers_formant_ratio",
    "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
    "spectral_flatness", "spectral_tilt",
    # breath / dynamics
    "rms_mean", "rms_std", "dynamic_range_db", "amplitude_stability",
    "sustain_max_sec", "shimmer_local", "spectral_flux",
]

# Grouping used by the manuscript / figures
FEATURE_GROUPS = {
    "Pitch stability": FEATURE_NAMES[0:9],
    "Vibrato": FEATURE_NAMES[9:14],
    "Resonance": FEATURE_NAMES[14:24],
    "Breath control": FEATURE_NAMES[24:31],
}


def extract_file(path: str) -> dict | None:
    """Extract all features from one audio file. Returns None if too short."""
    y, sr = librosa.load(path, sr=C.SR, mono=True)
    if y.size < int(C.MIN_DURATION * sr):
        return None
    y = librosa.util.normalize(y)
    snd = parselmouth.Sound(values=y.astype(np.float64), sampling_frequency=sr)

    feats = {}
    feats.update(pitch_features(snd))
    feats.update(vibrato_features(snd))
    feats.update(resonance_features(snd, y, sr))
    feats.update(breath_features(snd, y, sr))
    # Guarantee every column exists
    return {k: feats.get(k, np.nan) for k in FEATURE_NAMES}
