"""
Synthetic-signal validation of the feature extractors.

We synthesise sustained vowels with KNOWN vibrato rate and extent and a known
f0, run the real extractors, and check that the measured values recover the
ground truth within tolerance. This proves the pipeline is correct before any
real-data experiment, independently of VocalSet.

Run:  python -m tests.test_synthetic
"""
from __future__ import annotations
import os
import sys
import numpy as np
import parselmouth

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from src.features.pitch import pitch_features
from src.features.vibrato import vibrato_features


def synth_vowel(f0=220.0, vib_rate=6.0, vib_extent_cents=80.0,
                dur=3.0, sr=C.SR, seed=0):
    """Sawtooth-ish vowel with sinusoidal vibrato; extent = peak-to-peak cents."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    # peak-to-peak extent -> +/- half, in cents -> frequency ratio
    half = (vib_extent_cents / 2.0)
    ratio = 2 ** ((half / 1200.0) * np.sin(2 * np.pi * vib_rate * t))
    inst_f0 = f0 * ratio
    phase = 2 * np.pi * np.cumsum(inst_f0) / sr
    # a few harmonics for a voice-like spectrum
    y = sum((1.0 / k) * np.sin(k * phase) for k in range(1, 8))
    y += 0.01 * rng.standard_normal(y.size)        # light noise
    y *= np.hanning(y.size) ** 0.1                  # gentle fade
    return (y / np.max(np.abs(y))).astype(np.float64)


def main():
    rates = [4.5, 5.5, 6.0, 7.0]
    extents = [40.0, 80.0, 120.0, 160.0]
    f0s = [165.0, 220.0, 330.0, 440.0]

    gt = {"rate": [], "extent": []}
    meas = {"rate": [], "extent": []}
    rows = []
    for fr, ex, f0 in zip(rates, extents, f0s):
        y = synth_vowel(f0=f0, vib_rate=fr, vib_extent_cents=ex)
        snd = parselmouth.Sound(values=y, sampling_frequency=C.SR)
        pf = pitch_features(snd)
        vf = vibrato_features(snd)
        gt["rate"].append(fr); gt["extent"].append(ex)
        meas["rate"].append(vf["vibrato_rate"])
        meas["extent"].append(vf["vibrato_extent"])
        rows.append((f0, pf["f0_median"], fr, vf["vibrato_rate"],
                     ex, vf["vibrato_extent"]))

    print(f"{'f0_gt':>7}{'f0_meas':>9}{'rate_gt':>9}{'rate_meas':>11}"
          f"{'ext_gt':>8}{'ext_meas':>10}")
    for r in rows:
        print(f"{r[0]:7.1f}{r[1]:9.1f}{r[2]:9.2f}{r[3]:11.2f}"
              f"{r[4]:8.1f}{r[5]:10.1f}")

    # assertions: f0 within 2%, vibrato rate within 0.6 Hz
    ok = True
    for (f0, f0m, fr, frm, ex, exm) in rows:
        if abs(f0m - f0) / f0 > 0.02:
            print(f"  ! f0 off: {f0}->{f0m}"); ok = False
        if abs(frm - fr) > 0.6:
            print(f"  ! rate off: {fr}->{frm}"); ok = False
    print("\nVALIDATION", "PASSED" if ok else "FAILED")

    # figure
    from src.figures import fig_synthetic_validation
    fig_synthetic_validation(gt, meas)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
