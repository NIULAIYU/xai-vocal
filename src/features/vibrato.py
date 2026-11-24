"""
Vibrato features.

Vibrato is a quasi-periodic modulation of f0. Working on the pitch contour
in cents, sampled uniformly at fs = 1/HOP_SEC, we detrend the slow pitch
drift and analyse the residual on the longest sustained (voiced) segment:

    * vibrato_rate      dominant modulation frequency in the 3-9 Hz band (Hz)
    * vibrato_extent    peak-to-peak modulation depth (cents), from the
                        analytic-signal envelope
    * vibrato_strength  fraction of contour energy inside the vibrato band
                        (0-1); a presence/intensity proxy
    * vibrato_regularity  spectral concentration peak_power / band_power
    * vibrato_jitter    cycle-to-cycle variation of the instantaneous rate
"""
from __future__ import annotations
import numpy as np
from scipy import signal
import parselmouth

import config as C

FS = 1.0 / C.HOP_SEC          # pitch-contour sample rate (Hz)
VIB_LO, VIB_HI = 3.0, 9.0     # plausible vibrato-rate band (Hz)


def _longest_voiced(f0: np.ndarray) -> np.ndarray:
    """Return the longest run of consecutive voiced (f0>0) frames."""
    voiced = f0 > 0
    best_s = best_len = cur_s = 0
    cur_len = 0
    for i, v in enumerate(voiced):
        if v:
            if cur_len == 0:
                cur_s = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_s = cur_len, cur_s
        else:
            cur_len = 0
    return f0[best_s:best_s + best_len]


def vibrato_features(snd: parselmouth.Sound) -> dict:
    keys = ["vibrato_rate", "vibrato_extent", "vibrato_strength",
            "vibrato_regularity", "vibrato_jitter"]
    out = {k: np.nan for k in keys}

    pitch = snd.to_pitch_ac(pitch_floor=C.F0_MIN, pitch_ceiling=C.F0_MAX,
                            time_step=C.HOP_SEC)
    seg = _longest_voiced(pitch.selected_array["frequency"])
    # Need at least ~0.4 s to see a couple of vibrato cycles
    if seg.size < int(0.4 * FS):
        return out

    cents = 1200.0 * np.log2(seg / np.median(seg))

    # Detrend slow pitch drift with a zero-phase high-pass that removes
    # everything below 2.5 Hz while leaving the 3-9 Hz vibrato band intact.
    # (A short moving-average detrend would overlap the vibrato band and
    # attenuate the very modulation we are trying to measure.)
    try:
        sos_hp = signal.butter(2, 2.5, btype="high", fs=FS, output="sos")
        resid = signal.sosfiltfilt(sos_hp, cents - float(np.mean(cents)))
    except Exception:
        resid = cents - float(np.mean(cents))

    # Power spectrum of the residual
    n = resid.size
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    psd = np.abs(np.fft.rfft(resid * np.hanning(n))) ** 2

    band = (freqs >= VIB_LO) & (freqs <= VIB_HI)
    if not band.any() or psd[band].sum() <= 0:
        return out

    peak_idx = np.argmax(np.where(band, psd, 0.0))
    out["vibrato_rate"] = float(freqs[peak_idx])
    out["vibrato_strength"] = float(psd[band].sum() / (psd.sum() + 1e-12))
    out["vibrato_regularity"] = float(psd[peak_idx] / (psd[band].sum() + 1e-12))

    # Band-pass the residual around the detected rate, then measure depth
    lo = max(VIB_LO, out["vibrato_rate"] - 2.0)
    hi = min(VIB_HI, out["vibrato_rate"] + 2.0)
    try:
        sos = signal.butter(4, [lo, hi], btype="band", fs=FS, output="sos")
        bp = signal.sosfiltfilt(sos, resid)
        analytic = signal.hilbert(bp)
        # Peak-to-peak depth from the RMS of the band-limited contour:
        # for a sinusoid of amplitude A, std = A/sqrt(2) -> p2p = 2*sqrt(2)*std.
        out["vibrato_extent"] = float(2.0 * np.sqrt(2.0) * np.std(bp))  # cents p2p

        # Instantaneous rate from the unwrapped phase -> cycle-to-cycle jitter
        inst_freq = np.diff(np.unwrap(np.angle(analytic))) / (2 * np.pi) * FS
        inst_freq = inst_freq[(inst_freq > VIB_LO) & (inst_freq < VIB_HI)]
        if inst_freq.size > 2:
            out["vibrato_jitter"] = float(np.std(inst_freq) /
                                          (np.mean(inst_freq) + 1e-12))
    except Exception:
        pass

    return out
