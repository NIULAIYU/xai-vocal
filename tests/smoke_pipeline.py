"""
Integration smoke test: build a tiny VocalSet-shaped tree of synthetic clips,
then run the ENTIRE pipeline (scan -> features -> speaker-independent CV ->
final model -> SHAP -> figures). Proves the wiring end-to-end without the real
dataset. Not part of the paper; data is synthetic and thrown away.
"""
from __future__ import annotations
import os, sys, shutil, tempfile
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

SR = C.SR
TECHS = ["straight", "vibrato", "breathy", "belt", "trill", "vocal_fry"]


def _clip(tech, f0, rng, dur=1.6):
    t = np.arange(int(dur * SR)) / SR
    if tech == "vibrato":
        ratio = 2 ** ((40 / 1200) * np.sin(2 * np.pi * 6.0 * t))
    elif tech == "trill":
        ratio = 2 ** ((120 / 1200) * np.sign(np.sin(2 * np.pi * 8.0 * t)))
    elif tech == "vocal_fry":
        f0 = 70.0; ratio = 1 + 0.05 * rng.standard_normal(t.size)
    else:
        ratio = np.ones_like(t)
    inst = f0 * ratio
    phase = 2 * np.pi * np.cumsum(inst) / SR
    nharm = {"belt": 12, "breathy": 3}.get(tech, 7)
    y = sum((1.0 / k) * np.sin(k * phase) for k in range(1, nharm + 1))
    if tech == "breathy":
        y += 0.4 * rng.standard_normal(y.size)
    if tech == "belt":
        y *= 1.3
    y += 0.01 * rng.standard_normal(y.size)
    return (y / np.max(np.abs(y)) * 0.9).astype(np.float32)


def main():
    tmp = tempfile.mkdtemp(prefix="vs_smoke_")
    rng = np.random.default_rng(0)
    singers = ["female1", "female2", "female3", "male1", "male2"]
    for s in singers:
        base_f0 = 220 if s.startswith("female") else 130
        for tech in TECHS:
            for i in range(3):
                d = os.path.join(tmp, s, tech)
                os.makedirs(d, exist_ok=True)
                y = _clip(tech, base_f0 * (1 + 0.03 * i), rng)
                sf.write(os.path.join(d, f"{s}_{tech}_{i}.wav"), y, SR)

    # Point the pipeline at the temp tree and a temp feature cache
    C.DATA_DIR = tmp
    cache = os.path.join(tmp, "features.csv")

    from src.data_loader import scan
    from src.model import build_feature_matrix, evaluate, fit_final
    from src.explain import compute_shap
    from src import figures as F

    # restrict classes to the simulated ones for this smoke test
    C.TECHNIQUES = TECHS

    df = scan(tmp)
    print("scanned:", len(df), "files,", df.singer.nunique(), "singers")
    data = build_feature_matrix(df, n_jobs=2, cache=cache)
    print("feature matrix:", data.shape)

    results = evaluate(data, n_splits=5)
    model, classes = fit_final(data, "XGBoost")
    sh = compute_shap(model, data)

    F.fig_feature_grid(data, name="smoke_feature_grid")
    F.fig_model_compare(results, name="smoke_model_compare")
    F.fig_confusion(results, "XGBoost", name="smoke_confusion")
    F.fig_shap_bar(sh["importance"], name="smoke_shap_bar")
    F.fig_shap_heatmap(sh["importance"], sh["classes"], name="smoke_shap_heatmap")
    F.fig_vibrato_scatter(data, name="smoke_vibrato_scatter")
    F.fig_corr(data, name="smoke_corr")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\nSMOKE PASSED — full pipeline ran end-to-end.")


if __name__ == "__main__":
    main()
