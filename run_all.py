"""
End-to-end driver: dataset -> features -> speaker-independent evaluation ->
final model -> SHAP -> all figures and tables.

Usage:
    export VOCALSET_DIR=/path/to/extracted/VocalSet
    python run_all.py
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd

import config as C
from src.data_loader import scan
from src.model import build_feature_matrix, evaluate, fit_final
from src.explain import compute_shap
from src import figures as F


def main():
    # 1. discover + extract
    df = scan()
    print(df.groupby(["technique"]).size())
    data = build_feature_matrix(df)

    # 2. speaker-independent evaluation
    results = evaluate(data)
    with open(os.path.join(C.OUT_DIR, "cv_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # 3. final model + SHAP
    model, classes = fit_final(data, "XGBoost")
    sh = compute_shap(model, data)

    # 4. tables
    _write_tables(data, results)

    # 5. figures
    F.fig_pipeline()
    F.fig_feature_grid(data)
    F.fig_model_compare(results)
    F.fig_confusion(results, "XGBoost")
    F.fig_shap_bar(sh["importance"])
    F.fig_shap_heatmap(sh["importance"], sh["classes"])
    F.fig_vibrato_scatter(data)
    F.fig_corr(data)
    print("\nDone. Outputs in", C.OUT_DIR)


def _write_tables(data, results):
    # dataset summary
    data.groupby("technique").size().rename("n_clips").to_frame().to_csv(
        os.path.join(C.TAB_DIR, "dataset_summary.csv"))
    # per-feature mean by technique
    data.groupby("technique")[F_cols()].mean().to_csv(
        os.path.join(C.TAB_DIR, "feature_means_by_technique.csv"))
    # model scores
    rows = [{"model": m, "accuracy": v["accuracy"], "macro_f1": v["macro_f1"]}
            for m, v in results["models"].items()]
    pd.DataFrame(rows).to_csv(
        os.path.join(C.TAB_DIR, "model_scores.csv"), index=False)


def F_cols():
    from src.features.extract import FEATURE_NAMES
    return FEATURE_NAMES


if __name__ == "__main__":
    main()
