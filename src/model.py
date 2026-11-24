"""
Feature-matrix construction, speaker-independent evaluation, and model
training.

Design choices that matter for honesty of results:
  * Splits are GROUPED BY SINGER (GroupKFold) so no singer appears in both
    train and test -- otherwise accuracy is inflated by singer identity.
  * Features are imputed (median) and standardised inside each fold.
  * Out-of-fold predictions give an unbiased confusion matrix.
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)
from xgboost import XGBClassifier

import config as C
from src.data_loader import scan
from src.features.extract import extract_file, FEATURE_NAMES


# --------------------------------------------------------------------------- #
# Feature matrix
# --------------------------------------------------------------------------- #
def build_feature_matrix(df: pd.DataFrame, n_jobs: int = -1,
                         cache: str | None = None) -> pd.DataFrame:
    cache = cache or C.FEATURE_CSV
    if os.path.exists(cache):
        print(f"[model] loading cached features: {cache}")
        return pd.read_csv(cache)

    print(f"[model] extracting features from {len(df)} files ...")
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(extract_file)(p) for p in df.path)

    keep = [(r, m) for r, (_, m) in zip(results, df.iterrows()) if r is not None]
    feats = pd.DataFrame([r for r, _ in keep])
    meta = pd.DataFrame([m[["technique", "singer", "gender", "path"]].to_dict()
                         for _, m in keep]).reset_index(drop=True)
    out = pd.concat([meta, feats.reset_index(drop=True)], axis=1)
    out.to_csv(cache, index=False)
    print(f"[model] wrote {cache}  ({len(out)} rows)")
    return out


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def _pipeline(clf) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])


def make_models(n_classes: int) -> dict:
    return {
        "Logistic Regression": _pipeline(
            LogisticRegression(max_iter=2000, C=1.0)),
        "SVM (RBF)": _pipeline(SVC(C=10, gamma="scale")),
        "Random Forest": _pipeline(
            RandomForestClassifier(n_estimators=400, max_depth=None,
                                   class_weight="balanced", random_state=0)),
        # Note: the sklearn wrapper infers objective/num_class from y; setting
        # num_class explicitly errors on xgboost >= 3. Leave it to XGBoost.
        "XGBoost": _pipeline(
            XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                          subsample=0.9, colsample_bytree=0.8,
                          eval_metric="mlogloss", random_state=0,
                          tree_method="hist")),
    }


def evaluate(data: pd.DataFrame, n_splits: int = 5) -> dict:
    """Speaker-independent CV for every model. Returns a results dict."""
    X = data[FEATURE_NAMES].values
    y_txt = data["technique"].values
    classes = sorted(np.unique(y_txt).tolist())
    y = np.array([classes.index(t) for t in y_txt])
    groups = data["singer"].values

    n_splits = min(n_splits, data["singer"].nunique())
    gkf = GroupKFold(n_splits=n_splits)
    results = {"classes": classes, "models": {}}

    for name, model in make_models(len(classes)).items():
        y_pred = cross_val_predict(model, X, y, groups=groups, cv=gkf, n_jobs=-1)
        acc = accuracy_score(y, y_pred)
        f1m = f1_score(y, y_pred, average="macro")
        results["models"][name] = {
            "accuracy": float(acc),
            "macro_f1": float(f1m),
            "report": classification_report(y, y_pred, target_names=classes,
                                             output_dict=True, zero_division=0),
            "confusion": confusion_matrix(y, y_pred).tolist(),
            "y_true": y.tolist(),
            "y_pred": y_pred.tolist(),
        }
        print(f"[model] {name:20s} acc={acc:.3f} macroF1={f1m:.3f}")
    return results


def fit_final(data: pd.DataFrame, model_name: str = "XGBoost"):
    """Fit the chosen model on all data (for SHAP explanations)."""
    X = data[FEATURE_NAMES].values
    classes = sorted(data["technique"].unique().tolist())
    y = np.array([classes.index(t) for t in data["technique"]])
    model = make_models(len(classes))[model_name]
    model.fit(X, y)
    dump(model, os.path.join(C.MODEL_DIR, "final_model.joblib"))
    with open(os.path.join(C.MODEL_DIR, "classes.json"), "w") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    return model, classes


if __name__ == "__main__":
    df = scan()
    data = build_feature_matrix(df)
    res = evaluate(data)
    with open(os.path.join(C.OUT_DIR, "cv_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    fit_final(data)
