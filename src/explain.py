"""
SHAP explainability for the trained technique classifier.

Produces:
  * global mean |SHAP| per feature (overall importance)
  * per-class mean |SHAP| matrix (feature x technique)  -> which acoustic
    indicators drive each technique, the pedagogically useful view
  * raw SHAP values + transformed feature matrix for beeswarm/local plots
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import shap

import config as C
from src.features.extract import FEATURE_NAMES


def compute_shap(model, data: pd.DataFrame) -> dict:
    """model: fitted sklearn Pipeline ending in an XGBClassifier."""
    X = data[FEATURE_NAMES].values
    pre = model[:-1]                      # impute + scale
    clf = model[-1]
    Xt = pre.transform(X)

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xt)        # list[class] or (n, f, c) array

    # Normalise to array of shape (n_samples, n_features, n_classes)
    if isinstance(sv, list):
        arr = np.stack(sv, axis=-1)
    elif sv.ndim == 3:
        arr = sv
    else:                                 # binary -> single matrix
        arr = sv[:, :, None]

    classes = sorted(data["technique"].unique().tolist())
    # Per-class mean |SHAP|: (features x classes)
    per_class = np.abs(arr).mean(axis=0)                      # (f, c)
    glob = per_class.mean(axis=1)                            # (f,)

    imp = pd.DataFrame(per_class, index=FEATURE_NAMES, columns=classes)
    imp["overall"] = glob
    imp = imp.sort_values("overall", ascending=False)
    imp.to_csv(os.path.join(C.TAB_DIR, "shap_importance.csv"))

    np.savez_compressed(
        os.path.join(C.OUT_DIR, "shap_values.npz"),
        shap=arr, Xt=Xt, features=np.array(FEATURE_NAMES),
        classes=np.array(classes))

    print("[explain] SHAP computed:", arr.shape)
    return {"shap": arr, "Xt": Xt, "classes": classes,
            "importance": imp}


if __name__ == "__main__":
    import json
    from joblib import load
    from src.data_loader import scan
    from src.model import build_feature_matrix

    data = build_feature_matrix(scan())
    model = load(os.path.join(C.MODEL_DIR, "final_model.joblib"))
    compute_shap(model, data)
