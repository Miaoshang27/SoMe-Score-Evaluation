
# train_and_save.py
# End-to-end: load data, split, build pipeline, calibrate, tune thresholds,
# evaluate, and export a single joblib bundle for deployment.
from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve, roc_curve, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight

from engineer_some import EngineerSoMeFeatures

RANDOM_STATE = 42

@dataclass
class Bundle:
    pipeline_path: Path
    metrics_path: Path
    bundle_path: Path

    
def load_dataset(
    csv_path: Path,
    feature_cols,
    label_col: str,
    success_from: str = None,
    success_threshold: float = 50.0
):
    import pandas as pd
    df = pd.read_csv(csv_path)

    # Derive label if missing
    if label_col not in df.columns:
        if not success_from:
            raise ValueError(
                f"Missing '{label_col}' and no success_from column specified. "
                f"Pass --success-from <col> to derive the label (>= threshold → 1)."
            )
        if success_from not in df.columns:
            raise ValueError(
                f"'{label_col}' is missing and success_from column '{success_from}' "
                f"does not exist. Available columns: {list(df.columns)}"
            )
        df[label_col] = (df[success_from].fillna(0) >= float(success_threshold)).astype(int)

    # Validate features + label now exist
    missing = [c for c in feature_cols + [label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    X = df[feature_cols].copy()
    y = df[label_col].astype(int).copy()
    return X, y



def prevalence(y: pd.Series) -> float:
    return float(np.mean(y))

def build_base_pipeline(numeric_cols):
    # Impute -> Engineer -> Scale -> Logistic
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("eng", EngineerSoMeFeatures(input_feature_names=numeric_cols, clip_percentiles=(1, 99))),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    return pipe

def fit_and_calibrate(base_pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> CalibratedClassifierCV:
    # split a calibration set from train (20% of train)
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
    )
    # Fit base
    base_pipe.fit(X_tr, y_tr)
    # Calibrate with isotonic on held-out cal set
    calib = CalibratedClassifierCV(estimator=base_pipe, method="isotonic", cv="prefit")
    calib.fit(X_cal, y_cal)
    return calib

def eval_probs(y_true: np.ndarray, p: np.ndarray, label: str) -> Dict[str, float]:
    return {
        f"{label}_roc_auc": roc_auc_score(y_true, p),
        f"{label}_pr_auc": average_precision_score(y_true, p),
        f"{label}_brier": brier_score_loss(y_true, p),
        f"{label}_prevalence": float(np.mean(y_true)),
    }

def tune_thresholds(y_true: np.ndarray, p: np.ndarray, min_launch_precision: float = 0.70) -> Dict[str, float]:
    """
    Derive two thresholds:
    - t_review: single-threshold that maximizes F1 on validation
    - t_launch: smallest threshold with precision >= min_launch_precision, else fallback to 90th percentile
    """
    # F1 sweep for t_review
    prec, rec, thr = precision_recall_curve(y_true, p)
    # avoid division by zero
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    # precision_recall_curve returns an extra point for p=0; threshold array is len-1 of prec/rec
    t_candidates = np.r_[0.0, thr]  # align shapes
    t_review = float(t_candidates[np.argmax(f1)])

    # t_launch: smallest threshold with precision >= min_launch_precision
    # Sort by threshold ascending while keeping precision in sync
    # Interpolate precision at candidate thresholds; simpler: scan unique sorted probs
    uniq = np.unique(p)
    best = None
    for t in sorted(uniq):
        pred = (p >= t).astype(int)
        tp = int((pred & (y_true == 1)).sum())
        fp = int((pred & (y_true == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        if precision >= min_launch_precision:
            best = float(t)
            break
    if best is None:
        # fallback: top decile
        best = float(np.quantile(p, 0.9))
    t_launch = best

    return {"t_review": t_review, "t_launch": t_launch}

def apply_categories(p: np.ndarray, t_review: float, t_launch: float) -> np.ndarray:
    out = np.full_like(p, fill_value="Ignore", dtype=object)
    out[(p >= t_review) & (p < t_launch)] = "Review"
    out[p >= t_launch] = "Launch"
    return out

def report_at_threshold(y_true: np.ndarray, p: np.ndarray, t: float) -> Dict[str, float]:
    pred = (p >= t).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    return {"threshold": float(t), "precision": prec, "recall": rec, "f1": f1, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

def train_main(
    csv_path: str,
    label_col: str = "if_success",
    feature_cols = ("engagement_rate","follower_count_c","posting_frequency_c"),
    out_dir: str = "artifacts",
    min_launch_precision: float = 0.70,
    success_from: str = "followers_gained_last_30_days",
    success_threshold: float = 50.0
)-> Bundle:
    # Make sure output directory exists
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(
        Path(csv_path), 
        list(feature_cols), 
        label_col,
        success_from=success_from,
        success_threshold=success_threshold)

    # 60/20/20 split: train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_STATE)
    # -> Train 60%, Val 20%, Test 20%

    base_pipe = build_base_pipeline(feature_cols)
    calibrated = fit_and_calibrate(base_pipe, X_train, y_train)

    # Probabilities
    p_val = calibrated.predict_proba(X_val)[:,1]
    p_test = calibrated.predict_proba(X_test)[:,1]

    # Metrics
    metrics = {}
    metrics.update(eval_probs(y_val.to_numpy(), p_val, "val"))
    metrics.update(eval_probs(y_test.to_numpy(), p_test, "test"))

    # Tune thresholds on validation
    ts = tune_thresholds(y_val.to_numpy(), p_val, min_launch_precision=min_launch_precision)
    metrics["thresholds"] = ts

    # Category success rates on validation (sanity check)
    cats_val = apply_categories(p_val, ts["t_review"], ts["t_launch"])
    sr_val = pd.DataFrame({"cat": cats_val, "y": y_val.to_numpy()}).groupby("cat")["y"].agg(["count","mean"]).rename(columns={"mean":"success_rate"})
    metrics["val_category_counts"] = sr_val["count"].to_dict()
    metrics["val_category_success_rate"] = {k: float(v) for k,v in sr_val["success_rate"].to_dict().items()}

    # Snapshot thresholded binary reports (val & test)
    metrics["val_at_t_review"]  = report_at_threshold(y_val.to_numpy(), p_val, ts["t_review"])
    metrics["test_at_t_review"] = report_at_threshold(y_test.to_numpy(), p_test, ts["t_review"])
    metrics["test_at_t_launch"] = report_at_threshold(y_test.to_numpy(), p_test, ts["t_launch"])

    # Save bundle
    bundle = {
        "pipeline": calibrated,
        "feature_order": list(feature_cols),
        "thresholds": ts,
        "version": "some-score-" + pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "metrics_preview": {"val_roc_auc": metrics["val_roc_auc"], "test_roc_auc": metrics["test_roc_auc"],
                            "val_pr_auc": metrics["val_pr_auc"], "test_pr_auc": metrics["test_pr_auc"]},
    }
    bundle_path = out / "some_model.joblib"
    joblib.dump(bundle, bundle_path)

    # Save metrics JSON
    metrics_path = out / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save a lightweight model card
    card = out / "MODEL_CARD.md"
    with open(card, "w") as f:
        f.write(f"# SoMe Score Model Card\n\n")
        f.write(f"**Version:** {bundle['version']}\n\n")
        f.write(f"**Features:** {', '.join(feature_cols)}\n\n")
        f.write(f"**Splits:** train 60% / val 20% / test 20% (stratified)\n\n")
        f.write("## Validation metrics\n")
        f.write(f"- ROC AUC: {metrics['val_roc_auc']:.4f}\n- PR AUC: {metrics['val_pr_auc']:.4f}\n- Brier: {metrics['val_brier']:.4f}\n\n")
        f.write("## Test metrics\n")
        f.write(f"- ROC AUC: {metrics['test_roc_auc']:.4f}\n- PR AUC: {metrics['test_pr_auc']:.4f}\n- Brier: {metrics['test_brier']:.4f}\n\n")
        f.write("## Thresholds\n")
        f.write(json.dumps(bundle["thresholds"], indent=2))
        f.write("\n\n## Category success rate (VAL)\n")
        for k,v in metrics["val_category_success_rate"].items():
            f.write(f"- {k}: {v:.3f} (n={metrics['val_category_counts'].get(k,0)})\n")

    return Bundle(
        pipeline_path=bundle_path,
        metrics_path=metrics_path,
        bundle_path=bundle_path
    )

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV")
    ap.add_argument("--label", default="if_success", help="Label col name (created if missing)")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--min-launch-precision", type=float, default=0.70)
    ap.add_argument("--success-from", default="followers_gained_last_30_days",
                    help="Column to derive label from if --label is missing")
    ap.add_argument("--success-threshold", type=float, default=50.0,
                    help="Threshold: success_from >= threshold → label=1")
    args = ap.parse_args()

    b = train_main(
        csv_path=args.csv,
        label_col=args.label,
        out_dir=args.out,
        min_launch_precision=args.min_launch_precision,
        success_from=args.success_from,
        success_threshold=args.success_threshold,
    )
    print(f"Saved bundle to: {b.bundle_path}")
    print(f"Saved metrics to: {b.metrics_path}")

