"""Evaluation metrics and diagnostic plots for binary classifiers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from cardiorisk import config

ACCENT = "#e76f51"


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "false_negatives": float(fn),
    }


def plot_roc(y_true, proba, title: str) -> Figure:
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(
        fpr, tpr, color=ACCENT, lw=2, label=f"{title} (AUC = {roc_auc_score(y_true, proba):.3f})"
    )
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="grey", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title(f"ROC curve: {title} (test set)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_precision_recall(y_true, proba, title: str) -> Figure:
    precision, recall, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.step(recall, precision, where="post", color=ACCENT, lw=2, label=f"{title} (AP = {ap:.3f})")
    ax.axhline(np.mean(y_true), ls="--", lw=1, color="grey", label="Prevalence")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Precision-recall: {title} (test set)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_confusion(y_true, proba, title: str, threshold: float = 0.5) -> Figure:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        (np.asarray(proba) >= threshold).astype(int),
        display_labels=["No disease", "Disease"],
        cmap="Oranges",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"Confusion matrix: {title}")
    ax.grid(False)
    fig.tight_layout()
    return fig


def permutation_importances(pipeline, X: pd.DataFrame, y, repeats: int = 20) -> pd.DataFrame:
    """Importance of each *raw* input column, measured as ROC-AUC drop when shuffled."""
    result = permutation_importance(
        pipeline, X, y, scoring="roc_auc", n_repeats=repeats,
        random_state=config.RANDOM_STATE, n_jobs=1,
    )  # fmt: skip
    return (
        pd.DataFrame(
            {"feature": X.columns, "mean": result.importances_mean, "std": result.importances_std}
        )
        .sort_values("mean", ascending=False)
        .reset_index(drop=True)
    )


def plot_importances(importances: pd.DataFrame, title: str) -> Figure:
    data = importances.sort_values("mean")
    fig, ax = plt.subplots(figsize=(6, 4.8))
    ax.barh(data["feature"], data["mean"], xerr=data["std"], color=ACCENT, alpha=0.85)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Mean drop in ROC-AUC when shuffled")
    ax.set_title(f"Permutation importance: {title}")
    fig.tight_layout()
    return fig


def plot_model_comparison(table: pd.DataFrame) -> Figure:
    """Grouped bars of cross-validated mean +/- std for each model and metric."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    models = table["model"].tolist()
    width = 0.8 / len(models)
    colors = ["#264653", "#2a9d8f", "#e76f51", "#e9c46a"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(metrics))
    for i, row in table.reset_index(drop=True).iterrows():
        means = [row[f"cv_{m}_mean"] for m in metrics]
        stds = [row[f"cv_{m}_std"] for m in metrics]
        ax.bar(
            x + i * width - 0.4 + width / 2, means, width, yerr=stds, capsize=3,
            label=row["model"], color=colors[i % len(colors)],
        )  # fmt: skip
    ax.set_xticks(x, [m.replace("_", "-").upper() for m in metrics])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel(f"{config.CV_FOLDS}-fold CV score")
    ax.set_title("Model comparison (stratified cross-validation, mean +/- std)")
    ax.legend(fontsize=8, ncols=len(models), loc="upper center")
    fig.tight_layout()
    return fig
