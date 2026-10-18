"""Train, tune, compare and register heart-disease classifiers with MLflow tracking.

For each model family a parent MLflow run holds the best configuration, its
cross-validated and hold-out metrics, plots and the serialised pipeline. Every
grid-search candidate is logged as a nested child run. The champion (highest
mean CV ROC-AUC, never the test score) is registered in the MLflow Model
Registry under the alias "champion" and exported to models/ for the API.

Usage:
    python -m cardiorisk.train            # full grids
    python -m cardiorisk.train --quick    # tiny grids for CI smoke runs
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
from datetime import UTC, datetime

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from cardiorisk import __version__, config
from cardiorisk.data import load_clean, sha256sum
from cardiorisk.evaluation import (
    compute_metrics,
    permutation_importances,
    plot_confusion,
    plot_importances,
    plot_model_comparison,
    plot_precision_recall,
    plot_roc,
)
from cardiorisk.features import build_pipeline, split_features_target

log = logging.getLogger(__name__)

# Types skops does not trust by default. All model files are produced by this script.
SKOPS_TRUSTED_TYPES = [
    "cardiorisk.features.ClinicalFeatureAdder",
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
]

SCORING = {
    "accuracy": "accuracy",
    # Heavily regularised candidates can predict no positives; score that as 0, not a warning.
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def model_specs(quick: bool = False) -> dict[str, tuple]:
    """Estimator + hyper-parameter grid per model family."""
    rs = config.RANDOM_STATE
    if quick:
        return {
            "logistic_regression": (
                LogisticRegression(solver="saga", max_iter=5000, random_state=rs),
                {"model__C": [0.1, 1.0]},
            ),
            "random_forest": (
                RandomForestClassifier(n_estimators=50, random_state=rs),
                {"model__max_depth": [4, None]},
            ),
        }
    return {
        # l1_ratio 0 = ridge (L2), 1 = lasso (L1), 0.5 = elastic net
        "logistic_regression": (
            LogisticRegression(solver="saga", max_iter=10000, random_state=rs),
            {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__l1_ratio": [0.0, 0.5, 1.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=300, random_state=rs, n_jobs=1),
            {
                "model__max_depth": [None, 4, 8],
                "model__min_samples_leaf": [1, 4],
                "model__max_features": ["sqrt", 0.5],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=rs),
            {
                "model__n_estimators": [100, 300],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [2, 3],
                "model__subsample": [0.8, 1.0],
            },
        ),
    }


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, cwd=config.PROJECT_ROOT,
        )  # fmt: skip
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _clean_params(params: dict) -> dict:
    return {k.removeprefix("model__"): ("None" if v is None else v) for k, v in params.items()}


def _cv_summary(search: GridSearchCV, index: int | None = None) -> dict[str, float]:
    i = search.best_index_ if index is None else index
    res = search.cv_results_
    summary = {}
    for metric in SCORING:
        summary[f"cv_{metric}_mean"] = float(res[f"mean_test_{metric}"][i])
        summary[f"cv_{metric}_std"] = float(res[f"std_test_{metric}"][i])
    summary["cv_train_roc_auc_mean"] = float(res["mean_train_roc_auc"][i])
    return summary


def _log_candidates(search: GridSearchCV, family: str) -> None:
    ranks = search.cv_results_["rank_test_roc_auc"]
    for i, params in enumerate(search.cv_results_["params"]):
        with mlflow.start_run(run_name=f"{family}-cand-{i:02d}", nested=True):
            mlflow.set_tags({"model_family": family, "run_type": "grid_candidate"})
            mlflow.log_params(_clean_params(params))
            mlflow.log_metrics({**_cv_summary(search, i), "rank_roc_auc": float(ranks[i])})


def _log_figure(fig, artifact: str, save_as: str | None = None) -> None:
    mlflow.log_figure(fig, artifact)
    if save_as:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FIGURES_DIR / save_as, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _setup_mlflow() -> str:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            config.EXPERIMENT_NAME, artifact_location=config.MLFLOW_ARTIFACT_ROOT
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def train_family(family, estimator, grid, X_train, y_train, X_test, y_test, tags) -> dict:
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    search = GridSearchCV(
        build_pipeline(estimator), grid, scoring=SCORING, refit="roc_auc",
        cv=cv, n_jobs=-1, return_train_score=True,
    )  # fmt: skip

    with mlflow.start_run(run_name=family) as run:
        mlflow.set_tags({**tags, "model_family": family, "run_type": "model_family"})
        search.fit(X_train, y_train)
        _log_candidates(search, family)

        best = search.best_estimator_
        proba = best.predict_proba(X_test)[:, 1]
        cv_metrics = _cv_summary(search)
        test_metrics = {f"test_{k}": float(v) for k, v in compute_metrics(y_test, proba).items()}

        mlflow.log_params(_clean_params(search.best_params_))
        mlflow.log_params(
            {"n_candidates": len(search.cv_results_["params"]), "cv_folds": config.CV_FOLDS}
        )
        mlflow.log_metrics({**cv_metrics, **test_metrics})

        importances = permutation_importances(best, X_test, y_test)
        _log_figure(plot_roc(y_test, proba, family), "plots/roc_curve.png")
        _log_figure(plot_precision_recall(y_test, proba, family), "plots/pr_curve.png")
        _log_figure(plot_confusion(y_test, proba, family), "plots/confusion_matrix.png")
        _log_figure(plot_importances(importances, family), "plots/permutation_importance.png")
        mlflow.log_table(importances, "tables/permutation_importance.json")
        mlflow.log_table(
            pd.DataFrame(search.cv_results_).drop(columns="params").astype(str),
            "tables/cv_results.json",
        )

        signature = infer_signature(X_train, best.predict(X_train))
        info = mlflow.sklearn.log_model(
            best,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
            code_paths=[str(config.PROJECT_ROOT / "cardiorisk")],
            # MLflow serialises with skops (no arbitrary pickle); allow our own transformer.
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        log.info(
            "%-20s cv_roc_auc=%.4f +/- %.4f  test_roc_auc=%.4f  (%d candidates)",
            family, cv_metrics["cv_roc_auc_mean"], cv_metrics["cv_roc_auc_std"],
            test_metrics["test_roc_auc"], len(search.cv_results_["params"]),
        )  # fmt: skip
        return {
            "model": family,
            "run_id": run.info.run_id,
            "model_uri": info.model_uri,
            "best_params": _clean_params(search.best_params_),
            "estimator": best,
            "proba": proba,
            "importances": importances,
            **cv_metrics,
            **test_metrics,
        }


def run(quick: bool = False) -> dict:
    experiment_id = _setup_mlflow()
    df = load_clean()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    log.info(
        "Train %d rows / test %d rows (positive rate %.2f)", len(X_train), len(X_test), y.mean()
    )

    tags = {
        "data_sha256": sha256sum(config.CLEAN_FILE),
        "git_commit": _git_commit(),
        "quick_mode": str(quick),
        "package_version": __version__,
    }
    results = [
        train_family(name, est, grid, X_train, y_train, X_test, y_test, tags)
        for name, (est, grid) in model_specs(quick).items()
    ]

    # Selection uses cross-validation only; the test set stays an unbiased final check.
    champion = max(results, key=lambda r: r["cv_roc_auc_mean"])
    table = pd.DataFrame(
        [{k: v for k, v in r.items() if isinstance(v, str | float)} for r in results]
    )
    table = table.sort_values("cv_roc_auc_mean", ascending=False).round(4)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.drop(columns=["run_id", "model_uri"]).to_csv(
        config.REPORTS_DIR / "model_comparison.csv", index=False
    )

    with mlflow.start_run(run_name="model-selection"):
        mlflow.set_tags({**tags, "run_type": "selection", "champion": champion["model"]})
        mlflow.log_table(table, "tables/model_comparison.json")
        _log_figure(plot_model_comparison(table), "plots/model_comparison.png",
                    save_as="model_comparison.png")  # fmt: skip
        mlflow.log_metrics({"champion_cv_roc_auc": champion["cv_roc_auc_mean"]})

    # Report-ready copies of the champion's diagnostics.
    name, proba = champion["model"], champion["proba"]
    for fig, file in [
        (plot_roc(y_test, proba, name), "champion_roc_curve.png"),
        (plot_precision_recall(y_test, proba, name), "champion_pr_curve.png"),
        (plot_confusion(y_test, proba, name), "champion_confusion_matrix.png"),
        (plot_importances(champion["importances"], name), "champion_permutation_importance.png"),
    ]:
        fig.savefig(config.FIGURES_DIR / file, bbox_inches="tight", dpi=150)
        plt.close(fig)

    client = MlflowClient()
    version = mlflow.register_model(champion["model_uri"], config.REGISTERED_MODEL_NAME)
    client.set_registered_model_alias(config.REGISTERED_MODEL_NAME, "champion", version.version)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion["estimator"], config.MODEL_PATH)
    card = {
        "model_name": config.REGISTERED_MODEL_NAME,
        "model_family": champion["model"],
        "model_version": f"{champion['model']}-v{version.version}",
        "registry_version": int(version.version),
        "mlflow_run_id": champion["run_id"],
        "mlflow_experiment_id": experiment_id,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "best_params": champion["best_params"],
        "decision_threshold": 0.5,
        "features": config.FEATURE_COLUMNS,
        "metrics": {k: champion[k] for k in table.columns if k.startswith(("cv_", "test_"))},
        "data_sha256": tags["data_sha256"],
        "git_commit": tags["git_commit"],
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
    }
    config.MODEL_CARD_PATH.write_text(json.dumps(card, indent=2))
    log.info("Champion: %s (registry v%s) -> %s", name, version.version, config.MODEL_PATH)
    return card


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and register heart-disease models")
    parser.add_argument("--quick", action="store_true", help="tiny grids for CI smoke runs")
    args = parser.parse_args()
    matplotlib.use("Agg")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("mlflow", "alembic", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    card = run(quick=args.quick)
    print(json.dumps({k: card[k] for k in ("model_version", "metrics")}, indent=2))


if __name__ == "__main__":
    main()
