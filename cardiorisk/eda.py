"""Exploratory data analysis: report-ready figures and a summary for the cleaned data.

Usage:
    python -m cardiorisk.eda     # writes reports/figures/eda_*.png and reports/eda_summary.json
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from cardiorisk import config
from cardiorisk.data import load_clean, summarise

log = logging.getLogger(__name__)

CLASS_COLORS = {0: "#2a9d8f", 1: "#e76f51"}
CLASS_LABELS = {0: "No disease", 1: "Disease"}

CODE_LABELS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {1: "Typical", 2: "Atypical", 3: "Non-anginal", 4: "Asymptomatic"},
    "fbs": {0: "<=120", 1: ">120"},
    "restecg": {0: "Normal", 1: "ST-T abn.", 2: "LVH"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {1: "Up", 2: "Flat", 3: "Down"},
    "ca": {0: "0", 1: "1", 2: "2", 3: "3"},
    "thal": {3: "Normal", 6: "Fixed", 7: "Reversible"},
}

AXIS_TITLES = {
    "age": "Age (years)",
    "trestbps": "Resting BP (mm Hg)",
    "chol": "Cholesterol (mg/dl)",
    "thalach": "Max heart rate (bpm)",
    "oldpeak": "ST depression (mm)",
    "pct_max_hr": "Max HR / (220 - age)",
    "sex": "Sex",
    "cp": "Chest pain type",
    "fbs": "Fasting blood sugar",
    "restecg": "Resting ECG",
    "exang": "Exercise angina",
    "slope": "ST slope",
    "ca": "Vessels (fluoroscopy)",
    "thal": "Thallium test",
}


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
    plt.rcParams.update(
        {"figure.dpi": 110, "savefig.dpi": 150, "axes.titleweight": "bold", "axes.titlesize": 11}
    )


def _labelled_target(df: pd.DataFrame) -> pd.Series:
    return df[config.TARGET].map(CLASS_LABELS)


def plot_class_balance(df: pd.DataFrame) -> Figure:
    counts = df[config.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    bars = ax.bar(
        [CLASS_LABELS[k] for k in counts.index],
        counts.values,
        color=[CLASS_COLORS[k] for k in counts.index],
        width=0.55,
    )
    for bar, n in zip(bars, counts.values, strict=True):
        ax.annotate(
            f"{n}  ({n / counts.sum():.1%})",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            xytext=(0, 3),
            textcoords="offset points",
        )
    ax.set_ylabel("Patients")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_title("Target class balance")
    fig.tight_layout()
    return fig


def plot_missing_values(df: pd.DataFrame) -> Figure:
    missing = df[config.FEATURE_COLUMNS].isna().sum().sort_values()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    colors = ["#e76f51" if n else "#ced4da" for n in missing.values]
    ax.barh([AXIS_TITLES.get(c, c) for c in missing.index], missing.values, color=colors)
    for y, n in enumerate(missing.values):
        if n:
            ax.annotate(
                f"{n} ({n / len(df):.1%})",
                (n, y),
                xytext=(4, -4),
                textcoords="offset points",
                fontsize=9,
            )
    ax.set_xlabel("Missing values")
    ax.set_xlim(0, max(missing.max() * 1.6, 1))
    ax.set_title("Missing values per feature")
    fig.tight_layout()
    return fig


def plot_numeric_distributions(df: pd.DataFrame) -> Figure:
    data = df.assign(pct_max_hr=df["thalach"] / (220 - df["age"]), diagnosis=_labelled_target(df))
    columns = ["age", "trestbps", "chol", "thalach", "oldpeak", "pct_max_hr"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    palette = {CLASS_LABELS[k]: v for k, v in CLASS_COLORS.items()}
    for ax, col in zip(axes.flat, columns, strict=True):
        sns.histplot(
            data=data,
            x=col,
            hue="diagnosis",
            kde=True,
            stat="density",
            common_norm=False,
            element="step",
            palette=palette,
            ax=ax,
            bins=20,
        )
        ax.set_xlabel(AXIS_TITLES[col])
        ax.set_title(AXIS_TITLES[col])
        if ax is not axes.flat[0] and ax.get_legend():
            ax.get_legend().remove()
    fig.suptitle("Distribution of continuous features by diagnosis", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> Figure:
    corr = df[config.FEATURE_COLUMNS + [config.TARGET]].corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.7},
        ax=ax,
    )
    ax.set_title("Spearman correlation (features and target)")
    fig.tight_layout()
    return fig


def plot_categorical_risk(df: pd.DataFrame) -> Figure:
    columns = ["cp", "thal", "ca", "slope", "exang", "sex", "restecg", "fbs"]
    overall = df[config.TARGET].mean()
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True)
    for ax, col in zip(axes.flat, columns, strict=True):
        grouped = df.groupby(col)[config.TARGET].agg(["mean", "size"])
        labels = [CODE_LABELS[col].get(int(code), str(code)) for code in grouped.index]
        colors = sns.blend_palette([CLASS_COLORS[0], CLASS_COLORS[1]], as_cmap=True)(
            grouped["mean"].to_numpy()
        )
        ax.bar(labels, grouped["mean"], color=colors)
        for x, (rate, n) in enumerate(zip(grouped["mean"], grouped["size"], strict=True)):
            ax.annotate(
                f"{rate:.0%}\nn={n}",
                (x, rate),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )
        ax.axhline(overall, ls="--", lw=1, color="#495057")
        ax.set_title(AXIS_TITLES[col])
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis="x", labelrotation=20)
    axes[0, 0].set_ylabel("Disease rate")
    axes[1, 0].set_ylabel("Disease rate")
    fig.suptitle(
        f"Disease rate by category (dashed line = overall rate {overall:.0%})", fontweight="bold"
    )
    fig.tight_layout()
    return fig


def plot_heart_rate_vs_age(df: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, colour in CLASS_COLORS.items():
        subset = df[df[config.TARGET] == label]
        ax.scatter(
            subset["age"],
            subset["thalach"],
            s=22,
            alpha=0.75,
            color=colour,
            label=CLASS_LABELS[label],
            edgecolor="white",
            linewidth=0.4,
        )
    ages = np.linspace(df["age"].min(), df["age"].max(), 50)
    ax.plot(ages, 220 - ages, color="#264653", lw=1.5, label="Age-predicted max (220 - age)")
    ax.plot(ages, 0.85 * (220 - ages), color="#264653", lw=1, ls=":", label="85% of predicted max")
    ax.set_xlabel(AXIS_TITLES["age"])
    ax.set_ylabel(AXIS_TITLES["thalach"])
    ax.set_title("Exercise heart rate vs age")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


FIGURES: dict[str, Callable[[pd.DataFrame], Figure]] = {
    "eda_class_balance": plot_class_balance,
    "eda_missing_values": plot_missing_values,
    "eda_numeric_distributions": plot_numeric_distributions,
    "eda_correlation_heatmap": plot_correlation_heatmap,
    "eda_categorical_risk": plot_categorical_risk,
    "eda_heart_rate_vs_age": plot_heart_rate_vs_age,
}


def generate_all(df: pd.DataFrame, out_dir: Path = config.FIGURES_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, plot in FIGURES.items():
        fig = plot(df)
        path = out_dir / f"{name}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("matplotlib.category").setLevel(logging.WARNING)
    set_style()
    df = load_clean()
    for path in generate_all(df):
        log.info("Wrote %s", path.relative_to(config.PROJECT_ROOT))

    summary = summarise(df)
    summary["describe"] = df.describe().round(2).to_dict()
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote reports/eda_summary.json")


if __name__ == "__main__":
    main()
