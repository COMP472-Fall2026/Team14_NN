"""Neural-network experiments: one-factor-at-a-time sweeps on both tasks.

    uv run python scripts/nn_experiments.py                         # all sweeps, both tasks
    uv run python scripts/nn_experiments.py --task image --sweep lr
    uv run python scripts/nn_experiments.py --plot-only             # redraw figures from saved CSVs

Every sweep starts from the same BASE configuration and changes a single
hyperparameter, so each difference in the plots has one cause. All runs train
on the saved train split and are scored on the saved validation split
(data/splits/), with preprocessing fitted on the train split only.

Outputs
  results/neural_net.csv                     one row per run (final + best val accuracy, timing)
  results/neural_net_history.csv             one row per run per epoch (for learning curves)
  results/figures/nn_<task>_<sweep>.png      train vs. validation accuracy and loss per epoch

The "tuned" sweep compares BASE with the final configuration in
models/neural_net.py:TUNED_CONFIGS (what scripts/make_submission.py uses).
"""

import argparse
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from comp472_project.config import FIGURES_DIR, RESULTS_DIR, SEED, TASKS, seed_everything
from comp472_project.data import load_split
from comp472_project.models import NeuralNetClassifier
from comp472_project.models.neural_net import TUNED_CONFIGS
from comp472_project.preprocessing import L2Normalizer, SqrtTransform, Standardizer

# Shared starting point. Each sweep overrides exactly one of these.
BASE = {
    "preprocessing": "standard",
    "hidden_sizes": (256, 64),
    "activation": "relu",
    "dropout": 0.0,
    "lr": 1e-3,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "batch_size": 128,
    "epochs": 40,
    "scheduler": "none",
}

SWEEPS = {
    "preprocessing": ("preprocessing", ["none", "standard", "l2"]),
    "lr": ("lr", [1e-1, 1e-2, 1e-3, 1e-4]),
    "hidden": ("hidden_sizes", [(64,), (256,), (512,), (256, 64), (512, 256), (1024, 512)]),
    "dropout": ("dropout", [0.0, 0.2, 0.5]),
    # BASE vs. the final per-task configuration (TUNED_CONFIGS), which changes
    # several settings at once to fix the overfitting seen in the sweeps above.
    "tuned": ("config", ["base", "tuned"]),
}

TRANSFORMS = {
    "none": [],
    "standard": [Standardizer],
    "l2": [L2Normalizer],
    "sqrt+standard": [SqrtTransform, Standardizer],
}

RUNS_CSV = RESULTS_DIR / "neural_net.csv"
HISTORY_CSV = RESULTS_DIR / "neural_net_history.csv"


def fmt(value) -> str:
    if isinstance(value, tuple):
        return "-".join(map(str, value))
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def run_one(data, cfg: dict) -> tuple[dict, list[dict]]:
    """Train one configuration; return (summary row, per-epoch history)."""
    X_train, X_val = data.X_train, data.X_val
    for t in (cls() for cls in TRANSFORMS[cfg["preprocessing"]]):
        X_train = t.fit_transform(X_train, data.y_train)  # fitted on train only
        X_val = t.transform(X_val)

    model_args = {k: v for k, v in cfg.items() if k != "preprocessing"}
    model = NeuralNetClassifier(**model_args, seed=SEED)
    start = perf_counter()
    model.fit(X_train, data.y_train, X_val=X_val, y_val=data.y_val)
    fit_time = perf_counter() - start

    hist = model.history_
    best = max(hist, key=lambda r: r["val_acc"])
    summary = {
        "task": data.task,
        **{k: fmt(v) for k, v in cfg.items()},
        "final_train_acc": hist[-1]["train_acc"],
        "final_val_acc": hist[-1]["val_acc"],
        "best_val_acc": best["val_acc"],
        "best_epoch": best["epoch"],
        "fit_time_s": round(fit_time, 2),
    }
    return summary, hist


def run_sweep(task: str, sweep: str) -> None:
    param, values = SWEEPS[sweep]
    data = load_split(task)
    rows, hist_rows = [], []
    for value in values:
        seed_everything()
        if sweep == "tuned":
            cfg = {**BASE, **(TUNED_CONFIGS[task] if value == "tuned" else {})}
        else:
            cfg = {**BASE, param: value}
        summary, hist = run_one(data, cfg)
        summary = {"sweep": sweep, "setting": fmt(value), **summary}
        rows.append(summary)
        hist_rows += [{"task": task, "sweep": sweep, "setting": fmt(value), **h} for h in hist]
        print(
            f"  {task:>9} | {param}={fmt(value):<9} | final train {summary['final_train_acc']:.3f} "
            f"val {summary['final_val_acc']:.3f} | best val {summary['best_val_acc']:.3f} "
            f"@ epoch {summary['best_epoch']:>2} | {summary['fit_time_s']:.1f}s"
        )
    replace_rows(RUNS_CSV, pd.DataFrame(rows), task, sweep)
    replace_rows(HISTORY_CSV, pd.DataFrame(hist_rows), task, sweep)
    plot_sweep(task, sweep)


def replace_rows(path, new: pd.DataFrame, task: str, sweep: str) -> None:
    """Swap in this (task, sweep)'s rows so re-running a sweep doesn't duplicate it."""
    if path.exists():
        old = pd.read_csv(path)
        old = old[~((old["task"] == task) & (old["sweep"] == sweep))]
        new = pd.concat([old, new], ignore_index=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)


def plot_sweep(task: str, sweep: str) -> None:
    hist = pd.read_csv(HISTORY_CSV, dtype={"setting": str})
    hist = hist[(hist["task"] == task) & (hist["sweep"] == sweep)]
    if hist.empty:
        return
    param = SWEEPS[sweep][0]
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.2))
    colors = plt.get_cmap("tab10").colors
    for i, (setting, h) in enumerate(hist.groupby("setting", sort=False)):
        c = colors[i % len(colors)]
        ax_acc.plot(h["epoch"], h["val_acc"], color=c, lw=2, label=f"{setting}")
        ax_acc.plot(h["epoch"], h["train_acc"], color=c, lw=1.2, ls="--", alpha=0.8)
        ax_loss.plot(h["epoch"], h["val_loss"], color=c, lw=2, label=f"{setting}")
        ax_loss.plot(h["epoch"], h["train_loss_eval"], color=c, lw=1.2, ls="--", alpha=0.8)
    chance = 1 / TASKS[task]["n_classes"]
    ax_acc.axhline(chance, color="grey", lw=0.8, ls=":")
    ax_acc.set(xlabel="epoch", ylabel="accuracy", title=f"{task}: accuracy vs. {param}")
    ax_loss.set(xlabel="epoch", ylabel="cross-entropy loss", title=f"{task}: loss vs. {param}")
    ax_loss.set_yscale("log")
    ax_acc.legend(title=f"{param}  (solid = val, dashed = train)", fontsize=8, title_fontsize=8)
    for ax in (ax_acc, ax_loss):
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"nn_{task}_{sweep}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved {out.relative_to(RESULTS_DIR.parent)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=[*TASKS, "both"], default="both")
    parser.add_argument("--sweep", choices=[*SWEEPS, "all"], default="all")
    parser.add_argument("--epochs", type=int, help=f"override BASE epochs ({BASE['epochs']})")
    parser.add_argument("--plot-only", action="store_true", help="redraw figures from saved CSVs")
    args = parser.parse_args()

    if args.epochs:
        BASE["epochs"] = args.epochs
        for cfg in TUNED_CONFIGS.values():
            cfg["epochs"] = args.epochs
    tasks = list(TASKS) if args.task == "both" else [args.task]
    sweeps = list(SWEEPS) if args.sweep == "all" else [args.sweep]
    for task in tasks:
        for sweep in sweeps:
            print(f"[{task} / {sweep}]")
            plot_sweep(task, sweep) if args.plot_only else run_sweep(task, sweep)


if __name__ == "__main__":
    main()
