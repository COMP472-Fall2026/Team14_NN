"""Train the final model for each task, merge both tasks' predictions by id, verify, save.

    uv run python scripts/make_submission.py --name nn_v1
    uv run python scripts/make_submission.py --image knn --sentiment neural_net --name mixed_v1
    uv run python scripts/make_submission.py --name merged \\
        --image-csv results/predictions/knn_v2_image.csv \\
        --sentiment-csv results/predictions/nn_v1_sentiment.csv
    uv run python scripts/make_submission.py --verify submissions/<file>.csv

For each task the prediction comes either from a preset in FINAL_MODELS (the
model is re-trained on the *full* training file, preprocessing included, then
predicts the test file) or from a saved ``id,label`` CSV (--image-csv /
--sentiment-csv), so teammates' predictions can be mixed without retraining.

Each task's test predictions are also written to results/predictions/<name>_<task>.csv.
The merged file goes to submissions/<timestamp>_<name>.csv and is logged in
submissions/log.csv (fill in public_score after uploading).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from comp472_project.config import DATA_DIR, RESULTS_DIR, SEED, TASKS, seed_everything
from comp472_project.data import load_split, load_test, load_train
from comp472_project.evaluation import accuracy
from comp472_project.models import (
    DecisionTreeClassifier,
    KNNClassifier,
    NeuralNetClassifier,
    RandomForestClassifier,
)
from comp472_project.models.neural_net import TUNED_CONFIGS
from comp472_project.preprocessing import L2Normalizer, SqrtTransform, Standardizer
from comp472_project.submission import build_submission, save_submission, validate_submission

PREDICTIONS_DIR = RESULTS_DIR / "predictions"
N_TEST = {"image": 5000, "sentiment": 5000}
NN_TRANSFORMS = {"none": [], "standard": [Standardizer], "sqrt+standard": [SqrtTransform, Standardizer]}

# Final configuration of each model, per task: (model factory, transform classes).
# Update these as each model's tuning finishes (from its validation results).
FINAL_MODELS = {
    # Uses TUNED_CONFIGS from models/neural_net.py (chosen on the validation split).
    "neural_net": {
        task: (
            lambda task=task: NeuralNetClassifier(
                **{k: v for k, v in TUNED_CONFIGS[task].items() if k != "preprocessing"}, seed=SEED
            ),
            NN_TRANSFORMS[TUNED_CONFIGS[task]["preprocessing"]],
        )
        for task in TASKS
    },
    # Placeholders: owners of these models, set your tuned hyperparameters here.
    "knn": {
        # sqrt + cosine 1-NN: 56.3% val on images (vs. 51.9% without sqrt) (validation split).
        "image": (lambda: KNNClassifier(k=1, metric="cosine"), [SqrtTransform]),
        # kc's settings from submissions/log.csv (58.96% val).
        "sentiment": (lambda: KNNClassifier(k=50, metric="cosine", weighted=True), []),
    },
    "decision_tree": {
        "image": (lambda: DecisionTreeClassifier(max_depth=20, seed=SEED), []),
        "sentiment": (lambda: DecisionTreeClassifier(max_depth=10, seed=SEED), []),
    },
    "random_forest": {
        "image": (lambda: RandomForestClassifier(n_estimators=100, seed=SEED), []),
        "sentiment": (lambda: RandomForestClassifier(n_estimators=100, seed=SEED), []),
    },
}

_TRANSFORM_NAMES = {Standardizer: "standard", L2Normalizer: "l2", SqrtTransform: "sqrt"}


def _fit_predict(model, transforms, X_train, y_train, X_eval):
    for t in (cls() for cls in transforms):
        X_train = t.fit_transform(X_train, y_train)  # fitted on the training rows only
        X_eval = t.transform(X_eval)
    return np.asarray(model.fit(X_train, y_train).predict(X_eval))


def predict_task(task: str, model_name: str, check_val: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (test_ids, test_pred, notes) for one task using a FINAL_MODELS preset."""
    factory, transforms = FINAL_MODELS[model_name][task]
    notes = {f"{task}_model": model_name, f"{task}_params": describe(factory(), transforms)}

    if check_val:
        # Same configuration on the saved split, so the log records a validation score.
        seed_everything()
        data = load_split(task)
        val_pred = _fit_predict(factory(), transforms, data.X_train, data.y_train, data.X_val)
        notes[f"{task}_val_acc"] = round(accuracy(data.y_val, val_pred), 4)
        print(f"  {task:>9}: {model_name} validation accuracy {notes[f'{task}_val_acc']:.4f}")

    seed_everything()
    # final model: same settings, but trained on 100% of the training data
    _, X, y = load_train(task)  # all labelled rows
    test_ids, X_test = load_test(task)
    pred = _fit_predict(factory(), transforms, X, y, X_test)
    print(f"  {task:>9}: trained {model_name} on all {len(y):,} rows, predicted {len(pred):,} test rows")
    return test_ids, pred, notes


def read_predictions(task: str, path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load an ``id,label`` CSV and check it covers exactly this task's test ids."""
    df = pd.read_csv(path)
    if list(df.columns) != ["id", "label"]:
        raise ValueError(f"{path}: columns must be ['id', 'label'], got {list(df.columns)}")
    test_ids, _ = load_test(task)
    if df["id"].duplicated().any() or set(df["id"]) != set(test_ids):
        raise ValueError(f"{path}: ids must be exactly the {len(test_ids)} {task} test ids, once each")
    print(f"  {task:>9}: read {len(df):,} predictions from {path}")
    return df["id"].to_numpy(), df["label"].to_numpy(), {f"{task}_model": f"csv:{Path(path).name}"}


def describe(model, transforms) -> str:
    params = {k: v for k, v in vars(model).items() if not k.endswith("_") and k not in ("device",)}
    prep = "+".join(_TRANSFORM_NAMES.get(t, t.__name__) for t in transforms) or "none"
    return f"prep={prep}; " + ", ".join(f"{k}={v}" for k, v in params.items())


def verify(sub: pd.DataFrame) -> None:
    """Raise if anything is wrong; otherwise print what was checked."""
    validate_submission(sub)  # columns, every sample id exactly once, integer labels in range
    # double check against kaggle's own sample file
    sample_ids = pd.read_csv(DATA_DIR / "sample_submission.csv")["id"]
    assert len(sub) == 10_000 == len(sample_ids), f"expected 10,000 rows, got {len(sub)}"
    assert sub["id"].is_unique and set(sub["id"]) == set(sample_ids)
    assert sub["label"].notna().all()

    print(f"  rows: {len(sub):,} (unique ids: {sub['id'].nunique():,}, all match sample_submission.csv)")
    for task, cfg in TASKS.items():
        labels = sub.loc[sub["id"].str.startswith(cfg["test_id_prefix"]), "label"].astype(int)
        assert len(labels) == N_TEST[task], f"{task}: expected {N_TEST[task]} rows, got {len(labels)}"
        counts = np.bincount(labels, minlength=cfg["n_classes"])
        print(
            f"  {task:>9}: {len(labels):,} rows, labels {labels.min()}..{labels.max()}, "
            f"predicted per class {counts.min()}-{counts.max()}"
            + (f" {counts.tolist()}" if cfg["n_classes"] <= 5 else "")
        )
        if (counts == 0).any():
            print(f"  WARNING: {task} never predicts class(es) {np.flatnonzero(counts == 0).tolist()}")
    print("  OK: ready to upload")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", help="short name for the files, e.g. nn_v1")
    parser.add_argument("--image", choices=FINAL_MODELS, default="neural_net")
    parser.add_argument("--sentiment", choices=FINAL_MODELS, default="neural_net")
    parser.add_argument("--image-csv", type=Path, help="use saved image predictions instead")
    parser.add_argument("--sentiment-csv", type=Path, help="use saved sentiment predictions instead")
    parser.add_argument("--no-val", action="store_true", help="skip the validation-split check")
    parser.add_argument("--verify", type=Path, metavar="CSV", help="only verify an existing file")
    args = parser.parse_args()

    if args.verify:
        print(f"Verifying {args.verify}")
        try:
            verify(pd.read_csv(args.verify))
        except (ValueError, AssertionError) as e:
            print(f"  FAILED: {e}")
            sys.exit(1)
        return
    if not args.name:
        parser.error("--name is required (unless using --verify)")

    print("Predicting")
    results, notes = {}, {}
    for task, csv, model_name in [
        ("image", args.image_csv, args.image),
        ("sentiment", args.sentiment_csv, args.sentiment),
    ]:
        ids, pred, task_notes = (
            read_predictions(task, csv) if csv else predict_task(task, model_name, not args.no_val)
        )
        results[task] = (ids, pred)
        notes.update(task_notes)
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": ids, "label": pred}).to_csv(
            PREDICTIONS_DIR / f"{args.name}_{task}.csv", index=False
        )

    print("Merging by id and verifying")
    sub = build_submission(*results["image"], *results["sentiment"])
    verify(sub)
    if "image_val_acc" in notes and "sentiment_val_acc" in notes:
        notes["mean_val_acc"] = round((notes["image_val_acc"] + notes["sentiment_val_acc"]) / 2, 4)
        print(f"  expected score from validation: {notes['mean_val_acc']:.4f}")
    filename = save_submission(sub, args.name, **notes)
    print(f"Saved submissions/{filename}  (logged in submissions/log.csv)")


if __name__ == "__main__":
    main()
