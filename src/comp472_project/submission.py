"""Build, validate, and save Kaggle submission files."""

from datetime import datetime

import numpy as np
import pandas as pd

from comp472_project.config import DATA_DIR, SUBMISSIONS_DIR, TASKS


def _sample_ids() -> pd.Series:
    return pd.read_csv(DATA_DIR / "sample_submission.csv")["id"]


def validate_submission(sub: pd.DataFrame) -> None:
    """Raise ``ValueError`` listing every problem found; return None if valid."""
    problems = []
    if list(sub.columns) != ["id", "label"]:
        problems.append(f"columns must be ['id', 'label'], got {list(sub.columns)}")
        raise ValueError("Invalid submission:\n  " + "\n  ".join(problems))

    expected = set(_sample_ids())
    ids = sub["id"]
    if ids.duplicated().any():
        problems.append(f"{ids.duplicated().sum()} duplicated ids")
    if missing := expected - set(ids):
        problems.append(f"{len(missing)} test ids missing, e.g. {sorted(missing)[:3]}")
    if extra := set(ids) - expected:
        problems.append(f"{len(extra)} unknown ids, e.g. {sorted(extra)[:3]}")

    labels = pd.to_numeric(sub["label"], errors="coerce")
    if labels.isna().any():
        problems.append(f"{labels.isna().sum()} missing or non-numeric labels")
    elif (labels % 1 != 0).any():
        problems.append("labels must be integers")
    else:
        for task, cfg in TASKS.items():
            mask = ids.str.startswith(cfg["test_id_prefix"])
            bad = ~labels[mask].between(0, cfg["n_classes"] - 1)
            if bad.any():
                problems.append(
                    f"{bad.sum()} {task} labels outside 0..{cfg['n_classes'] - 1}"
                )

    if problems:
        raise ValueError("Invalid submission:\n  " + "\n  ".join(problems))


def build_submission(
    image_ids: np.ndarray,
    image_pred: np.ndarray,
    sentiment_ids: np.ndarray,
    sentiment_pred: np.ndarray,
) -> pd.DataFrame:
    """Combine both tasks' predictions, matched to the sample file by id."""
    preds = pd.concat(
        [
            pd.DataFrame({"id": image_ids, "label": np.asarray(image_pred)}),
            pd.DataFrame({"id": sentiment_ids, "label": np.asarray(sentiment_pred)}),
        ],
        ignore_index=True,
    )
    sub = _sample_ids().to_frame().merge(preds, on="id", how="left", validate="one_to_one")
    validate_submission(sub)
    sub["label"] = sub["label"].astype(int)
    return sub


def save_submission(sub: pd.DataFrame, name: str, **notes) -> str:
    """Write ``submissions/<timestamp>_<name>.csv`` and log it.

    ``notes`` (e.g. ``model="neural_net"``, ``image_val_acc=0.91``) are recorded in
    ``submissions/log.csv`` with an empty ``public_score`` column to fill in
    after uploading. Returns the file name.
    """
    validate_submission(sub)
    created = datetime.now()
    filename = f"{created:%Y%m%d-%H%M%S}_{name}.csv"
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub.to_csv(SUBMISSIONS_DIR / filename, index=False)

    log_path = SUBMISSIONS_DIR / "log.csv"
    row = pd.DataFrame([{"file": filename, "created": f"{created:%Y-%m-%d %H:%M}", **notes, "public_score": ""}])
    log = pd.concat([pd.read_csv(log_path), row], ignore_index=True) if log_path.exists() else row
    log.to_csv(log_path, index=False)
    return filename
