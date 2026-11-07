# COMP 472 Project

Comparing k-Nearest Neighbours, Decision Trees, Random Forests and a feed-forward Neural Network on two kinds of pre-extracted features:

| Task | Features | Classes | Train | Test |
|---|---|---|---|---|
| Visual object recognition | 500-d image features | 50 household objects | 20,000 | 5,000 |
| Sentiment analysis (TweetEval) | 768-d mean-pooled BERT embeddings | 3 (negative / neutral / positive) | 12,000 | 5,000 |

Plus one clustering experiment on the feature space. The Kaggle score is
`(image accuracy + sentiment accuracy) / 2`. See
[`proj_description.md`](proj_description.md) and
[`data/data_description.md`](data/data_description.md) for the full brief.

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).
uv installs the right Python version (3.13) automatically.

1. Clone the repo:

```bash
git clone https://github.com/kcmnd/comp472_project.git && cd comp472_project
```

2. The competition CSVs are too large for GitHub, so they are not in the repo. Download them from the [Kaggle competition data page](https://www.kaggle.com/competitions/comp-472-project-fall-2026-concordia/data) and put these four files directly in `data/`:

```
data/image_train.csv
data/image_test.csv
data/sentiment_train.csv
data/sentiment_test.csv
```

3. Install and prepare:

```bash
uv sync                                  # create .venv, install deps + this package (editable)
uv run python scripts/prepare_data.py    # parse CSVs into data/cache/, check splits
uv run pytest                            # sanity checks
```

`uv sync` installs the code in `src/comp472_project/` as an editable package, so `import comp472_project` works from notebooks, scripts and tests, and edits take effect without reinstalling (restart the notebook kernel to pick them up, or use `%load_ext autoreload` / `%autoreload 2`).

`prepare_data.py` takes about a second. It caches each CSV as a `.npz` file in `data/cache/` (git-ignored) so later loads are near-instant. It also creates the train/validation splits in `data/splits/` if they don't exist yet. The splits are committed, so don't regenerate them (see [Validation](#validation)).

### Notebooks

```bash
uv run jupyter lab
```

This runs Jupyter from the project environment, so the default kernel already has everything installed.

## Repository layout

```
data/                      competition CSVs (download from Kaggle, git-ignored, never modified)
  splits/                  fixed train/val split per task (committed)
  cache/                   parsed .npz cache (git-ignored, rebuilt automatically)
src/comp472_project/
  config.py                paths, SEED, VAL_FRACTION, task metadata, seed_everything()
  data.py                  load_train / load_test / load_split, split creation
  preprocessing.py         Standardizer, L2Normalizer (fit on train only)
  models/                  KNNClassifier, DecisionTreeClassifier,
                           RandomForestClassifier, NeuralNetClassifier
  clustering.py            KMeans
  evaluation.py            accuracy, confusion matrix, evaluate(), log_results()
  submission.py            build / validate / save Kaggle submissions
scripts/                   runnable entry points (prepare_data, validate_submission, ...)
notebooks/                 EDA, per-model experiments, clustering, error analysis
tests/                     pytest checks for data, preprocessing, submission, models
results/                   experiment tables (CSV) and figures/ used in the report
submissions/               every uploaded CSV + log.csv
report/                    report and slide material
```

## Usage

### Loading data

```python
from comp472_project.data import load_split, load_test, load_train

data = load_split("image")        # or "sentiment"
data.X_train, data.y_train        # 80% of image_train.csv  (16000, 500)
data.X_val, data.y_val            # held-out 20%            (4000, 500)
data.ids_val                      # ids, for error analysis

ids, X, y = load_train("image")   # all 20,000 labelled rows (final model training)
test_ids, X_test = load_test("image")
```

### Evaluating a model

Every model has the same interface (`fit(X, y)` returns self, `predict(X)` returns labels), so a single harness works for all of them on both tasks:

```python
from comp472_project.config import seed_everything
from comp472_project.evaluation import evaluate, log_results
from comp472_project.models import KNNClassifier
from comp472_project.preprocessing import Standardizer

seed_everything()
for k in [1, 3, 5, 10, 20]:
    metrics, val_pred = evaluate(
        KNNClassifier(k=k),
        load_split("sentiment"),
        transforms=[Standardizer()],   # fitted on the train split only
        train_metrics=False,
    )
    log_results({"model": "knn", "k": k, "scaling": "standard", **metrics}, "knn")
```

`evaluate` returns `val_acc`, `train_acc` (unless disabled), `fit_time_s` and `predict_time_s`, plus the validation predictions. `log_results` appends rows to `results/<name>.csv`, so tables and plots can be regenerated from saved results instead of copied by hand.

For error analysis, `evaluation.py` also has `confusion_matrix`, `per_class_accuracy` and `most_confused_pairs`. Scikit-learn metrics (`classification_report` etc.) are allowed too.

### Making a submission

Train the final models on the **full** training data, predict both test files, then:

```python
from comp472_project.submission import build_submission, save_submission

sub = build_submission(img_test_ids, img_pred, txt_test_ids, txt_pred)
save_submission(sub, "neural_net_v1", model="neural_net", image_val_acc=0.91, sentiment_val_acc=0.66)
```

`build_submission` matches predictions to `sample_submission.csv` by id (the row order differs from the test files) and validates the result. `save_submission` writes `submissions/<timestamp>_neural_net_v1.csv` and appends a row to `submissions/log.csv`. Fill in its `public_score` column after uploading. To check any CSV before uploading:

```bash
uv run python scripts/validate_submission.py submissions/<file>.csv
```

## Validation

- Each task uses a single **stratified 80/20 train/validation split** of the training data with seed `42`: 320/80 rows per class for images and 3,200/800 per class for sentiment. It is saved in `data/splits/<task>.csv` so every experiment uses the same validation rows. Regenerating it (`prepare_data.py --new-split`) makes earlier results incomparable.
- All model selection and hyperparameter tuning uses this validation split. The Kaggle public leaderboard (2,500 rows per task) is only a sanity check.
- Anything that learns from data (scalers, PCA, ...) is fitted on the train split only. Pass it via `evaluate(..., transforms=[...])`.

## Reproducibility

| | |
|---|---|
| Python | 3.13 (`.python-version`) |
| Key libraries | numpy 2.5.4, pandas 3.1.0, torch 2.15.0, scikit-learn 1.9.1, matplotlib 3.11.2 (exact versions pinned in `uv.lock`) |
| Seed | `42` (`comp472_project.config.SEED`). Call `seed_everything()` at the start of each experiment. |
| Hardware | CPU is sufficient; a GPU is used automatically for the neural network if available. |

## Tests

```bash
uv run pytest
```

`tests/test_models.py` checks that each classifier solves an easy toy
problem. Models that aren't implemented yet are skipped, so these tests start
running as each algorithm is filled in.
