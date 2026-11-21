"""Blended / adjusted submissions (k-NN + neural net), trained on all labelled rows.

    uv run python scripts/make_blend_submission.py --name blend_v1
    uv run python scripts/make_blend_submission.py --name blend_bal_v1 --balance-image
    uv run python scripts/make_blend_submission.py --name blend_bal_prior_v1 --balance-image --prior-correct-sentiment

Image: k-NN class scores (sqrt + standardised features, cosine similarity,
mean of the 3 most similar training images of each class) are turned into
probabilities and blended with the tuned neural network's probabilities
(80% k-NN / 20% NN, chosen on the validation split: 56.5% -> 57.7%).

--balance-image: the brief states the image test set has exactly 100 images
per class. Assign labels so each class gets exactly 100, maximising the total
log-probability (Hungarian algorithm). Uses no test labels, only the stated
class counts. Validation (80 per class): 57.7% -> 59.0%.

--prior-correct-sentiment: estimate the sentiment test set's class mix from the
network's own probabilities on the unlabelled test features (EM; Saerens et
al., 2002) and re-weight the probabilities by new_prior / train_prior. On
validation subsets with a simulated 48%-neutral mix it recovered the mix and
added 2.5-3 points; on the real test set it cannot be checked offline.

Both adjustments use statistics of the whole (unlabelled) test set; disclose
them in the report if a submission using them is selected.
"""

import argparse

import numpy as np
from scipy.optimize import linear_sum_assignment

from comp472_project.config import SEED, seed_everything
from comp472_project.data import load_split, load_test, load_train
from comp472_project.models import NeuralNetClassifier
from comp472_project.models.neural_net import TUNED_CONFIGS
from comp472_project.preprocessing import SqrtTransform, Standardizer
from comp472_project.submission import build_submission, save_submission

KNN_TOP = 3          # class score = mean of the 3 most similar training images of that class
KNN_TEMPERATURE = 0.02
NN_WEIGHT = 0.2


def _unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _nn(task):
    cfg = {k: v for k, v in TUNED_CONFIGS[task].items() if k != "preprocessing"}
    return NeuralNetClassifier(**cfg, seed=SEED)


def image_proba(X_train, y_train, X_eval):
    prep = [SqrtTransform(), Standardizer()]
    for t in prep:
        X_train = t.fit_transform(X_train, y_train)
        X_eval = t.transform(X_eval)
    classes = np.unique(y_train)
    sim = _unit(X_eval) @ _unit(X_train).T
    score = np.stack(
        [np.sort(sim[:, y_train == c], axis=1)[:, -KNN_TOP:].mean(1) for c in classes], axis=1
    )
    p_knn = np.exp((score - score.max(1, keepdims=True)) / KNN_TEMPERATURE)
    p_knn /= p_knn.sum(1, keepdims=True)
    seed_everything()
    p_nn = _nn("image").fit(X_train, y_train).predict_proba(X_eval)
    return classes, (1 - NN_WEIGHT) * p_knn + NN_WEIGHT * p_nn


def balanced_labels(proba, classes, per_class):
    """Exactly `per_class` predictions per class, maximising total log-probability."""
    cost = -np.repeat(np.log(proba + 1e-12), per_class, axis=1)
    rows, cols = linear_sum_assignment(cost)
    out = np.empty(len(proba), dtype=int)
    out[rows] = classes[cols // per_class]
    return out


def sentiment_proba(X_train, y_train, X_eval):
    s = Standardizer().fit(X_train)
    seed_everything()
    model = _nn("sentiment").fit(s.transform(X_train), y_train)
    return model.classes_, model.predict_proba(s.transform(X_eval))


def em_prior_correction(proba, train_prior, iters=200):
    prior = train_prior.copy()
    for _ in range(iters):
        q = proba * prior / train_prior
        q /= q.sum(1, keepdims=True)
        prior = q.mean(0)
    return prior, q


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--balance-image", action="store_true")
    ap.add_argument("--prior-correct-sentiment", action="store_true")
    ap.add_argument("--no-val", action="store_true")
    args = ap.parse_args()
    notes = {"model": "knn+nn blend (image), nn (sentiment)", "balance_image": args.balance_image,
             "prior_correct_sentiment": args.prior_correct_sentiment}

    if not args.no_val:
        d = load_split("image")
        classes, p = image_proba(d.X_train, d.y_train, d.X_val)
        pred = (balanced_labels(p, classes, np.bincount(d.y_val).max())
                if args.balance_image else classes[p.argmax(1)])
        notes["image_val_acc"] = round(float(np.mean(pred == d.y_val)), 4)
        d = load_split("sentiment")
        classes, p = sentiment_proba(d.X_train, d.y_train, d.X_val)
        notes["sentiment_val_acc"] = round(float(np.mean(classes[p.argmax(1)] == d.y_val)), 4)
        notes["mean_val_acc"] = round((notes["image_val_acc"] + notes["sentiment_val_acc"]) / 2, 4)
        print(f"validation: image {notes['image_val_acc']:.4f}  sentiment {notes['sentiment_val_acc']:.4f}"
              f"  mean {notes['mean_val_acc']:.4f}")

    _, X, y = load_train("image")
    img_ids, X_test = load_test("image")
    classes, p = image_proba(X, y, X_test)
    img_pred = balanced_labels(p, classes, len(X_test) // len(classes)) if args.balance_image else classes[p.argmax(1)]

    _, X, y = load_train("sentiment")
    txt_ids, X_test = load_test("sentiment")
    classes, p = sentiment_proba(X, y, X_test)
    if args.prior_correct_sentiment:
        train_prior = np.bincount(y, minlength=len(classes)) / len(y)
        prior, p = em_prior_correction(p, train_prior)
        notes["sentiment_test_prior_est"] = " / ".join(f"{v:.3f}" for v in prior)
        print("estimated sentiment test mix (neg/neu/pos):", prior.round(3))
    txt_pred = classes[p.argmax(1)]
    print("sentiment predicted mix:", (np.bincount(txt_pred, minlength=3) / len(txt_pred)).round(3))

    sub = build_submission(img_ids, img_pred, txt_ids, txt_pred)  # validates all 10,000 rows
    print(f"saved submissions/{save_submission(sub, args.name, **notes)}")


if __name__ == "__main__":
    main()
