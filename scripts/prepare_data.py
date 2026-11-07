"""Parse the CSVs into the .npz cache and create the train/validation splits.

    uv run python scripts/prepare_data.py              # keep existing splits
    uv run python scripts/prepare_data.py --new-split  # regenerate them

Regenerating a split changes which examples are in validation, so results
recorded before and after are no longer comparable.
"""

import argparse

import numpy as np

from comp472_project.config import SEED, TASKS, VAL_FRACTION
from comp472_project.data import create_split, load_split, load_test, split_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--new-split", action="store_true", help="overwrite existing splits")
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    for task in TASKS:
        path = split_path(task)
        if path.exists() and not args.new_split:
            status = "existing split"
        else:
            create_split(task, args.val_fraction, args.seed, overwrite=True)
            status = f"new split (val_fraction={args.val_fraction}, seed={args.seed})"

        data = load_split(task)
        test_ids, X_test = load_test(task)
        counts = np.bincount(data.y_val, minlength=data.n_classes)
        print(
            f"{task:>9}: train {data.X_train.shape}, val {data.X_val.shape}, "
            f"test {X_test.shape} | val per class {counts.min()}-{counts.max()} | {status}"
        )


if __name__ == "__main__":
    main()
