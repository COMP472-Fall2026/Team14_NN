import numpy as np
import pandas as pd
import pytest

from comp472_project.config import DATA_DIR
from comp472_project.submission import build_submission, validate_submission


@pytest.fixture
def ids():
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")["id"]
    img = sample[sample.str.startswith("img_")].to_numpy()
    txt = sample[sample.str.startswith("text_")].to_numpy()
    return img, txt


def test_build_submission_matches_by_id_not_order(ids):
    img, txt = ids
    img_pred, txt_pred = np.arange(len(img)) % 50, np.arange(len(txt)) % 3
    sub = build_submission(img[::-1], img_pred[::-1], txt, txt_pred)
    lookup = dict(zip(sub["id"], sub["label"]))
    assert lookup[img[0]] == img_pred[0]
    assert lookup[img[7]] == img_pred[7]
    assert len(sub) == 10_000


def test_validate_rejects_bad_submissions(ids):
    img, txt = ids
    good = build_submission(img, np.zeros(len(img), int), txt, np.zeros(len(txt), int))

    missing = good.iloc[1:]
    out_of_range = good.assign(label=np.where(good["id"].str.startswith("text_"), 3, 0))
    duplicated = pd.concat([good, good.iloc[:1]])
    wrong_cols = good.rename(columns={"label": "pred"})

    for bad in (missing, out_of_range, duplicated, wrong_cols):
        with pytest.raises(ValueError):
            validate_submission(bad)
