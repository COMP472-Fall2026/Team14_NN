"""Check a submission CSV against the competition format before uploading.

    uv run python scripts/validate_submission.py submissions/<file>.csv
"""

import argparse
import sys

import pandas as pd

from comp472_project.submission import validate_submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path")
    args = parser.parse_args()

    try:
        validate_submission(pd.read_csv(args.path))
    except ValueError as e:
        print(e)
        sys.exit(1)
    print(f"OK: {args.path}")


if __name__ == "__main__":
    main()
