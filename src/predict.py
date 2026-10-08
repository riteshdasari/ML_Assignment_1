"""Load fitted pipelines and generate the two assignment submission CSVs."""

import argparse
from pathlib import Path

import joblib
import numpy as np

from models import ROOT, read_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for problem in (1, 2):
        artifact = joblib.load(args.model_dir / f"var{problem}.joblib")
        columns, values = read_csv(args.data_dir / f"IMT2024080_test_var{problem}(in).csv")
        if set(columns) != set(artifact["features"]):
            raise ValueError(f"Question {problem}: test columns do not match training features")
        X = values[:, [columns.index(name) for name in artifact["features"]]]
        prediction = artifact["pipeline"].predict(X)
        if prediction.shape != (len(X),) or not np.isfinite(prediction).all():
            raise ValueError(f"Question {problem}: invalid predictions")
        path = args.output_dir / f"IMT2024080_pred_var{problem}.csv"
        np.savetxt(path, prediction, delimiter=",", header="y", comments="")
        print(f"Question {problem}: wrote {len(prediction)} predictions to {path}")


if __name__ == "__main__":
    main()
