"""Validate selected settings, then fit and save models using all training rows."""

import argparse
import json
import warnings

import joblib
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from models import ROOT, SETTINGS, load_training, make_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default=str(ROOT / "data"))
    parser.add_argument("--model-dir", type=str, default=str(ROOT / "models"))
    parser.add_argument("--skip-cv", action="store_true", help="Fit final models without repeating CV")
    args = parser.parse_args()
    from pathlib import Path
    destination = Path(args.model_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report = {"scikit_learn": sklearn.__version__, "folds": 5, "seed": 42, "problems": []}
    with threadpool_limits(limits=2):
        for problem in (1, 2):
            features, X, y = load_training(args.data_dir, problem)
            record = {"problem": problem, **SETTINGS[problem], "training_rows": len(y)}
            if not args.skip_cv:
                scores = []
                for fold, (train, valid) in enumerate(KFold(5, shuffle=True, random_state=42).split(X), 1):
                    model = make_model(problem)
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter("always")
                        model.fit(X[train], y[train])
                    prediction = model.predict(X[valid])
                    scores.append({"fold": fold, "mse": float(mean_squared_error(y[valid], prediction)),
                                   "r2": float(r2_score(y[valid], prediction)),
                                   "warnings": [str(item.message) for item in captured]})
                record.update(fold_scores=scores,
                              mean_cv_mse=float(np.mean([score["mse"] for score in scores])),
                              mean_cv_r2=float(np.mean([score["r2"] for score in scores])))
            model = make_model(problem)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                model.fit(X, y)
            record["final_fit_warnings"] = [str(item.message) for item in captured]
            record["polynomial_features"] = int(model[0].n_output_features_)
            joblib.dump({"pipeline": model, "features": features, "settings": SETTINGS[problem]},
                        destination / f"var{problem}.joblib")
            report["problems"].append(record)
            print(f"Question {problem}: saved {SETTINGS[problem]['model']} (degree {SETTINGS[problem]['degree']})")
            if not args.skip_cv:
                print(f"  Mean CV MSE: {record['mean_cv_mse']:.10f}; mean CV R2: {record['mean_cv_r2']:.10f}")
    (destination / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
