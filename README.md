# ML Assignment 1: Polynomial Regression

**Ritesh Dasari · IMT2024080**

Polynomial regression for two geothermal power plant tasks. The notebooks contain the complete searches across ordinary least squares (OLS), Ridge (L2), Lasso (L1), and Elastic Net. The Python scripts reproduce validation of the selected settings, train the final models, and generate test predictions.

## Selected models

| Question | Task | Model | Degree | Alpha | Mean CV MSE | Mean CV R² |
| --- | --- | --- | --- | --- | --- | --- |
| 1 (var1) | Steam turbine optimization, six inputs | Lasso | 5 | 0.00975 | 0.310765 | 0.969196 |
| 2 (var2) | Thermal reservoir mapping, three inputs | Ridge | 12 | 2.5 | 0.251404 | 0.994409 |

These scores are independent reruns of the selected settings using five shuffled folds and seed 42. Each fold fits on 80% of the training rows and validates on the remaining 20%. Polynomial expansion and scaling are fitted inside each fold. After selection, each final pipeline is fitted on all 1,000 training rows.

The original warm-start Lasso search for question 1 gives a slightly different MSE (0.310716); the independent rerun starts each fit from scratch. Test targets are hidden, so CV scores are not test accuracy measurements.

## Repository contents

```text
notebooks/       Original Colab notebooks, with saved search results and plots
src/            Selected-model training, reusable pipelines, and inference
validation/     An optional targeted convergence and neighboring-settings check
```

- [Question 1 notebook](notebooks/Untitled0.ipynb)
- [Question 2 notebook](notebooks/Untitled1.ipynb)

## Train and predict locally

Use Python 3.12 or later in a virtual environment. From the repository root:

Create a local `data/` folder and place your downloaded assignment files inside it:

```text
IMT2024080_train_var1(in).csv
IMT2024080_test_var1(in).csv
IMT2024080_train_var2(in).csv
IMT2024080_test_var2(in).csv
sample_submission.csv
```

The datasets, submission files, and PDF report are separate assignment deliverables and are not uploaded in this code repository.

```bash
python -m venv .venv
```

Activate it on Windows with `.venv\Scripts\activate`, or on macOS/Linux with `source .venv/bin/activate`. Then run:

```bash
python -m pip install -r requirements.txt
python src/train.py
python src/predict.py
```

Training saves the fitted pipelines and fold metrics in `models/`. Inference writes `results/IMT2024080_pred_var1.csv` and `results/IMT2024080_pred_var2.csv`, each with the single header `y`, preserving test row order. Generated results and model files are ignored by Git.

For a quick final fit without repeating validation, use `python src/train.py --skip-cv`. Both scripts accept `--data-dir` and `--model-dir`; inference also accepts `--output-dir`. Paths default to this repository, even when the scripts are called from another working directory.

## Reproduce the full searches in Colab

Open either notebook in Google Colab and upload the relevant CSVs from `data/` to `/content/`. Question 1 also uses `sample_submission.csv`. Run cells in order. Google Drive and download cells are specific to Colab. The original question 1 notebook exports `IMT2024080_predictions_var1.csv`; the submitted file uses the required name `IMT2024080_pred_var1.csv`.

The notebooks preserve the original code and outputs. The large degree/penalty sweeps can take much longer than the selected-model scripts. The report documents the ranges and the rationale for refining promising regions.

## Verification

- Fresh final-model predictions match the supplied prediction files within an absolute tolerance of `1e-6`.
- Both final pipelines fit without convergence warnings; both submission files have 1,000 finite predictions.
- Question 2's best original Elastic Net setting hit its 30,000-iteration limit in all five folds. Continuing those fits to convergence produced mean CV MSE **0.253226**, still above Ridge's **0.251404**.
- Targeted neighboring degrees and penalties retained the chosen settings on the original folds. This establishes the best settings among the comparisons performed, rather than a guaranteed global optimum.

To repeat the optional local and Elastic Net checks, run `python validation/check_optimality.py`. It writes its records to `results/optimality_check.json` and does not repeat the full notebook grids.
