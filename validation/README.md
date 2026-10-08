# Targeted validation

Run `python validation/check_optimality.py` after placing your downloaded datasets in the local `data/` folder. The script compares neighboring degrees and penalties, uses a second CV seed, and continues the original question 2 Elastic Net fits until convergence. It writes `results/optimality_check.json`.

For the selected final models, `python src/train.py` writes fresh five-fold scores and final-fit warnings to `models/validation.json`. Model comparisons use mean fold scores.

The saved notebook search outputs were not all regenerated. The final selected models and the targeted checks were independently rerun.
