# Phase 3 Submission Format

Phase 3 is open-ended. Your final predictions may come from self-evolution, custom scripts, API calls, fine-tuning, ensembling, notebooks, or any other reproducible method.

Whatever method you use, Kaggle expects this CSV shape:

```csv
id,Usage,predicted_value
HIG/2004/page_140.pdf-3,Public,31.0
```

If your method already produces predictions, save a CSV or JSON file with test IDs and numeric values, then run:

```bash
python format_submission.py --predictions my_predictions.csv --output-file submission.csv
```

Accepted prediction columns or JSON keys:

- `id`
- `predicted_value`, `prediction`, `answer`, or `value`

The formatter uses `data/test.json` for the official row order and writes every required test row exactly once. Any missing, blank, null, NaN, or non-numeric prediction is replaced with `0.0`.

The self-evolution strategy path still works too:

```bash
python submit.py --strategy-path ./runs/exp_self/iter_best_strategy.json --output-file ./submission.csv
```

Before upload, check:

- `submission.csv` has exactly the columns `id,Usage,predicted_value`.
- The row count matches the Kaggle test set.
- No prediction cells are blank.
- Failed predictions are `0.0`, not empty or null.
