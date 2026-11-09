# Customer Churn Model Evaluation

## Objective and data split

The target is `Exited` (`1` = exited, `0` = retained). The local dataset has
10,000 rows and was split with stratification into 8,000 original training
rows and 2,000 held-out test rows. The original training partition was split
again into 6,000 internal training and 2,000 validation rows.

| Partition | Rows | Use |
|---|---:|---|
| Internal training | 6,000 | Fit candidates during model/threshold selection |
| Validation | 2,000 | Compare candidates and select the model/threshold |
| Original training | 8,000 | Refit selected algorithm for final model |
| Held-out test | 2,000 | Final evaluation and post-selection group audit; not used for selection |

Candidate model comparison, threshold selection, and explanation analysis
use validation data only. After model selection, the held-out test set was
used for the final evaluation and a descriptive group audit. Neither test
analysis was used to change the selected model, threshold, or features.

## Candidate comparison and selection

The candidates are Majority-Class Dummy, Logistic Regression, Balanced
Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
Pipelines fit preprocessing only on their internal training data.

At the common comparison threshold of 0.50:

| Candidate | Accuracy | Precision | Recall | F1 | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| Majority-Class Dummy | 0.7965 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2035 |
| Logistic Regression | 0.8215 | 0.6420 | 0.2776 | 0.3877 | 0.7775 | 0.5265 |
| Balanced Logistic Regression | 0.7290 | 0.4020 | 0.6806 | 0.5055 | 0.7792 | 0.5260 |
| Decision Tree | 0.7550 | 0.4426 | 0.7862 | 0.5664 | 0.8286 | 0.6039 |
| Random Forest | 0.8245 | 0.5579 | 0.6634 | 0.6061 | 0.8498 | 0.6498 |
| Gradient Boosting | 0.8560 | 0.7767 | 0.4103 | 0.5370 | 0.8515 | 0.6699 |

The deployed candidate was selected by searching thresholds 0.10–0.90 in
0.05 increments. For each model, the highest validation F1 among thresholds
with precision at least 0.50 was selected; if none met the floor, the
maximum-F1 threshold was used. The model with the highest resulting F1 was
chosen, with ties broken by Average Precision, precision, fewer false
positives, then model name. This selected **Gradient Boosting** at **0.30**.

Validation selection metrics at that threshold were precision 0.6357, recall
0.6044, F1 0.6196, ROC-AUC 0.8515, and Average Precision 0.6699. The
validation confusion matrix was TN 1,452, FP 141, FN 161, TP 246.

## Final held-out test results

The selected algorithm was refit on all 8,000 original training rows, then
evaluated once on the 2,000 test rows at the already-selected threshold 0.30.

| Metric | Test result |
|---|---:|
| Accuracy | 0.8465 |
| Precision | 0.6244 |
| Recall | 0.6167 |
| F1-score | 0.6205 |
| ROC-AUC | 0.8615 |
| Average Precision | 0.7032 |

| Actual / predicted | Retained | Churned |
|---|---:|---:|
| Retained | 1,442 | 151 |
| Churned | 156 | 251 |

Thus, 251 of 407 churned test records were identified, while 156 were missed;
151 retained records were false alerts. These results are one held-out split,
not a guarantee of future performance.

## Threshold and operational interpretation

At the selected validation threshold, 387 of 2,000 validation records would
be flagged: 246 true positives and 141 false alerts, with 161 churners missed.
This represents a 19.35% flag rate and a 60.44% validation recall. The
precision floor is a modeling criterion, not a financial objective. Campaign
costs, customer lifetime values, treatment effectiveness, and capacity
constraints were not provided, so this report does not claim savings or ROI.

## Limitations

- The dataset's provenance, collection process, and representativeness were
  not independently verified.
- There is no prospective or external-bank validation.
- Probability calibration was not separately measured or applied.
- Validation estimates also informed candidate and threshold selection and
  should not be read as unbiased final performance estimates.
- A single held-out split does not quantify variation across resamples.
- Predictions and explanations are associations, not causal evidence.
- The original test set must not be used for further tuning.

## Artifacts

- `model-selection-validation.json`
- `model-comparison-validation.csv`
- `model-threshold-comparison-validation.csv`
- `classification-report-validation.json`
- `final-model-test-evaluation.json`
- `final-model-evaluation.md`
- Validation plots under `figures/`
