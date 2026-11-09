# Executive Summary for Public-Sector Stakeholders

## What was assessed

This educational prototype estimates whether a customer record is associated
with past churn (`Exited = 1`). It was tested on a locally supplied dataset
of 10,000 rows. The dataset's original source, collection process, permission
status, and representativeness have not been independently verified.

Six candidate classifiers were compared using training and validation
records. The chosen Gradient Boosting model and its 0.30 classification
threshold were selected using validation data only. The 2,000-record
held-out test set was not used for model selection; it was used for final
performance evaluation and a post-selection subgroup audit.

## Results on the held-out test sample

| Measure | Result |
|---|---:|
| Correct classifications | 84.65% |
| Precision among flagged records | 62.44% |
| Churn cases identified | 61.67% |
| F1-score | 62.05% |
| ROC-AUC | 86.15% |

On the test sample, the model identified 251 of 407 churned records, missed
156, and incorrectly flagged 151 retained records. These figures describe
one historical dataset split; they do not predict performance for another
institution, time period, or population.

## Fairness and public-interest considerations

The post-selection held-out test audit showed substantial group differences,
especially by AgeGroup and Geography. The AgeGroup equal-opportunity gap was
0.8490 and its disparate-impact ratio was 0.0119; the corresponding
Geography values were 0.1856 and 0.4700. These are descriptive screening
indicators, not proof of legal compliance, unlawful discrimination, or its
absence. The fairness audit uses the same held-out records as the performance
results above; it is not an independent second test sample and was not used
to revise the model or threshold.

## Recommended interpretation

The model may be considered only as a research prototype for prioritizing
human review. It does not show that a customer will leave, explain why a
customer left, or establish that a proposed intervention will prevent churn.
It should not be used by itself to deny services, change terms, restrict
benefits, or make another consequential customer decision.

## Conditions before any operational consideration

1. Verify dataset provenance, lawful use, measurement quality, and relevance
   to the intended population.
2. Conduct independent temporal and external validation and quantify
   uncertainty.
3. Investigate subgroup disparities with domain experts and affected
   stakeholders; assess applicable legal and policy obligations.
4. Define the real costs, benefits, review capacity, and appeal process
   before selecting an operating threshold.
5. Assess calibration if probability values will guide decisions; monitor
   performance and subgroup outcomes over time.
6. Require meaningful human oversight and prohibit solely automated
   consequential actions.

No campaign outcomes, customer lifetime values, intervention costs, or
causal evidence were available. Accordingly, no financial savings or
retention-effect claims are made.
