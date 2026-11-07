# Final Model Evaluation

## 1. Overview

This report presents the final evaluation of the selected customer churn prediction model on the held-out test dataset.

The selected model is Balanced Logistic Regression, with a classification threshold of 0.65.

The original test dataset was reserved for final evaluation and was not used for model selection, feature engineering decisions, or threshold optimization.

## 2. Model Configuration

| Property                 | Value                          |
| ------------------------ | ------------------------------ |
| Model identifier         | `logistic_regression_balanced` |
| Algorithm                | Balanced Logistic Regression   |
| Classification threshold | 0.65                           |
| Training records         | 8,000                          |
| Test records             | 2,000                          |
| Positive class           | `Exited = 1`                   |

Customers with predicted churn probabilities greater than or equal to 0.65 are classified as potential churners. Customers below this threshold are classified as retained.

## 3. Test Evaluation Metrics

| Metric            | Test Score |
| ----------------- | ---------: |
| Accuracy          |     0.8000 |
| Precision         |     0.5081 |
| Recall            |     0.5405 |
| F1-score          |     0.5238 |
| ROC-AUC           |     0.7948 |
| Average Precision |     0.5460 |

Average Precision is calculated using predicted churn probabilities rather than threshold-based predictions. It summarizes the precision-recall curve and provides an additional measure of positive-class performance.

## 4. Confusion Matrix

The following confusion matrix shows the classification results on the 2,000 held-out test records.

| Actual / Predicted | Retained (0) | Churned (1) |
| ------------------ | -----------: | ----------: |
| Retained (0)       |        1,380 |         213 |
| Churned (1)        |          187 |         220 |

The results indicate:

- True negatives: 1,380 customers correctly classified as retained.
- False positives: 213 retained customers incorrectly classified as potential churners.
- False negatives: 187 actual churners incorrectly classified as retained.
- True positives: 220 customers correctly identified as potential churners.

The model identifies 220 of the 407 actual churners in the test dataset, corresponding to a recall of approximately 54.05%.

Its precision of approximately 50.81% means that roughly half of its positive churn predictions correspond to actual churn cases.

## 5. Classification Report

| Class        | Precision | Recall | F1-score | Support |
| ------------ | --------: | -----: | -------: | ------: |
| Retained (0) |    0.8807 | 0.8663 |   0.8734 |   1,593 |
| Churned (1)  |    0.5081 | 0.5405 |   0.5238 |     407 |

The model performs better on the retained class than on the churn class.

This difference is important because overall accuracy alone does not adequately describe performance on the minority churn class.

## 6. Evaluation Methodology

The final evaluation follows this procedure:

1. The original dataset was divided using stratified 80/20 sampling.
2. The original training partition contained 8,000 records, while the held-out test partition contained 2,000 records.
3. Model selection and classification threshold optimization were performed using training and validation data.
4. The final model was fitted using the original training partition.
5. Feature preprocessing was fitted as part of the training pipeline.
6. The selected classification threshold of 0.65 was applied to the held-out test predictions.
7. Final metrics, confusion matrix, classification report, and Average Precision were calculated using the test data.

The held-out test results were not used to tune the model, threshold, or features.

## 7. Limitations

The evaluation has several limitations:

- Results represent performance on one held-out test partition.
- Performance may differ on future customer populations or production data.
- Churn recall remains approximately 54.05%, meaning a substantial proportion of actual churners are missed.
- Precision is approximately 50.81%, so positive predictions require further business evaluation before customer outreach.
- The classification threshold was selected using validation F1, not optimized against actual retention costs or financial outcomes.
- The evaluation does not establish that the model will deliver measurable business improvements.

## 8. Conclusion

The final Balanced Logistic Regression model achieves 80% accuracy and a ROC-AUC of approximately 0.7948 on the held-out test dataset.

Its churn-class F1-score is approximately 0.5238, with a recall of 0.5405 and precision of 0.5081.

The results demonstrate useful predictive signal, but also show that churn identification remains imperfect. In particular, false positives and missed churners should be considered before using predictions to support customer retention decisions.

The model should be treated as a decision-support component rather than an autonomous basis for customer actions. Further validation, calibration analysis, monitoring, and business-impact evaluation would be necessary before production use.
