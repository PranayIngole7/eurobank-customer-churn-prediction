# Feature Engineering

The training scripts and `PredictionService` use the shared
`engineer_features` implementation in `src/feature_engineering.py`. It copies
the input frame, validates values, preserves `HasBalance` and `AgeGroup`, and
adds the following canonical columns. Compatibility aliases are retained for
older callers; the aliases are not separate model inputs.

| Canonical feature | Formula | Business interpretation | Limitations and edge cases |
|---|---|---|---|
| `BalanceToSalaryRatio` | `Balance / EstimatedSalary` | Relates a customer's current bank balance to their estimated annual salary. | Not a debt-to-income ratio and does not measure affordability or financial health. If salary is zero, the implementation returns `0.0` to avoid division by zero; that value is a computational convention, not evidence that the balance is zero. Negative, missing, non-numeric, and non-finite salary values are rejected. If the salary column is absent, the backward-compatible implementation emits `0.0`; production inference supplies the raw field. |
| `ProductDensityIndicator` | `NumOfProducts / Tenure` | A simple count of products relative to recorded customer tenure; it can highlight product breadth relative to relationship duration. | This is not a time-normalized acquisition rate unless tenure's unit and observation timing are established. It can overstate differences for short-tenure customers and does not encode product value or usage. Tenure 0 yields `0.0` rather than division by zero. Tenure must be an integer from 0 to 10; product count must be an integer from 1 to 4. Missing, non-numeric, and non-finite values in present columns are rejected. |
| `EngagementProductInteraction` | `IsActiveMember * NumOfProducts` | Combines the binary active-member flag with the number of products, retaining the product count for active members and zero for inactive members. | It assumes the active-member indicator is meaningful and comparable across customers; it does not establish that activity causes retention. The active flag must be 0 or 1 and product count an integer from 1 to 4. |
| `AgeTenureInteraction` | `Age * Tenure` | A multiplicative interaction that allows a model to represent combined age and tenure patterns. | The product is scale-dependent and does not itself imply a causal or linear business relationship. Age and tenure are validated before multiplication; tenure must be an integer from 0 to 10. |

The implementation does not use `Exited` when constructing features. The
derived inputs are deterministic transformations of customer attributes, so
they can be computed before splitting without fitting any statistics across
partitions. Learned preprocessing remains inside the estimator pipeline and is
fit on that model's training partition.

For model selection and inference, raw customer fields are transformed with
the same function and selected in the model's persisted feature order. Invalid
values inside present source columns raise explicit errors rather than being
silently imputed. The zero-salary and zero-tenure conventions are documented
above because a zero-valued derived feature can otherwise be mistaken for an
observed economic or engagement measurement.

For backwards compatibility, `engineer_features` still accepts the legacy
partial input containing only `Age` and `Balance`. If optional source columns
are absent, affected derived outputs use explicit zero defaults: missing
`EstimatedSalary` yields a zero ratio; missing `Tenure` yields zero density
and age-tenure interaction; missing `NumOfProducts` yields zero product
density; and the engagement interaction is zero if either source column is
absent. These defaults indicate unavailable inputs, not measured customer
values. The training and `PredictionService` schemas require the complete raw
feature set, so those fallbacks are not expected in model selection or normal
inference.
