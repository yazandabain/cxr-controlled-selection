# Method details

## Inputs and reference availability

The sampling functions take one row per patient. A report state and a binary target are different variables: state controls eligibility, while the target controls supervision and evaluation. Blank report labels are not missing source records. Unknown outcomes must not be coded as negatives. These distinctions have to be established upstream; a numeric array alone cannot recover them.

`select_one_study_per_patient` sorts by normalized patient and study identifiers, then lexical image identifier. It selects the lowest study identifier, not necessarily the earliest timestamp. Use explicit timestamps if a chronological rule is required. Selection must precede report-state filtering to avoid selecting a patient's study based on whether its report is eligible.

## Controlled memberships

`sample_training_panels` returns explicit, random and class-matched all-state arms at supplied budgets and seeds. It uses deterministic canonical patient ordering and separate random priorities. Budgets are nested within an arm. Sampling is without replacement; overlapping membership between arms is allowed. The class-matched arm exactly matches the explicit arm's target-positive and target-negative counts at each seed and budget.

`sample_addback` constructs two class-matched pools: explicit plus uncertain, and explicit plus blank. It samples new memberships from each eligible pool; it does **not** keep every anchor patient and append extra rows. Equal size and class counts do not imply equal underlying case mix.

`cross_memberships` exchanges the positive and negative membership sources of two equal-size, class-matched parent panels. The four arms preserve class counts. `factorial_contrasts` calculates conditional source differences, their interaction and the total contrast. This is a computational factorial comparison, not randomized clinical assignment or causal identification.

## Model fitting

The logistic grid is `C = (0.01, 0.1, 1, 10)`. The optional tree grid uses maximum depths `(2, 4, 6)` with 300 trees and otherwise fixed parameters in `models.py`. All candidates see the same supplied validation population. Validation AUROC selects the candidate; ties within `1e-8` of the global maximum select the smallest candidate value. Nonconverged logistic fits fail rather than silently contributing results.

The fitting functions receive test features but no test-label argument. This prevents test-label tuning through that function interface, not through a user's external workflow. They return predictions and fitted state in memory. Do not commit those outputs when fitting restricted data.

## Metrics and paired uncertainty

`weighted_auc_ap` and `CachedMetric` use half credit for score ties in AUROC. Average precision is the noninterpolated weighted sum of precision at distinct score thresholds, not trapezoidal PR area. Tests compare weighted results with scikit-learn and explicit integer-weight replication.

`patient_weights` draws patient multiplicities and propagates each multiplicity to all that patient's rows. Use the same draw for every compared prediction array. `shared_training_weights` samples the union of two selected training panels, so shared patients retain shared multiplicity. A weighted refit can rerun model tuning, but these helpers alone do not rerun cohort construction, matching, encoders or preprocessing.

`paired_interval` takes percentiles of aligned within-replicate differences. `simultaneous_band` estimates a standardized maximum-deviation critical value across a declared family of contrasts. It is an approximate bootstrap band, not a guarantee of exact finite-sample coverage. Choose the contrast family before examining its intervals.

`contrasts` preserves the original fixed-budget group-key convention (`3072__...` and `full__...`). For other budgets, construct aligned contrasts explicitly as in the synthetic example; do not rename groups to imply a budget that was not used.

The example reports conditional, fixed-prediction patient-bootstrap intervals. It fails if a bootstrap draw loses an outcome class rather than silently dropping the draw. Small or rare-outcome cohorts need an uncertainty design suited to that problem.

## AUROC pair accounting

`PairAccounting` divides the weighted positive-negative comparison numerator into group-pair cells. Each cell's contribution uses the **global** pair denominator. Contributions sum exactly to total AUROC. Cell-specific AUROC instead uses that cell's denominator and is unavailable when the cell has no pairs. A change in contributions can reflect both pair opportunities and ranking; contribution size alone is not evidence of a group-specific mechanism.

## Boundaries

The code accepts prepared features. It neither downloads data nor constructs a linked cohort, trains an image encoder, proves feature timing or establishes an independent clinical reference. Those are prerequisites for a defensible application, not properties supplied by this package.
