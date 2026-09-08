# Final QA

Overall status: **PASS**

- [x] All 25 base artifacts have matching protocol hashes
- [x] Each outer context contains 40 unique patients
- [x] Every outer context has 32 inner-OOF and 8 outer-test patients
- [x] Composition vectors sum to one
- [x] Fixed subtype ordering contains 27 classes
- [x] Fixed lineage ordering contains five classes
- [x] Patient disease is unique
- [x] Disease predictions contain 40 outer-test patients per representation
- [x] Every disease representation has five reported outer folds
- [x] Technical control contains only log cell count
- [x] Equal-cell sensitivity used 50 seeds per finite budget and representation
- [x] Patient-label permutations reached the requested count
- [x] Global multinomial control was evaluated
- [x] Fold-ID predictability was evaluated
- [x] Global multinomial null contains the chance reference
- [x] Fold-ID predictability is not substantially above chance

## Scope Notes

- Disease metrics, permutations, and bootstrap analyses use patients, not cells, as the statistical unit.
- XGBoost receives subtype labels only; disease labels enter only the downstream logistic regression.
- Equal-cell sensitivity samples without replacement within each patient.
- The formal run used 10,000 patient-label permutations.
- Neural subtype predictors were not rerun; they are optional and do not delay the disease-label-free primary analysis.
