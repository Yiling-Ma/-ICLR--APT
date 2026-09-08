# Strict Nested Cross-Fitting Design

For every outer fold `f`:

1. Keep the eight patients in `f` untouched.
2. For each remaining fixed fold `g`, train the fine-subtype XGBoost model on patients outside `f` and `g`, then predict all cells in `g`.
3. Concatenate the four predictions from step 2 to obtain inner-OOF subtype probabilities for all 32 outer-development patients.
4. Train one fine-subtype XGBoost model on all 32 outer-development patients and predict all cells in `f`.
5. Aggregate probabilities within patient. Train the disease model on step-3 patient features and evaluate on step-4 patient features.

Preprocessing is re-fit inside every base-model fit. Disease labels are never provided to the base subtype model. Soft lineage composition is obtained by summing the 27 fixed subtype probabilities according to the frozen subtype-to-lineage mapping.
