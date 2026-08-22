"""XGBoost training + recursive multi-step prediction.

Sibling of forecasting.lgbm — same panel/feature engineering, same
input/output schema, different gradient-boosting library. Wired as a
peer candidate in the multi-model picker. Kept as a separate package
so per-library defaults (learning rate, n_estimators, categorical
handling) can diverge without cross-contamination.
"""
