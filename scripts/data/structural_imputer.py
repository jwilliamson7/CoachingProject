#!/usr/bin/env python
"""
Constant fill for structurally missing feature values.

Every missing value in data/master_data.csv has a structural cause recorded by an
indicator column (cf_no_unit_record, cf_no_drive_record, rf_no_trend, org_expansion,
hire_no_roster_data, hire_no_retention_data; see CLAUDE.md "Data policies"): the value
does not exist (a coach who never led an NFL unit has no unit statistics), it is not
unobserved. So nothing is estimated here. Missing cells are set to the column median of
the TRAINING partition; with the indicator modeled alongside the feature, the constant
carries no information and the feature's coefficient comes only from rows that have it.
This replaces the earlier SVD matrix completion, which filled undefined values from
other features as if they were merely unobserved.

Same fit/transform interface as SVDImputer, so it drops into every split routine.
"""

import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class StructuralImputer(BaseEstimator, TransformerMixin):
    """Fill NaNs with training-partition column medians (0 if a column is all-NaN)."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN columns
            med = np.nanmedian(X, axis=0)
        self.fill_values_ = np.where(np.isnan(med), 0.0, med)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} columns, got {X.shape[1]}")
        rows, cols = np.where(np.isnan(X))
        X[rows, cols] = self.fill_values_[cols]
        return X
