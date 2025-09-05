
# engineer_some.py
# Leakage-safe feature engineering for SoMe model.
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class EngineerSoMeFeatures(BaseEstimator, TransformerMixin):
    """
    Adds log-transformed follower count, percentile clipping, and an interaction term.
    - fit(): learns per-column clipping thresholds on TRAIN ONLY.
    - transform(): applies the stored thresholds.
    Exposes get_feature_names_out for downstream introspection.
    """

    def __init__(self, input_feature_names=None, clip_percentiles=(1, 99)):
        self.input_feature_names = input_feature_names
        self.clip_percentiles = clip_percentiles
        self._to_clip = ["engagement_rate", "posting_frequency_c", "log_follower_count"]
        self._out_order = ["engagement_rate", "posting_frequency_c", "log_follower_count", "er_x_pf"]

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            X_df = X.copy()
        else:
            if self.input_feature_names is None:
                raise ValueError("Provide input_feature_names when X is a NumPy array.")
            self.feature_names_in_ = list(self.input_feature_names)
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)

        # Create log feature to learn its clipping thresholds
        if "follower_count_c" in X_df.columns:
            X_df["log_follower_count"] = np.log1p(X_df["follower_count_c"].clip(lower=0))
        else:
            X_df["log_follower_count"] = 0.0

        lo_p, hi_p = self.clip_percentiles
        self.clip_thresholds_ = {}
        for c in [col for col in self._to_clip if col in X_df.columns]:
            arr = X_df[c].to_numpy()
            lo, hi = np.percentile(arr, [lo_p, hi_p])
            # guard in case the column is constant
            if np.isclose(lo, hi):
                lo, hi = float(lo), float(hi + 1e-9)
            self.clip_thresholds_[c] = (float(lo), float(hi))

        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            for col in self.feature_names_in_:
                if col not in X_df.columns:
                    raise ValueError(f"Missing expected column: {col}")
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)

        # Derived features
        if "follower_count_c" in X_df.columns:
            X_df["log_follower_count"] = np.log1p(X_df["follower_count_c"].clip(lower=0))
        else:
            X_df["log_follower_count"] = 0.0

        # Apply stored clipping
        for c, (lo, hi) in self.clip_thresholds_.items():
            if c in X_df.columns:
                X_df[c] = X_df[c].clip(lo, hi)

        # Interaction
        if "engagement_rate" in X_df.columns and "posting_frequency_c" in X_df.columns:
            X_df["er_x_pf"] = X_df["engagement_rate"] * X_df["posting_frequency_c"]
        else:
            X_df["er_x_pf"] = 0.0

        return X_df[self._out_order].to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.array(self._out_order, dtype=object)
