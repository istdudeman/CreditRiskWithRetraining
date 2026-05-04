"""
feature_selection.py
====================
Modular Preprocessing – Weight of Evidence (WoE) & Information Value (IV)

This script provides a reusable class that:
  1. Bins continuous features into quantile-based buckets.
  2. Computes WoE for every bin of every feature.
  3. Computes IV per feature to rank predictive power.
  4. Transforms the original DataFrame by replacing raw values with WoE values.
  5. Selects the top-k features by IV.

Usage
-----
    from feature_selection import WoETransformer

    woe = WoETransformer(target_col="Default", bins=10, iv_threshold=0.02)
    X_train_woe = woe.fit_transform(train_df)
    X_test_woe  = woe.transform(test_df)
    print(woe.get_iv_summary())
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constants – IV interpretation thresholds (Siddiqi, 2006)
# ---------------------------------------------------------------------------
IV_BANDS: List[Tuple[float, float, str]] = [
    (0.00, 0.02, "Not Predictive"),
    (0.02, 0.10, "Weak Predictor"),
    (0.10, 0.30, "Medium Predictor"),
    (0.30, 0.50, "Strong Predictor"),
    (0.50, float("inf"), "Suspicious / Over-predicting"),
]


def _interpret_iv(iv_value: float) -> str:
    """Return a human-readable label for an IV value."""
    for lo, hi, label in IV_BANDS:
        if lo <= iv_value < hi:
            return label
    return "Unknown"


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------
class WoETransformer:
    """Weight-of-Evidence transformer with built-in IV feature selection.

    Parameters
    ----------
    target_col : str
        Name of the binary target column (1 = event / default, 0 = non-event).
    bins : int, default 10
        Number of quantile bins for continuous features.
    iv_threshold : float, default 0.02
        Minimum IV to keep a feature (features below this are dropped on
        ``transform``).  Set to 0.0 to keep everything.
    top_k : int or None, default None
        If set, keep only the top-*k* features ranked by IV regardless of
        ``iv_threshold``.
    epsilon : float, default 0.0001
        Small constant added to event / non-event counts to avoid log(0).
    """

    def __init__(
        self,
        target_col: str = "Default",
        bins: int = 10,
        iv_threshold: float = 0.02,
        top_k: Optional[int] = None,
        epsilon: float = 0.0001,
    ) -> None:
        self.target_col = target_col
        self.bins = bins
        self.iv_threshold = iv_threshold
        self.top_k = top_k
        self.epsilon = epsilon

        # Populated after fit()
        self._woe_maps: Dict[str, pd.DataFrame] = {}  # feature -> bin WoE table
        self._iv_table: Optional[pd.DataFrame] = None
        self._selected_features: List[str] = []
        self._bin_edges: Dict[str, np.ndarray] = {}    # for continuous features
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_continuous(series: pd.Series, threshold: int = 15) -> bool:
        """Heuristic: treat as continuous if numeric and nunique > threshold."""
        import pandas.api.types as ptypes
        return ptypes.is_numeric_dtype(series) and series.nunique() > threshold

    def _bin_continuous(
        self, series: pd.Series, fit: bool = True
    ) -> pd.Series:
        """Quantile-bin a continuous feature.  On transform, reuse fitted edges."""
        col = series.name
        if fit:
            try:
                binned, edges = pd.qcut(
                    series, q=self.bins, retbins=True, duplicates="drop"
                )
            except ValueError:
                # Fallback to equal-width bins when quantile binning fails
                binned, edges = pd.cut(
                    series, bins=self.bins, retbins=True, duplicates="drop"
                )
            self._bin_edges[col] = edges
        else:
            edges = self._bin_edges[col]
            binned = pd.cut(series, bins=edges, include_lowest=True)
        return binned

    def _compute_woe_iv_for_feature(
        self,
        df: pd.DataFrame,
        feature: str,
    ) -> Tuple[pd.DataFrame, float]:
        """Compute WoE table and IV for a single (already-binned) feature."""
        grouped = df.groupby(feature, observed=False)[self.target_col].agg(["sum", "count"])
        grouped.columns = ["events", "total"]
        grouped["non_events"] = grouped["total"] - grouped["events"]

        total_events = grouped["events"].sum()
        total_non_events = grouped["non_events"].sum()

        grouped["pct_events"] = (grouped["events"] + self.epsilon) / (
            total_events + self.epsilon
        )
        grouped["pct_non_events"] = (grouped["non_events"] + self.epsilon) / (
            total_non_events + self.epsilon
        )
        grouped["woe"] = np.log(grouped["pct_non_events"] / grouped["pct_events"])
        grouped["iv_component"] = (
            grouped["pct_non_events"] - grouped["pct_events"]
        ) * grouped["woe"]

        iv = grouped["iv_component"].sum()
        grouped["feature"] = feature
        return grouped.reset_index(), iv

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "WoETransformer":
        """Fit the transformer: compute WoE maps and IV for every feature.

        Parameters
        ----------
        df : pd.DataFrame
            Training data **including** the target column.

        Returns
        -------
        self
        """
        features = [c for c in df.columns if c != self.target_col]
        iv_records = []

        for feat in features:
            series = df[feat].copy()

            # Bin continuous features
            if self._is_continuous(series):
                binned = self._bin_continuous(series, fit=True)
            else:
                binned = series

            temp = pd.DataFrame({feat: binned, self.target_col: df[self.target_col]})
            woe_table, iv_value = self._compute_woe_iv_for_feature(temp, feat)

            self._woe_maps[feat] = woe_table
            iv_records.append(
                {
                    "feature": feat,
                    "iv": iv_value,
                    "interpretation": _interpret_iv(iv_value),
                }
            )

        self._iv_table = (
            pd.DataFrame(iv_records)
            .sort_values("iv", ascending=False)
            .reset_index(drop=True)
        )

        # Select features
        mask = self._iv_table["iv"] >= self.iv_threshold
        selected = self._iv_table.loc[mask, "feature"].tolist()

        if self.top_k is not None:
            selected = selected[: self.top_k]

        self._selected_features = selected
        self._is_fitted = True
        return self

    def transform(
        self, df: pd.DataFrame, keep_target: bool = False
    ) -> pd.DataFrame:
        """Replace raw feature values with their WoE scores.

        Only the features that passed the IV filter are included.

        Parameters
        ----------
        df : pd.DataFrame
            Data to transform (may or may not contain the target column).
        keep_target : bool, default False
            If True, append the target column to the output.

        Returns
        -------
        pd.DataFrame  – WoE-encoded data with selected features only.
        """
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before .transform().")

        result = pd.DataFrame(index=df.index)

        for feat in self._selected_features:
            series = df[feat].copy()
            woe_table = self._woe_maps[feat]

            if feat in self._bin_edges:
                binned = self._bin_continuous(series, fit=False)
            else:
                binned = series

            woe_lookup = woe_table.set_index(feat)["woe"]
            mapped = binned.map(woe_lookup)
            # Convert to float to avoid Categorical setitem errors
            result[feat] = mapped.astype(float)

        # Fill any unmapped bins (unseen categories) with 0 WoE
        result.fillna(0.0, inplace=True)

        if keep_target and self.target_col in df.columns:
            result[self.target_col] = df[self.target_col].values

        return result

    def fit_transform(
        self, df: pd.DataFrame, keep_target: bool = False
    ) -> pd.DataFrame:
        """Convenience: fit + transform in one call."""
        self.fit(df)
        return self.transform(df, keep_target=keep_target)

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------
    def get_iv_summary(self) -> pd.DataFrame:
        """Return a DataFrame with IV and interpretation for every feature."""
        if self._iv_table is None:
            raise RuntimeError("Call .fit() first.")
        return self._iv_table.copy()

    def get_selected_features(self) -> List[str]:
        """Return the list of features that passed the IV filter."""
        if not self._is_fitted:
            raise RuntimeError("Call .fit() first.")
        return list(self._selected_features)

    def get_woe_map(self, feature: str) -> pd.DataFrame:
        """Return the WoE mapping table for a single feature."""
        if feature not in self._woe_maps:
            raise KeyError(f"No WoE map for feature '{feature}'.")
        return self._woe_maps[feature].copy()


# ---------------------------------------------------------------------------
# Quick demo / self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    n = 2000
    demo = pd.DataFrame(
        {
            "income": np.random.normal(50_000, 15_000, n),
            "age": np.random.randint(18, 70, n),
            "loan_amount": np.random.normal(20_000, 8_000, n),
            "employment_years": np.random.randint(0, 30, n),
            "region": np.random.choice(["A", "B", "C", "D"], n),
            "noise": np.random.rand(n),  # should have low IV
        }
    )
    # Create a target that correlates with income & loan_amount
    prob = 1 / (
        1
        + np.exp(
            -(
                -3
                + 0.00005 * demo["income"]
                - 0.00008 * demo["loan_amount"]
                + 0.01 * demo["employment_years"]
            )
        )
    )
    demo["Default"] = (np.random.rand(n) < prob).astype(int)

    woe = WoETransformer(target_col="Default", bins=10, iv_threshold=0.02)
    transformed = woe.fit_transform(demo, keep_target=True)

    print("=" * 60)
    print("IV Summary")
    print("=" * 60)
    print(woe.get_iv_summary().to_string(index=False))
    print(f"\nSelected features ({len(woe.get_selected_features())}): "
          f"{woe.get_selected_features()}")
    print(f"\nTransformed shape: {transformed.shape}")
    print(transformed.head())
