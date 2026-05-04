"""
imbalance_handler.py
====================
Modular Preprocessing – SMOTE Class-Imbalance Handler

This script wraps SMOTE (Synthetic Minority Over-sampling Technique) in a
clean, reusable interface designed for credit-default modelling where the
"Default" minority class is ≈14.8 % of the data.

Usage
-----
    from imbalance_handler import SmoteResampler

    resampler = SmoteResampler(target_col="Default", strategy="auto")
    X_resampled, y_resampled = resampler.fit_resample(train_df)
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE


# ---------------------------------------------------------------------------
# Helper – pretty class-distribution summary
# ---------------------------------------------------------------------------
def _class_summary(y: pd.Series, label: str = "") -> str:
    counts = y.value_counts().sort_index()
    total = counts.sum()
    lines = [f"  {label} Class Distribution (n={total}):"]
    for cls, cnt in counts.items():
        pct = cnt / total * 100
        lines.append(f"    Class {cls}: {cnt:>7,} ({pct:5.1f}%)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------
class SmoteResampler:
    """Apply SMOTE to balance a binary target variable.

    Parameters
    ----------
    target_col : str, default "Default"
        Name of the binary target column.
    strategy : float | str, default "auto"
        ``sampling_strategy`` passed to SMOTE.
        - ``"auto"`` → minority is resampled to match majority.
        - A float like ``0.5`` → minority / majority ratio after resampling.
    k_neighbors : int, default 5
        Number of nearest neighbours used by SMOTE to synthesise samples.
    random_state : int, default 42
        Seed for reproducibility.
    verbose : bool, default True
        Print before / after class distributions.
    """

    def __init__(
        self,
        target_col: str = "Default",
        strategy: Union[float, str] = "auto",
        k_neighbors: int = 5,
        random_state: int = 42,
        verbose: bool = True,
    ) -> None:
        self.target_col = target_col
        self.strategy = strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.verbose = verbose

        self._smote: Optional[SMOTE] = None
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit_resample(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features / target, apply SMOTE, return rebalanced data.

        Parameters
        ----------
        df : pd.DataFrame
            Full training data **including** the target column.

        Returns
        -------
        X_resampled : pd.DataFrame
            Feature matrix after oversampling.
        y_resampled : pd.Series
            Target vector after oversampling.
        """
        if self.target_col not in df.columns:
            raise KeyError(
                f"Target column '{self.target_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        self._feature_names = list(X.columns)

        if self.verbose:
            print("=" * 60)
            print("SMOTE Resampling Report")
            print("=" * 60)
            print(_class_summary(y, "BEFORE"))

        self._smote = SMOTE(
            sampling_strategy=self.strategy,
            k_neighbors=self.k_neighbors,
            random_state=self.random_state,
        )

        X_res, y_res = self._smote.fit_resample(X, y)

        # Convert back to DataFrame / Series for convenience
        X_resampled = pd.DataFrame(X_res, columns=self._feature_names)
        y_resampled = pd.Series(y_res, name=self.target_col)

        if self.verbose:
            print(_class_summary(y_resampled, "AFTER "))
            print("=" * 60)

        return X_resampled, y_resampled

    def fit_resample_combined(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Same as ``fit_resample`` but returns a single combined DataFrame.

        Useful when the downstream pipeline expects a single table.
        """
        X_resampled, y_resampled = self.fit_resample(df)
        return pd.concat([X_resampled, y_resampled], axis=1)


# ---------------------------------------------------------------------------
# Quick demo / self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    n = 5000

    # Simulate an imbalanced dataset (~14.8 % Default rate)
    default_rate = 0.148
    n_default = int(n * default_rate)
    n_non_default = n - n_default

    features = pd.DataFrame(
        {
            "income": np.concatenate(
                [
                    np.random.normal(55_000, 12_000, n_non_default),
                    np.random.normal(38_000, 10_000, n_default),
                ]
            ),
            "loan_amount": np.concatenate(
                [
                    np.random.normal(18_000, 6_000, n_non_default),
                    np.random.normal(25_000, 7_000, n_default),
                ]
            ),
            "credit_score": np.concatenate(
                [
                    np.random.normal(700, 50, n_non_default),
                    np.random.normal(600, 60, n_default),
                ]
            ),
        }
    )
    target = pd.Series(
        [0] * n_non_default + [1] * n_default, name="Default"
    )
    demo_df = pd.concat([features, target], axis=1)

    print(f"Original shape: {demo_df.shape}")
    resampler = SmoteResampler(target_col="Default", strategy="auto")
    X_res, y_res = resampler.fit_resample(demo_df)
    print(f"Resampled shape: X={X_res.shape}, y={y_res.shape}")
