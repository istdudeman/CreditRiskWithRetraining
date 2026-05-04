"""
drift_monitor.py
================
Modular Preprocessing – Population Stability Index (PSI) &
Characteristic Stability Index (CSI)

These metrics act as early-warning "senses" for model drift:
  * **PSI** measures how much the *score / prediction* distribution has
    shifted between a reference (training) population and a new population.
  * **CSI** measures how much *individual feature* distributions have
    shifted, helping pin-point the root cause of drift.

Interpretation Guide
--------------------
| PSI / CSI Value | Interpretation                    |
|-----------------|-----------------------------------|
| < 0.10          | No significant shift              |
| 0.10 – 0.25     | Moderate shift – investigate      |
| > 0.25          | Major shift – action required     |

Usage
-----
    from drift_monitor import DriftMonitor

    monitor = DriftMonitor(bins=10)
    psi = monitor.calculate_psi(train_scores, new_scores)
    csi_report = monitor.calculate_csi(train_df, new_df)
    monitor.print_report(psi, csi_report)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import requests
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Interpretation helper
# ---------------------------------------------------------------------------
def _interpret_stability(value: float) -> str:
    if value < 0.10:
        return "No significant shift"
    elif value < 0.25:
        return "Moderate shift – investigate"
    else:
        return "Major shift – action required"


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------
class DriftMonitor:
    """Calculate PSI and CSI for model-drift detection.

    Parameters
    ----------
    bins : int, default 10
        Number of equal-frequency (quantile) bins used to discretise
        continuous distributions.
    epsilon : float, default 0.0001
        Small constant to avoid division-by-zero and log(0).
    """

    def __init__(self, bins: int = 10, epsilon: float = 0.0001) -> None:
        self.bins = bins
        self.epsilon = epsilon

    # ------------------------------------------------------------------
    # PSI
    # ------------------------------------------------------------------
    def calculate_psi(
        self,
        reference: Union[np.ndarray, pd.Series],
        current: Union[np.ndarray, pd.Series],
        bin_edges: Optional[np.ndarray] = None,
    ) -> float:
        """Compute the Population Stability Index between two 1-D distributions.

        Parameters
        ----------
        reference : array-like
            Distribution from the reference (training / validation) period.
        current : array-like
            Distribution from the current (production / new) period.
        bin_edges : np.ndarray or None
            Pre-computed bin edges. If ``None``, quantile bins are derived
            from ``reference``.

        Returns
        -------
        float – PSI value.
        """
        ref = np.asarray(reference, dtype=float)
        cur = np.asarray(current, dtype=float)

        if bin_edges is None:
            bin_edges = self._quantile_edges(ref)

        ref_counts = self._bin_counts(ref, bin_edges)
        cur_counts = self._bin_counts(cur, bin_edges)

        # Convert to proportions
        ref_pct = ref_counts / ref_counts.sum()
        cur_pct = cur_counts / cur_counts.sum()

        # Add epsilon to avoid log(0)
        ref_pct = np.where(ref_pct == 0, self.epsilon, ref_pct)
        cur_pct = np.where(cur_pct == 0, self.epsilon, cur_pct)

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    # ------------------------------------------------------------------
    # CSI
    # ------------------------------------------------------------------
    def calculate_csi(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compute the Characteristic Stability Index for each feature.

        CSI is simply PSI applied per-feature rather than on the model score.

        Parameters
        ----------
        reference_df : pd.DataFrame
            Reference (training) data.
        current_df : pd.DataFrame
            Current (production / new) data.
        features : list[str] or None
            Features to evaluate.  Defaults to all shared numeric columns.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``csi``, ``interpretation``.
        """
        if features is None:
            shared = set(reference_df.columns) & set(current_df.columns)
            features = sorted(
                [
                    c
                    for c in shared
                    if pd.api.types.is_numeric_dtype(reference_df[c])
                ]
            )

        records: List[Dict[str, object]] = []
        for feat in features:
            ref_vals = reference_df[feat].dropna().values
            cur_vals = current_df[feat].dropna().values

            if len(ref_vals) == 0 or len(cur_vals) == 0:
                csi_val = np.nan
            else:
                edges = self._quantile_edges(ref_vals)
                csi_val = self.calculate_psi(ref_vals, cur_vals, bin_edges=edges)

            records.append(
                {
                    "feature": feat,
                    "csi": round(csi_val, 6) if not np.isnan(csi_val) else np.nan,
                    "interpretation": (
                        _interpret_stability(csi_val)
                        if not np.isnan(csi_val)
                        else "Insufficient data"
                    ),
                }
            )

        return (
            pd.DataFrame(records)
            .sort_values("csi", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Detailed bin-level breakdown (useful for debugging)
    # ------------------------------------------------------------------
    def psi_bin_detail(
        self,
        reference: Union[np.ndarray, pd.Series],
        current: Union[np.ndarray, pd.Series],
    ) -> pd.DataFrame:
        """Return a per-bin breakdown of PSI contributions.

        Useful for identifying *which* bins have shifted the most.
        """
        ref = np.asarray(reference, dtype=float)
        cur = np.asarray(current, dtype=float)

        bin_edges = self._quantile_edges(ref)
        ref_counts = self._bin_counts(ref, bin_edges)
        cur_counts = self._bin_counts(cur, bin_edges)

        ref_pct = ref_counts / ref_counts.sum()
        cur_pct = cur_counts / cur_counts.sum()

        ref_pct = np.where(ref_pct == 0, self.epsilon, ref_pct)
        cur_pct = np.where(cur_pct == 0, self.epsilon, cur_pct)

        psi_components = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)

        labels = []
        for i in range(len(bin_edges) - 1):
            labels.append(f"({bin_edges[i]:.4f}, {bin_edges[i+1]:.4f}]")

        return pd.DataFrame(
            {
                "bin": labels,
                "ref_pct": np.round(ref_pct, 6),
                "cur_pct": np.round(cur_pct, 6),
                "psi_component": np.round(psi_components, 6),
            }
        )

    @staticmethod
    def print_report(
        psi_value: float,
        csi_df: pd.DataFrame,
    ) -> None:
        """Print a formatted stability report to stdout."""
        print("=" * 60)
        print("Model Drift Report")
        print("=" * 60)

        print(f"\n  PSI (Score Distribution): {psi_value:.6f}")
        print(f"  Interpretation:           {_interpret_stability(psi_value)}")

        print("\n  CSI (Feature-level Stability):")
        print("  " + "-" * 56)
        header = f"  {'Feature':<25} {'CSI':>10}  {'Interpretation'}"
        print(header)
        print("  " + "-" * 56)
        for _, row in csi_df.iterrows():
            csi_str = f"{row['csi']:.6f}" if not np.isnan(row["csi"]) else "     N/A"
            print(f"  {row['feature']:<25} {csi_str:>10}  {row['interpretation']}")
        print("=" * 60)

    def trigger_retrain(
        self,
        webhook_url: str = "http://localhost:5678/webhook-test/retrain-trigger",
    ) -> None:
        """Trigger a model retraining pipeline via a webhook.
        
        This can be called when significant drift is detected
        (e.g., PSI or CSI > 0.25).
        """
        try:
            print(f"Triggering retraining webhook: {webhook_url}")
            response = requests.post(webhook_url)
            response.raise_for_status()
            print("Successfully triggered retrain.")
        except requests.exceptions.RequestException as e:
            print(f"Failed to trigger retrain: {e}")

    def _quantile_edges(self, arr: np.ndarray) -> np.ndarray:
        """Compute quantile bin edges from the reference data."""
        percentiles = np.linspace(0, 100, self.bins + 1)
        edges = np.percentile(arr, percentiles)
        edges = np.unique(edges)  # drop duplicates
        edges[0] = -np.inf
        edges[-1] = np.inf
        return edges

    def _bin_counts(self, arr: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Count observations falling into each bin."""
        counts = np.histogram(arr, bins=edges)[0].astype(float)
        return counts



if __name__ == "__main__":
    np.random.seed(42)
    n = 5000

    # --- Simulate reference and current distributions ---
    ref_scores = np.random.normal(0.35, 0.15, n)
    # Current scores have drifted slightly higher
    cur_scores = np.random.normal(0.40, 0.16, n)

    ref_df = pd.DataFrame(
        {
            "income": np.random.normal(50_000, 12_000, n),
            "loan_amount": np.random.normal(20_000, 7_000, n),
            "credit_score": np.random.normal(680, 55, n),
            "age": np.random.normal(40, 12, n),
        }
    )
    # Current data: income shifted, others stable
    cur_df = pd.DataFrame(
        {
            "income": np.random.normal(46_000, 14_000, n),  # drift!
            "loan_amount": np.random.normal(20_500, 7_200, n),
            "credit_score": np.random.normal(678, 56, n),
            "age": np.random.normal(40, 12, n),
        }
    )

    monitor = DriftMonitor(bins=10)

    psi = monitor.calculate_psi(ref_scores, cur_scores)
    csi = monitor.calculate_csi(ref_df, cur_df)

    monitor.print_report(psi, csi)

    print("\nPSI Bin Detail:")
    print(monitor.psi_bin_detail(ref_scores, cur_scores).to_string(index=False))

    # Trigger retrain if major shift detected in PSI or any CSI
    if psi > 0.25 or any(csi["csi"] > 0.25):
        print("\n[ALERT] Major drift detected! Initiating retrain process...")
        monitor.trigger_retrain()
