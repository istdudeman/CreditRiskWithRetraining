"""
preprocessing
=============
Modular preprocessing package for credit-risk modelling.

Modules
-------
- feature_selection : WoE transformations & IV-based feature selection
- imbalance_handler : SMOTE resampling for class-imbalanced data
- drift_monitor     : PSI & CSI calculations for model-drift detection
"""

from .feature_selection import WoETransformer
from .imbalance_handler import SmoteResampler
from .drift_monitor import DriftMonitor

__all__ = ["WoETransformer", "SmoteResampler", "DriftMonitor"]
