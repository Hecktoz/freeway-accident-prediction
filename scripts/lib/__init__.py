"""Shared library used by data scripts, training scripts, and notebooks.

Modules:
- calendar: Taiwan day-class calendar helpers
- data_loaders: read station / profile / accident / sample artifacts
- windows: SampleWindow config and slicing helpers
- visualization: heatmap renderers for observed / historical / predicted / diff
- inference: load trained model and run predict_one_accident
- metrics: MAE / RMSE / F1 / impact-time scoring
"""

from .windows import SampleWindow, DEFAULT_WINDOW

__all__ = ["SampleWindow", "DEFAULT_WINDOW"]
