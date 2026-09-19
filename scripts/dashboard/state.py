"""Resident application state loaded once at startup and shared read-only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class AppState:
    accidents_df: pd.DataFrame
    vd_records_df: pd.DataFrame
    vd_dim_df: pd.DataFrame
    samples: object | None                   # AccidentSamples | None
    heatmap_png_index: dict[str, Path]       # accident_id -> PNG path
    sample_id_index: dict[str, int]          # accident_id -> row index in samples.matrix
    route_index: dict[tuple[str, str], pd.DataFrame]
    checkpoints: dict[str, Path | None]
    data_root: Path
    models_root: Path
