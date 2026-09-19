"""Startup eager loaders — called once, results stored in AppState.

No Streamlit; all caching is explicit (module-level singletons or lru_cache).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from pyarrow.lib import ArrowInvalid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_MODELS_ROOT = PROJECT_ROOT / "output" / "models"
DEFAULT_VDXML = DEFAULT_DATA_ROOT / "metadata" / "vd_static.xml"
DEFAULT_HEATMAP_DIR = PROJECT_ROOT / "output" / "accident_heatmaps_5m"

PROFILE_REQUIRED_COLUMNS = {
    "vdid", "direction", "mileage_m", "day_class",
    "peak_class", "minute_bucket", "mean_speed_x10",
}


def _t(label: str, t0: float) -> None:
    print(f"  {label}: {time.perf_counter() - t0:.2f}s", flush=True)


def load_vd_records(xml_path: Path = DEFAULT_VDXML) -> list[dict[str, Any]]:
    from scripts.visualization.build_vd_station_map import parse_vd_xml
    records = parse_vd_xml(xml_path)
    return [r for r in records if not (r["lat"] == 0 and r["lon"] == 0)]


def load_vd_records_df(xml_path: Path = DEFAULT_VDXML) -> pd.DataFrame:
    return pd.DataFrame(load_vd_records(xml_path))


def load_vd_dim(data_root: Path = DEFAULT_DATA_ROOT) -> pd.DataFrame:
    from scripts.lib.data_loaders import read_vd_dim
    return read_vd_dim(data_root)


def load_accidents(data_root: Path = DEFAULT_DATA_ROOT) -> pd.DataFrame:
    from scripts.lib.data_loaders import read_accidents
    df = read_accidents(data_root)
    df = df[df["report_time"].notna() & df["mileage_km"].notna()].copy()
    df = df[df["direction"].isin(["N", "S", "E", "W"])]
    df = df[df["freeway"].isin(["N1", "N3", "N4", "N6"])]
    df["report_date"] = df["report_time"].dt.date
    # Compute processing duration in minutes
    if "completed_time" in df.columns and "report_time" in df.columns:
        dur = (df["completed_time"] - df["report_time"]).dt.total_seconds() / 60.0
        df["processing_min"] = dur.round(0).astype("Int64")
    else:
        df["processing_min"] = pd.NA
    return df.reset_index(drop=True)


def load_samples(data_root: Path = DEFAULT_DATA_ROOT):
    """Load NPZ + metadata parquet; returns AccidentSamples or None."""
    from scripts.lib.data_loaders import load_samples as _load
    try:
        return _load(data_root, resolution_min=5)
    except FileNotFoundError:
        return None


def build_heatmap_png_index(heatmap_dir: Path = DEFAULT_HEATMAP_DIR) -> dict[str, Path]:
    """Scan the heatmap PNG directory and build accident_id -> Path."""
    if not heatmap_dir.exists():
        return {}
    return {p.stem: p for p in heatmap_dir.iterdir() if p.suffix == ".png"}


def build_sample_id_index(samples) -> dict[str, int]:
    """Map accident_id string -> row index in samples.matrix."""
    if samples is None:
        return {}
    return {str(aid): i for i, aid in enumerate(samples.accident_ids)}


def load_all(
    data_root: Path | None = None,
    models_root: Path | None = None,
    xml_path: Path | None = None,
    heatmap_dir: Path | None = None,
) -> "AppState":
    """Load all resident data and return AppState. Prints timing."""
    from scripts.dashboard.state import AppState
    from scripts.dashboard import model_runtime
    from scripts.dashboard import geo

    dr = data_root or DEFAULT_DATA_ROOT
    mr = models_root or DEFAULT_MODELS_ROOT
    xp = xml_path or DEFAULT_VDXML
    hd = heatmap_dir or DEFAULT_HEATMAP_DIR

    print("Dashboard startup — loading resident data …", flush=True)
    t0 = time.perf_counter()

    t = time.perf_counter()
    accidents_df = load_accidents(dr)
    _t(f"accidents ({len(accidents_df):,} rows)", t)

    t = time.perf_counter()
    vd_records_df = load_vd_records_df(xp)
    _t(f"vd_records ({len(vd_records_df):,} rows)", t)

    t = time.perf_counter()
    vd_dim_df = load_vd_dim(dr)
    _t(f"vd_dim ({len(vd_dim_df):,} rows)", t)

    t = time.perf_counter()
    samples = load_samples(dr)
    if samples is None:
        print("  [warn] accident_samples_5m.npz not found — NPZ heatmap path disabled", flush=True)
    else:
        _t(f"samples ({len(samples):,} accidents, matrix {samples.matrix.shape})", t)

    t = time.perf_counter()
    heatmap_png_index = build_heatmap_png_index(hd)
    _t(f"heatmap_png_index ({len(heatmap_png_index):,} PNGs)", t)

    sample_id_index = build_sample_id_index(samples)

    t = time.perf_counter()
    vd_dim_vdids = tuple(vd_dim_df["vdid"].astype(str).tolist()) if not vd_dim_df.empty else ()
    route_index = geo.build_route_index(vd_records_df, vd_dim_vdids)
    _t(f"route_index ({len(route_index)} routes)", t)

    t = time.perf_counter()
    checkpoints = model_runtime.discover_checkpoints(mr)
    _t("checkpoints", t)

    total = time.perf_counter() - t0
    print(f"Startup complete in {total:.2f}s", flush=True)

    return AppState(
        accidents_df=accidents_df,
        vd_records_df=vd_records_df,
        vd_dim_df=vd_dim_df,
        samples=samples,
        heatmap_png_index=heatmap_png_index,
        sample_id_index=sample_id_index,
        route_index=route_index,
        checkpoints=checkpoints,
        data_root=dr,
        models_root=mr,
    )


def latest_station_date(vd_dim: pd.DataFrame) -> str | None:
    if vd_dim.empty or "last_source_date" not in vd_dim.columns:
        return None
    last = pd.to_datetime(vd_dim["last_source_date"]).max()
    if pd.isna(last):
        return None
    return last.strftime("%Y-%m-%d")
