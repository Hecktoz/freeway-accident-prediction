"""Load curated artifacts from the Drive data root.

All loaders accept `data_root` as a `Path` and default to `<repo>/data`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .windows import SampleWindow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
RAW_LANE_DATASET = "vd_raw_lane_1m_full"


def _data_root(data_root: Path | str | None) -> Path:
    return Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT


def station_partition_path(
    data_root: Path | str | None,
    dataset: str,
    date_str: str,
) -> Path:
    ts = pd.Timestamp(date_str)
    return _data_root(data_root) / dataset / f"source_year={ts:%Y}" / f"source_month={ts:%m}" / f"vd_station_{ts:%Y%m%d}.parquet"


def raw_lane_partition_path(
    date_str: str | pd.Timestamp,
    data_root: Path | str | None = None,
) -> Path:
    """Return the raw lane-level VD parquet path for one source date."""
    ts = pd.Timestamp(date_str)
    return _data_root(data_root) / RAW_LANE_DATASET / f"source_year={ts:%Y}" / f"source_month={ts:%m}" / f"vd_{ts:%Y%m%d}.parquet"


def raw_lane_paths_for_window(
    center_time: str | pd.Timestamp,
    before_min: int,
    after_min: int,
    data_root: Path | str | None = None,
    require_exists: bool = True,
) -> list[Path]:
    """Return daily raw parquet paths needed by a timestamp-centered window."""
    center = pd.Timestamp(center_time)
    start = center - pd.Timedelta(minutes=int(before_min))
    end = center + pd.Timedelta(minutes=int(after_min))
    dates = pd.date_range(start.normalize(), end.normalize(), freq="D")
    paths = [raw_lane_partition_path(ts, data_root=data_root) for ts in dates]
    if require_exists:
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(missing[0])
    return paths


def read_station_day(
    date_str: str,
    resolution_min: int = 5,
    data_root: Path | str | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read one day of station-level VD data."""
    if resolution_min not in (1, 5):
        raise ValueError("resolution_min must be 1 or 5")
    dataset = f"vd_station_{resolution_min}m"
    path = station_partition_path(data_root, dataset, date_str)
    if not path.exists():
        raise FileNotFoundError(path)
    return pq.read_table(str(path), columns=columns).to_pandas()


def read_station_dates(
    dates: Iterable[str],
    resolution_min: int = 5,
    data_root: Path | str | None = None,
    columns: list[str] | None = None,
    skip_missing: bool = True,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for d in dates:
        try:
            parts.append(read_station_day(d, resolution_min=resolution_min, data_root=data_root, columns=columns))
        except FileNotFoundError:
            if not skip_missing:
                raise
    if not parts:
        return pd.DataFrame(columns=columns or [])
    return pd.concat(parts, ignore_index=True)


def read_station_dataset(
    resolution_min: int = 5,
    data_root: Path | str | None = None,
    filters: list | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read the entire station partitioned dataset with optional pyarrow filters."""
    if resolution_min not in (1, 5):
        raise ValueError("resolution_min must be 1 or 5")
    base = _data_root(data_root) / f"vd_station_{resolution_min}m"
    if not base.exists():
        raise FileNotFoundError(base)
    dataset = ds.dataset(str(base), format="parquet", partitioning="hive")
    return dataset.to_table(columns=columns, filter=_filter_expression(dataset, filters)).to_pandas()


def _filter_expression(dataset: ds.Dataset, filters):
    if not filters:
        return None
    return ds.dataset(dataset.files, format="parquet", partitioning="hive").scanner(filter=filters).filter


def read_vd_dim(data_root: Path | str | None = None) -> pd.DataFrame:
    path = _data_root(data_root) / "metadata" / "vd_dim.parquet"
    return pq.read_table(str(path)).to_pandas()


def read_accidents(
    data_root: Path | str | None = None,
    accident_csv: Path | str | None = None,
    freeway: str | None = None,
) -> pd.DataFrame:
    """Load accident_data.csv and normalize core fields.

    Returns columns:
    - `accident_id` (str)
    - `freeway` (str, e.g. N1)
    - `direction` (str, in {N,S,E,W})
    - `mileage_km` (float)
    - `mileage_m` (int)
    - `report_time` (Timestamp)
    - `arrived_time`, `completed_time`, `cleared_time` (Timestamp or NaT)
    """
    path = Path(accident_csv) if accident_csv else _data_root(data_root) / "accident_data.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    for col in ("發送", "到達", "處理完成", "開始清理"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df[df["發送"].notna()].copy()
    df["accident_id"] = df["任務編號"].astype(str)

    dir_map = {"北": "N", "南": "S", "東": "E", "西": "W", "N": "N", "S": "S", "E": "E", "W": "W"}
    df["direction"] = df["方向"].map(lambda x: dir_map.get(str(x).strip(), str(x).strip()) if pd.notna(x) else "")
    df["freeway"] = df["國道"].astype(str)

    df["mileage_km"] = pd.to_numeric(df["里程數"], errors="coerce")
    df["mileage_m"] = (df["mileage_km"] * 1000).round().astype("Int64")

    df["report_time"] = df["發送"]
    df["arrived_time"] = df.get("到達")
    df["completed_time"] = df.get("處理完成")
    df["cleared_time"] = df.get("開始清理")

    if freeway:
        df = df[df["freeway"] == freeway].copy()

    keep = [
        "accident_id", "freeway", "direction", "mileage_km", "mileage_m",
        "report_time", "arrived_time", "completed_time", "cleared_time",
    ]
    extras = [c for c in df.columns if c not in keep and not c.startswith("Unnamed")]
    return df[keep + extras].reset_index(drop=True)


def read_historical_profile(
    resolution_min: int = 5,
    data_root: Path | str | None = None,
    freeway: str | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read the historical-profile dataset, optionally filtered by freeway."""
    base = _data_root(data_root) / f"historical_speed_profile_{resolution_min}m"
    if not base.exists():
        raise FileNotFoundError(base)
    if freeway is not None:
        freeway_dir = base / f"freeway={freeway}"
        if freeway_dir.exists():
            files = sorted(freeway_dir.glob("*.parquet"))
            if not files:
                raise FileNotFoundError(freeway_dir)
            dataset = ds.dataset(files, format="parquet")
            return dataset.to_table(columns=columns).to_pandas()
    parquet_files = sorted(base.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(base)
    dataset = ds.dataset(parquet_files, format="parquet", partitioning="hive")
    table = dataset.to_table(columns=columns).to_pandas()
    if freeway is not None:
        if "freeway" not in table.columns:
            raise FileNotFoundError(base / f"freeway={freeway}")
        table = table[table["freeway"] == freeway].drop(columns=["freeway"], errors="ignore").copy()
    return table


@dataclass
class AccidentSamples:
    matrix: np.ndarray              # shape: (n_accidents, n_space, n_time), float32
    accident_ids: np.ndarray        # str array, shape (n_accidents,)
    space_offsets_m: np.ndarray     # shape: (n_space,), int
    time_offsets_min: np.ndarray    # shape: (n_time,), int
    metadata: pd.DataFrame          # one row per accident
    window: SampleWindow

    def __len__(self) -> int:
        return self.matrix.shape[0]


def load_samples(
    data_root: Path | str | None = None,
    resolution_min: int = 5,
) -> AccidentSamples:
    """Load the accident samples NPZ plus its metadata parquet."""
    root = _data_root(data_root)
    npz_path = root / f"accident_samples_{resolution_min}m.npz"
    meta_path = root / f"accident_samples_{resolution_min}m_metadata.parquet"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    arr = np.load(str(npz_path), allow_pickle=False)
    metadata = pq.read_table(str(meta_path)).to_pandas()
    window_keys = ("upstream_km", "downstream_km", "time_before_min", "time_after_min", "spatial_resolution_km", "resolution_min")
    window = SampleWindow(**{k: int(arr[f"window_{k}"]) for k in window_keys})

    return AccidentSamples(
        matrix=arr["matrix"].astype(np.float32),
        accident_ids=arr["accident_ids"],
        space_offsets_m=arr["space_offsets_m"].astype(np.int32),
        time_offsets_min=arr["time_offsets_min"].astype(np.int32),
        metadata=metadata,
        window=window,
    )


def duckdb_connection(data_root: Path | str | None = None, register_raw: bool = False):
    """Open a DuckDB connection with project parquet datasets registered as views.

    Registered views (only when the underlying files exist):
    - `raw_lane_1m`                         - hive-partitioned raw lane parquet, only when `register_raw=True`
    - `station_1m`, `station_5m`           - hive-partitioned station parquet
    - `profile_5m`                          - hive-partitioned historical profile parquet
    - `vd_dim`                              - VD dimension table
    - `accidents_raw`                       - raw `accident_data.csv`
    - `accident_samples_metadata_5m`        - accident samples metadata parquet

    Use this for ad-hoc SQL exploration in notebooks and for aggregation-heavy
    work (groupby with percentiles, large GROUP BY scans). For per-day or
    per-accident streaming, prefer pandas. See `docs/COLAB_MIGRATION.md` ->
    Tooling Conventions.
    """
    import duckdb

    root = _data_root(data_root)
    con = duckdb.connect()

    def _glob(p: Path) -> str:
        return p.as_posix() + "/**/*.parquet"

    if register_raw and (root / RAW_LANE_DATASET).exists():
        con.execute(f"CREATE VIEW raw_lane_1m AS SELECT * FROM read_parquet('{_glob(root / RAW_LANE_DATASET)}', hive_partitioning=true)")
    if (root / "vd_station_1m").exists():
        con.execute(f"CREATE VIEW station_1m AS SELECT * FROM read_parquet('{_glob(root / 'vd_station_1m')}', hive_partitioning=true)")
    if (root / "vd_station_5m").exists():
        con.execute(f"CREATE VIEW station_5m AS SELECT * FROM read_parquet('{_glob(root / 'vd_station_5m')}', hive_partitioning=true)")
    if (root / "historical_speed_profile_5m").exists():
        con.execute(f"CREATE VIEW profile_5m AS SELECT * FROM read_parquet('{_glob(root / 'historical_speed_profile_5m')}', hive_partitioning=true)")
    vd_dim_path = root / "metadata" / "vd_dim.parquet"
    if vd_dim_path.exists():
        con.execute(f"CREATE VIEW vd_dim AS SELECT * FROM '{vd_dim_path.as_posix()}'")
    accident_csv = root / "accident_data.csv"
    if accident_csv.exists():
        con.execute(f"CREATE VIEW accidents_raw AS SELECT * FROM read_csv_auto('{accident_csv.as_posix()}', header=true, ignore_errors=true)")
    samples_meta = root / "accident_samples_5m_metadata.parquet"
    if samples_meta.exists():
        con.execute(f"CREATE VIEW accident_samples_metadata_5m AS SELECT * FROM '{samples_meta.as_posix()}'")
    return con


def load_accident_csv(
    accident_id: str,
    data_root: Path | str | None = None,
    resolution_min: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """Load a single per-accident debug CSV and its metadata sidecar."""
    import json
    root = _data_root(data_root)
    csv_dir = root / f"accident_samples_csv_{resolution_min}m"
    csv_path = csv_dir / f"{accident_id}.csv"
    meta_path = csv_dir / f"{accident_id}.json"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    matrix_df = pd.read_csv(csv_path, index_col=0)
    matrix_df.columns = matrix_df.columns.astype(int)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return matrix_df, meta
