# Colab Migration Plan

## Goal

Run all downstream project work in Colab using Google Drive + Parquet.

The local machine is only responsible for raw VD download when necessary.

## Current Completed Step

Raw VD download is complete:

```text
data/vd_raw_lane_1m_full/
data/_vd_download_manifests/
data/metadata/vd_download_filter.json
```

## Migration Status: Complete

All canonical data layers are built and scripts run from Drive. The locked
training dataset is `dataset_20260511_5m_v2`; active work is now model
refinement / case-study visualization / conference paper preparation, not data
migration. GUI delivery is deferred to future work.
See [docs/ROADMAP.md](ROADMAP.md).

| Data layer | Status |
| --- | --- |
| `data/vd_station_1m/`, `data/vd_station_5m/` | Built; latest full station report covers 2023-08-06 – 2026-04-30, while the locked training sample range is 2024-04-01 – 2025-05-31 |
| `data/historical_speed_profile_5m/` | Built (2,385,567 rows, 4 freeways) |
| `data/accident_samples_5m.npz` | Built (2,036 accidents, 61 × 61 matrix) |
| `output/feature_engineering/incident_features_33.csv` | Built (2,043 × 33) |

## Code Repository

Source code is also versioned on GitHub for sharing. Large VD data layers stay exclusively in Drive. See `.gitignore` at the project root for the full exclusion list.

## Colab Setup

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
pip install uv
cd "/content/drive/MyDrive/陽交大高速公路計畫/main"
uv run scripts/data/build_station_vd_drive.py --start-date 2024-04-01 --end-date 2024-04-01
```

For full range:

```bash
uv run scripts/data/build_station_vd_drive.py --start-date 2024-04-01 --end-date 2025-05-31
```

## Running the Dashboard on Colab

Colab reads from Drive's backend directly (co-located), so data I/O is much
faster than Drive for Desktop's virtual filesystem on a local machine.

```python
# Cell 1 — setup
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/陽交大高速公路計畫/main
!pip install uv
!npm install -g localtunnel # for browser access
```

```python
# Cell 2 — launch dashboard
!uv run scripts/dashboard/run.py --colab
```

A `localtunnel` URL will be printed (e.g. `https://xxx.loca.lt`). Open it in
your browser. If localtunnel is not available, run these in two separate cells:

```python
# Cell A — start Gradio in the background
!nohup uv run scripts/dashboard/run.py -- --server-port 7860 > dashboard.log 2>&1 &
```

```python
# Cell B — start tunnel
!lt --port 7860
```

## Colab Constraints

- Process one raw day at a time.
- Avoid loading all raw Parquet files at once.
- Write outputs immediately per day.
- Expect Drive I/O to dominate some steps.
- Prefer Parquet/NPZ over many small CSV files.

## Migration Boundary

Do not migrate these assumptions:

- monthly raw Parquet files
- old `data/processed/vd_1m.parquet` single-file outputs
- scripts that parse `vdid` repeatedly instead of using `freeway/direction/mileage_m`
- direct training from raw lane-level rows

## Code Layout

```
scripts/
├─ data/
│  ├─ download_vd_drive.py         # raw VD download
│  ├─ build_station_vd_drive.py    # raw -> station 1m / 5m
│  ├─ build_historical_profile.py  # station 5m -> per-VD historical profile
│  ├─ build_accident_samples.py    # station 5m + accidents -> NPZ + per-accident CSV
│  └─ process_accident_data.py     # raw 派遣回報 xlsx -> accident_data.csv
├─ features/
│  └─ build_incident_features.py   # accident_data.csv + geometry/weather -> 33-feature table
├─ lib/                             # shared modules used by scripts and notebooks
│  ├─ data_loaders.py               # read_station / read_profile / read_accidents / load_samples
│  ├─ windows.py                    # SampleWindow + slice helpers
│  ├─ visualization.py              # heatmap (observed / historical / predicted / diff)
│  ├─ inference.py                  # load checkpoint, predict_accident_impact
│  └─ metrics.py                    # MAE / RMSE / F1 / impact-time scoring
├─ models/                          # one training script per architecture
│  ├─ unet_matrix.py
│  ├─ unet_feature_fusion.py
│  └─ timesfm_covariate.py
└─ visualization/
   └─ accident_heatmap.py           # CLI heatmap renderer (already migrated)

notebooks/
├─ 00_setup_and_mount.ipynb
├─ 01_build_data_layers.ipynb
├─ 02_explore_data.ipynb
├─ train/
│  ├─ unet_matrix.ipynb
│  ├─ unet_feature_fusion.ipynb
│  ├─ timesfm_covariate.ipynb
│  └─ _template.ipynb
└─ apply/
   └─ accident_workflow.ipynb
```

## Sample Window Specification

`build_accident_samples.py` is the canonical accident-window slicer. It uses one
window superset; downstream training and inference subset by slicing the
already-built matrices.

| Parameter | Value | Notes |
| --- | --- | --- |
| `upstream_km` | 50 | Direction-aware. For S/E: `mileage_acc - 50` to `mileage_acc + 10`. For N/W: `mileage_acc - 10` to `mileage_acc + 50`. |
| `downstream_km` | 10 | Same convention. |
| `time_before_min` | 60 | Window starts at `report_time - 60 min`. |
| `time_after_min` | 240 | Window ends at `report_time + 240 min`. |
| `spatial_resolution_km` | 1 | Cell size along mileage. |
| `resolution_min` | 5 | 5-minute cells in the canonical run. 1-minute cells reserved for future use. |

Resulting matrix per accident: `61 (space) x 61 (time)` cells at 5-minute resolution.

The current research/model default is a sliced subset of this superset:
`-10..+25 km` and `-30..+145 min` (36 x 36). This is the code-level
representation of the `-30~+150 min` research horizon because the post-incident
target cells are `0, 5, ..., 145`.

Spatial axis is normalized to **upstream-distance** so matrices are stackable
across accidents and directions:

- `+50000 m` = 50 km upstream of the accident
- `0 m` = at the accident
- `-10000 m` = 10 km downstream

CSV outputs keep the **absolute mileage** index for human-readable debugging.
NPZ outputs use the **upstream-distance** axis.

## Data Layers Built by Notebook 01

1. `data/vd_station_1m/`, `data/vd_station_5m/` (already produced).
2. `data/historical_speed_profile_5m/` (see `docs/DATA_DESIGN.md`).
3. `data/accident_samples_5m.npz` and `data/accident_samples_csv_5m/` (per-accident).
4. `output/feature_engineering/incident_features_33.csv`.

Each step is per-day or per-accident streaming; never materialize all raw or
station rows at once.

## Notebook Specification

The notebook tree mirrors the user-facing workflow: setup -> build layers ->
explore -> train (per-architecture) -> apply.

### `notebooks/00_setup_and_mount.ipynb`

- Mount Drive at `/content/drive`.
- `cd` to `/content/drive/MyDrive/陽交大高速公路計畫/main`.
- `pip install uv` and verify Python version.
- Print `data/` size summary by reading `data/metadata/station_build_report.json`.
- Smoke-import `scripts.lib` and run a 1-day sanity slice from `vd_station_5m`.

### `notebooks/01_build_data_layers.ipynb`

For each layer, one cell calls the corresponding script via `uv run`. Default
date range read from `central_vd_list.txt` and accident CSV bounds.

1. Run `scripts/data/build_station_vd_drive.py` for the missing date range.
2. Run `scripts/data/build_historical_profile.py` for the full available range, e.g. `--start-date 2021-06-01 --end-date 2025-05-31 --force`.
3. Run `scripts/data/build_accident_samples.py` (writes both NPZ and per-accident CSV).
4. Run `scripts/features/build_incident_features.py`.
5. Print final dataset sizes and any rows in `station_build_report.json` with `failed`/`missing_raw` status.

### `notebooks/02_explore_data.ipynb`

Pure exploration, uses `scripts.lib` only:

- Plot accident count per day, per freeway, per direction.
- Plot VD coverage map (mileage distribution) per freeway/direction.
- Sample 3 accidents and render observed vs historical heatmaps side by side via `lib.visualization.compare_heatmaps`.
- Surface accidents with sample metadata `filled_ratio < 0.8` for triage; this is the current default training keep/drop rule.

### `notebooks/train/<architecture>.ipynb`

One notebook per architecture. Each notebook follows the same skeleton (see
`_template.ipynb`):

1. **Setup** — mount, install, import.
2. **Load samples** — `lib.data_loaders.load_samples()` returns the matrix stack, ids, axes, metadata, and source `SampleWindow`.
3. **Subset window (optional)** — `lib.windows.subset(matrix, axes, window=...)`.
4. **Split** — train/val/test by accident_id (deterministic seed).
5. **Train** — call the architecture training entry from `scripts/models/<name>.py`.
6. **Save artifact** — write trained weights and config to `output/models/<name>/<run_id>/`.
7. **Inline eval** — `lib.metrics.evaluate(model, val_set)` -> small summary table.

Architectures planned in this round:

- `unet_matrix.ipynb` — speed-matrix U-Net (also runs the spatial-window sweep that used to be in `compare_unet_spatial_windows.py`).
- `unet_feature_fusion.ipynb` — feature-fusion U-Net plus optional TIID head.
- `timesfm_covariate.ipynb` — covariate-enhanced forecasting baseline. **Deferred from Phase 3 round-1**; see [ROADMAP.md](ROADMAP.md) Phase C for the active ablation matrix (data × architecture × feature-time).

A new architecture is added by copying `_template.ipynb` and writing a matching
script under `scripts/models/`. Notebook cells stay short because heavy logic
lives in `scripts/lib` and `scripts/models/`.

### `notebooks/apply/accident_workflow.ipynb`

The end-user workflow notebook. Round 1 supports lookup by `accident_id`; round
2 can add arbitrary user-input accidents; the locked notebook flow uses known accident ids for reproducibility.

1. **Pick accident** — `accident_id = "2024040004"`.
2. **Observed heatmap** — `obs = lib.windows.observed(accident_id)` -> `lib.visualization.heatmap(obs, title="Observed")`.
3. **Historical heatmap** — `hist = lib.windows.historical(accident_id)` (joins `historical_speed_profile_5m` to the same window) -> heatmap.
4. **Model prediction heatmap** — `model = lib.inference.load("unet_feature_fusion", run_id=..., experiment="direct_concat")`, then run the shared inference path for t+0 / t+5 / t+15 checkpoints.
5. **Diff heatmaps** — `obs - hist` (excess congestion observed) and `obs - pred` (model error) side by side.
6. **Impact-time estimate** — `lib.metrics.impact_time(obs, hist, threshold=...)` and the same on `pred`.
7. **Optional** — same pipeline on a hypothetical accident dict after GUI validation.

This notebook should not depend on any model-architecture-specific helpers.
Switching from `unet_feature_fusion` to another model is a single string change
in step 4.

## Tooling Conventions

The project mixes pandas and DuckDB. Pick the right one per task; do not switch
back and forth inside one script.

| Pattern | Tool | Why |
| --- | --- | --- |
| Per-day raw -> station streaming (`build_station_vd_drive.py`) | pandas + pyarrow | Each call processes a single day's frame; pandas overhead is negligible at this size and the existing pyarrow schema is already canonical. |
| Per-accident window slicing (`build_accident_samples.py`) | pandas | ~2000 accidents x small per-accident frames. DuckDB has no advantage here and would complicate the per-accident JSON sidecar / NPZ stacking flow. |
| Aggregate with percentiles over many groups (`build_historical_profile.py`) | **DuckDB** | `quantile_cont` is vectorized C++. Pandas `groupby(...).agg(lambda s: np.percentile(s, q))` is dominated by Python dispatch when group count is large (1.5M groups for 1300 VDs x day x peak x minute). DuckDB also writes hive-partitioned Parquet in one statement. |
| Ad-hoc notebook exploration over partitioned datasets | DuckDB via `lib.data_loaders.duckdb_connection()` | One-line SQL is more readable than chained pandas filters and avoids loading whole partitions into RAM. |
| Stacking matrices into NPZ for model training | numpy | Standard. |

If a future task wants groupby-with-percentiles, GROUP BY with COUNT/AVG over
millions of rows, or window functions across the full station dataset, default
to DuckDB. Reach for pandas when working a single day at a time, joining small
metadata tables, or producing per-row Python objects (e.g. JSON sidecars).

This decision was made after a 156 s / 2-day pandas profile build; see
`PITFALLS.md` entry 11.

## Historical Implementation Order

This sequence is complete and retained only as context:

1. `scripts/lib/` skeleton + this spec.
2. `process_accident_data.py` and `build_incident_features.py`.
3. `build_historical_profile.py` and `build_accident_samples.py`.
4. Migrate `unet_matrix`, `unet_feature_fusion`, `timesfm_covariate` to NPZ sample I/O.
5. Notebook shells in the order listed above.
