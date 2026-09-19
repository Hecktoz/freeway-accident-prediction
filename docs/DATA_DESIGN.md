# Data Design

This project uses Google Drive + Parquet as the canonical storage plan for the Colab migration.

## Design Principles

- Keep raw data reproducible but do not train directly from raw lane rows.
- Use daily Parquet files to keep Drive operations tractable.
- Store route and mileage fields explicitly; do not parse `vdid` during every query.
- Store speeds as integer tenths (`speed_x10`) in curated layers.
- Keep the 1-minute station layer as canonical. Build 5-minute data as a cache.
- Do not encode accident window size into the base datasets. Select route/mileage windows at sample-generation time.

## Layer 1: Raw Lane Data

Dataset:

```text
data/vd_raw_lane_1m_full/
```

Partition/path:

```text
source_year=YYYY/source_month=MM/vd_YYYYMMDD.parquet
```

Purpose:

- Reproducible source data from MOTC XML.
- Lane-level rows.
- Includes a buffered VD universe around `central_vd_list.txt`.

Schema:

| Column | Type | Meaning |
| --- | --- | --- |
| `source_date` | date32 | Requested source date |
| `source_hhmm` | int16 | Requested upstream minute file |
| `timestamp` | timestamp[ms] | Taiwan local wall time, no timezone metadata |
| `vdid` | string | VD station ID |
| `freeway` | string | `N1`, `N3`, `N4`, `N6`, etc. |
| `direction` | string | `N`, `S`, `E`, `W` |
| `mileage_m` | int32 | Mileage in meters |
| `link_id` | string | Upstream link ID |
| `lane_id` | int8 | Lane ID |
| `lane_type` | int8 | Lane type from source |
| `speed` | int16 | Raw lane speed |
| `occupancy` | int16 | Raw lane occupancy |
| `status` | int8 | Raw station status |
| `vol_s`, `spd_s` | int16 | Small vehicle volume/speed |
| `vol_l`, `spd_l` | int16 | Large vehicle volume/speed |
| `vol_t`, `spd_t` | int16 | Trailer volume/speed |

Current status:

- Training lock accident/sample range: 2024-04-01 to 2025-05-31.
- Current Drive scan: 1,759 daily raw files from 2021-06-29 to 2026-04-30.
- The last downloader filter records `exact_vd_count=1334` from `central_vd_list.txt` plus a 70 km route/direction buffer. The buffered observed VD universe is larger than the exact list.
- Do not widen the training sample date range from the lock range without creating a new dataset-lock report.

## Layer 2: Station 1-Minute Data

Dataset:

```text
data/vd_station_1m/
```

Partition/path:

```text
source_year=YYYY/source_month=MM/vd_station_YYYYMMDD.parquet
```

Purpose:

- Canonical modeling and feature-engineering traffic layer.
- One row per timestamp and VD.
- Derived from raw lane data by aggregating valid lanes.

Cleaning rule:

- Keep raw lane rows where `0 < speed <= 140`.
- Invalid speed rows are removed before station aggregation.
- Occupancy participates in the occupancy mean only when `0 <= occupancy <= 100`.
- Negative or missing vehicle volumes are filled with 0 before summing.

Schema:

| Column | Type | Meaning |
| --- | --- | --- |
| `source_date` | date32 | Source date |
| `timestamp` | timestamp[ms] | Taiwan local wall time |
| `vdid` | string | VD station ID |
| `freeway` | string | Freeway code |
| `direction` | string | Direction code |
| `mileage_m` | int32 | Mileage in meters |
| `speed_x10` | int16 | Mean station speed in km/h times 10 |
| `occupancy_x10` | int16 | Mean station occupancy times 10 |
| `vol_s`, `vol_l`, `vol_t` | int32 | Summed valid-lane volumes |
| `n_lanes_valid` | int16 | Valid lane rows used |

Current status:

- About 10 MiB/day for 2024-04-01 sample.
- The latest full station build report covers 2023-08-06 to 2026-04-30: 999 days, 2,958,942,967 station-1m rows, 12,155.763 MiB.
- A filesystem scan currently sees 1,759 daily station files from 2021-06-29 to 2026-04-30; older files are retained for historical profile construction, not automatically part of a training lock.

## Layer 3: Station 5-Minute Data

Dataset:

```text
data/vd_station_5m/
```

Purpose:

- Cache for current 5-minute matrix/model workflows.
- Derived from station 1-minute data, not directly from raw.

Aggregation:

- `timestamp = floor(timestamp, 5min)`
- `speed_x10`, `occupancy_x10`: rounded mean of 1-minute station values
- `vol_s`, `vol_l`, `vol_t`: sum
- `n_lanes_valid`: sum

Current status:

- About 2.8 MiB/day for 2024-04-01 sample.
- The latest full station build report covers 2023-08-06 to 2026-04-30: 700,465,715 station-5m rows, 3,461.221 MiB.
- This is the primary traffic layer for the locked 5-minute samples and current model workflows.

## Metadata

### `data/metadata/vd_dim.parquet`

One row per VD observed in station/raw data.

Columns:

- `vdid`
- `freeway`
- `direction`
- `mileage_m`
- `first_source_date`
- `last_source_date`
- `n_source_days`
- `n_rows_station_1m`

### `data/metadata/station_build_report.json`

Build summary with per-day row counts, output sizes, skipped days, and error messages.

### `data/metadata/vd_download_filter.json`

Records the VD filter universe used by `download_vd_drive.py`.

## Historical Profile Layers

Dataset:

```text
data/historical_speed_profile_5m/
```

Purpose:

- Event-agnostic per-VD historical baseline for reconstructing accident windows.
- Built from station data, never from raw lane rows.
- Current built range: 2021-06-01 to 2025-05-31.
- Current rows: 2,385,567 across N1/N3/N4/N6.

Schema:

- `resolution_min`
- `vdid`
- `direction`
- `mileage_m`
- `day_class` (`weekday`, `weekend`, `national_holiday`)
- `minute_bucket` (every `resolution_min` minutes of day, 0…1435)
- `peak_class` — **derived convenience label, not an independent grouping dimension.** Fully determined by `minute_bucket` (peak ⇔ 07:00–09:00 or 17:00–19:00); the rows are already keyed at `minute_bucket` granularity, so this column adds no partitioning and changes no statistics. Kept only as a ready-made filter; recompute from `minute_bucket` if absent.
- `mean_speed_x10`
- `p10_speed_x10`
- `p50_speed_x10`
- `p90_speed_x10`
- `std_speed_x10`
- `n_obs`
- `coverage_ratio`

`national_holiday` uses the project Taiwan public/adjusted holiday calendar in `scripts/lib/calendar.py`; makeup workdays are classified as `weekday`. Do not precompute a fixed upstream/downstream accident window in this layer. Select VD rows by `freeway`, `direction`, and `mileage_m` when generating accident samples.

### Per-accident historical-baseline matrices

```text
data/historical_matrices_5m.npz (dispatch)
data/historical_matrices_5m_sms.npz (SMS, with --tag sms)
```

Derived layer built by `scripts/data/build_historical_matrices.py`: one
historical-baseline (typical non-accident) speed matrix per accident, aligned
1:1 with `accident_samples_5m.npz` over the same superset window. The canonical
U-Net input is now exactly three channels: pre-incident observed VD speed,
observed-VD validity, and causal historical speed. Missing historical values are
filled from the nearest valid spatial row in the same time column; the former
historical-validity fourth channel is removed. Observed VD says *what is
happening*; the historical baseline says *the speed level to recover toward*.

## Locked Accident Samples

Dataset:

```text
data/accident_samples_5m.npz
data/accident_samples_5m_metadata.parquet
data/accident_samples_csv_5m/
```

Canonical sample window:

- Space: 50 km upstream + 10 km downstream, 1 km cells.
- Time: 60 min before to 240 min after report time, 5-minute cells.
- Matrix shape: 61 space x 61 time.
- Space axis: upstream-distance in meters; `-10000` is 10 km downstream, `0` is the accident point, `50000` is 50 km upstream.
- Time axis: `-60, -55, ..., 240` minutes from report time.

Research/model subset used by the current U-Net and benchmark runs:

- Space: 25 km upstream + 10 km downstream -> 36 cells.
- Time: 30 min before to 145 min after -> 36 cells. In project shorthand this is the `-30~+150 min` horizon: at 5-minute resolution the target columns are `0, 5, ..., 145`, covering 30 post-incident 5-minute intervals.
- Input cells: `-30` to `-5` min. The `t=0` cell is the first prediction target.
- Plotting scripts keep the full 61 x 61 view by default for audit/exploration, and can render the research/model subset with `--research-window`.
- Training scripts apply the research/model subset by default and drop samples with metadata `filled_ratio < 0.80`. Other quality flags are diagnostic unless explicitly enabled.

Current lock:

- Dataset version: `dataset_20260511_5m_v2`.
- Accident range: 2024-04-01 to 2025-05-31.
- Samples: 2,036 accidents, 0 failed.
- Current coverage rule keeps 1,825 / 2,036 accidents (`filled_ratio >= 0.80`) and removes 211 accidents. Generated by `uv run scripts/data/check_accident_quality.py`.
- Feature table (legacy, locked): `output/feature_engineering/incident_features_33.csv`, 2,043 x 33; 32 model features after dropping `task_id`.
- Feature table (current, availability schema): `output/feature_engineering/incident_features_44.csv`, 2,043 x 44 = 33 value + 11 `avail_<col>` → `feature_dim=43`. Built by `build_incident_features.py` (default output). The 32 value columns are byte-identical to the locked file; only `avail_<col>` indicator columns are appended, so this is a schema extension, not a data-range change. Checkpoints trained on the 32-dim file must be retrained.
- Weather feature columns are still zero-filled and carry `avail_*=0` (unknown). Adding real weather requires a new dataset version and flips those indicators to 1.
- SMS mixed-training is implemented via the availability schema: missing incident fields are marked `avail_<col>=0` (unknown) instead of a fake 0 (no explicit source flag — the availability pattern itself distinguishes SMS, which has `avail=0` for most incident fields). SMS is used only outside dispatch coverage (report date < 2024-04-01). Artifacts: `data/accident_data_sms.csv` (`process_sms_data.py`), `data/accident_samples_5m_sms.npz` (`build_accident_samples.py --tag sms`), `output/feature_engineering/incident_features_sms_44.csv` (`build_incident_features.py --output-name incident_features_sms_44.csv`). See `notebooks/06_build_sms_samples.ipynb`.
