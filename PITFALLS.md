# Known Pitfalls & Solutions

This file reflects the Drive + Parquet + Colab migration state as of 2026-04-25.

## Data & Preprocessing

### 1. VD Data Coverage Gaps

Problem: Raw VD files may miss minutes or entire stations.

Symptoms:

- Station matrices have sparse or all-missing regions.
- Historical averages have low `n_obs`.
- Model samples are filtered out for poor pre-event coverage.

Solutions:

- Treat upstream 404 minute files as expected when manifest marks them `server_404`.
- Keep `n_lanes_valid` in station datasets.
- Write per-day station build stats and inspect low-row days.
- During matrix generation, keep a coverage threshold and record skipped task IDs.

### 2. Accident ID Misalignment

Problem: Task IDs in `data/accident_data.csv` must match feature rows and generated matrix/sample IDs.

Solutions:

- Always load task IDs as strings.
- Normalize with `astype(str).str.strip()`.
- Cross-check intersections before training.

### 3. Speed Validity and Normalization

Problem: Raw speed values include invalid codes and out-of-range values.

Rules:

- For station aggregation, exclude raw lane rows with `speed <= 0` or `speed > 140`.
- Before model normalization, clip station speeds to `[0, 140]`.
- Store station speed as `speed_x10` int16; convert with `speed_kmh = speed_x10 / 10`.

### 4. Timezone Handling

Problem: Upstream XML has `+08:00` timestamps, while accident CSV times are local Taiwan wall times.

Current rule:

- Raw Drive parquet stores `timestamp` as Taiwan local wall time without timezone metadata.
- Do not compare timezone-aware and timezone-naive timestamps in downstream scripts.

### 5. Drive Performance

Problem: Google Drive is slow with many small files.

Rules:

- Prefer one Parquet per source day for raw/station data.
- Avoid matrix CSV as a long-term format.
- Use consolidated Parquet, NPZ, or larger array bundles for generated training samples.

### 6. Colab Memory

Problem: One raw day can expand to hundreds of MiB in memory.

Rules:

- Process raw files day-by-day.
- Do not concatenate all raw days before groupby.
- Build `vd_station_1m` first, then derive `vd_station_5m`.
- For full-range jobs, write each day immediately and append summary metadata only.

### 7. Raw vs Station Semantics

Problem: Raw lane rows are not model-ready.

Rules:

- `vd_raw_lane_1m_full` is a reproducible source layer.
- `vd_station_1m` is the canonical model/data-science layer.
- `vd_station_5m` is a cache derived from station 1m.
- Historical profiles should be derived from station data, not raw lane rows.

### 11. Pandas groupby with Python lambdas does not scale

Problem: `df.groupby(...).agg(p10=lambda s: np.percentile(s, 10), ...)` calls a Python function once per group. With 1.5M groups (typical for the historical-speed profile across ~1300 VDs x day_class x peak_class x minute_bucket), the Python overhead dominates regardless of input size. A 2-day profile build took ~156 s, mostly inside the lambda dispatch loop.

Symptoms:

- Aggregation runs orders of magnitude slower than reading the input.
- CPU sits at one core; profile flame graph is dominated by `_aggregate_named` / `apply`.

Rules:

- For groupby-with-percentiles over many groups, use DuckDB's `quantile_cont` instead of pandas lambdas. DuckDB runs the aggregation in vectorized C++ and writes hive-partitioned Parquet directly.
- Pandas is fine for streaming per-day or per-accident work where each call processes one small frame.
- See `scripts/data/build_historical_profile.py` for the canonical DuckDB pattern.

## Modeling

### 8. Small Validation Sets

Small validation sets make congestion F1 unstable. Use the full usable sample set unless debugging a single issue.

### 9. Congestion Class Imbalance

Congested cells are sparse. Track congestion F1 alongside MAE/RMSE and keep weighted losses available.

### 10. Missing Feature / Matrix Alignment

Feature fusion requires matrix sample IDs and feature row IDs to align. Validate intersections before training.

## Operational Checks

Before long Colab runs:

- Confirm Drive is mounted.
- Confirm `data/vd_raw_lane_1m_full` exists.
- Confirm `data/accident_data.csv` exists.
- Confirm available Colab disk space before writing large temporary outputs.
- Run one-day station build before full-range processing.
