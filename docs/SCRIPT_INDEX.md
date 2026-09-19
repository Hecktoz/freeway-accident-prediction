# Script Index

## Active Scripts

### `scripts/data/download_vd_drive.py`

Local-first downloader for Drive + Parquet raw VD data. Used when raw MOTC XML
must be downloaded or resumed.

```bash
uv run scripts/data/download_vd_drive.py --start-date 2024-04-01 --end-date 2025-05-31
```

Output:
- `data/vd_raw_lane_1m_full/`
- `data/_vd_download_manifests/`
- `data/metadata/vd_download_filter.json`

### `scripts/demo/live_vd_smoke.py`

CLI sanity-check for `scripts.lib.live_vd.fetch_live_station_5m`. Hits MOTC
for a small accident-shaped window and prints the VDs / row counts / fetch
status. GUI delivery is deferred, so this is currently a future deployment
smoke / debugging utility rather than an active paper milestone.

```bash
uv run scripts/demo/live_vd_smoke.py --start "2024-04-01 02:00" --end "2024-04-01 02:09" \
  --freeway N1 --direction S --mileage-km 147
```

### `scripts/demo/inference_smoke.py`

End-to-end smoke for `scripts.lib.inference.predict_accident_impact`. Wires
the real live fetcher or the local `--offline` station-5m reader into the
same inference path used by dashboard/notebook case studies and the future GUI.
By default it loads the time-progressive checkpoint for `--feature-set`;
`--no-forward` keeps the historical-baseline fallback path available for smoke
tests.

```bash
uv run scripts/demo/inference_smoke.py --report-time "2024-04-01 02:27" \
  --freeway N1 --direction S --mileage-km 147

uv run scripts/demo/inference_smoke.py --report-time "2024-04-01 02:27" \
  --freeway N1 --direction S --mileage-km 147 --offline --no-forward
```

### `scripts/demo/raw_vd_query_benchmark.py`

Interactive-demo backend for comparing raw lane-level VD window queries with
DuckDB over daily Parquet files versus pandas `read_csv` over a same-day CSV
cache. The script resolves only the raw daily files needed by the requested
time window, including cross-midnight windows.

```bash
uv run scripts/demo/raw_vd_query_benchmark.py --date 2024-04-01 --time 09:30 --freeway N1 --direction S --mileage-km 60 --upstream-km 50 --downstream-km 10 --before-min 60 --after-min 240 --repeat 3 --prepare-csv
```

Output:
- `output/query_benchmark/raw_lane_csv_cache/vd_YYYYMMDD.csv` (on-demand CSV cache)

### `scripts/data/build_station_vd_drive.py`

Builds curated station-level traffic datasets from raw lane data.

```bash
uv run scripts/data/build_station_vd_drive.py --start-date 2024-04-01 --end-date 2025-05-31
```

Output:
- `data/vd_station_1m/`
- `data/vd_station_5m/`
- `data/metadata/vd_dim.parquet`
- `data/metadata/station_build_report.json`

### `scripts/data/process_accident_data.py`

Reads `派遣回報紀錄YYYYMM.xlsx` files and produces a consolidated
`accident_data.csv` with cancelled rows filtered out and freeway codes
standardized (`國1` -> `N1`).

```bash
uv run scripts/data/process_accident_data.py
uv run scripts/data/process_accident_data.py --input-dir 派遣系統資料 --output data/accident_data.csv
```

Output:
- `data/accident_data.csv`

### `scripts/data/process_sms_data.py`

Standardizes the 交控事故簡訊 (SMS) monthly Excel files
(`data/accident_交控事故簡訊紀錄表/*.xlsx`) into a dispatch-compatible CSV so the
existing sample/feature builders consume them unchanged. Normalizes mixed
ROC/AD years, strips mileage unit suffixes (`158.5公里`), looks up mainline lane
count from `data/geometry`, derives `剩餘車道數` from SMS blocked-lane columns,
and injects structured special-accident flags (翻覆/起火/危險物) into the text.
Keeps only the period outside dispatch coverage (default report date <
2024-04-01) and rows within observed VD mileage coverage.

```bash
uv run scripts/data/process_sms_data.py
uv run scripts/data/process_sms_data.py --end-date-exclusive 2024-04-01 --no-central-only
```

Output:
- `data/accident_data_sms.csv` (~4,479 pre-dispatch rows)

Then build SMS matrices/features with the tagged builders:

```bash
uv run scripts/data/build_accident_samples.py --accident-csv data/accident_data_sms.csv --tag sms --start-date 2023-06-01 --end-date 2024-03-31 --no-csv --force
uv run scripts/features/build_incident_features.py --accident-csv data/accident_data_sms.csv --output-name incident_features_sms_44.csv
```

See `notebooks/06_build_sms_samples.ipynb` for the Colab end-to-end flow.

### `scripts/data/build_historical_profile.py`

Builds the per-VD historical speed profile from station 5-minute data. The
profile is event-agnostic: it stores per-(vdid, day_class, peak_class,
minute_bucket) statistics so accident windows can be reconstructed cheaply
later. `day_class` is one of `weekday`, `weekend`, or `national_holiday`.

```bash
uv run scripts/data/build_historical_profile.py --start-date 2021-06-01 --end-date 2025-05-31 --force
```

Output:
- `data/historical_speed_profile_5m/freeway=<F>/profile_*.parquet`
- `data/metadata/historical_profile_report.json`

### `scripts/data/build_historical_matrices.py`

Builds one historical-baseline (typical non-accident) speed matrix per accident,
aligned 1:1 with the sample NPZ over the same canonical superset window. Output
feeds the `--input-type hist` / `vd+hist` training runs as extra "history" input
channels (the model's design intent — observed VD tells *what is happening*,
history tells *the speed level to recover toward*). Reuses
`scripts.lib.inference.build_historical_baseline` + the historical speed profile.

```bash
uv run scripts/data/build_historical_matrices.py
uv run scripts/data/build_historical_matrices.py --tag sms
```

Output:
- `data/historical_matrices_5m.npz` (or `..._sms.npz` with `--tag sms`)

### `scripts/data/build_congestion_masks.py`

Precomputes the locked `origin_cc` accident-congestion mask for each accident,
aligned to `accident_samples_5m.npz` and `historical_matrices_5m.npz`. This is
the mask used by the post-2026-06-03 loss / metric contract.

```bash
uv run scripts/data/build_congestion_masks.py
uv run scripts/data/build_congestion_masks.py --tag sms
uv run scripts/data/build_congestion_masks.py --limit 50
```

Output:
- `data/congestion_masks_origin_cc_5m.npz`
- `data/congestion_masks_origin_cc_5m_sms.npz` when `--tag sms` is used.

### `scripts/data/build_accident_samples.py`

Builds per-accident spacetime sample matrices using the canonical superset
window (50 km up + 10 km down, -60 to +240 min, 1 km x 5 min). Produces a
debug-friendly CSV per accident plus one stacked NPZ for training. `--accident-csv`
points at an alternate source (e.g. the SMS CSV); `--tag <name>` suffixes all
outputs (NPZ / metadata / report / CSV dir) so a second source does not
overwrite the dispatch artifacts; `--no-csv` skips per-accident CSVs.

```bash
uv run scripts/data/build_accident_samples.py --limit 5
uv run scripts/data/build_accident_samples.py
uv run scripts/data/build_accident_samples.py --accident-csv data/accident_data_sms.csv --tag sms --no-csv --force
```

Output:
- `data/accident_samples_csv_5m/<accident_id>.csv` and `.json` sidecar (per accident)
- `data/accident_samples_5m.npz` (stacked matrices on direction-normalized upstream-distance axis)
- `data/accident_samples_5m_metadata.parquet` (one row per accident)
- `data/metadata/accident_samples_report.json`

### `scripts/features/build_incident_features.py`

Builds the incident feature table with descriptive statistics. Joins the
accident CSV with road geometry segments and optional weather covariates. The
default output uses the **availability schema**: 34 value cols including
`task_id`, 12 `avail_<col>` indicators, and 1 QC-only `avail_mainline` column =
47 cols (`feature_dim=45` after excluding `task_id` and `avail_mainline`). The
`avail_<col>` flags mark whether the source provided each missing-prone field,
so a real `0` is distinguishable from "unknown". Weather CSVs may use either the
contract names (`temp_c`, `humidity_pct`, `wind_speed_mps`, `rain_mm`,
`visibility_km`) or the current Chinese unit columns. `--output-name` controls
the filename. Stats/summary filenames derive from the output stem.

```bash
uv run scripts/features/build_incident_features.py
uv run scripts/features/build_incident_features.py --weather-csv data/weather/weather_per_task.csv
uv run scripts/features/build_incident_features.py --weather-csv data/weather_data/accident_with_weather_Interpolated.csv
uv run scripts/features/build_incident_features.py --accident-csv data/accident_data_sms.csv --output-name incident_features_sms_44.csv
```

Output (default):
- `output/feature_engineering/incident_features_44.csv`
- `output/feature_engineering/incident_features_44_statistics.csv`
- `output/feature_engineering/incident_features_44_summary.md`

### `scripts/data/check_accident_quality.py`

Pre-training quality diagnosis and keep-list writer. The default active filter
is only `filled_ratio >= 0.80`; F1-F6 traffic-shape filters are reported as
diagnostics unless explicitly enabled.

```bash
uv run scripts/data/check_accident_quality.py
uv run scripts/data/check_accident_quality.py --enable-f1 --enable-f2
```

Output:
- `output/quality/accident_quality.csv`
- `output/quality/accident_keep_ids.csv`

### `scripts/data/fill_vd_gaps.py`

Maintenance utility for auditing VD download manifests and retrying missing /
errored minutes. Use only when raw download manifests show gaps or the VD
universe needs to be expanded; it is not part of the default paper rerun path.

```bash
uv run scripts/data/fill_vd_gaps.py --start-date 2024-04-01 --end-date 2025-05-31
uv run scripts/data/fill_vd_gaps.py --start-date 2024-04-01 --end-date 2025-05-31 --fill
uv run scripts/data/fill_vd_gaps.py --start-date 2024-04-01 --end-date 2025-05-31 --fill --expand-vd
```

### `scripts/data/parse_line_chat.py`

Parses the LINE group chat export into structured initial / intermediate /
final report CSVs. LINE is not part of the locked paper mainline, but these
tables remain useful for data-source audits.

```bash
uv run scripts/data/parse_line_chat.py
```

Output:
- `data/line_reports_initial.csv`
- `data/line_reports_intermediate.csv`
- `data/line_reports_final.csv`

### `scripts/visualization/plot_heatmap.py` (canonical)

Canonical batch renderer. Reads the accident samples NPZ and renders one PNG
per accident using the unified style from `scripts/lib/visualization`.
This is the primary output script for reports and slides.

```bash
uv run scripts/visualization/plot_heatmap.py --style report --limit 5
uv run scripts/visualization/plot_heatmap.py --style slide --accident-id 2024040004
uv run scripts/visualization/plot_heatmap.py --style report --accident-id 2024040004 --research-window
uv run scripts/visualization/plot_heatmap.py --style notebook --start-date 2024-04-01 --end-date 2024-04-30
uv run scripts/visualization/plot_heatmap.py --style report --force
```

Output:
- `output/heatmaps_<style>/`
- `output/heatmaps_<style>_research/` when `--research-window` is used.

### `scripts/visualization/accident_heatmap.py`

Station-Parquet-based batch renderer. Builds heatmap matrices directly from
`vd_station_5m` rather than the pre-built NPZ. Useful when the NPZ is not
available or when adjusting the spatial/temporal window beyond the canonical
superset.

```bash
uv run scripts/visualization/accident_heatmap.py --limit 5 --force
uv run scripts/visualization/accident_heatmap.py --accident-id 2024040004 --force
uv run scripts/visualization/accident_heatmap.py --accident-id 2024040004 --research-window --force
```

Output:
- `output/accident_heatmaps_5m/`

### `scripts/gui/app.py` (future/deferred)

Deferred Phase 5 desktop GUI draft for the Highway Bureau. Wraps
`scripts.lib.inference.predict_accident_impact`. Accepts an accident's time /
freeway / direction / mileage and renders the predicted heatmap, impact
duration, recovery curves, and coverage panel. It supports the t+0 / t+5 /
t+15 time-progressive checkpoints listed in `reports/feature_time_grouping.md`.
See `docs/DEPLOYMENT.md`. This is not an active milestone while the project is
focused on model refinement, case studies, visualization, and conference paper
preparation.

```bash
uv run scripts/gui/app.py
uv run scripts/gui/app.py --host 0.0.0.0 --port 7860
```

Output: launches a local Gradio server (default `http://127.0.0.1:7860`).

### `scripts/visualization/accident_heatmap_web.py`

Interactive Dash browser for accident heatmaps. It lets users select an
accident, adjust upstream/downstream distance and before/after time windows,
and compare the observed speed heatmap with the historical average speed
profile when `data/historical_speed_profile_5m/` is available.

```bash
uv run scripts/visualization/accident_heatmap_web.py
uv run scripts/visualization/accident_heatmap_web.py --host 0.0.0.0 --port 8050
uv run scripts/visualization/accident_heatmap_web.py --smoke-test --accident-id 2024040004
```

### `scripts/visualization/build_vd_station_map.py`

Builds a single-page HTML inventory/map for VD stations by joining
`data/metadata/vd_static.xml` with `vd_dim.parquet`.

```bash
uv run scripts/visualization/build_vd_station_map.py
uv run scripts/visualization/build_vd_station_map.py --fetch
```

Output:
- `output/vd_station_map.html`

### `scripts/visualization/corridor_profile_heatmap.py`

Renders an absolute-mileage corridor heatmap from the historical profile. This
is useful for showing recurring congestion and for slide / paper context figures.

```bash
uv run scripts/visualization/corridor_profile_heatmap.py --freeway N1 --direction S \
  --mileage-min 150 --mileage-max 190 --style slide \
  --output output/slide_materials/corridors/N1_S_taichung_weekday.png
```

### `scripts/visualization/export_slide_heatmaps.py`

Exports clean historical-speed heatmaps for fixed slide scenarios.

```bash
uv run scripts/visualization/export_slide_heatmaps.py
```

Output:
- `output/slide_materials/hist_speed_<freeway>_<dir>_<day_class>.png`

### `scripts/visualization/export_accident_diff_heatmaps.py`

Exports observed and observed-minus-historical accident heatmaps for a curated
slide set.

```bash
uv run scripts/visualization/export_accident_diff_heatmaps.py
```

Output:
- `output/slide_materials/accidents/`

### `scripts/visualization/filter_slide_materials.py`

Generates the F1-F6 quality-filter slide figures from
`output/quality/accident_quality.csv`.

```bash
uv run scripts/visualization/filter_slide_materials.py
uv run scripts/visualization/filter_slide_materials.py --output-dir output/my_slides
```

## Shared Library: `scripts/lib/`

Used by data scripts, training scripts, and notebooks. Notebooks import these
modules so heavy logic stays out of cells.

- `data_loaders.py` — `read_accidents`, `read_station_day`, `read_station_dates`, `read_historical_profile`, `read_vd_dim`, `load_samples`, `load_accident_csv`.
- `windows.py` — `SampleWindow`, `slice_station_df`, `rasterize_window`, `coverage`, `subset`, `matrix_to_csv_dataframe`, `absolute_mileage_axis_m`.
- `visualization.py` — canonical heatmap renderer. Public API: `heatmap`, `diff_heatmap`, `compare_heatmaps`. Named style presets: `STYLE_NOTEBOOK`, `STYLE_REPORT`, `STYLE_SLIDE`. Shared constants: `SPEED_MIN=0`, `SPEED_MAX=140`, `DIFF_ABS_MAX=40`. Full spec in `docs/VIZ_SPEC.md`.
- `inference.py` — two surfaces: `load(name, run_id, experiment, models_root)` loads trained U-Net checkpoints; `predict_accident_impact(accident, data_root, fetch_live_vd, forward_fn)` orchestrates validate → live fetch/offline fetch → historical baseline → tensor assembly → model forward → post-process. The forward step still accepts an injectable `forward_fn` for fake-model smoke tests. Contract in `docs/DEPLOYMENT.md` remains for future GUI work.
- `live_vd.py` — pure library. `fetch_live_station_5m(start, end, freeway, direction, mileage_m, ...)` hits MOTC for the requested time range, filters at parse-time to only VDs within the accident's spatial window, returns station-5m DataFrame + `FetchStatus`. No disk writes. Used by `predict_accident_impact` and the future GUI. Smoke test: `scripts/demo/live_vd_smoke.py`.
- `metrics.py` — `mae`, `rmse`, `f1_congestion`, `impact_time_min`.
- `calendar.py` — Taiwan day-class helper used by historical-profile and visualization code.
- `congestion_mask.py` — canonical `origin_cc` mask logic used by loss, eval, diagnostics, and mask precomputation.
- `model_architectures.py` — shared `SimpleUNet` / `FeatureFusionUNet` / `WindowConfig` definitions used by inference and dashboards.

## Baseline Scripts

Paper Table 2 baselines share `scripts/baselines/_harness.py`, which reuses the
same coverage filter, time split, target window, `origin_cc` mask, and metric
panel as the U-Net candidates. As of 2026-06-18, LSTM and XGBoost smoke outputs
exist; full 80-epoch / 3-seed results still need to be run for the paper table.

### `scripts/baselines/_harness.py`

Shared loader / scorer for paper baselines. When executed directly it scores the
historical-baseline prediction on the validation split as a self-test.

```bash
uv run scripts/baselines/_harness.py
```

### `scripts/baselines/lstm_percell.py`

Baseline 1a: spatially blind per-cell LSTM. Each sample uses only the six
5-minute observed-VD steps from `-30` through `-5` minutes, represented as
speed plus validity. It receives no historical profile, future covariate,
incident feature, space offset, or neighbouring-cell input, and directly emits
the 30 post-incident steps. The supported matched-support loss ablation is
`full_field_mae`, `full_field_speed60_weighted`, and
`full_field_aic_weighted`.

```bash
uv run scripts/baselines/lstm_percell.py --smoke
uv run scripts/baselines/lstm_percell.py --epochs 80 --seeds 42,7,123 --loss full_field_mae
uv run scripts/baselines/lstm_percell.py --epochs 80 --seeds 42,7,123 --loss full_field_speed60_weighted
uv run scripts/baselines/lstm_percell.py --epochs 80 --seeds 42,7,123 --loss full_field_aic_weighted
```

### `scripts/baselines/xgboost_cellwise.py`

Baseline 1b: cell-wise tabular XGBoost over pre-event summaries, historical
baseline, position/horizon, and leakage-free t0 incident features.

```bash
uv run scripts/baselines/xgboost_cellwise.py --smoke
uv run scripts/baselines/xgboost_cellwise.py --seeds 42,7,123
```

## Model Scripts

All model scripts read `data/accident_samples_5m.npz`, then slice the locked
61 x 61 superset to the training window requested by CLI. Current U-Net
defaults are 25 km upstream + 10 km downstream and -30 to +145 minutes
(`-30~+150 min` in horizon shorthand: target cells `0, 5, ..., 145`).
The matrix spatial axis is downstream-to-upstream (`-10 ... 0 ... +25/+50`);
loss and operational metrics use `space_offset_km >= 0` as upstream.

Both U-Net scripts are locked to `--input-type vd+hist` and exactly three matrix
channels: observed VD speed, observed-VD validity, and causal historical speed.
The historical speed plane comes from the aligned historical-matrix NPZ and is
nearest-space-filled within each time column. The former historical-validity
fourth channel and coordinate/input-type ablations are excluded. The training
target remains the observed VD post-incident field.
By default, `unet_matrix.py`, `unet_feature_fusion.py`, and
`timesfm_covariate.py` drop metadata `filled_ratio < 0.80` and do not apply
extra pre-window coverage or traffic-shape filters unless requested by CLI.

### `scripts/models/unet_matrix.py`

Speed-matrix U-Net baseline. Reads accident sample NPZ, slices a target window,
and trains a 4-layer U-Net with congestion-weighted L1 loss.

```bash
uv run scripts/models/unet_matrix.py --epochs 50
uv run scripts/models/unet_matrix.py --upstream-km 26 --downstream-km 5 --time-after-min 115
```

Output (per run):
- `output/models/unet_matrix/<run_id>/checkpoints/best_model.pth` and `last_model.pth`
- `output/models/unet_matrix/<run_id>/plots/curves.png`, `predictions_best4.png`, `predictions_worst4.png`
- `output/models/unet_matrix/<run_id>/eval/metrics.csv`, `summary.txt`
- `output/models/unet_matrix/<run_id>/run_config.json`

### `scripts/models/unet_feature_fusion.py`

Feature-fusion U-Net with three injection methods (`direct_concat`,
`mid_inject`, `attention`) and an optional TIID decay head. Joins the 33-feature
incident table by `accident_id`.

```bash
uv run scripts/models/unet_feature_fusion.py --epochs 30
uv run scripts/models/unet_feature_fusion.py --methods direct_concat,mid_inject --with-tiid
```

Output (per run):
- `output/models/unet_feature_fusion/<run_id>/<experiment>/{checkpoints,plots,eval}/`
- `output/models/unet_feature_fusion/<run_id>/comparison_summary.csv` and `.md`

### `scripts/models/timesfm_covariate.py`

> **Status: deferred (not in Phase 3 round-1).** Kept as a future / weak baseline. Champion selection per [ROADMAP.md](ROADMAP.md) Phase C runs only `unet_matrix` and `unet_feature_fusion`.

Lightweight ridge-regression covariate benchmark. Reduces each accident to a
spatial-mean speed series, compares context-only vs context + 33 incident
features + blockage decay + cyclic time covariates.

```bash
uv run scripts/models/timesfm_covariate.py
uv run scripts/models/timesfm_covariate.py --max-samples 200
```

Output (per run):
- `output/models/timesfm_covariate/<run_id>/comparison_summary.csv` and `.md`
- `output/models/timesfm_covariate/<run_id>/per_sample_metrics.csv`
- `output/models/timesfm_covariate/<run_id>/run_config.json`

## Experiment Scripts

### `scripts/experiments/run_loss_window_ablation.py`

Single-process loss / upstream-window ablation for `unet_matrix` (eval_protocol_v1).
Loads data once and reuses the train/val/test DataLoaders across all grid cells
(matrix window fixed at upstream 25 / downstream 10 km; only the loss-focus window,
mask, and weight vary), then a unified evaluator re-scores every checkpoint on a
FIXED metric panel (eval windows {10,15,20,25} km x masks {origin_cc, speed<60} +
cell_MAE / recovery_AE@10 / impact_AE@20 / F1@20), decoupled from each run's loss
zone so windows are comparable. Top-k cells by val origin_cc_MAE@20 get extra seeds.

```bash
uv run scripts/experiments/run_loss_window_ablation.py --smoke   # 2 cells x 2 epochs, timing
uv run scripts/experiments/run_loss_window_ablation.py           # full grid, 50 epochs
```

Output: `output/models/loss_window_ablation/<run_ts>/{summary.csv, selection.csv,
per_accident_metrics.csv, report.md, cells/<cell_id>/best_model.pth}` + a report copy
to `reports/loss_window_ablation_<date>.md`.

### `scripts/experiments/select_composite.py`

Composite sensitivity table over an ablation `summary.csv` (eval_protocol_v1 §5).
Baseline-relative weighted sum of `origin_cc_MAE@20` / `cell_MAE` / `impact_AE@20` /
`recovery_AE@10` (cell_mae baseline = 1.000); reports three preference schemes.
It is no longer the sole champion selector; paper comparison uses the full metric panel.

```bash
uv run scripts/experiments/select_composite.py            # newest ablation run
uv run scripts/experiments/select_composite.py --summary <path/summary.csv>
```

Output: `reports/loss_window_composite_<date>.md`.

### `scripts/experiments/run_unet_backbone_ablation.py`

Orchestrates locked-protocol long runs for `unet_matrix` and
`unet_feature_fusion` across backbone choices. This is how the current top
candidates (`matrix/simple`, `feature_fusion/attention/residual_attention`) were
reduced.

```bash
uv run scripts/experiments/run_unet_backbone_ablation.py --dry-run
uv run scripts/experiments/run_unet_backbone_ablation.py --epochs 80
```

Output:
- `output/models/unet_backbone_ablation/<run_ts>/`
- report copies under `reports/` when generated.

### `scripts/experiments/run_speed_pace_ablation.py`

Matrix U-Net ablation for speed + true-pace hybrid loss. Kept as a diagnostic /
negative-result experiment; not the locked paper loss.

```bash
uv run scripts/experiments/run_speed_pace_ablation.py --smoke
uv run scripts/experiments/run_speed_pace_ablation.py --epochs 30
```

### `scripts/experiments/run_origin_cc_pace_aux_ablation.py`

Origin-cc weighted loss plus true-pace auxiliary grid. Used to test whether the
pace auxiliary term improves the locked `origin_cc` setup.

```bash
uv run scripts/experiments/run_origin_cc_pace_aux_ablation.py --smoke
uv run scripts/experiments/run_origin_cc_pace_aux_ablation.py --epochs 30
```

## Analysis And Diagnostics

### `scripts/analysis/audit_vd_inventory.py`

Data re-audit step 1 (2026-09). Scans the whole `vd_station_5m` layer one day
at a time with DuckDB and writes a resumable per-day cache plus daily / monthly /
per-VD summaries, corridor 1-km x month coverage, missing-day and anomaly-day
lists, figures, and a comparison of candidate accident-cohort windows (locked
2024-04..2025-05 vs extended 2023-08..2026-04 vs full span). Read-only with
respect to `data/` and `reports/`. Colab shell:
`notebooks/audit/01_vd_inventory_colab.ipynb`.

```bash
uv run scripts/analysis/audit_vd_inventory.py --resume                      # full span, resumable
uv run scripts/analysis/audit_vd_inventory.py --start-date 2024-04-01 --end-date 2024-04-07
uv run scripts/analysis/audit_vd_inventory.py --summary-only                # rebuild tables/figures from daily/
uv run scripts/analysis/audit_vd_inventory.py --corridor N1:100.8-251.1 N3:110.9-269.0
```

Output (`output/data_audit/vd_inventory/`):
- `daily/vd_day_YYYYMMDD.parquet` per-(day, VD) stats; resume cache
- `daily_summary.csv`, `monthly_summary.csv`, `vd_summary.csv`, `corridor_km_month.csv`, `vd_churn_daily.csv`
- `missing_days.csv`, `anomaly_days.csv`, `file_inventory.csv`
- `report.json`, `SUMMARY.md`, `figures/*.png`

### `scripts/analysis/audit_accident_sources.py`

Data re-audit step 1 (2026-09). Reconciles dispatch, traffic-control SMS
(事故處理時間紀錄表), LINE, and debris logs: monthly coverage per source inside the
paper corridor, field completeness, dispatch duplicates, dispatch-SMS and
dispatch-LINE matching at several tolerances with report-delay, duration, and
lane-blockage comparisons. Read-only on `data/`. Output:
`output/data_audit/accident_sources/`.

```bash
uv run scripts/analysis/audit_accident_sources.py
```

### `scripts/analysis/audit_mainline_only_rebuild.py`

Data re-audit step 2b (2026-09). Rebuilds the 473 locked test accidents with the
locked recipe (A, all VDs; reproduces the frozen target exactly) and with
mainline-only VDs (B), reports how congested cells, observed TRT, and ACR change,
and re-scores the frozen ensemble predictions against both targets. Output:
`output/data_audit/mainline_rebuild/`.

```bash
uv run scripts/analysis/audit_mainline_only_rebuild.py
uv run scripts/analysis/audit_mainline_only_rebuild.py --limit 40 --output-dir tmp/mainline_smoke
```

### `scripts/data/fetch_cwa_hourly_archive.py`

Fetches hourly CWA station observations (2023-2026) for stations within 15 km of the
corridor from the public CODiS mirror (no API key), writes
`data/weather/cwa_hourly_corridor.parquet` plus station/coverage tables.

```bash
uv run scripts/data/fetch_cwa_hourly_archive.py --start-year 2023 --end-year 2026
```

### `scripts/features/build_weather_candidates.py`

Builds the timestamped per-accident weather candidate CSV that
`build_incident_features.py --weather-csv` needs for causal matching (nearest station
per variable, 3-hour lookback). Output: `data/weather/weather_per_task_observations.csv`.

```bash
uv run scripts/features/build_weather_candidates.py
uv run scripts/features/build_weather_candidates.py --accident-csv data/accident_data_sms.csv --output data/weather/weather_per_task_observations_sms.csv
```

### `scripts/analysis/measure_congestion_extent.py`

Measures accident-caused upstream congestion extent under several mask variants
and writes per-accident CSVs plus reports.

```bash
uv run scripts/analysis/measure_congestion_extent.py
uv run scripts/analysis/measure_congestion_extent.py --variants origin_cc --primary-variant origin_cc
```

### `scripts/analysis/review_origin_cc_weight.py`

Builds visual/report material explaining the locked `origin_cc` weighted loss.

```bash
uv run scripts/analysis/review_origin_cc_weight.py
```

Output:
- `output/analysis/origin_cc_weight_review/`
- `reports/origin_cc_weight_review_<YYYYMMDD>.md`

### `scripts/analysis/measure_mask_recurrent_overlap.py`

Quantifies whether the stored `origin_cc` masks leak into recurrent congestion,
pre-accident cells, downstream cells, or outside-cone cells. This supports the
paper's mask-validation section.

```bash
uv run scripts/analysis/measure_mask_recurrent_overlap.py
uv run scripts/analysis/measure_mask_recurrent_overlap.py --plot --min-time-min 0 --min-space-km 0 --max-space-km 20
```

### `scripts/analysis/plot_mask_construction.py`

Generates the paper-style stage-by-stage figure explaining how `origin_cc` masks
are circled.

```bash
uv run scripts/analysis/plot_mask_construction.py
uv run scripts/analysis/plot_mask_construction.py --accident-id 2024110787
```

Output:
- `output/analysis/mask_construction_<accident_id>.png`
- `output/analysis/mask_gallery.png`
- `output/analysis/mask_failure_modes.png`

### Recurring-Congestion Helpers

These scripts are one-off diagnostics for understanding recurring congestion and
background bottlenecks. They are not default pipeline steps, but their outputs
are referenced by reports and paper limitations.

- `scripts/analysis/find_recurring_congestion.py`
- `scripts/analysis/find_persistent_low_speed_vds.py`
- `scripts/analysis/find_congested_baseline_accidents.py`
- `scripts/analysis/measure_mask_recurrent_overlap.py`
- `scripts/analysis/plot_origin_cc_pace_aux_ablation.py`

## Notebooks

User-facing entry points. Each notebook is a thin orchestrator over
`scripts/...` CLI commands and `scripts.lib`.

- `notebooks/00_setup_and_mount.ipynb` - Drive mount, uv install, dataset sanity check.
- `notebooks/01_build_data_layers.ipynb` - run station / profile / sample / feature builders in order.
- `notebooks/02_explore_data.ipynb` - accident counts, coverage histogram, sample observed heatmap.
- `notebooks/03_accident_sms_dispatch_review.ipynb` - SMS / dispatch label-quality review.
- `notebooks/03_db_explorer.ipynb` - local data exploration helper.
- `notebooks/04_raw_vd_query_benchmark.ipynb` - Colab widget demo comparing DuckDB + raw Parquet query speed with pandas CSV.
- `notebooks/05_dashboard.ipynb` - Colab dashboard launcher.
- `notebooks/06_build_sms_samples.ipynb` - SMS prep: standardize SMS xlsx, build SMS VD matrices (`--tag sms`) and availability-schema features for the pre-dispatch period.
- `notebooks/build_station_full_range.ipynb` - full-range station build runner.
- `notebooks/build_historical_profile.ipynb` - historical-profile build runner.
- `notebooks/train/_template.ipynb` - copy-from skeleton when adding a new architecture.
- `notebooks/train/unet_matrix.ipynb` - baseline U-Net + window-sweep cell.
- `notebooks/train/unet_feature_fusion.ipynb` - feature fusion + optional TIID.
- `notebooks/train/timesfm_covariate.ipynb` - ridge covariate benchmark (deferred from Phase 3 round-1).
- `notebooks/train/ablation_time_progressive.ipynb` - t+0/t+5/t+15 exploratory ablation record.
- `notebooks/train/unet_unified_comparison.ipynb` - unified U-Net comparison / TIID exploration record.
- `notebooks/apply/accident_workflow.ipynb` - end-user workflow: observed + historical baseline + t+0/t+5/t+15 model prediction.

Pre-existing notebooks left in place but not part of the canonical workflow:
- `notebooks/accident_heatmap_colab.ipynb` - older heatmap explorer.

## Remaining Research Items

- Verify dashboard prediction rendering against `notebooks/apply/accident_workflow.ipynb`.
- Fix a paper case-study set and export observed / historical / predicted / diff figures.
- Run full 80-epoch / 3-seed top-candidate reruns, baseline rows, and TIID diagnostics for the conference-paper table.
- Add real weather CSV/API features only as a new dataset version; current locked feature rows have weather columns zero-filled.
- If model selection uses the 2026-05-25 upstream-axis loss/metric correction, rerun the affected ablation before updating the official champion.
- GUI live-fetch target-network smoke and PyInstaller clean Windows 11 packaging are future work.

## Will Not Carry Forward

- `scripts/data/download_vd.py`, `download_vd_local.py`, `process_vd.py` — superseded by `*_drive.py`.
- `scripts/spacetime/convert_spacetime_csv_to_matrix.py` — old long-form CSV converter is obsolete now that `build_accident_samples.py` writes matrices directly.
- `scripts/pipeline/run_pipeline.py` — wrapper deprecated by Colab-first notebook orchestration.
- `scripts/presentation/*`, `scripts/reporting/*` — one-off briefing artifacts; new presentations will be re-built on demand.
- `scripts/app/server.py` — web server irrelevant to the Colab workflow.
- Root-level `vd_preprocessing.py`, `spacetime.py`, `feature_parsers.py`, `config.py` — utilities folded into the new scripts/lib modules and inline.
