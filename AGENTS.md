Read `README.md` for current repository context.
DO NOT modify lines before `## TODOs` in `README.md` unless the user explicitly requests a project-title/core-goal rewrite.
Use `uv run scripts/...py` to execute scripts and validate functionality.
DO NOT use files in `legacy/` as implementation baselines.
Keep `AGENTS.md` and `CLAUDE.md` synchronized exactly.
Reference `PITFALLS.md` before debugging data, preprocessing, or training issues.
This Drive directory is the new main project root; old OneDrive working copies are transitional only.

# Agent Knowledge Base

## Current Objective

Refine the model and prepare a conference-paper-ready research package.

Current priority: lock the post-2026-06-03 modeling protocol, rerun/confirm
top model candidates, produce case-study visualizations, and assemble paper
figures/tables. GUI delivery is deferred to future work; keep existing GUI
files/specs as references, but do not spend active effort on packaging or
bureau deployment unless explicitly requested.

The Colab-first data workflow remains the execution baseline. The only expected
local step is raw VD download through Drive for Desktop when new data are
needed:

```powershell
uv run scripts/data/download_vd_drive.py --start-date 2024-04-01 --end-date 2025-05-31
```

All downstream processing should be runnable in Colab from:

```text
/content/drive/MyDrive/陽交大高速公路計畫/main
```

## Execution Convention

Use direct script-path execution so `uv` applies each script's inline dependency block.

Examples:

```bash
uv run scripts/data/build_station_vd_drive.py --start-date 2024-04-01 --end-date 2024-04-01
uv run scripts/data/download_vd_drive.py --start-date 2024-04-01 --end-date 2024-04-01 --dry-run
```

In Colab, install `uv` first if needed:

```bash
pip install uv
```

### Local Python Interpreter (Windows)

On this Windows host, use one of these — do NOT try `python3`:

- `uv run <script>` — preferred for project scripts (applies inline deps)
- `uv run python -c "..."` — preferred for ad-hoc snippets
- `python` or `py` — both resolve to Python 3.12.2 if needed without uv

`python3` is **not installed** and triggers the Microsoft Store stub (exit 49,
"Python was not found..."). Never invoke `python3` on this machine.

## Repository Layout

- Documentation:
  - `README.md`
  - `PITFALLS.md`
  - `docs/DATA_DESIGN.md`
  - `docs/COLAB_MIGRATION.md`
  - `docs/SCRIPT_INDEX.md`
  - `docs/DEPLOYMENT.md` (deferred/future GUI delivery contract)
  - `docs/VIZ_SPEC.md` (heatmap input contract, style presets, output path conventions)
- Data scripts:
  - `scripts/data/download_vd_drive.py`
  - `scripts/data/build_station_vd_drive.py`
- Visualization:
  - `scripts/lib/visualization.py` — canonical renderer; exports `STYLE_NOTEBOOK/REPORT/SLIDE` and `SPEED_MIN/MAX`, `DIFF_ABS_MAX`
  - `scripts/visualization/plot_heatmap.py` — **standard batch CLI** (NPZ source, `--style report|slide|notebook`)
  - `scripts/visualization/accident_heatmap.py` — station-Parquet-based batch renderer
- Data:
  - `data/vd_raw_lane_1m_full/`
  - `data/vd_station_1m/`
  - `data/vd_station_5m/`
  - `data/metadata/`
  - `data/_vd_download_manifests/`

## Data Format Summary

### Raw VD Parquet

Path:

```text
data/vd_raw_lane_1m_full/source_year=YYYY/source_month=MM/vd_YYYYMMDD.parquet
```

Schema:

- `source_date` date32
- `source_hhmm` int16
- `timestamp` timestamp
- `vdid` string
- `freeway` string
- `direction` string
- `mileage_m` int32
- `link_id` string
- `lane_id` int8
- `lane_type` int8
- `speed` int16
- `occupancy` int16
- `status` int8
- `vol_s/spd_s/vol_l/spd_l/vol_t/spd_t` int16

### Station VD Parquet

Path:

```text
data/vd_station_{1m,5m}/source_year=YYYY/source_month=MM/vd_station_YYYYMMDD.parquet
```

Schema:

- `source_date` date32
- `timestamp` timestamp
- `vdid` string
- `freeway` string
- `direction` string
- `mileage_m` int32
- `speed_x10` int16
- `occupancy_x10` int16
- `vol_s/vol_l/vol_t` int32
- `n_lanes_valid` int16

### Metadata

- `data/metadata/vd_dim.parquet`
- `data/metadata/station_build_report.json`
- `data/metadata/vd_download_filter.json`

## Configuration Rules

- Prefer CLI arguments and project-root-relative paths.
- Avoid hardcoded local absolute paths in scripts.
- In Colab, default data root should resolve from the script location or be passed as `--data-root`.
- Use `mileage_m` integer for filtering; convert to km only for display.
- Keep timestamp as Taiwan local wall time without timezone metadata for compatibility with accident CSV times.

## Validation Checklist

After data script changes:

```bash
uv run scripts/data/build_station_vd_drive.py --help
uv run scripts/data/build_station_vd_drive.py --start-date 2024-04-01 --end-date 2024-04-01 --force
```

Then inspect:

- `data/vd_station_1m/.../vd_station_20240401.parquet`
- `data/vd_station_5m/.../vd_station_20240401.parquet`
- `data/metadata/vd_dim.parquet`
- `data/metadata/station_build_report.json`

## Do Not Carry Forward

- Old monthly raw files: `data/VD/vd_YYYYMM.parquet`
- Old raw schema with string timestamps only
- Old `data/processed/vd_1m.parquet` and `vd_5m.parquet` as canonical assumptions
- Matrix CSV as the long-term exchange format
