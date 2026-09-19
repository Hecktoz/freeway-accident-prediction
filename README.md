# Freeway Accident Impact-Time Prediction

Predicting how long a freeway accident disrupts traffic and how the resulting congestion spreads and recovers from vehicle-detector speed data and accident dispatch records. Two Taiwan freeways, a causal historical-baseline channel, a U-Net encoder–decoder, and an accident-induced-congestion mask used to keep evaluation focused on what a traffic manager actually cares about.

## Problem Framing

Framed as a supervised space–time regression task, closer to image-to-image translation than classical forecasting. Input: a pre-incident observation window plus a causal historical-baseline channel. Target: the full post-incident speed field, from which impact-duration and recovery-time are derived. An accident-induced-congestion mask addresses the label imbalance in both loss weighting and evaluation.

- **Input**: a short window of pre-incident speed observations, plus a causal historical-baseline channel — the expected speed for that location, weekday type, and time of day, estimated only from data recorded before the study period so it introduces no label leakage.
- **Target**: the full post-incident space–time speed field (distance × time), from which downstream scalar quantities — impact duration, recovery time — are derived at inference time.
- **Feature sources**: two raw sources are joined and feature-engineered into this representation — per-lane vehicle-detector (VD) speed telemetry (station-aggregated, 5-minute resolution) and the accident dispatch log (report time, freeway, direction, mileage).
- **Label imbalance**: only a small fraction of space–time cells are meaningfully affected by any given accident. An accident-induced-congestion (AIC) mask isolates that subset, both to reweight the training loss and to define evaluation metrics that aren't dominated by trivially easy free-flow cells.

## Modeling Approach

- **Representation**: each sample is encoded as a multi-channel space × time matrix rather than a flat feature vector, so the model can exploit spatial and temporal locality directly instead of treating each cell as independent.
- **Architecture**: a convolutional encoder–decoder (U-Net) performs the space–time regression. A feature-fusion variant conditions the network on structured incident metadata (lane blockage, weather, etc.) through a secondary input branch merged at the bottleneck.
- **Training objective**: a masked, weighted loss that upweights AIC-affected cells rather than optimizing a uniform MAE/MSE across the full matrix — a loss design driven directly by the label-imbalance problem above.
- **Validation strategy**: a chronological train / validation / test split, not random k-fold, to prevent temporal leakage from recurring seasonal patterns. A random split with matched subset sizes is retained only as a robustness check, never for model selection.
- **Baselines**: historical-mean and persistence models serve as non-learning references, so any reported lift is measured against a genuinely competitive floor rather than a strawman.

This project is part of ongoing research; a paper describing the full methodology and results is in preparation. Detailed metrics, ablations, and result tables are intentionally omitted from this repository pending submission — see `docs/` for data schema and pipeline design documentation, and the codebase for the full implementation.

## Repository Structure

```text
scripts/
├─ data/            # Ingestion & preprocessing: VD download, cleaning, accident-sample construction
├─ features/        # Feature engineering: structured incident feature extraction
├─ lib/             # Shared modules: data loaders, windowing, visualization, metrics, inference
├─ models/          # Model architectures (U-Net variants, TimesFM covariate baseline)
├─ baselines/       # Non-learning and classical baselines
├─ experiments/      # Loss / window / backbone ablation runners
├─ analysis/        # Failure-mode and congestion-propagation diagnostics
├─ visualization/   # Heatmap and figure generation CLIs
├─ dashboard/       # Internal Gradio dashboard for exploratory data analysis
└─ gui/             # Prototype desktop delivery app (future work, not actively developed)

notebooks/           # Colab-first workflow, run in numbered order
├─ 00_setup_and_mount.ipynb   # Environment setup
├─ 01_build_data_layers.ipynb # Data pipeline execution
├─ 02_explore_data.ipynb      # EDA / sanity checks
├─ train/                     # Training notebooks, one per model architecture
└─ apply/                     # Post-training inference / case-study workflow

docs/                # Data schema, pipeline design, and visualization spec documentation
packaging/           # Packaging assets for the prototype desktop app
```

Large raw data artifacts (raw/station-level VD parquet files, engineered feature arrays, model checkpoints) are not included in this repository. They're reproducible end-to-end from the scripts above and are kept in the team's private data store rather than version-controlled.

## Tech Stack

Python, PyTorch (model training), DuckDB + Parquet (feature store / data layer), Gradio (internal EDA dashboard), `uv` for reproducible script-level dependency pinning.

## Status

Research in progress — the data pipeline and model training are complete; current work is final model evaluation and case-study analysis ahead of a conference submission.

## Team

- **Advisor**: Prof. Yu-Ting Hsu
- **Contributors**: Che An Lin, Wen-Yuan Wu

This repository reflects Che An Lin's contribution to the project (data collection and cleaning, feature engineering, pipeline development, and U-Net model training), shared with the team's permission.
