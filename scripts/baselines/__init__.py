"""Paper baseline models for the accident speed-recovery comparison (Table 2).

Every baseline is trained and scored on the SAME protocol as the U-Net
candidates (eval_protocol_v1), so its row drops straight into the model
comparison table and `scripts/experiments/select_composite.py`:

- 0a `historical_mean`   — non-learning (predict the historical baseline)
- 0b `persistence`       — non-learning (hold last pre-accident speed)
- 1a `lstm_percell`      — per-cell LSTM using only the six pre-incident VD steps
- 1b `xgboost_cellwise`  — per-cell gradient boosting (vd+hist+t0 features)
- `cellmae_unet`         — SimpleUNet trained with cell_mae loss (composite anchor)

Shared data loading, panel scoring and output writing live in `_harness.py`.
"""
