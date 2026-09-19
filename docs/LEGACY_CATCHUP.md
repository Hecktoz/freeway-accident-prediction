# Legacy Team Catch-up

更新時間：2026-06-18（Asia/Taipei）

這份文件給曾經做 `legacy/` 版本的成員快速重新接上目前主線。重點先講清楚：
**這個 Drive 目錄才是新主專案根目錄**，`legacy/` 和舊 OneDrive 工作副本只作為歷史參考，不再當作實作 baseline。

## 先讀哪些文件

建議照這個順序看：

1. [README.md](../README.md)：專案目標、資料狀態、目前 phase。
2. [PAPER_ROADMAP.md](PAPER_ROADMAP.md)：conference paper 的單一事實來源；目前主線以 TRB 20-page 研究包為目標。
3. [ROADMAP.md](ROADMAP.md)：從 dataset lock 到模型 refinement、case study、paper package 的完整工作地圖。
4. [SCRIPT_INDEX.md](SCRIPT_INDEX.md)：目前可用 scripts / notebooks 的入口索引。
5. [DATA_DESIGN.md](DATA_DESIGN.md)：raw / station / historical profile / accident sample 的資料格式。
6. [VIZ_SPEC.md](VIZ_SPEC.md)：paper / report heatmap 的視覺化合約。
7. [../PITFALLS.md](../PITFALLS.md)：資料、preprocessing、training 的已知坑位。

歷史 reports 可以看，但要分清楚時間點。特別注意：`reports/v1_champion.md` 是舊 champion record，
已在座標修正與 post-2026-06-03 loss/mask 鎖定前產生，**不能當 paper champion 引用**。

## 本次文件檢查結果

- README、ROADMAP、PAPER_ROADMAP 已經反映目前主線：GUI 交付 deferred，短期優先 conference-paper-ready research package。
- `reports/eval_protocol_v1.md` 已同步成 PAPER_ROADMAP 的口徑：**full metric panel 並報，composite 只作 sensitivity，不作唯一 champion selector**。
- `docs/SCRIPT_INDEX.md` 已補上目前實際存在的 baselines、experiments、analysis diagnostics、congestion mask、slide/paper visualization scripts。
- Markdown 死圖 / 錯路徑已修正；TIID 介紹補回兩張簡單 SVG 示意圖。
- `AGENTS.md` 與 `CLAUDE.md` 目前內容一致。

## 跟 legacy 最大的差異

- 資料層已從舊 monthly / processed CSV 思維，改成 **Drive + daily Parquet + NPZ**：
  raw lane 1m → station 1m / 5m → historical profile → accident sample matrix。
- 不再使用舊 `data/VD/vd_YYYYMM.parquet`、舊 `data/processed/vd_1m.parquet` / `vd_5m.parquet`、或 matrix CSV 當 canonical input。
- Scripts 以 `uv run scripts/...py` 直接執行，讓每支 script 的 inline dependency block 生效。
- 主要 downstream workflow 以 Colab + Google Drive 為準；本機只負責必要 raw VD 下載與少量 smoke test。
- GUI / PyInstaller / bureau delivery 目前列為 future work，不是本輪 active milestone。

## 目前資料進度

目前訓練鎖版是 `dataset_20260511_5m_v2`：

- 事故範圍：2024-04-01 ~ 2025-05-31。
- 事故樣本：`data/accident_samples_5m.npz`，2,036 accidents，61 x 61 matrix，0 failed。
- 預設訓練 keep rule：只剔除 `filled_ratio < 0.80`，保留 1,825 accidents；其他 shape filters 只作 diagnostics。
- Raw / station VD：`data/vd_raw_lane_1m_full/`、`data/vd_station_1m/`、`data/vd_station_5m/`。
- Historical baseline：`data/historical_speed_profile_5m/`，另有 `data/historical_matrices_5m.npz` 對齊每筆事故。
- Congestion mask：`data/congestion_masks_origin_cc_5m.npz`，用於 locked loss / eval。
- Incident features：舊鎖版 `incident_features_33.csv` 保留；新 availability schema `incident_features_44.csv` 已存在，用於後續資料擴充 / SMS / weather 口徑。
- SMS 資料已整理為 `data/accident_data_sms.csv`、`data/accident_samples_5m_sms.npz`、`incident_features_sms_44.csv`，但目前 paper 主線仍是 `dispatch_only`，SMS 不混入 locked main experiment。

## 目前模型與評估進度

已完成：

- Round-1 U-Net / feature-fusion ablation，保留為歷史紀錄。
- 2026-05-25 修正 upstream-distance 軸：`-10 km` 是下游，`0~+25/+50 km` 是上游。
- 2026-06-03 鎖定 loss/window：`upstream_congestion_weighted`、origin_cc mask、上游 0~20 km、w=4。
- 2026-06-16/18 完成 origin_cc mask leak 診斷，確認 effective recurrent contamination 約 2%，決定不再改 mask 圈法。
- Top candidates 收斂為：
  - Track A：`matrix/simple`
  - Track B：`feature_fusion/attention/residual_attention`
- `scripts/lib/inference.py` 已有 `load()` 與 `predict_accident_impact()`，dashboard / notebook / future GUI 共用此推論入口。
- Baseline scripts 已 scaffold：
  - `scripts/baselines/_harness.py`
  - `scripts/baselines/lstm_percell.py`
  - `scripts/baselines/xgboost_cellwise.py`
  2026-06-18 已有 smoke outputs，但 full 80ep / 3-seed paper results 尚未完成。

尚未完成：

- post-2026-06-03 locked protocol 下的 top-2 candidate 3-seed rerun。
- paper Table 2 的完整 baseline rows（0a historical mean、0b persistence、1a LSTM、1b XGBoost、cell_MAE anchor）。
- TIID A/B 與 decay diagnostic 的正式定位。
- 5-8 筆固定 case-study accidents 與 observed / historical / predicted / diff / recovery figures。
- paper-specific model-selection report / 更新後 champion record。

## 論文主線現在怎麼講

目前論文不是主張「新架構」，而是主張：

1. 用事故前 VD 時空速度場 + historical baseline，預測事故後 0-150 分鐘、上下游路段的速度恢復。
2. 用 `origin_cc` accident-causality mask，把事故造成的非經常性壅塞從經常性壅塞切出來，並同時作為 loss weighting 與 evaluation target。
3. 用真實中部高速公路資料做 empirical characterization，包含壅塞深度難預測、loss 是次要旋鈕、case-study visualization 與 failure modes。

正式比較不要只看單一指標。paper table 需要同時報：

- `origin_cc_MAE@20`
- `cell_MAE`
- `impact_time_AE@20`
- `recovery_AE@10`
- `F1@20`
- composite sensitivity

## 常用入口

資料層：

```powershell
uv run scripts/data/build_station_vd_drive.py --help
uv run scripts/data/build_historical_profile.py --help
uv run scripts/data/build_historical_matrices.py
uv run scripts/data/build_congestion_masks.py
```

模型 / baseline smoke：

```powershell
uv run scripts/models/unet_matrix.py --epochs 5 --eval-protocol v1 --input-type vd+hist --smoke
uv run scripts/baselines/_harness.py
uv run scripts/baselines/lstm_percell.py --smoke
uv run scripts/baselines/xgboost_cellwise.py --smoke
```

視覺化 / case study：

```powershell
uv run scripts/visualization/plot_heatmap.py --style report --research-window
uv run scripts/analysis/plot_mask_construction.py
uv run scripts/dashboard/run.py
```

本機 Windows 注意：不要用 `python3`。用 `uv run ...`，或臨時 snippets 用 `uv run python -c "..."`。

## 現在加入專案要避免什麼

- 不要從 `legacy/` 複製實作當新的 baseline。
- 不要重啟 GUI packaging，除非明確要求。
- 不要把舊 round-1 champion 當 paper 結論。
- 不要把 t+5 / t+15 feature timing 當主實驗；目前全 grounded on t0，t5/t15 因 feature 品質凍結。
- 不要 zero-fill weather 當真值；若天氣特徵補齊，要升新的 dataset version 並用 availability mask。
- 不要混用舊座標口徑；上游是 `space_offset_km >= 0`。

## 最短接手路徑

1. 讀 README + PAPER_ROADMAP，確認現在不是 GUI delivery，而是 paper package。
2. 跑一個 `uv run scripts/baselines/_harness.py` 或 model smoke，確認環境能讀 data / masks / splits。
3. 依 PAPER_ROADMAP §7 補 baseline full runs、top-2 3-seed reruns、case-study pack。
4. 所有新結果都要能追溯到 run id、dataset version、eval protocol、checkpoint hash。
