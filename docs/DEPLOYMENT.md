# 交付規劃 (Deployment)

> **Status (2026-06-16)**：GUI 交付已 deferred 為 future work。目前 active priority 是模型 refinement、case-study visualization 與 conference paper preparation。本文件保留作為未來若重啟高公局桌面版交付時的設計合約；不要把 PyInstaller 打包、局內網 live VD 測試或 Windows 11 乾淨環境 smoke 當成本輪 milestone，除非明確重啟。

最終要交付給高速公路局的成品：一個在局內單機執行的 GUI app，讓非技術背景的操作員輸入剛剛通報的事故，得到模型預測的速度時空 heatmap、影響時間估計、與各里程的恢復曲線。

Colab / notebook 工作流是開發用；局端工作站跑這個 GUI app。

## 操作員流程

1. 操作員在桌面開啟 app（Windows 雙擊執行）。
2. 表單輸入：事故時間、國道、方向、里程（km），以及 optional 的結構化特徵。
3. 點 **預測**。進度列依序顯示：取得即時 VD → 組合輸入張量 → 跑模型 → 渲染。
4. 結果區：heatmap 圖、影響分鐘數、各里程恢復曲線、覆蓋率資訊。

不需要跑腳本、不需要 mount Drive、不需要 Colab。

## 系統架構

```
┌───────────────────────────────────────────┐
│ GUI 層 — scripts/gui/app.py (Gradio)       │
│ • 表單元件                                 │
│ • 進度 / 狀態                              │
│ • 結果面板                                 │
└──────────────────┬────────────────────────┘
                    │ predict_accident_impact(accident, data_root, fetch_live_vd)
                    ▼
┌───────────────────────────────────────────┐
│ 推論層 — scripts/lib/inference.py          │
│ 1. 取即時 VD（事故前 ~75 分鐘）            │
│ 2. 在記憶體內組成 station-5m               │
│ 3. 讀取 historical_speed_profile_5m       │
│ 4. 組裝模型輸入張量（同 NPZ 形狀）         │
│ 5. 載入 checkpoint 跑 forward              │
│ 6. 推導影響時間 + 恢復曲線                 │
└──────────────────┬────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────┐
│ 局端工作站本地 bundle                       │
│ • Python（PyInstaller 一鍵資料夾）          │
│ • model_checkpoint.pt                      │
│ • historical_speed_profile_5m/ (~<100MB)   │
│ • metadata/vd_dim.parquet                  │
│ • central_vd_list.txt                      │
└───────────────────────────────────────────┘
```

## Bundle 中要打包的東西

| 檔案 | 工作站路徑 | 更新頻率 |
| --- | --- | --- |
| App 啟動程式 | `freeway_accident_app/` | 每次發版 |
| 模型 checkpoint | `freeway_accident_app/models/<arch>.pt` | 每次發版 |
| 歷史速度 profile | `freeway_accident_app/data/historical_speed_profile_5m/` | 每季 |
| VD 維度表 | `freeway_accident_app/data/metadata/vd_dim.parquet` | 每季 |
| 中部 VD 名單 | `freeway_accident_app/central_vd_list.txt` | 每季 |

**不**打包進去（訓練用，留在 Drive）：

- `vd_raw_lane_1m_full/` （~100 GB）
- `vd_station_1m/` （~9 GB）
- `vd_station_5m/` （~2.7 GB）
- `accident_samples_*` 各種訓練輸出

## 推論合約 (Phase 4)

GUI 唯一依賴的函式，定義在 [scripts/lib/inference.py](../scripts/lib/inference.py)。

```python
def predict_accident_impact(
    accident: AccidentInput,
    data_root: Path | str | None = None,
    fetch_live_vd: Callable[[datetime, datetime], pd.DataFrame] | None = None,
    forward_fn: Callable[[ModelInput], np.ndarray] | None = None,
    model_name: str | None = None,
    run_id: str | None = None,
    experiment: str | None = None,
    window: SampleWindow | None = None,
) -> AccidentPrediction:
    ...
```

### `AccidentInput`

| 欄位 | 型別 | 說明 |
| --- | --- | --- |
| `report_time` | `datetime` | 台灣本地時間，naive（不含時區），與 accident CSV 一致 |
| `freeway` | `str` | `N1` / `N3` / `N4` / `N6` |
| `direction` | `str` | `N` / `S` / `E` / `W` |
| `mileage_km` | `float` | 公里，例如 `147.0` |
| `features` | `dict | None` | 33 維結構化特徵（feature-fusion 模型用） |

### `AccidentPrediction`

| 欄位 | 形狀 / 型別 | 說明 |
| --- | --- | --- |
| `heatmap` | `np.ndarray (n_space, n_time)` | 預測速度（km/h） |
| `window` | `SampleWindow` | 軸的設定，與 heatmap 對齊 |
| `duration_min` | `float` | 速度回到 free-flow 所需分鐘數 |
| `recovery_curves` | `dict[float, np.ndarray]` | key = 絕對里程 km，value = 該里程切片速度序列 |
| `coverage` | `dict` | `vd_count`、`filled_ratio`、`missing_minutes`、`live_fetch_status` |
| `input_echo` | `dict` | 操作員輸入的快照，用於顯示和稽核 |

### 強制要求

- **純函式（pure function）。** `data_root` 由參數傳入；不能寫死 Colab 路徑或 `~/...` 預設。
- **Live fetch 可注入（injectable）。** `fetch_live_vd(start, end) -> DataFrame` 是參數，方便測試傳假資料。預設實作打 MOTC。
- **推論時不寫硬碟。** 即時 VD 直接組成記憶體 DataFrame，不寫日 Parquet、不寫 NPZ。Log 可以寫。
- **可重現（deterministic）。** 同一個 `accident` + 同一段 VD 讀值 + 同一個 checkpoint → 結果完全相同。任何隨機性要設 seed。
- **覆蓋率要顯示。** 操作員看到 fill ratio 和缺漏分鐘數，才能判斷是否信任這次預測。

## Live VD 取資料

[scripts/data/download_vd_drive.py](../scripts/data/download_vd_drive.py) 已經有非同步 MOTC 取檔 + XML 解析。推論用同樣的邏輯，但不寫日 Parquet。

抽到 [scripts/lib/live_vd.py](../scripts/lib/live_vd.py)：

```python
def fetch_live_station_5m(
    start: datetime,
    end: datetime,
    freeway: str,
    direction: str,
    mileage_m: int,
    *,
    upstream_km: int = 50,
    downstream_km: int = 10,
    concurrency: int = 16,
) -> tuple[pd.DataFrame, dict]:
    ...
```

關鍵設計：**只處理事故空間窗口內的 VD**。針對該事故的 (freeway, direction, mileage) 計算空間範圍，再用 `VDFilter(mode="ranges")` 在 XML 解析階段就把不相關的 VD 丟掉，所以即使全國有 2,000+ 個 VD，每分鐘的解析只會處理該事故路段需要的幾十個。

`download_vd_drive.py` 的日 Parquet 路徑保留不動；新模組只共用 fetch + parse 的核心。

## 打包 (Packaging)

三個選項，依操作員體驗排序：

1. **PyInstaller 一鍵資料夾**（建議）。打包成一個資料夾複製給對方。雙擊 `freeway_accident_app.exe` 啟動。約 300 MB。工作站不需要安裝 Python。
2. **uv 攜帶版**。把 repo + 一個 `run.bat` 包進去執行 `uv run scripts/gui/app.py`。建置簡單但工作站要先有 `uv`。
3. **Docker**。隔離乾淨，但局端 IT 大概會拒絕。

預設用 PyInstaller，目標 Windows 11（局端工作站）。

## 季度更新 (Quarterly Refresh)

Bundle 中有兩部分會隨時間衰減：

- `historical_speed_profile_5m/` — 用新累積的 `vd_station_5m` 重建一次。
- `vd_dim.parquet` — 有新 VD 上線或舊 VD 退役時重新匯出。

更新分發機制兩擇一（首次發版前要決定）：

- **Pull**：app 啟動時去檢查版本 URL，發現新版就提示就地升級。操作員按「是」。
- **Push**：直接 email / USB 把新資料夾給對方，操作員把新的覆蓋舊的。

## 風險清單（首次發版前必須處理）

- **MOTC API 在局內網路是否可達。** 如果防火牆擋住 `tisvcloud.freeway.gov.tw` 對外 HTTPS，整個 live-fetch 設計會失敗。**做 GUI 前先測試這一點。**
- **事故當下 VD 故障。** 覆蓋率面板要永遠顯示；訓練/品質 keep-list 的資料門檻是 `filled_ratio >= 0.8`，推論端低於 0.8 時應顯示「VD 覆蓋率偏低，模型結果可能不可靠」警示。
- **時鐘偏差。** MOTC 檔案以分鐘為刻度，「現在」由工作站時鐘算出。結果面板要顯示實際解析出的時間區間，讓任何偏差都看得到。
- **未來時間。** 拒絕 `report_time > now`。
- **里程超出 VD 覆蓋範圍。** 用 `vd_dim.parquet` 的路線範圍驗證，超出時給清楚錯誤訊息，不要回空 heatmap。

## v1 不做 (Out of Scope)

- 多人 / 並發推論（單機單操作員）。
- 歷史事故重播（資料已有，要再做也快，但不是這次要的）。
- 雲端部署（GPU / hosted URL）。客戶沒要求。
- 第二份事故來源（`交控事故簡訊紀錄表`）整合，見 [DATA_REVIEW.md §7](DATA_REVIEW.md#7-兩份事故資料對照)。

## 階段相依 (Phase Dependencies)

GUI app 依賴下列前置工作：

- Phase 2 全量跑 `build_historical_profile.py`（完成）。
- Phase 3/4 至少一個架構訓出 checkpoint 並可由 `load()` 載入（完成；t+0/t+5/t+15 見 `reports/feature_time_grouping.md`）。
- Phase 4 把 `predict_accident_impact` 補完到上面的合約（完成）。

若未來重啟 GUI 交付，剩餘風險會集中在環境端：PyInstaller 實際打包、乾淨 Windows 11 smoke test、以及局內網路是否可連到 MOTC live VD API。
