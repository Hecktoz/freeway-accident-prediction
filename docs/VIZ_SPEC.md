# Heatmap Visualization Spec

這份文件定義本專案所有熱圖（速度時空圖）的輸入合約、Style 規格、輸出路徑慣例，
以及各入口點（腳本、notebook、dashboard）的對應關係。
任何新的繪圖程式碼都應遵守本規格。

---

## 1. 輸入合約

所有繪圖函式以下列四個陣列為標準輸入。

| 參數 | Shape | dtype | 說明 |
|------|-------|-------|------|
| `matrix` | `(n_space, n_time)` | float | 速度值 km/h，缺值填 `NaN` |
| `space_axis_km` | `(n_space,)` | float | 相對事故位置（km）；**上游為正、下游為負** |
| `time_offsets_min` | `(n_time,)` | float | 相對事故發送時間（分鐘）；事故前為負 |
| `accident_meta` | dict | None | — | 見下表（均可選） |

### `accident_meta` 可接受的鍵

| 鍵 | 型別 | 用途 |
|----|------|------|
| `accident_id` | str | 標題標注 |
| `freeway` | str | 標題（例：`N1`） |
| `direction` | str | 標題（`N/S/E/W`） |
| `mileage_km` | float | 標題裡的絕對里程 |
| `report_time` | Timestamp | x 軸改顯示絕對時間（HH:MM） |
| `arrived_time` | Timestamp | None | 到達時間線（藍色虛線） |
| `completed_time` | Timestamp | None | 處理完成時間線（綠色虛線） |

### 軸方向慣例

```
y 軸（space_axis_km）
 +50 km  ← 上游 (upstream)
 …
 0 km    ← 事故點（紅點 / 水平虛線）
 …
 −10 km  ← 下游 (downstream)

x 軸（time_offsets_min）
 −60 min    0 min    +240 min
 （事故前）（發送）（事故後）
             ↑
        紅色垂直虛線
```

### NPZ 來源的軸計算

從 `load_samples()` 讀到的 `AccidentSamples` 轉換方式：

```python
space_axis_km = samples.space_offsets_m.astype(float) / 1000.0
time_offsets_min = samples.time_offsets_min.astype(float)
matrix = samples.matrix[i]  # shape (n_space, n_time)
```

Locked `data/accident_samples_5m.npz` uses a 61 × 61 superset window:

- space offsets: `-10 … 0 … +50 km`（上游為正）
- time offsets: `-60 … 0 … +240 min`

Current U-Net training/inference checkpoints slice this superset to 36 × 36:

- space offsets: `-10 … 0 … +25 km`
- time offsets: `-30 … 0 … +145 min`
- the pre-accident input is `-30 … -5 min`; `t=0` is predicted, not provided as input.
- Project shorthand `-30~+150 min` maps to this 5-minute endpoint convention: target columns are `0, 5, ..., 145`, which cover a 150-minute post-incident horizon without adding a 37th column at `+150`.
- `scripts/visualization/plot_heatmap.py` and `scripts/visualization/accident_heatmap.py` render the 61 × 61 audit window by default; pass `--research-window` to render the 36 × 36 research/model subset.

---

## 2. Style 預設

**定義於** `scripts/lib/visualization.py`，`STYLE_MAP` 字典可用名稱查詢。

| 預設 | figsize | dpi | font_size | title_size | 用途 |
|------|---------|-----|-----------|------------|------|
| `STYLE_NOTEBOOK` | 12 × 7 | 100 | 10 pt | 11 pt | Colab inline 顯示 |
| `STYLE_REPORT` | 14 × 8 | 150 | 11 pt | 12 pt | 論文 / PDF 報告 (**預設**) |
| `STYLE_SLIDE` | 14 × 8 | 150 | 13 pt | 15 pt | 投影片嵌入（自然比例貼入排版） |

所有預設共用相同的色彩規格：

| 項目 | 值 |
|------|-----|
| speed cmap | `RdYlGn` |
| speed 範圍 | 0 – 140 km/h |
| diff cmap | `RdBu` |
| diff 範圍 | ±40 km/h |

這些常數（`SPEED_MIN`, `SPEED_MAX`, `DIFF_ABS_MAX`, `SPEED_CMAP`, `DIFF_CMAP`）
從 `scripts/lib/visualization.py` 匯出，`scripts/dashboard/plotting.py` 也直接
import 這些常數，確保靜態輸出與互動 dashboard 色階一致。

### TRB 論文 heatmap 固定規格

TRB 論文內呈現 observed、historical、LSTM 或 U-Net speed field 的 heatmap
一律使用 `RdYlGn`（低速紅、中速黃、高速綠）、20–140 km/h，colorbar ticks
固定為 20、40、60、80、100、120、140 km/h。同一 figure 內的 panels 必須
共用相同範圍與 ticks。`cividis` 或其他藍–黃色階不得用於論文 speed heatmap。

差值圖仍使用以 0 為中心的 `RdBu` 類 diverging scale 與 ±40 km/h；若差值
定義的正負方向與共用函式相反，應反轉 colormap，讓顏色語意保持一致。

要微調單一屬性，用 `dataclasses.replace()`：

```python
from dataclasses import replace
from scripts.lib.visualization import STYLE_REPORT

my_style = replace(STYLE_REPORT, dpi=200)
```

---

## 3. 公開 API（`scripts/lib/visualization`）

```python
from scripts.lib.visualization import heatmap, diff_heatmap, compare_heatmaps, STYLE_REPORT

# 單張速度熱圖
fig = heatmap(matrix, space_axis_km, time_offsets_min, accident_meta,
              style=STYLE_REPORT, output_path="output/fig.png")

# 差值熱圖（observed − baseline）
fig = diff_heatmap(obs - baseline, space_axis_km, time_offsets_min, accident_meta,
                    style=STYLE_SLIDE)

# 多張並列比較
fig = compare_heatmaps(
    [observed, predicted, observed - predicted],
    ["觀測", "預測", "誤差"],
    space_axis_km, time_offsets_min, accident_meta,
    style=STYLE_REPORT,
    output_path="output/compare.png",
)
```

`compare_heatmaps` 自動偵測差值矩陣（含負值）並套用 diff colormap。

---

## 4. 入口點對應

| 用途 | 入口 | CLI / import |
|------|------|-------------|
| **批次輸出（NPZ 來源）** | `scripts/visualization/plot_heatmap.py` | `uv run scripts/visualization/plot_heatmap.py --style report` |
| **批次輸出（station Parquet 來源）** | `scripts/visualization/accident_heatmap.py` | `uv run scripts/visualization/accident_heatmap.py --limit 5` |
| **互動瀏覽（Dash）** | `scripts/visualization/accident_heatmap_web.py` | `uv run scripts/visualization/accident_heatmap_web.py` |
| **Notebook inline** | `scripts/lib/visualization.py` | `from scripts.lib.visualization import heatmap, STYLE_NOTEBOOK` |
| **Dashboard（Plotly）** | `scripts/dashboard/plotting.py` | `from scripts.dashboard.plotting import speed_heatmap` |
| **GUI（Gradio；future/deferred）** | `scripts/gui/app.py` | `uv run scripts/gui/app.py` |

---

## 5. 輸出路徑慣例

```
output/
├── heatmaps_notebook/       # plot_heatmap.py --style notebook
│   └── <accident_id>.png
├── heatmaps_report/         # plot_heatmap.py --style report ← 預設
│   └── <accident_id>.png
├── heatmaps_slide/          # plot_heatmap.py --style slide
│   └── <accident_id>.png
└── accident_heatmaps_5m/    # accident_heatmap.py 的舊輸出（保留相容）
    ├── <accident_id>.png
    └── metadata/<accident_id>.json
```

---

## 6. 在 Notebook 使用

```python
# 安裝 uv（Colab 首次執行）
# !pip install uv

import sys
sys.path.insert(0, "/content/drive/MyDrive/陽交大高速公路計畫/main")

from scripts.lib.visualization import heatmap, STYLE_NOTEBOOK
from scripts.lib.data_loaders import load_samples

samples = load_samples()
space_km = samples.space_offsets_m / 1000.0
time_min = samples.time_offsets_min.astype(float)
meta = samples.metadata.iloc[0]

accident_meta = dict(
    accident_id=str(meta["accident_id"]),
    freeway=str(meta.get("freeway", "")),
    direction=str(meta.get("direction", "")),
    mileage_km=float(meta["mileage_km"]),
    report_time=meta.get("report_time"),
)

fig = heatmap(samples.matrix[0], space_km, time_min, accident_meta,
              style=STYLE_NOTEBOOK)
fig.show()
```

---

## 7. 投影片貼入建議

- 使用 `STYLE_SLIDE`（150 dpi，13 pt 字型）
- PNG 尺寸約 2100 × 1200 px，貼入 PowerPoint 後縮放至版面寬度即可（不需裁切）
- 不需要把圖變成 16:9 — 熱圖本身的長寬比由資料決定（時間軸長度 × 空間範圍）

```bash
uv run scripts/visualization/plot_heatmap.py --style slide --accident-id 2024040004
```

PNG 位置：`output/heatmaps_slide/2024040004.png`
