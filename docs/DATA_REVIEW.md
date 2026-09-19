# 資料全面檢視文件

> 撰寫日期：2026-04-29
> 涵蓋範圍：VD 資料管線、靜態維度、事故資料（派遣系統 + 交控簡訊紀錄）

---

## 目錄

1. [原始 VD：XML → `vd_raw_lane_1m_full`](#1-原始-vd-xml--vd_raw_lane_1m_full)
2. [站點聚合：`vd_station_1m` / `vd_station_5m`](#2-站點聚合-vd_station_1m--vd_station_5m)
3. [Heatmap 視覺化](#3-heatmap-視覺化)
4. [VD 靜態資訊：`vd_dim.parquet`](#4-vd-靜態資訊-vd_dimparquet)
5. [事故資料：派遣系統](#5-事故資料派遣系統)
6. [事故資料：交控事故簡訊紀錄表](#6-事故資料交控事故簡訊紀錄表)
7. [兩份事故資料對照](#7-兩份事故資料對照)

---

## 1. 原始 VD：XML → `vd_raw_lane_1m_full`

### 1.1 上游資料來源

高速公路 VD 資料由交通部公路局（MOTC）每分鐘發布一個壓縮 XML 檔案：

```
https://tisvcloud.freeway.gov.tw/history/motc20/VD/{YYYYMMDD}/VDLive_{HHMM}.xml.gz
```

- 一天 1,440 個檔案（00:00 ~ 23:59，每分鐘一個）
- 每個檔案包含所有 VD 站點在該分鐘的車道流量觀測值
- 格式：gzip 壓縮的 XML

---

### 1.2 XML 結構與欄位說明

每個 XML 檔案的層次結構如下（簡化示意）：

```xml
<MOTC_VDLives>
  <VDLives>

    <!-- 一個 VD 站點 -->
    <VDLive>
      <VDID>VD-N1-N-22.990-N-LOOP</VDID>
      <DataCollectTime>2024-04-01T00:00:00+08:00</DataCollectTime>
      <Status>0</Status>  <!-- 0=正常, 1=異常 -->

      <LinkFlows>
        <LinkFlow>
          <LinkID>0000100102280A</LinkID>

          <Lanes>
            <!-- 一條車道 -->
            <Lane>
              <LaneID>0</LaneID>
              <LaneType>2</LaneType>
              <Speed>101</Speed>        <!-- 混合速度 km/h -->
              <Occupancy>4</Occupancy>  <!-- 佔有率 % -->

              <Vehicles>
                <Vehicle>
                  <VehicleType>S</VehicleType>  <!-- S=小型車 -->
                  <Volume>6</Volume>
                  <Speed>100</Speed>
                </Vehicle>
                <Vehicle>
                  <VehicleType>L</VehicleType>  <!-- L=大型車 -->
                  <Volume>0</Volume>
                  <Speed>0</Speed>
                </Vehicle>
                <Vehicle>
                  <VehicleType>T</VehicleType>  <!-- T=聯結車 -->
                  <Volume>1</Volume>
                  <Speed>96</Speed>
                </Vehicle>
              </Vehicles>
            </Lane>

            <!-- 第二條車道 -->
            <Lane>
              <LaneID>1</LaneID>
              ...
            </Lane>
          </Lanes>
        </LinkFlow>
      </LinkFlows>
    </VDLive>

    <!-- 下一個 VD 站點 ... -->
  </VDLives>
</MOTC_VDLives>
```

**XML 層次：** `VDLive` → `LinkFlows/LinkFlow` → `Lanes/Lane` → `Vehicles/Vehicle`

每一個 `<Lane>` 節點產生 parquet 中的一列（一個時間點、一個 VD、一條車道）。

---

### 1.3 VDID 命名規則

VDID 格式：`VD-{國道}-{方向}-{里程}-{後綴}`

解析正則：`^VD-(N\d+)-([NSEW])-([\d.]+)`

| VDID 範例 | 解析出的國道 | 方向 | 里程 (km) |
|---|---|---|---|
| `VD-N1-N-22.990-N-LOOP` | N1 | N (北) | 22.990 |
| `VD-N1-N-23-I-EN-1-圓山` | N1 | N (北) | 23.000 |
| `VD-N1-S-147-N-LOOP` | N1 | S (南) | 147.000 |
| `VD-N3-S-150-N-LOOP` | N3 | S (南) | 150.000 |
| `VD-N6-W-2-N-LOOP` | N6 | W (西) | 2.000 |

里程（km × 1000 取整）存為 `mileage_m`（公尺整數），顯示時再除以 1000。

---

### 1.4 下載與轉換流程（`download_vd_drive.py`）

```
上游 MOTC API
  ↓ 非同步 HTTP 下載（64 並發，httpx）
.xml.gz 檔案
  ↓ gzip 解壓縮（process pool）
.xml 原始文字
  ↓ lxml XML 解析
VD 原始列（Python tuples）
  ↓ VD 篩選（buffered-vd-list 模式）
篩選後列
  ↓ Timestamp 正規化（去掉 +08:00，保留本地時間）
  ↓ 型別轉換（PyArrow）
  ↓ 排序（freeway / direction / mileage_m / vdid / timestamp / lane_id）
  ↓ 寫入暫存 Parquet（zstd 壓縮）
本機暫存 Parquet
  ↓ 複製到 Google Drive
  ↓ 設為 online-only（attrib -P +U）
data/vd_raw_lane_1m_full/source_year=YYYY/source_month=MM/vd_YYYYMMDD.parquet
```

**VD 篩選邏輯（buffered-vd-list 模式）**

以 `central_vd_list.txt` 為目標 VD 清單，分析每條路線的里程範圍，再向兩端各加 70 km 緩衝，保留這個擴大範圍內的所有 VD。這確保事故前後各 50 km 分析窗口內的 VD 資料完整。

**Timestamp 處理**

原始 XML：`2024-04-01T00:00:00+08:00`
儲存值：`2024-04-01 00:00:00`（去掉時區資訊，保留台灣本地時間）
型別：目前 Parquet 讀出為 `timestamp[ms]`（無時區）

保留本地時間的原因：與事故 CSV 中的時間欄位直接對齊，避免時區轉換錯誤。

**每日進度追蹤（Manifest）**

每個日期對應一個 manifest JSON 記錄每分鐘的下載狀態：

```json
{
  "date": "2024-04-01",
  "complete": true,
  "counts": {
    "downloaded": 1439,
    "server_404": 1
  },
  "minutes": {
    "0000": "downloaded",
    "0001": "downloaded",
    ...
  }
}
```

狀態值：`downloaded` / `server_404` / `http_error` / `parse_error`

---

### 1.5 Parquet Schema

路徑：`data/vd_raw_lane_1m_full/source_year=YYYY/source_month=MM/vd_YYYYMMDD.parquet`

| 欄位 | 型別 | 說明 |
|---|---|---|
| `source_date` | date32 | 要求的來源日期 |
| `source_hhmm` | int16 | 分鐘檔案序號（0 = 00:00，1439 = 23:59） |
| `timestamp` | timestamp[ms] | 台灣本地時間（無時區，直接使用） |
| `vdid` | string | VD 站點 ID |
| `freeway` | string | 國道代碼（N1/N3/N4/N6） |
| `direction` | string | 行向（N/S/E/W） |
| `mileage_m` | int32 | 里程（公尺整數） |
| `link_id` | string | 路段 ID |
| `lane_id` | int8 | 車道 ID（0-based） |
| `lane_type` | int8 | 車道類型（目前觀測值均為 2） |
| `speed` | int16 | 車道混合速度（km/h，含異常值） |
| `occupancy` | int16 | 車道佔有率（%，含異常值） |
| `status` | int8 | VD 站狀態（0=正常, 1=異常） |
| `vol_s` | int16 | 小型車流量（輛/分鐘） |
| `spd_s` | int16 | 小型車速度（km/h） |
| `vol_l` | int16 | 大型車流量（輛/分鐘） |
| `spd_l` | int16 | 大型車速度（km/h） |
| `vol_t` | int16 | 聯結車流量（輛/分鐘） |
| `spd_t` | int16 | 聯結車速度（km/h） |

分區欄（Hive 分區，不在 schema 中但讀取時出現）：`source_year`, `source_month`

壓縮格式：`zstd`，Row Group Size：131,072

---

### 1.6 範例資料

以下為 2024-04-01 `VD-N1-N-22.990-N-LOOP`（國道1號北向22.990K）的前幾列：

| source_date | source_hhmm | timestamp | vdid | freeway | direction | mileage_m | link_id | lane_id | lane_type | speed | occupancy | status | vol_s | spd_s | vol_l | spd_l | vol_t | spd_t |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2024-04-01 | 0 | 2024-03-31 23:58:00 | VD-N1-N-22.990-N-LOOP | N1 | N | 22990 | 0000100102280A | 0 | 2 | 106 | 4 | 0 | 5 | 105 | 0 | 0 | 0 | 0 |
| 2024-04-01 | 0 | 2024-03-31 23:58:00 | VD-N1-N-22.990-N-LOOP | N1 | N | 22990 | 0000100102280A | 1 | 2 | 96 | 5 | 0 | 6 | 95 | 0 | 0 | 0 | 0 |
| 2024-04-01 | 1 | 2024-03-31 23:59:00 | VD-N1-N-22.990-N-LOOP | N1 | N | 22990 | 0000100102280A | 0 | 2 | 101 | 4 | 0 | 6 | 100 | 0 | 0 | 0 | 0 |
| 2024-04-01 | 1 | 2024-03-31 23:59:00 | VD-N1-N-22.990-N-LOOP | N1 | N | 22990 | 0000100102280A | 1 | 2 | 88 | 4 | 0 | 4 | 90 | 1 | 78 | 0 | 0 |

> `source_hhmm = 0` 對應 `timestamp = 2024-03-31 23:58:00`（下載的第0分鐘檔案記錄上一日23:58的資料，是正常現象。）

**觀測到的值域（2024-04-01 樣本）：**

| 欄位 | 觀測值域 | 備注 |
|---|---|---|
| freeway | N1, N3, N4, N6 | |
| direction | N, S, E, W | |
| lane_type | 2 | 目前僅見此值 |
| status | 0, 1 | 0=正常 |
| speed | -99 ~ 256 | 負值/極大值為異常 |
| occupancy | -99 ~ 111 | 負值/超過100為異常 |
| vol_s | -99 ~ 254 | 負值為異常 |

---

### 1.7 規模

| 指標 | 數值 |
|---|---|
| 訓練鎖版樣本範圍 | 2024-04-01 至 2025-05-31 |
| 該範圍天數 | 426 天 |
| 目前 Drive raw/station 檔案掃描 | 2021-06-29 至 2026-04-30，共 1,759 個 daily parquet |
| 單日列數（2024-04-01 樣本） | 約 688 萬列 |
| 總列數（估計） | 約 30.5 億列 |
| 總大小 | 約 16 GiB |

---

## 2. 站點聚合：`vd_station_1m` / `vd_station_5m`

### 2.1 處理目標

從車道層級（per lane, per minute）聚合成**站點層級**（per VD station, per minute），並清洗無效量測值，使資料可直接用於模型訓練與特徵工程。

由 `scripts/data/build_station_vd_drive.py` 執行。

---

### 2.2 處理流程（Raw → Station 1m）

```
vd_raw_lane_1m_full/vd_YYYYMMDD.parquet
  ↓ 讀取欄位：source_date, timestamp, vdid, freeway,
             direction, mileage_m, speed, occupancy,
             vol_s, vol_l, vol_t
  ↓ 清洗：保留 0 < speed <= 140 的車道列
  ↓ 清洗：occupancy 只在 0 <= occupancy <= 100 時才參與平均計算
  ↓ 以 (source_date, timestamp, vdid, freeway, direction, mileage_m) 分組
  ↓ 聚合（見下表）
  ↓ speed_x10 = round(mean_speed × 10)
  ↓ occupancy_x10 = round(mean_occupancy × 10) [nullable Int16]
  ↓ 排序（source_date / freeway / direction / mileage_m / vdid / timestamp）
  ↓ 寫入 zstd Parquet
vd_station_1m/source_year=YYYY/source_month=MM/vd_station_YYYYMMDD.parquet
```

**清洗規則說明：**

| 條件 | 行為 | 原因 |
|---|---|---|
| `speed <= 0` | 排除整條 lane 列 | 感測器離線（常見值：0, -99） |
| `speed > 140` | 排除整條 lane 列 | 明顯異常超速讀值（最大觀測值 256） |
| `occupancy < 0 或 > 100` | 佔有率計算時排除，但 lane 列保留 | 車流量計算仍有效 |

**聚合邏輯：**

| 輸出欄位 | 計算方式 | 來源欄位 |
|---|---|---|
| `speed_x10` | `round(mean(speed) × 10)` | `speed`（有效車道） |
| `occupancy_x10` | `round(mean(occupancy) × 10)` | `occupancy`（有效值） |
| `vol_s` | `sum(vol_s)` | `vol_s`（有效車道，負值先填 0） |
| `vol_l` | `sum(vol_l)` | `vol_l` |
| `vol_t` | `sum(vol_t)` | `vol_t` |
| `n_lanes_valid` | `count(speed)` | 通過篩選的車道數 |

---

### 2.3 speed_x10 / occupancy_x10 設計說明

為什麼不直接存浮點數：避免 float 的儲存開銷和精度問題，用整數存一位小數。

| 原始速度 | speed_x10 | 還原時 |
|---|---|---|
| 98.7 km/h | 987 | `987 / 10.0 = 98.7` |
| 101.3 km/h | 1013 | `1013 / 10.0 = 101.3` |
| 45.0 km/h | 450 | `450 / 10.0 = 45.0` |

`occupancy_x10` 為 **nullable Int16**（pandas extension type），允許 `pd.NA`（當該時間點所有車道的 occupancy 都無效時）。

---

### 2.4 站點 5 分鐘聚合（Station 1m → Station 5m）

```
vd_station_1m/vd_station_YYYYMMDD.parquet
  ↓ timestamp_5m = floor(timestamp, 5min)
  ↓ 以 (source_date, timestamp_5m, vdid, freeway, direction, mileage_m) 分組
  ↓ speed_x10 = round(mean(speed_x10))
  ↓ occupancy_x10 = round(mean(occupancy_x10))
  ↓ vol_s, vol_l, vol_t = sum
  ↓ n_lanes_valid = sum（所有 1m 內的有效車道數加總）
vd_station_5m/source_year=YYYY/source_month=MM/vd_station_YYYYMMDD.parquet
```

> 注意：5m 的 `n_lanes_valid` 是 5 個 1m 值的加總。
> 例如一個 VD 有 3 條有效車道，在 5 分鐘內都有資料，則 5m 的 `n_lanes_valid = 15`。

---

### 2.5 Schema（station_1m 與 station_5m 相同）

路徑：
- `data/vd_station_1m/source_year=YYYY/source_month=MM/vd_station_YYYYMMDD.parquet`
- `data/vd_station_5m/source_year=YYYY/source_month=MM/vd_station_YYYYMMDD.parquet`

| 欄位 | 型別 | 說明 |
|---|---|---|
| `source_date` | date32 | 來源日期 |
| `timestamp` | timestamp[ms] | 台灣本地時間（毫秒精度） |
| `vdid` | string | VD 站點 ID |
| `freeway` | string | 國道代碼 |
| `direction` | string | 行向 |
| `mileage_m` | int32 | 里程（公尺） |
| `speed_x10` | int16 | 站點平均速度 × 10（km/h × 10） |
| `occupancy_x10` | Int16 | 站點平均佔有率 × 10（nullable） |
| `vol_s` | int32 | 小型車流量（有效車道加總） |
| `vol_l` | int32 | 大型車流量（有效車道加總） |
| `vol_t` | int32 | 聯結車流量（有效車道加總） |
| `n_lanes_valid` | int16 | 有效車道數 |

---

### 2.6 範例資料

**station_1m（VD-N1-N-22.990-N-LOOP，2024-04-01 凌晨）：**

| timestamp | speed_x10 | occupancy_x10 | vol_s | vol_l | vol_t | n_lanes_valid |
|---|---|---|---|---|---|---|
| 2024-03-31 23:58:00 | 1010 | 45 | 11 | 0 | 0 | 2 |
| 2024-03-31 23:59:00 | 945 | 40 | 10 | 1 | 0 | 2 |
| 2024-04-01 00:00:00 | 880 | 70 | 18 | 0 | 1 | 2 |
| 2024-04-01 00:01:00 | 910 | 50 | 10 | 0 | 1 | 2 |
| 2024-04-01 00:02:00 | 835 | 40 | 9 | 1 | 0 | 2 |

> 解讀：`speed_x10 = 1010` → 速度 101.0 km/h；`n_lanes_valid = 2` → 此 VD 有 2 條有效車道。

**station_5m（同一 VD，5 分鐘取整後）：**

| timestamp | speed_x10 | occupancy_x10 | vol_s | vol_l | vol_t | n_lanes_valid |
|---|---|---|---|---|---|---|
| 2024-03-31 23:55:00 | 978 | 42 | 21 | 1 | 0 | 4 |
| 2024-04-01 00:00:00 | 912 | 49 | 58 | 2 | 3 | 10 |
| 2024-04-01 00:05:00 | 928 | 39 | 46 | 5 | 1 | 10 |

> 解讀：`n_lanes_valid = 10` = 2 車道 × 5 分鐘。

---

### 2.7 規模

| 指標 | station_1m | station_5m |
|---|---|---|
| 目前 `vd_dim.parquet` VD 數 | 3,229 個 | 3,229 個 |
| 單日列數（2024-04-01 樣本） | 約 245 萬列 | 約 54 萬列 |
| 單日大小（2024-04-01 樣本） | 約 10 MiB | 約 2.8 MiB |
| 全範圍估計大小 | 約 4-5 GiB | 約 1-1.5 GiB |

---

## 3. Heatmap 視覺化

### 3.1 目的

以每個事故為單位，產生一張**時間-空間二維速度熱圖**，用顏色直觀展現事故前後的交通衝擊範圍與恢復過程。

- X 軸：時間（以事故通報時間為 t=0）
- Y 軸：里程（以事故位置為中心，向上下游延伸）
- 像素顏色：該格的平均車速（紅=低速=壅塞，綠=高速=順暢）

標準批次輸出建議使用 `scripts/visualization/plot_heatmap.py` 讀取鎖版 NPZ，輸出至 `output/heatmaps_<style>/`。
`scripts/visualization/accident_heatmap.py` 則保留為 station-Parquet 來源的相容 renderer，預設 window 已同步為鎖版 sample window。

---

### 3.2 輸入資料

| 資料 | 路徑 | 用途 |
|---|---|---|
| 事故派遣 CSV | `data/accident_data.csv` | 事故位置、時間、方向 |
| Station Parquet | `data/vd_station_5m/` 或 `vd_station_1m/` | 各 VD 的分鐘速度 |
| VD 維度 | `data/metadata/vd_dim.parquet` | 確認路線的 VD 里程邊界 |

---

### 3.3 空間窗口設計

以事故里程 `acc_km` 為中心，依行向定義上下游：

**南向（S）/ 東向（E）** — 里程增加方向為下游：
```
空間窗口 = [acc_km - upstream_km, acc_km + downstream_km]
        = [acc_km - 50, acc_km + 10]
```

**北向（N）/ 西向（W）** — 里程減小方向為下游：
```
空間窗口 = [acc_km - downstream_km, acc_km + upstream_km]
        = [acc_km - 10, acc_km + 50]
```

鎖版 sample window 參數：
- `upstream_km = 50`
- `downstream_km = 10`
- `spatial_resolution_km = 1.0`（每格 1 km）

以 147K 南向事故為例：空間軸 = [97K, 157K]，共 61 個格。

---

### 3.4 時間窗口設計

以事故**發送時間**（交控中心接獲通報）為 t=0：

| 參數 | 預設值 | 說明 |
|---|---|---|
| `time_before_min` | 60 | 事故前 60 分鐘 |
| `time_after_min` | 240 | 事故後 240 分鐘 |
| `resolution_min` | 5 | 每格 5 分鐘 |
| 時間格數 | 61 格 | `(60+240)/5 + 1`，包含兩端點 |

---

### 3.5 矩陣建構流程

矩陣維度：`61（時間）× 61（空間）`，初始化為 `NaN`。

```
1. 篩選 station_df：
   - freeway == 事故國道
   - direction == 事故行向
   - timestamp 在 [t-60min, t+240min]
   - mileage_km 在有效 VD 覆蓋範圍內（route bounds）
   - mileage_km 在 [space_start - 2km, space_end + 2km]（插值邊際）

2. 計算 time_bin：
   time_bin = floor((timestamp - time_start) / 5min)

3. 對每個 time_bin：
   a. 取得所有 VD 的 (mileage_km, speed_kmh) 配對
   b. 若只有 1 個 VD：填入最近空間格（距離須 < 0.5 km）
   c. 若多個 VD：以里程為 x 軸進行 np.interp 線性插值，
      填入空間軸上所有有效格（在 route bounds 內）
   d. 超出 VD 覆蓋範圍的空間格保持 NaN（顯示為白色）
```

---

### 3.6 圖表元素說明

```
標題：2024-04-01 02:27 | N1-S | 147.0K | [+50/-10km] | 2024040004
      ↑事故時間        ↑路線 ↑事故里程 ↑空間窗口      ↑任務編號

色彩軸（RdYlGn）：
  紅色（0 km/h）─────── 黃色（70 km/h）─────── 綠色（140 km/h）
  壅塞                  中等                   順暢

  白色格：該格無有效 VD 資料（NaN）

標注線：
  紅色垂直虛線 (t=0)：事故通報時間（發送）
  藍色垂直線 ：救援人員抵達時間（到達）
  綠色垂直線 ：事故處理完成時間（處理完成）
  紅色圓點（●） ：事故位置（里程）

Y 軸：里程刻度（KM 單位），南向/東向時軸方向反轉（使上游在上方）
X 軸：時間刻度（HH:MM），每 15 分鐘標記一次
```

**現有範例圖片：**

| 圖片 | 路徑 |
|---|---|
| 任務 2024040088 | [`output/accident_heatmaps_5m/2024040088.png`](../output/accident_heatmaps_5m/2024040088.png) |
| 任務 2024040101 | [`output/accident_heatmaps_5m/2024040101.png`](../output/accident_heatmaps_5m/2024040101.png) |

---

### 3.7 Coverage 品質指標

每個 heatmap 同時輸出一個 JSON metadata 記錄填充率等品質資訊：

```json
{
  "accident": {
    "accident_id": "2024040004",
    "freeway": "N1",
    "direction": "S",
    "mileage_km": 147.0,
    "report_time": "2024-04-01T02:27:00"
  },
  "coverage": {
    "vd_count": 35,
    "filled_cells": 2860,
    "total_cells": 3328,
    "filled_ratio": 0.859,
    "space_window_start": 97.0,
    "space_window_end": 161.0,
    "missing_station_dates": []
  }
}
```

---

## 4. VD 靜態資訊：`vd_dim.parquet`

### 4.1 概述

`data/metadata/vd_dim.parquet` 是 VD 維度表，記錄在 `vd_station_1m` 層出現過的**每一個唯一 VD 站點**的靜態屬性與資料統計。

由 `build_station_vd_drive.py` 在處理完 station 資料後同步更新，每次跑完整日期範圍後重新合併所有天的 VD 出現記錄。

---

### 4.2 Schema

| 欄位 | 型別 | 說明 |
|---|---|---|
| `vdid` | string | VD 站點 ID |
| `freeway` | string | 國道代碼（N1/N3/N4/N6） |
| `direction` | string | 行向（N/S/E/W） |
| `mileage_m` | int32 | 里程（公尺整數） |
| `first_source_date` | date32 | 此 VD 最早出現在 station_1m 的日期 |
| `last_source_date` | date32 | 此 VD 最晚出現在 station_1m 的日期 |
| `n_source_days` | int32 | 此 VD 出現的日期數（用於評估資料連續性） |
| `n_rows_station_1m` | int64 | 此 VD 在 station_1m 的總列數 |

---

### 4.3 用途

1. **確定路線的 VD 覆蓋範圍**：查詢某國道+方向的 `min(mileage_m)` 與 `max(mileage_m)`，作為 heatmap 的空間邊界
2. **篩選穩定的 VD**：用 `n_source_days / 總日數` 計算覆蓋率，排除安裝時間短的 VD
3. **維度對齊**：模型訓練時以 vdid 為索引，從此表查詢對應的里程與路線

---

### 4.4 範例資料

（下表是 2024-04-01 至 2025-05-31 鎖版範圍的早期範例；目前 `vd_dim.parquet` 會隨較長 station range 更新）

| vdid | freeway | direction | mileage_m | first_source_date | last_source_date | n_source_days | n_rows_station_1m |
|---|---|---|---|---|---|---|---|
| VD-N1-N-22.990-N-LOOP | N1 | N | 22990 | 2024-04-01 | 2025-05-31 | 406 | 458,688 |
| VD-N1-N-23-I-EN-1-圓山 | N1 | N | 23000 | 2024-04-01 | 2025-05-31 | 416 | 458,175 |
| VD-N1-N-23-O-SE-1A-圓山 | N1 | N | 23000 | 2024-04-01 | 2025-05-31 | 418 | 456,108 |
| VD-N1-S-147-N-LOOP | N1 | S | 147000 | 2024-04-01 | 2025-05-31 | 410 | 462,240 |

> 同一個 `mileage_m` 可能有多個 `vdid`（不同的車道組合或進出口匝道）。

---

### 4.5 規模

| 指標 | 數值 |
|---|---|
| 目前 `vd_dim.parquet` 總 VD 站數 | 3,229 個 |
| 覆蓋國道 | N1, N3, N4, N6 |
| 覆蓋方向 | N, S, E, W |

---

## 5. 事故資料：派遣系統

### 5.1 原始格式

Excel 格式，**一個月一個檔案**，命名規則：

```
派遣系統資料/派遣回報紀錄YYYYMM.xlsx
```

由 `scripts/data/process_accident_data.py` 讀取並處理。

---

### 5.2 原始欄位（完整清單）

從 Excel 讀取後，保留以下欄位：

| 欄位 | 說明 | 備注 |
|---|---|---|
| `任務編號` | 事故唯一識別碼 | 例：2024040004 |
| `國道` | 事故發生的國道 | 原始值：國1, 國3... |
| `方向` | 行向（中文） | 南、北、東、西 |
| `里程數` | 事故里程（km，float） | |
| `路段` | 里程區段描述 | 例：140銅鑼-150三義 |
| `位置` | 占用車道描述 | 例：第三車道/第四車道 |
| `主線車道數` | 事故段主線總車道數 | |
| `輔助車道數` | 路肩/輔助車道數 | |
| `剩餘車道數` | 事故後可通行車道數 | |
| `性質` | 事故性質（事故 / 交通事故 / ...） | |
| `事故類別` | A1/A2/A3 分類 | |
| `車種` | 肇事車種（可能多種，分號分隔） | 例：小客車;小貨車 |
| `發送` | 交控中心接獲事故通報時間 | **heatmap t=0** |
| `到達` | 救援人員抵達現場時間 | |
| `處理完成` | 事故清理完成時間 | |
| `開始清理` | 開始清理現場時間 | 常為 NaN |
| `受傷人數` | 事故受傷人數 | |
| `死亡人數` | 事故死亡人數 | |
| `任務狀態` | 原始欄位（處理後移除） | 篩選用 |

---

### 5.3 處理步驟

1. 讀取所有 `派遣回報紀錄*.xlsx` 檔案
2. 保留欄位（COLUMNS_TO_KEEP）
3. **篩選掉取消任務**：`任務狀態 == 任務取消` 者移除
4. **篩選事故性質**：只保留 `性質 in {事故, 交通事故}`（排除道路坑洞、異物等非事故任務）
5. 移除 `任務狀態` 欄（篩選後不再需要）
6. **標準化國道名稱**：`國1 → N1`，`國3 → N3`...
7. 合併所有月份，輸出 `data/accident_data.csv`（UTF-8 with BOM）

---

### 5.4 輸出 Schema（`data/accident_data.csv`）

| 欄位 | 範例值 | 說明 |
|---|---|---|
| `任務編號` | 2024040004 | 事故唯一 ID（string） |
| `國道` | N1 | 標準化後（N1/N3...） |
| `方向` | 南 | 中文方向 |
| `里程數` | 147.0 | 里程（km） |
| `路段` | 140銅鑼-150三義 | 區段描述 |
| `位置` | 第三車道/第四車道 | 車道描述 |
| `主線車道數` | 3 | |
| `輔助車道數` | 1 | |
| `剩餘車道數` | 2 | |
| `性質` | 事故 | |
| `事故類別` | A2 | A1=死亡 / A2=受傷 / A3=財損 |
| `車種` | 小客車 | |
| `發送` | 2024/04/01 02:27 | 接獲通報時間 |
| `到達` | 2024/04/01 02:52 | 救援抵達時間 |
| `處理完成` | 2024/04/01 03:05 | 完成時間 |
| `開始清理` | NaN | 常缺 |
| `受傷人數` | 1.0 | |
| `死亡人數` | 0.0 | |

---

### 5.5 範例資料

| 任務編號 | 國道 | 方向 | 里程數 | 事故類別 | 發送 | 到達 | 處理完成 | 受傷 | 死亡 |
|---|---|---|---|---|---|---|---|---|---|
| 2024040004 | N1 | 南 | 147.0 | A2 | 2024/04/01 02:27 | 2024/04/01 02:52 | 2024/04/01 03:05 | 1 | 0 |
| 2024040006 | N1 | 北 | 157.8 | A2 | 2024/04/01 03:29 | 2024/04/01 03:50 | 2024/04/01 04:39 | 2 | 0 |
| 2024040049 | N1 | 北 | 124.6 | A2 | 2024/04/01 15:41 | 2024/04/01 15:53 | 2024/04/01 16:44 | 1 | 0 |

---

### 5.6 規模

| 指標 | 數值 |
|---|---|
| 總筆數（篩選後） | 2,043 筆 |
| 時間範圍 | 2024-04-01 起 |
| 欄位數 | 18 個 |

---

## 6. 事故資料：交控事故簡訊紀錄表

### 6.1 原始格式

ZIP 壓縮包，內含多個 Excel 月份檔案：

```
data/11206_11405交控事故簡訊紀錄表-20251125T072343Z-1-001.zip
└── 11206_11405交控事故簡訊紀錄表/
    ├── 112年06月事故處理時間紀錄表.xlsx
    ├── 112年07月事故處理時間紀錄表.xlsx
    ├── ...
    └── 114年05月事故處理時間紀錄表.xlsx
```

**涵蓋月份：** 112年6月（2023/06）至 114年5月（2025/05）

**各檔案大小差異極大：** 從 57 KB（114年5月）到 27 MB（112年11月）。
部分早期月份（112年9-11月）檔案異常龐大，可能包含完整的簡訊歷史記錄或格式差異。

---

### 6.2 完整欄位說明（49 個欄位）

#### 6.2.1 事故時間（分解欄）

| 欄位 | 型別 | 說明 |
|---|---|---|
| `年` | int64 | 民國年（例：113 = 2024） |
| `月` | int64 | 月份（1-12） |
| `日` | int64 | 日期 |
| `時` | int64 | 小時（24hr） |
| `分` | int64 | 分鐘 |

> 時間分開存放五個欄位，使用時需組合：`{民國年+1911}/{月}/{日} {時}:{分}`

#### 6.2.2 詳細時間流程（HH:MM:SS 格式字串）

| 欄位 | 說明 | 113年4月非空率 |
|---|---|---|
| `事件發生` | 事故實際發生時間 | 407/407 (100%) |
| `交控中心\n接獲通報` | 交控中心收到通報的時間 | 0/407 (0%) |
| `CCTV\n監看現場` | 透過 CCTV 開始監看現場的時間 | 0/407 |
| `CMS\n發布資訊` | 可變情報板（CMS）發布資訊的時間 | 0/407 |
| `交控中心\n通報工務段` | 交控中心通知工務段的時間 | 0/407 |
| `事故處理\n小組出發` | 事故處理小組從基地出發時間 | 0/407 |
| `事故處理\n小組抵達` | 事故處理小組抵達現場時間 | 0/407 |
| `事故處理\n小組完成` | 小組完成處理時間 | 0/407 |
| `事件排除` | 事故完全排除、車道恢復通行的時間 | 407/407 (100%) |
| `處理分鐘` | 事件排除 - 事件發生（分鐘整數） | 407/407 (100%) |

> 注意：113年4月的詳細時間欄除事件發生/排除外均為空值，但這可能是資料填寫方式的差異，其他月份或其他年度可能不同，使用前應驗證各月份的填寫狀況。

#### 6.2.3 位置資訊

| 欄位 | 說明 | 範例 |
|---|---|---|
| `國道名稱` | 國道全名 | 國道1號 |
| `方向` | 行向（含「向」字） | 南向、北向、東向、西向 |
| `里程` | 事故里程（km，float） | 166.0 |

#### 6.2.4 事故資訊

| 欄位 | 說明 |
|---|---|
| `事故類型` | 文字描述（NaN 表示未分類，常見） |
| `死亡` | 死亡人數（float，0.0 表示無） |
| `受傷` | 受傷人數（float） |
| `肇事車輛` | 肇事車輛數（int） |

#### 6.2.5 車道佔用（個別車道，有值=佔用，NaN=未佔用）

| 欄位 | 說明 |
|---|---|
| `內路肩` | 內側路肩 |
| `內車道` | 最內側車道 |
| `中內車道` | 次內側車道 |
| `中車道` | 中間車道 |
| `中外車道` | 次外側車道 |
| `外車道` | 最外側車道 |
| `外路肩` | 外側路肩 |
| `匝道` | 匝道 |

#### 6.2.6 特殊事故標記（有值=是，NaN=否）

| 欄位 | 說明 |
|---|---|
| `翻覆事故\n註記` | 車輛翻覆 |
| `施工事故\n註記` | 施工區域事故 |
| `危險物品車輛\n註記` | 危險品車輛涉及 |
| `車輛起火\n註記` | 車輛起火 |
| `冒煙車事故\n註記` | 冒煙車輛 |
| `主線中斷\n註記` | 主線完全中斷 |

#### 6.2.7 車輛資訊

| 欄位 | 說明 |
|---|---|
| `車輛1` ~ `車輛12` | 各肇事車輛類型（string，最多 12 輛） |

#### 6.2.8 簡訊內容

| 欄位 | 說明 |
|---|---|
| `簡訊內容` | 交控中心對外發送的事故通報簡訊全文 |

---

### 6.3 簡訊內容範例

簡訊格式遵循固定樣板，包含：等級、日期、里程、車輛數、事故描述、占用車道、傷亡、排除時間、回堵狀況。

```
中控通報3級04/30,18:36國1南向166k(結報)3小客車追撞事故占內車道,無人受傷,18:55排除,無回堵,來源3隊
中控通報3級04/30,17:42國1北向199k(結報)2小客車追撞事故占內車道,無人受傷,17:47排除,回堵1K,來源3隊
中控通報3級04/30,17:34國1南向165k(結報)2小客車追撞事故占內車道,無人受傷,17:43排除,回堵1K紓解中,來源3隊
中控通報3級04/30,17:31國6西向2.9k(結報)2小客車追撞事故占外車道,17:37排除,無人受傷,無回堵,來源7隊
```

**簡訊結構解析：**
```
中控通報{等級}級{MM/DD},{HH:MM}{國道}{方向}{里程}(結報){車輛數}{車種}追撞事故
占{車道},受傷情況,{排除時間}排除,回堵情況,來源{隊}隊
```

---

### 6.4 範例資料

（113年4月，前 3 筆）

| 年 | 月 | 日 | 時 | 分 | 國道名稱 | 方向 | 里程 | 事件發生 | 事件排除 | 處理分鐘 | 受傷 | 肇事車輛 | 內車道 | 中內車道 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 113 | 4 | 30 | 18 | 55 | 國道1號 | 南向 | 166.0 | 18:36:00 | 18:55:00 | 19 | NaN | 3 | NaN | 1 |
| 113 | 4 | 30 | 17 | 47 | 國道1號 | 北向 | 199.0 | 17:42:00 | 17:47:00 | 5 | NaN | 2 | 1 | NaN |
| 113 | 4 | 30 | 17 | 43 | 國道1號 | 南向 | 165.5 | 17:34:00 | 17:43:00 | 9 | NaN | 2 | NaN | 1 |

---

### 6.5 規模

| 指標 | 數值 |
|---|---|
| 涵蓋時間 | 112年6月 ~ 114年5月（2023/06 ~ 2025/05） |
| 月份數 | 24 個月 |
| 欄位數 | 49 個 |
| 各月約略筆數 | 300 ~ 1,000 筆不等 |
| 113年4月樣本 | 407 筆 |
| 特色欄位 | 9個時間點、8個車道標記、12輛車明細、簡訊全文 |

---

## 7. 兩份事故資料對照

### 7.1 欄位對照表

| 比較項目 | 派遣系統資料 | 交控簡訊紀錄表 |
|---|---|---|
| **原始格式** | Excel（月份檔） | Excel（月份檔，ZIP 壓縮） |
| **唯一識別碼** | `任務編號`（例：2024040004） | 無（以時間+位置識別） |
| **國道格式** | `N1`, `N3`... | `國道1號`, `國道3號`... |
| **方向格式** | `南`, `北`, `東`, `西` | `南向`, `北向`, `東向`, `西向` |
| **時間點數** | 3 個（發送/到達/處理完成） | 9 個（事件發生/接獲/CCTV/CMS/通報/出發/抵達/完成/排除） |
| **車道資訊** | 主線/輔助/剩餘車道數 | 個別車道是否佔用（8 個欄位） |
| **事故類別** | A1/A2/A3 分類 | 文字描述（常為 NaN） |
| **車種資訊** | 合併字串（分號分隔） | 最多 12 輛分欄 |
| **涵蓋時間** | 2024-04 起 | 2023-06 ~ 2025-05 |
| **特色欄位** | 受傷/死亡人數 | 簡訊全文、特殊事故標記 |
| **目前使用狀態** | 已整合（heatmap 主要來源） | **尚未整合** |

---

### 7.2 時間欄位對照

| 事故流程節點 | 派遣資料欄位 | 簡訊紀錄表欄位 |
|---|---|---|
| 事故實際發生 | （無） | `事件發生` |
| 交控中心接獲通報 | `發送` | `交控中心\n接獲通報` |
| CCTV 監看 | （無） | `CCTV\n監看現場` |
| CMS 發布資訊 | （無） | `CMS\n發布資訊` |
| 通知工務段 | （無） | `交控中心\n通報工務段` |
| 救援人員出發 | （無） | `事故處理\n小組出發` |
| 救援人員抵達 | `到達` | `事故處理\n小組抵達` |
| 救援完成 | `處理完成` | `事故處理\n小組完成` |
| 事件完全排除 | （無） | `事件排除` |

> **派遣資料的「發送」** ≈ **簡訊紀錄表的「交控中心\n接獲通報」**
> 兩者語意相同，是交控中心得知事故的時間點，也是 heatmap 的 t=0。

---

### 7.3 整合建議

**連結鍵（join key）：**

兩份資料無法直接以 ID join，建議使用模糊匹配：
- `國道`（標準化後）
- `方向`（標準化後）
- `里程`（允許 ±0.5 km 誤差）
- `時間`（允許 ±5 分鐘誤差）

**簡訊資料的優勢：**
1. **更早期的歷史資料**：涵蓋 2023/06 起，而派遣資料從 2024/04 起
2. **更細的時間流程**：可以計算 CCTV 反應時間、CMS 發布延遲等指標
3. **精確的車道佔用**：知道是哪條車道被占用，可用於估計通行能力損失
4. **簡訊內容**：包含回堵距離等派遣資料沒有的資訊

**簡訊資料的注意事項：**
1. 部分月份的詳細時間欄位（CCTV、CMS 等）可能全為空值，需逐月驗證
2. 國道名稱和方向格式需標準化才能和 VD 資料對齊
3. 早期月份（112年）的檔案異常龐大，需確認格式是否相同
4. 時間欄位為 `HH:MM:SS` 字串，需結合 年/月/日 欄位才能得到完整 timestamp

---

## 附錄：資料管線架構圖

```
MOTC API（每分鐘 XML.gz）
    │
    ▼
download_vd_drive.py
    │ [1,440 HTTP requests/day, async]
    │ [XML parse, VD filter, type cast]
    ▼
vd_raw_lane_1m_full/
    source_year=YYYY/source_month=MM/vd_YYYYMMDD.parquet
    │ [~688萬列/天, 每列=1車道1分鐘]
    │
    ▼
build_station_vd_drive.py
    │ [speed filter: 0 < speed <= 140]
    │ [lane → station aggregation]
    ▼
vd_station_1m/               vd_station_5m/
    vd_station_YYYYMMDD.parquet   vd_station_YYYYMMDD.parquet
    [~245萬列/天, 每列=1 VD 1分鐘]  [~54萬列/天, 每列=1 VD 5分鐘]
    │                              │
    │ ─────────────────────────────────── │
    ▼
accident_heatmap.py
    │ [讀取 station_5m]
    │ [讀取 accident_data.csv]
    │ [時空矩陣 + 線性插值]
    ▼
output/accident_heatmaps_5m/
    {任務編號}.png
    metadata/{任務編號}.json

─────────────────────────────────────────

派遣系統資料/派遣回報紀錄YYYYMM.xlsx
    │
    ▼
process_accident_data.py
    │ [filter: 非取消, 性質=事故]
    │ [standardize: 國1→N1]
    ▼
data/accident_data.csv  ← heatmap 目前使用的事故來源

data/11206_11405...zip  ← 尚未整合的第二事故來源
    （交控事故簡訊紀錄表）
```
