# TIID 詳細介紹

TIID 是本專題在 Feature Fusion U-Net 上加入的事故影響衰退輔助機制，全名可寫作 **Traffic Incident Impact Decay**。它的核心想法是：事故造成的壅塞不是一個固定不變的偏移量，而是一個會隨時間逐步消散的影響。模型除了預測事故後每個時空格的速度，也額外預測一條「事故影響衰退曲線」，用來修正速度場，並提供更容易解釋的事故影響時間訊號。

![TIID concept](assets/tiid_concept.svg)

## 為什麼需要 TIID

Baseline U-Net 的主任務是從事故前的時空速度矩陣，直接預測事故後的速度矩陣。這種做法可以學到壅塞形狀與恢復趨勢，但它把所有資訊都壓在一張輸出的速度場裡：模型可能知道「速度會慢慢回升」，但我們不容易明確看出它認為事故影響有多大、何時開始消退、是否已接近恢復。

TIID 把這件事拆開。主任務仍然是速度預測，輔助任務則要求模型產生一條 `decay_curve`。這條曲線代表事故後各時間點仍殘留多少事故影響；數值越大，代表模型認為事故壓低速度的程度越強，數值越接近 0，代表事故影響接近消散。

這樣做有三個好處：

1. **更符合交通直覺**：事故影響通常在事故後早期最大，之後隨排除、車流疏解與速度恢復而下降。
2. **讓特徵有更清楚的用法**：車道阻斷、天候、幾何、方向與里程等事故特徵，不只影響速度場，也會影響影響衰退速度。
3. **增加可解釋性**：除了 MAE、RMSE、F1，也可以檢查 decay curve 是否合理，例如是否過早歸零、是否沒有反映嚴重事故。

## 在本專題中的資料定義

每一筆事故會被整理成一張速度時空矩陣。橫軸是相對事故時間的時間格，縱軸是事故附近的空間位置。模型輸入使用事故前時間窗；模型輸出預測事故後時間窗。

在目前實作中，速度會先被裁切並正規化：

```text
speed_norm = clip(speed_kmh, 0, SPEED_MAX_KMH) / SPEED_MAX_KMH
```

TIID 的 target 由真實速度矩陣自動估出，不需要人工標註事故影響時間。做法是：

```text
baseline = mean(pre_event_speed)
future_mean(t) = mean(post_event_speed_at_time_t_over_space)
impact(t) = clip((baseline - future_mean(t)) / baseline, 0, 1)
decay_target(t) = cumulative_minimum(impact(t))
```

也就是說，先用事故前平均速度近似該筆事故的正常基準，再看事故後每個時間點平均速度比基準低多少。若事故後速度明顯低於事故前，`impact(t)` 會較高；若速度接近事故前狀態，`impact(t)` 會趨近 0。

目前實作再用 cumulative minimum 讓 target 單調不增加。這是 TIID 的重要假設：事故影響應該整體往消散方向走，而不是在衰退曲線上反覆變大。真實交通可能因二次事故、回堵波或尖峰時段而出現再次惡化，所以這個單調假設是建模簡化，不是交通現象的完整描述。

![TIID training objective](assets/tiid_training_objective.svg)

## 模型架構

TIID 是加在 `FeatureFusionUNet` 上的 optional head。一般 feature-fusion U-Net 會先用 encoder 把事故前速度場壓到 bottleneck，再用 decoder 還原事故後速度場。Feature fusion 的方式可選：

- `direct_concat`：把事故特徵轉成額外 channel，直接併到輸入。
- `mid_inject`：把事故特徵轉成 bottleneck embedding，加到 U-Net bottleneck。
- `attention`：把事故特徵轉成 bottleneck channel gate，調整各 channel 的權重。

開啟 TIID 後，模型會從 bottleneck 表徵與事故特徵產生 `decay_curve`：

```text
pooled = global_average_pool(bottleneck)
decay_curve = sigmoid(decay_head(concat(pooled, incident_features)))
decay_curve = cumulative_minimum(decay_curve)
```

接著模型把這條 curve 轉成速度懲罰項，套回 base forecast：

```text
penalty(t) = decay_curve(t) * baseline_speed * tiid_scale
prediction = clamp(base_prediction - penalty, 0, 1)
```

這代表 TIID 不是另外預測一個完全獨立的結果，而是在 U-Net 原本的速度預測上加上一個「事故影響仍然存在，因此速度應被下修」的結構化修正。`tiid_scale` 是可學習參數，用來控制 decay curve 對速度修正的整體強度。

## Loss 設計

TIID 訓練時包含兩個 loss。第一個是主任務的速度場 loss，目前使用 congestion-weighted L1，讓壅塞區域的錯誤權重較高：

```text
speed_loss = weighted_L1(predicted_speed, target_speed, valid_mask)
```

第二個是輔助的 decay loss：

```text
decay_loss = L1(decay_pred, decay_target)
```

總 loss 為：

```text
total_loss = speed_loss + tiid_lambda * decay_loss
```

`tiid_lambda` 是重要超參數。若太小，TIID head 可能學不到有意義的衰退曲線；若太大，模型可能過度追求曲線形狀，反而犧牲速度場預測。README 目前建議可從 `0.1/0.3/0.5` 掃描，觀察 MAE、RMSE、congestion F1 與 decay curve 診斷圖的變化。

## 如何解讀 TIID 輸出

TIID 的 `decay_curve` 可視為「事故影響殘留比例」的 proxy，但它不是官方定義的事故清除時間，也不是直接等同於派遣資料中的處理完成時間。比較保守的說法是：

- `decay_curve` 高：模型認為事故影響仍明顯，速度仍需要被事故因素下修。
- `decay_curve` 下降：模型認為事故影響正在消散。
- `decay_curve` 接近 0：模型認為事故造成的額外速度壓低效果已很弱。

若要從曲線推估「影響時間」，可以設定門檻，例如第一次低於 `0.05` 或 `0.10` 的時間點。但這個門檻需要用 validation set 與實務定義校準，不能直接當成絕對真值。

## 目前結果與限制

目前程式已能訓練 `mid_inject + TIID` 版本，也能輸出比較結果與診斷圖。不過目前觀察到的狀態是：TIID 可訓練、架構合理，但主指標尚未穩定勝過 attention feature fusion。既有報告腳本也記錄了這個結論：TIID head 目前容易輸出接近 0 的衰退曲線，表示 decay signal 可能太弱，或 loss 尚未聚焦到真正受事故影響的時空區域。

主要限制包括：

1. **baseline 定義較粗**：目前用事故前整體平均速度當基準，沒有區分上游、下游、車道、尖峰時段與歷史常態速度。
2. **decay target 可能偏弱**：若事故後平均速度沒有明顯低於事故前，target curve 會很淺，模型容易學成接近 0。
3. **空間資訊被壓成一條曲線**：目前 decay curve 只沿時間變化，沒有明確表示不同里程位置的衰退差異。
4. **單調假設過於保守**：真實壅塞可能因回堵波、二次干擾或尖峰背景而反覆惡化。
5. **主任務與輔助任務可能拉扯**：若 decay penalty 下修太強，會改善壅塞區域但傷害自由流區域；若太弱，TIID 幾乎沒有作用。

## 下一版改進方向

短期最值得做的是強化 decay target 的品質。可以改成只在事故上游或高壅塞風險區域計算 impact，而不是對整個空間窗平均；也可以引入同路段、相似時段的歷史平均速度作為 baseline，避免把尖峰自然壅塞誤認為事故衝擊。

第二步是重新掃描訓練權重與作用區域。`tiid_lambda` 可以擴大測試到 `0.1/0.3/0.5/1.0`，同時檢查 `tiid_scale` 是否學到非零值。loss 也可以只在事故後早期或壅塞 mask 區域計算，讓 TIID 專注於事故影響最明顯的部分。

第三步是把 decay 從一維時間曲線升級成時空影響圖。例如讓 TIID head 輸出 `space × time` 的 impact map，或輸出上游、事故點、下游三段不同的 decay curve。這會更貼近交通回堵在空間上傳播與消退的特性，但模型複雜度與診斷成本也會提高。

## 建議報告說法

可以把 TIID 描述為「在速度場預測之外，額外加入事故影響衰退的結構化輔助任務」。目前較穩妥的結論是：

> TIID 的目的不是取代 U-Net，而是讓模型在預測事故後速度場時，同時學習一條可解釋的事故影響衰退曲線。初版實作已可訓練並可視覺化，但目前 decay signal 偏弱，主指標尚未超過最佳 attention fusion；下一步會調整 target 定義、loss 權重與作用區域。

這樣的說法能清楚交代方法價值，也避免過度宣稱目前結果已經有效。
