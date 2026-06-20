# PLAN.md — REFIT 即時協調式需量反應(DR)系統

> 多住戶在 REFIT 上做**即時協調式需量反應**:每戶本地 LSTM 預測 baseload,中央 DR Coordinator 以 online rolling-horizon 協調各戶的 deferrable 電器排程,達成跨戶削峰(load leveling)。Coordinator 只看加總負載(secure aggregation),原始 state/排程/log 不出戶。
>
> **註**:FL(聯邦學習)暫不做,降為選用延伸(見文末)。題目中的「multi-agent」保留,「federated learning」移除 — 已與指導教授確認框架後再定案。

---

## 0. 系統定位與核心貢獻

- **目標**:降系統 PAR / peak kW(grid/VPP 角度跨戶削峰),**非**單戶 cost 最小化。
- **協調方式**:中央協調(coordinated),**非**純價格分散回應。價值在避免「大家延到同一便宜時段」造成的 herding / rebound peak。
- **時序**:**即時 online rolling-horizon**,非 day-ahead 一次性。
- **核心方法**:online **shadow-price 協調** + rolling horizon。
- **兩個平面**:
  - 預測層 — 每戶**各自的 local LSTM**(無權重交換)。
  - 控制層 — 中央 **DR Coordinator** 只收**加總負載**(secure aggregation),原始排程/state 不出戶。
- **隱私機制 = 控制層 secure-agg(唯一一層,需寫紮實)**:coordinator 全程只看到 Σ 負載,看不到任一戶的排程或 state。
- server 端只有一個角色:**DR Coordinator**(不再有 FL Server,命名無衝突)。

---

## 固定參數

| 項目 | 值 |
|---|---|
| 控制層解析度 Δ | **10 min → 144 slots/day** |
| 原始解析度 | REFIT CLEAN 版 ~8s,resample 到 10-min 平均功率(W) |
| 能量換算 | `energy_kWh = Σ(slot_avg_W) / 6000` |
| 預測 horizon H | 預設 6h = 36 格(ablation: 4/6/8h) |
| comfort 容忍 Δ_max | 預設 4–8h、不跨日 |
| tariff | 英國 Economy 7 雙費率(寫死,標為假設) |
| baseload 定義 | `量測 Aggregate − Σ(deferrable channels)`,clip≥0 |
| seed / config | 全程固定 seed,所有設定存檔 |

---

## 資料集與選戶

deferrable 五類:**WM 洗衣 / DW 洗碗 / TD 滾筒烘衣 / WD 洗烘一體(獨立類別)/ DR 烘衣(同 TD)**。冰箱/冷凍/電暖器(恆溫連續)與微波/水壺/烤麵包機/電腦/電視(即時舒適)一律排除。

### Per-house deferrable map(channel 0 = Aggregate;N → ApplianceN)

| House | Deferrable | 備註 |
|---|---|---|
| 1 | 4:TD 5:WM 6:DW | |
| 2 | 2:WM 3:DW | |
| 3 | 4:TD 5:DW 6:WM | |
| 4 | 4:WM 5:WM | 兩台洗衣機、無 DW/TD |
| 5 | 3:WM 4:DW | ch2(原 TD)實為除濕機 → 已移出 |
| 6 | 2:WM 3:DW | |
| 7 | 4:TD 5:WM 6:DW | ⚠ ch6 簽章變更 @2014-05-20 → flag_after |
| 8 | 3:DR 4:WM | ch3 Dryer ≈ TD |
| 9 | 2:WD 3:WM 4:DW | |
| 10 | 5:WM 6:DW | |
| 11 | 3:WM 4:DW | ❌ Aggregate 受太陽能污染 |
| 13 | 3:WM 4:DW 5:TD | ⚠ ch5 簽章不穩 → flag_all |
| 15 | 2:TD 3:WM 4:DW | |
| 16 | 5:WM 6:DW | |
| 17 | 3:TD 4:WM | 無 DW |
| 18 | 4:WD 5:WM 6:DW | |
| 19 | 2:WM | 僅 1 台,long-tail 樣本 |
| 20 | 3:TD 4:WM 5:DW | 最乾淨 |
| 21 | 2:TD 3:WM 4:DW | ❌ Aggregate 受太陽能污染 |

### 收錄決策
- **排除**:House **12**(零 deferrable)、House **11 / 21**(太陽能污染 baseload 與削峰目標)。
- **主集合 = 17 戶**;含 11/21 的 PV-aware 延伸 = 19 戶(另開實驗)。
- House 19 留作異質性/long-tail,非主力。

---

## Phase 0 — 環境與資料底層
- 沿用 refit-fl pinned env(numpy<2.0 / torch 2.3.x),固定 seed,config 全存檔。
- 選戶規則寫死:完整度達標 + 至少 1 個 deferrable;記錄被排除戶與原因。
- 從 metadata 建 **per-house deferrable map**(各戶 channel→電器不同,**不可用 channel index 硬編**)。

## Phase 1 — 預處理與 cycle 抽取 ✅ 已完成並驗證
- resample 10-min 平均功率;限制 forward-fill 長度,缺漏過多的整天丟棄並記錄。
- baseload = `Aggregate − Σ(deferrable)`,clip≥0。
- **cycle 抽取順序:先合併短 gap、再過濾長度**(反了會把洗碗段間/泡水切碎)。
- 全部 deferrable cycle 視為 **non-interruptible**。

抽取參數(10-min 平均功率,**已按真實資料定案**):

| 型別 | on 門檻 | min_len | max_len(超過標 anomaly) | merge_gap |
|---|---|---|---|---|
| WM | ≥40 W | 3 格 | **22 格** | **≤1 格** |
| DW | ≥40 W | 4 格 | 20 格 | ≤2 格 |
| TD | ≥80 W | 3 格 | 16 格 | ≤1 格 |
| WD | ≥40 W | 6 格 | 30 格 | ≤3 格 |
| DR | ≥80 W | 3 格 | 16 格 | ≤1 格 |

定案依據(Phase 1 真實資料診斷):
- **WM merge_gap 2→1**:H2 把兩次洗衣(中間 20-min 停頓)橋接成一個大 cycle,anomaly_long 30→2,ok 277→321。
- **WM max_len 18→22**:H6/H20 為長週期使用戶,21–22 slot 為真實 cycle(ratio 正常)。
- **TD 維持 16(否決 →20)**:放寬的唯一需求來自 H5 ch2,而該 channel 經證實為除濕機污染、不可採信;乾淨戶 TD anomaly 全 ≤3%。
- **DR 全維持**:H8 DR 全 111 個 cycle 每格 ≥80W(`slots_below_80=0`),無冷卻段被切,降門檻反而少 cycle。

污染遮罩:`(7,6) flag_after 2014-05-20`、`(13,5) flag_all`。
**H5 ch2 處置(決定:移出 deferrable)**:該 channel 實為除濕機(月 on-slots 250–450、冬季 800+),不符 deferrable 定義,且若保留會在 `baseload = Aggregate − Σ(deferrable)` 中被全程扣除、導致 H5 baseload 系統性低估(clip-to-0 增多)。故**自 `DEFERRABLE_MAP[5]` 移除 ch2**,原 `(5,2) cut_after` 遮罩一併移除。H5 改為只含 WM(ch3)+ DW(ch4)。

抽完先看 `summary_by_type.csv` + 長度/能量直方圖再往下:WM 每週 2–5 次合理;WD 應比 WM 長且雙峰;anomaly_pct >5% 要查 max_len 或感測。

## Phase 2 — 預測層(每戶 local LSTM,只預測 baseload)
- 每戶**各自訓練一個 LSTM**,無權重交換(FL 暫不做)。
- 輸入過去 24h(144 步)+ 時間特徵(hour、day-of-week),預測未來 H 的 baseload。
- **chronological split 70/10/20,絕不 shuffle**(呼應先前抓到的 random-split leakage)。
- 指標:每戶 **RMSE / MAE / MAPE**(不用 R²,避開灌水爭議)。
- ★ **程式保留 FedAvg-ready 介面**:本地訓練迴圈包成可被外層權重聚合呼叫的形式,日後要加 FL 當延伸/ablation 零成本。

### Phase 2 定案（已完成，17 戶全數通過）

**管線與模型參數**

| 項目 | 定案值 |
|---|---|
| 架構 | per-house local LSTM，hidden=64，layers=2，dropout=0.1 |
| 輸入 | look_back=144（24h）；預測 horizon=36（6h） |
| 特徵 | baseload_scaled + hour_sin/cos + dow_sin/cos（共 5 維） |
| 正規化 | TrainOnlyScaler（StandardScaler，只用 train 段 fit，禁止 val/test 統計量洩漏） |
| Gap 處理 | ≤3 slots → linear interpolate（limit_direction="forward"，無 backfill 洩漏）；長缺漏維持 NaN，切窗跳過 |
| 訓練 | lr=1e-3，patience=10，early stopping；lr=5e-4 實驗無改善，維持原值 |
| Split | chronological 70/10/20，禁止 shuffle；兩個邊界均有斷言保護 |
| FL 介面 | `get_weights()` / `set_weights()` / `local_train()`，日後加 FL 零成本 |
| 輸出 | `out_phase2_17h/`（results.json、config.json 入版控；model_house*.pt 不入版控） |

**跨戶評估指標定案**
- **主要**：RMSE_scaled（σ，以 train std 為單位）—— 跨戶量級不同時直接可比
- **輔助**：nRMSE_range（= RMSE / (max−min)，per split）—— 注意 range 離群值會使分母爆炸失真
  - ⚠ H18 test nRMSE_range = 1.2% 為**假訊號**：test 段有巨大尖峰使 range ≈ 26,000 W，分母失真；H18 真實水準看 RMSE_scaled = 0.98σ
- 廢棄：RMSE/mean（舊 NRMSE）—— mean 受基線低谷壓低分母，H20 良好卻顯示 67.5%，已移除
- MAPE 保留為參考（mask: actual > 50 W），不作主要判斷依據

**17 戶 test 結果摘要（依 RMSE_scaled 排序）**

| 戶號 | RMSE(W) | MAE(W) | RMSE_scaled(σ) | nRMSE_range(%) | 備註 |
|------|---------|--------|---------------|----------------|------|
| H3   | 393  | 244 | 0.52 | 7.1 | 最佳 |
| H13  | 342  | 210 | 0.67 | 7.4 | |
| H16  | 294  | 213 | 0.62 | 8.7 | |
| H4   | 214  | 149 | 0.72 | 5.6 | |
| H5   | 428  | 271 | 0.70 | 4.0 | 除濕機移除後正常 ✓ |
| H15  | 108  |  70 | 0.76 | 8.0 | |
| H6   | 225  | 155 | 0.78 | 6.0 | |
| H7   | 333  | 211 | 0.77 | 9.1 | |
| H1   | 424  | 208 | 0.78 | 4.3 | val 偏高(冬季)→ test 正常 |
| H20  | 221  | 139 | 0.82 | 7.4 | |
| H19  | 228  | 124 | 0.87 | 5.8 | |
| H9   | 578  | 272 | 0.89 | 5.8 | |
| **H8**  | **740** | **421** | **0.89** | **7.2** | ⚠ 偏高 |
| **H10** | **505** | **307** | **0.94** | **9.5** | ⚠ 偏高 |
| H18  | 312  | 165 | 0.98 | ~~1.2%~~ | nRng 為假訊號，見上方說明 |
| **H2**  | **783** | **333** | **1.10** | **6.9** | ⚠ 高不確定性 |
| **H17** | **590** | **225** | **1.14** | **5.2** | ⚠ 高不確定性 |

整體：test RMSE_scaled 約 0.5–1.1σ；nRMSE_range 約 4–10%；**無任何戶超過 15% 停損線**。

**高不確定性戶（Phase 3 / 4 注意事項）**

H2、H8、H10、H17 的 test RMSE_scaled ≥ 0.89σ，其中 H2（1.10σ）與 H17（1.14σ）誤差超過 1 個 train std。原因：baseload 含大量不可預測短暫尖峰（電熱器、大功率烹飪設備），LSTM 只能學到平均基線。
**Phase 3 模擬器需記錄這四戶的 baseload 預測絕對不確定性較高；Phase 4 協調時應對其預測降低信任（加大 uncertainty buffer 或 worst-case margin）。**

## Phase 3 — 因果模擬器(成敗關鍵)
- `observe(t)` **只回傳 ≤ t 的資料**;baseload 一律用預測值;job 只在實際 `r_j` 才釋放(不可偷看未來)。
- Job 模型:`r_j` 取自 trace、`d_j = r_j + Δ_max`、功率曲線取自該 cycle。
- 拒絕處理:被拒 job 釘成 must-run,下個 tick 重排;無真人 → acceptance model `p(accept | 延後時數, 省的錢)`。

## Phase 4 — 協調層 + baselines

### Must-run 規則（Phase 4a 定案）

當一個已釋放 job 在 `schedule_house` 中找不到合法啟動 slot（`s_min > s_max`），轉為 **must-run**：

| 條件 | 啟動點 | `deadline_missed` |
|---|---|---|
| `d_j − dur ≥ t` | `start = d_j_slot − duration`（最晚仍能趕上 deadline 的點） | `False` |
| `d_j − dur < t` | `start = 0`（立刻從 t 起跑） | `True` |

- Must-run job **non-interruptible**，整段連續執行，不參與協調最佳化。
- Must-run 的 `power_profile` **必須計入聚合負載**（`compute_aggregate_load`），coordinator 和其他排程要把它當成固定背景負載繞著它排。Must-run 不可從負載帳中消失。
- 驗證規則：must-run job 仍驗 ① non-interruptible、② release（start ≥ 0）；豁免 ③ deadline 與 ④ horizon 約束（改記錄 `deadline_missed`）。
- `deadline_missed` 計入 Phase 5 指標：**deadline-miss rate** = must-run 中 `deadline_missed=True` 的比例。

主方法 online shadow-price loop（雙層：外層 rolling horizon，內層協調）:

```
λ_warm = 0
for t in timeline:                      # 外層
    obs = sim.observe(t)                # 只 ≤ t
    baseload_hat = LSTM_h.forecast(obs, H)   # 每戶各自的 local 模型
    jobs = release_deferrable(obs)
    λ = λ_warm
    for k in range(K):                  # 內層協調
        broadcast(λ over [t, t+H])
        sched = {h: argmin_local(jobs[h], baseload_hat[h], λ, deadline, comfort)}
        L = secure_aggregate(sched)     # coordinator 只看加總
        if max(L) - target < ε: break
        λ = clip(λ + α*(L - target), 0, ∞)
    commit_only_now(t, sched)           # ⑧ 決策 → ⑨ LLM 建議 → ⑪⑫ accept/exec → ⑬ local log
    λ_warm = λ                          # warm-start 下個 tick
```

關鍵技巧:**λ warm-start**(相鄰 tick 擁塞模式相近,幾次迭代即收斂)、**commit-first / MPC**(每 tick 只執行當下格)、**terminal cost**(防 horizon myopia)、**target = horizon 內 aggregate running mean**(load leveling)。

對照組:`No-DR` / `Greedy 純價格(故意製造 herding)` / `Day-ahead 一次性 MILP(近 oracle 上界)`。

### Phase 4d — 擴展評估（17 戶 × 最長乾淨窗，指標定案）

**動機**：Phase 4c 以 3 戶 2 天驗通機制，但整窗 PAR 受低載稀釋（-17% 而非單點 -32%）；需更大規模才能體現協調價值。

**評估規模**：
- 全 **17 戶**（排除 H11/21/12）。
- 每戶使用 **各自最長乾淨連續窗**（Phase 3 已算，全戶 ≥ 19 天，平均 32.3 天）。

**PAR 指標定案（雙報）**：
1. **整窗 PAR**（保守）：max(load_per_tick) / mean(load_per_tick)，含所有無 job 低載格。
2. **每日尖峰降幅**（主打）：各日最大 5 分鐘均值 peak → coord vs greedy 的百分比降幅，反映削峰本質，不被低載稀釋。

**完整 baselines**：
- **No-DR**：所有 job 在 r_j 立刻啟動（理論最差）。
- **Greedy**：base ToU price，rolling commit-first（Phase 4c greedy）。
- **Online-coordinated**：shadow-price best-so-far，rolling，warm-start=best_lam（本方法）。
- **Oracle MILP**（單點參考，不做全窗）：每日抽樣 1–2 個 tick 跑 oracle，估計協調效率上界。

**其他指標**：
- 協調效率 = (greedy−coord)/(greedy−oracle) × 100%（抽樣 tick）。
- 平均延後時數 = mean(commit_time − r_j)（分鐘）。
- deadline-miss rate = committed 中 deadline_missed=True 的比例。
- 每 tick 平均協調 iter 數與 wall-clock runtime（佐證即時可行性，目標 < 1 s/tick）。

**輸出格式**：
- 每戶一欄的指標總表（整窗 PAR / 每日尖峰降幅 / 協調效率 / deadline-miss rate）。
- 跨戶 box-plot 或 summary（median / IQR / outliers）。
- 代表戶（H3/H8/H20）的逐 tick 負載曲線（coord vs greedy）。

### Phase 4d 定案（17 戶 × 18 天社區，SCHED_REF bug 已修正）

**Bug 修正紀錄**
原始 `phase4d_eval.py` 的 `run_rolling()` 中，`schedule_house` / `run_coordination` 誤用
for 迴圈最後一戶（H20）的 `t_real` 作為所有戶的共同時間基準。
不同 `win_start` 的戶（H3=Jan、H8=Jan、H7=Jun、H10=May）因此出現虛假的「已超時」或
「未來 job」，造成 deadline-miss 率虛報至 31.9%，Greedy 與 Coord PAR 一致。
**修正方式**：引入 `SCHED_REF = 2000-01-01 UTC` 作為共同基準；每戶 active jobs 在進入
`schedule_house` 前以 `_normalize_for_sched()` 平移至 SCHED_REF 框架；LSTM forecast 仍
用各戶真實 `t_real`（因果性不受影響）。

**日期分類（日峰時刻 Job% 標準，診斷 2 結論）**
- 可協調日（Job% ≥ 10%）：Day [3, 7, 11, 14, 16]（5 天）
- 背景主導日（Job% < 10%）：其餘 13 天

**整窗 PAR（保守指標）**

| 方法 | PAR |
|---|---|
| No-DR | 2.8534 |
| Greedy | 2.8700 |
| Online-Coord | 2.8370 |

★ 整窗最高峰落在背景主導日（deferrable job W 佔比 < 10%），協調本來就削不動。
整窗 PAR 為保守下界；以「每日尖峰降幅」為主打指標。

**每日尖峰降幅（%，vs No-DR）— 主打指標**

| 分組 | n 天 | Greedy | Coord |
|---|---|---|---|
| 全 18 天 | 18 | +0.40% | +4.92% |
| 可協調日 | 5 | 0.00% | +10.72%±1.90% |
| 背景主導日 | 13 | — | +2.69% |

**協調效率 = (Greedy峰−Coord峰)/(Greedy峰−Oracle峰)×100%**

| 分組 | mean±std |
|---|---|
| 全 18 天 | 74.6% |
| 可協調日（5 天） | 88.8% |

**Job 層級指標**

| 指標 | 值 |
|---|---|
| Committed jobs | — |
| Deadline-miss rate | 3.6% |
| Avg delay | 2.47 h |
| Runtime | 0.093 s/tick |
| Fallback ticks | 0 |

**已知侷限**
- Day 9 coord 降幅 −0.9%（微幅墊高），為 rolling commit-first 的局部決策在背景低載日
  多排 job 於峰前時刻造成，屬方法特性而非 bug。
- 可協調日僅 5/18 天（28%），整窗 PAR 難以彰顯協調效益；論文宜以每日尖峰降幅與
  協調效率為主，整窗 PAR 為補充。

**輸出檔案**：`results/phase4d_final.json`、`results/day14_peak.png`


## Phase 5 — 實驗矩陣與指標

| 軸 | 取值 |
|---|---|
| 策略 | No-DR / Greedy-price(herding)/ Day-ahead-coordinated / **Online-coordinated(本方法)** |
| 預測品質 | perfect / local-LSTM / degraded |
| Acceptance | 1.0 / 模型化 / 敏感度掃描 |

指標三類:
- **系統**:PAR 降幅、peak kW、load factor、tariff cost。
- **用戶**:平均延後 min、deadline-miss rate、acceptance rate。
- **協調可行性**:iters/tick、runtime/tick(支撐「即時可行」)、收斂率。

## Phase 6 — robustness 與 ablation
- **forecast-degradation**:逐步加大預測誤差,量化 online 重排的自我修正 vs day-ahead 的脆弱(online 的賣點)。
- Ablation:warm-start on/off、H∈{4,6,8}h、terminal cost on/off、secure-agg on/off(只傳加總應 lossless,證明隱私不犧牲效能)。

---

## 選用延伸(暫不做)
- **FL**:FedAvg / personalized FL(FedAvg + local fine-tune)。賣點是隱私/部署/cold-start,非 accuracy。程式保留 FedAvg-ready 介面以便加回。
- **PV-aware**:納入 House 11/21,處理太陽能倒灌的 baseload。

---

## Reviewer 陷阱清單

| 陷阱 | 處理 |
|---|---|
| 因果洩漏(online 版 leakage) | `observe(t)` 只暴露 ≤ t;forecast 一律用預測值;程式層鎖死 |
| DR 策略分層混亂 | 一個目標(削峰=load leveling)→ 一個機制(shadow-price)→ behavioral DR 當交付通道 |
| herding / rebound | 中央協調 + shadow-price 機制保證錯峰;Greedy baseline 故意呈現問題 |
| 隱私宣稱(無 FL 後) | 控制層 secure-agg 為唯一機制 → 明確界定 coordinator 只見 Σ 負載;ablation 證明 lossless |
| 預測誤差傷 non-interruptible job | online 重排自我修正;但 commit 後不可停 → robustness 實驗量化代價 |
| horizon myopia | terminal cost,別讓 job 被推出窗外 |

---

## 目前進度
- **Phase 1 已定案完成**：真實資料跑通，PARAMS 經診斷定案（WM merge_gap=1 / max_len=22，其餘維持），H5 ch2 移出 deferrable。`out/` 為正式輸出。
- **Phase 2 已定案完成**：全 17 戶 local LSTM 訓練通過，無任何戶觸發斷言或破 15% 停損線。結果存於 `out_phase2_17h/`（config.json、results.json 入版控）。管線參數與評估指標定案詳見 Phase 2 定案節。
- **Phase 3a 已完成**：`phase3_simulator.py` 骨架 + 第一層因果鎖。`observe(t)` 與 `get_job()` 均已驗證：三個測試情境（正常存取 / 偷看未來 baseload / 未釋放 job）全部實際 raise，非靜默返回空值。
- **Phase 3b 已完成**：Phase 2 LSTM 接入 `forecast(t)`，第二層因果鎖上線。
  - Lock 2a：`forecast(t)` 若 t > current_t → raise CausalViolationError。
  - Lock 2b：`_assert_no_future()` 在組裝 LSTM 輸入前檢查切片，含任何 index > t 立即 raise。
  - Scaler 從 results.json 讀取（不重新 fit）。Gap 處理與 Phase 2 完全一致（handle_gaps，forward-only）。
  - 反作弊測試通過：forecast 結果 ≠ 真實未來值（t_valid=2015-03-25 18:00，H20）。
  - **全 17 戶 gap 診斷完成（`diag_phase3b_gaps17.py`）**：
    - test 段 None% 平均 27.7%（範圍 11.9–36.0%），val 段平均 2.7%；呈系統性「資料尾聲品質下降」型態（REFIT 2015 末期傳感器掉訊），非演算法 bug。
    - 全 17 戶 test 段最長乾淨連續窗均 ≥ 19 天（平均 32.3d，H8 最佳 61.4d），全部有 ≥2 段 ≥7 天乾淨窗。
    - Phase 4 可行性確認：coordinator 可在乾淨連續窗內執行；`forecast()` 回 None 時需 fallback（延用最近有效預測或跳過協調輪）。
- **Phase 4a 已完成（含 must-run 規則）**：`phase4a_schedule.py`，單戶 deferrable 排程子問題 + must-run 分類 + 負載計入驗證。
  - `schedule_house()`：infeasible job 不再靜默丟棄，轉為 `MustRunJob`，含 `deadline_missed` 旗與計算啟動點。
  - `assert_schedule_valid()` / `assert_must_run_valid()`：scheduled ①②③④；must-run ①②，③④ 豁免。
  - `compute_aggregate_load(include_must_run=True/False)`：可開/關 must-run 項的計入，用於 coordinator 做負載對比。
  - 驗證結果（H20, t=2015-03-27 20:00 UTC，HARD RULE 全 OK）：
    - **TEST A**：job 310（WM, d_slot=2, dur=8, d_slot-dur=-6<0）→ must-run start=0, deadline_missed=True ✓
    - **TEST B**：SYN-TIGHT（dur=6, d_slot=8, horizon=4）→ horizon 短於 dur 強制 infeasible，但 d_slot-dur=2≥0 → must-run start=2, deadline_missed=False ✓
    - **TEST C**：負載帳目驗證 — diff 在 slots 0–7 精確等於 326.4 W，slot 8+ 為 0.0 ✓
    - **TEST E**：故意給 must-run 錯誤 end_slot → ① violation raise ✓
- **Phase 4b 已完成（含收斂修正與 oracle 定位）**：`phase4b_coordinator.py`，3 戶 shadow-price 協調（H3/H8/H20），單 horizon，乾淨窗 t=2015-04-10 10:00 UTC。
  - `tick_all_houses()`：各戶用同一 λ 解 schedule_house，coordinator 只收 Σ load（隱私邊界）。
  - `run_coordination()`：subgradient λ 更新，衰減步長 α₀/√(k+1)，target = mean(greedy L)（固定，能量守恆）。
    - **best-so-far 追蹤**（非單調 subgradient 必要）：回傳 best_L / best_results / best_lam，不回 last。
    - best_lam 為 Phase 4c rolling warm-start 的種子（必須用 best_lam，不可用 last λ）。
  - `oracle_milp()`：PuLP (CBC) + scipy fallback，min max(L) MILP，同 t/jobs/horizon。
  - **三方 PAR 對照（4 active jobs，全 s_min=0 → greedy herding）**：
    | 方法 | PAR | peak W | Δ vs greedy |
    |---|---|---|---|
    | Greedy (herding) | 2.392 | 4326 | — |
    | Online coord (best-so-far) | **1.628** | 2944 | −32.0% |
    | Oracle MILP (day-ahead OPT) | **1.344** | 2430 | −43.8% |
    - **協調效率 = (2.392−1.628)/(2.392−1.344)×100% = 72.9%**（離散 subgradient 已捕捉 ~73% 的可用削峰空間）
    - Oracle 最佳排程：H3-WM→13、H8-DR→0、H20-WM→18、H20-TD→4（shadow-price 找到 H20-TD→20，與最優的 →4 有 16 slot 落差）
  - Red-flag 驗證：31 次迭代 PAR > greedy（最高 2.420），best_par 全程不採壞解（is_best=True@bad = 0）。
  - Mean conservation：greedy 與 coord diff = 0.000W。
  - 振盪診斷：離散問題的對偶間隙——best PAR 在 iter 38 定案後不再改善；屬結構性而非演算法 bug。
  - HARD RULE 全 OK。
- **Phase 4c 已完成（rolling horizon + commit-first + warm-start）**：`phase4c_rolling.py`。
  - 外層逐 tick（Δ=10 min）推進；每 tick 呼叫 `sim.observe(t)` + `sim.forecast(t)`，跑內層 shadow-price 協調（50 iter），取 best_lam 為下 tick warm-start。
  - **commit-first**：僅 `start_slot=0` 的 scheduled / must-run job 被鎖定，後續 tick 不可撤銷（`_build_hd_list` filter 結構保證）。
  - **None fallback**：`forecast=None` 時將最近有效預測左移 1 slot 延用，標記 fallback tick。
  - 已 commit 的 running job 加回各戶 forecast（`_running_bg` background correction），協調時正確計入背景負載。
  - **驗證結果（2 天 × H3/H8/H20，t=2015-04-10 10:00 UTC，288 ticks）**：
    | 方法 | 整窗 PAR | peak W | fallback ticks | 耗時 |
    |---|---|---|---|---|
    | Greedy (rolling) | 3.563 | — | 0 | — |
    | Coordinated (rolling) | **2.949** | — | 0 | 5.8 s |
    - 整窗 PAR 降幅 −17.2%（3.563→2.949）。
    - warm-start 驗證：tick k 的 `lam_init = best_lam[k-1]`，PASS（浮點精確）。
    - commit 不可撤：dict filter 結構保證，no duplicate job_id，PASS。
    - 耗時 5.8 s / 288 ticks = 0.02 s/tick → **即時可行佐證**。
  - **已知侷限**：整窗 PAR 被「無 active job 低載時段」稀釋（單點 −32% 被攤成 −17.2%）；3 戶 2 天 job 太稀疏，需放大到 17 戶全窗才能體現協調價值。
  - HARD RULE 全 OK。
- **Phase 5–6 尚未開始。**
- **下一步：Phase 4d** — 擴展評估規模與指標定案（見下方 Phase 4d 節）。
- **git 待清理備忘**：`model_house*.pt` 與過程目錄 `out_phase2/`、`out_phase2_baseline/`、`out_phase2_lr5e4/` 被誤 commit 並 push。處理方式：`.gitignore` 已補規則 → 執行 `git rm --cached` 移出追蹤 → 新增 commit（不要 reset 或改寫歷史）。`out_phase2_17h/config.json` 與 `results.json` 保留在版控。
