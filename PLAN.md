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
主方法 online shadow-price loop(雙層:外層 rolling horizon,內層協調):

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
- **下一步：Phase 3** — 因果模擬器（`observe(t)` 只回傳 ≤ t 資料，baseload 用 LSTM 預測值，job 在實際 `r_j` 釋放，不可偷看未來）。
- Phase 4–6 尚未開始。
- **git 待清理備忘**：`model_house*.pt` 與過程目錄 `out_phase2/`、`out_phase2_baseline/`、`out_phase2_lr5e4/` 被誤 commit 並 push。處理方式：`.gitignore` 已補規則 → 執行 `git rm --cached` 移出追蹤 → 新增 commit（不要 reset 或改寫歷史）。`out_phase2_17h/config.json` 與 `results.json` 保留在版控。
