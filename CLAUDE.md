# REFIT 即時協調式 DR — 專案說明(for Claude Code)

每戶本地 LSTM 預測 baseload,中央 DR Coordinator 以 online rolling-horizon + shadow-price 協調各戶 deferrable 電器排程,跨戶削峰(load leveling)。Coordinator 只看加總負載(secure aggregation),原始 state/排程/log 不出戶。FL 暫不做。

## 真相來源
`PLAN.md` 是唯一規格來源。實作一律以 PLAN.md 為準;**與 PLAN.md 衝突或規格不明時,停下來問,不要自行改設計**。

## 不可違反的硬規則(HARD RULES)
1. 時間解析度固定 **Δ=10min(144 slots/day)**。
2. **chronological split 70/10/20,禁止 shuffle**。任何 train/test 切分必須按時間先後。
3. **因果性**:模擬器 `observe(t)` 只能回傳 ≤ t 的資料;forecast 一律用預測值,**禁止讀取 actual 未來值**。
4. cycle 抽取順序:**先 `merge_short_gaps` 再過濾長度**,不可顛倒。
5. 排除 House **11/21**(太陽能)、**12**(無 deferrable);deferrable 一律用 per-house `DEFERRABLE_MAP`,**禁止用 channel index 硬編**。
6. `baseload = Aggregate − Σ(deferrable)`,clip≥0,**不重組 aggregate**。
7. deferrable cycle 為 **non-interruptible**。
8. FL 暫不做,但 Phase 2 本地訓練要**保留 FedAvg-ready 介面**(本地訓練包成可被外層權重聚合呼叫)。
9. 評估**不用 R²**(避免 leakage 灌水爭議),用 RMSE / MAE / MAPE。

## 工作方式(重要)
- **一次只做一個 Phase**。做完 → 跑驗證 → 把 sanity 輸出(圖/表)給我看 → 我確認後才進下一個。**禁止一次實作整個 plan**。
- 每個 Phase 交付要附:① 可跑指令 ② sanity 輸出 ③ 自我檢查「是否違反任何 HARD RULE」。
- 固定 seed;所有 config 存檔。
- env:numpy<2.0 / torch 2.3.x。改動相依套件前先問。

## 進度
- **Phase 1**(`phase1_cycles.py`)已完成並驗證(self-test 注入數=抽出數)。
- **下一步**:真實 CLEAN 資料跑 Phase 1 → 看 `summary_by_type.csv` 回調 `PARAMS` → 進 Phase 2 每戶 local LSTM。
