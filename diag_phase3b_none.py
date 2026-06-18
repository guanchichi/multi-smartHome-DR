"""
診斷: Phase 3b forecast 在 test 段開頭為何回 None
目標:
  1. 定位 t0=2015-03-23 18:00 的 144 格窗口，找出 NaN 原因
  2. 掃全 test 段，統計 None 分布（集中 gap 還是系統性邊界問題）
  3. 判定是 (a) 真實資料 gap 還是 (b) off-by-one / 邊界 bug
只診斷，不修改任何程式。
"""

import numpy as np
import pandas as pd
from pathlib import Path

from phase3_simulator import Simulator

HOUSE     = 20
LOOK_BACK = 144
SLOT_MIN  = 10

# ─────────────────────────────────────────────────────────────────────────────
# 載入 simulator（不需要 load_lstm，純看資料結構）
# ─────────────────────────────────────────────────────────────────────────────
sim = Simulator(HOUSE)

bl_raw   = sim._baseload      # Phase 1 輸出，可能有缺失 timestamp
bl_interp = sim._bl_interp    # handle_gaps 後的均勻格線

print("=" * 64)
print("PART 0: 基本比較 — _baseload vs _bl_interp")
print("=" * 64)
print(f"  _baseload    : {len(bl_raw)} rows  "
      f"[{bl_raw.index[0]}  →  {bl_raw.index[-1]}]")
print(f"  _bl_interp   : {len(bl_interp)} rows  "
      f"[{bl_interp.index[0]}  →  {bl_interp.index[-1]}]")

# 理論均勻格數（從 bl_raw 起訖計算）
td      = bl_raw.index[-1] - bl_raw.index[0]
uniform_slots = int(td.total_seconds() / (SLOT_MIN * 60)) + 1
print(f"  理論均勻格數 : {uniform_slots}")
extra = len(bl_interp) - uniform_slots
print(f"  _bl_interp 多出格數 : {extra}  "
      f"({'正常 ±幾格' if abs(extra) < 5 else '⚠ 超出預期，需確認'})")
print(f"  _bl_interp NaN 總數 : {bl_interp.isna().sum()}")
print(f"  _baseload  NaN 總數 : {bl_raw.isna().sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: 定位 t0 窗口
# ─────────────────────────────────────────────────────────────────────────────
t0_req = pd.Timestamp("2015-03-23 18:00:00", tz="UTC")
sim._jump_to(t0_req)
t0 = sim.current_t

print(f"\n{'=' * 64}")
print(f"PART 1: t0 窗口診斷  (t0={t0})")
print("=" * 64)
print(f"  _jump_to({t0_req}) → current_t = {t0}")
if t0 != t0_req:
    print(f"  ⚠ jump 結果 ≠ 請求時間 (diff = {t0 - t0_req})")

# _bl_interp 中 t0 的位置
pos_interp = int(bl_interp.index.searchsorted(t0))
print(f"\n  t0 在 _bl_interp 的位置 : pos={pos_interp} / {len(bl_interp)-1}")

if pos_interp < LOOK_BACK:
    print(f"  → None 原因: pos_interp={pos_interp} < LOOK_BACK={LOOK_BACK}  (歷史不足)")
else:
    window_series = bl_interp.iloc[pos_interp - LOOK_BACK : pos_interp]
    win_start = window_series.index[0]
    win_end   = window_series.index[-1]
    nan_count = int(window_series.isna().sum())

    print(f"  窗口範圍 : {win_start} → {win_end}")
    print(f"  NaN 數   : {nan_count} / {LOOK_BACK}")

    if nan_count == 0:
        print(f"  → 窗口乾淨，無 NaN —— 不應回 None！可能是 off-by-one bug")
    else:
        print(f"  → 窗口含 NaN，forecast 正確回 None")

        # 找 NaN run
        nan_mask = window_series.isna().values
        runs = []
        i = 0
        while i < len(nan_mask):
            if nan_mask[i]:
                j = i
                while j < len(nan_mask) and nan_mask[j]:
                    j += 1
                runs.append((i, j - 1, j - i,
                              window_series.index[i],
                              window_series.index[j - 1]))
                i = j
            else:
                i += 1

        print(f"\n  NaN runs in window ({len(runs)} 段):")
        for rs, re, rlen, ts, te in runs:
            print(f"    slots [{rs:3d}–{re:3d}]  len={rlen:4d}  "
                  f"{ts} → {te}")

        # 比對 _baseload 在同範圍內有幾格
        bl_in_win = bl_raw[(bl_raw.index >= win_start) & (bl_raw.index <= win_end)]
        expected  = LOOK_BACK
        present   = len(bl_in_win)
        missing   = expected - present
        print(f"\n  _baseload 在同範圍內: {present} 格 (uniform 應有 {expected}，"
              f"缺 {missing} 格)")
        if missing > 0:
            print(f"  → 原始資料在此範圍有 {missing} 格缺失 ← gap 為真實資料問題")
        else:
            print(f"  → _baseload 格數完整；NaN 來自 handle_gaps 插值限制")

        # 對應的 _baseload 在 NaN run 時段
        for rs, re, rlen, ts, te in runs:
            bl_in_run = bl_raw[(bl_raw.index >= ts) & (bl_raw.index <= te)]
            print(f"    NaN run [{ts} → {te}]: "
                  f"_baseload 有 {len(bl_in_run)} 格 (應有 {rlen}，"
                  f"缺 {rlen - len(bl_in_run)})")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: 掃全 test 段（純陣列操作，不跑 LSTM）
# ─────────────────────────────────────────────────────────────────────────────
test_start = pd.Timestamp("2015-03-23 18:00:00", tz="UTC")

print(f"\n{'=' * 64}")
print(f"PART 2: 全 test 段掃描  (test_start={test_start})")
print("=" * 64)

bl_arr = bl_interp.values.astype(np.float64)
bl_idx = bl_interp.index

# test 段在 _bl_interp 的位置範圍
test_start_pos = int(bl_idx.searchsorted(test_start))
test_end_pos   = len(bl_idx)
total_test     = test_end_pos - test_start_pos
print(f"  test 段在 _bl_interp: pos [{test_start_pos}, {test_end_pos-1}]  "
      f"共 {total_test} 格")

# 對每個 test 位置，判斷 forecast 是否為 None
none_mask = np.zeros(total_test, dtype=bool)
for i, pos in enumerate(range(test_start_pos, test_end_pos)):
    if pos < LOOK_BACK:
        none_mask[i] = True   # 歷史不足
    else:
        win = bl_arr[pos - LOOK_BACK : pos]
        none_mask[i] = np.isnan(win).any()

none_count  = int(none_mask.sum())
valid_count = total_test - none_count
print(f"  forecast=None : {none_count} 格 ({100*none_count/total_test:.1f}%)")
print(f"  forecast=OK   : {valid_count} 格 ({100*valid_count/total_test:.1f}%)")

# 找 None 的連續段
none_global_pos = np.where(none_mask)[0] + test_start_pos  # 在 _bl_interp 的絕對位置
if len(none_global_pos) > 0:
    runs = []
    rs = none_global_pos[0]
    re = none_global_pos[0]
    for p in none_global_pos[1:]:
        if p == re + 1:
            re = p
        else:
            runs.append((rs, re))
            rs = re = p
    runs.append((rs, re))

    print(f"\n  None 連續段 ({len(runs)} 段):")
    for rs, re in runs:
        rlen    = re - rs + 1
        ts_run  = bl_idx[rs]
        te_run  = bl_idx[re]
        # 此 run 最早的 NaN 槽在哪
        if rs >= LOOK_BACK:
            # 找窗口內 NaN 的最早位置
            w = bl_arr[rs - LOOK_BACK : rs]
            first_nan_in_win = int(np.argmax(np.isnan(w)))
            nan_ts = bl_idx[rs - LOOK_BACK + first_nan_in_win]
        else:
            nan_ts = bl_idx[0]
        print(f"    [{ts_run}  →  {te_run}]  "
              f"len={rlen} 格  (窗口內首個 NaN 約 {nan_ts})")
else:
    print("  無 None → test 段完全乾淨")

# ─────────────────────────────────────────────────────────────────────────────
# PART 3: 判定結論
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 64}")
print("PART 3: 判定")
print("=" * 64)

if none_count == 0:
    print("  → test 段無 None：之前的 WARN 可能是 jump 落點問題，需進一步確認。")
elif none_count / total_test > 0.10:
    print(f"  ⚠ None 比例 {100*none_count/total_test:.1f}% 偏高，需確認是否系統性 bug。")
elif len(runs) <= 5:
    print(f"  → None 集中在 {len(runs)} 個 gap 區段，比例 "
          f"{100*none_count/total_test:.1f}%。")
    print(f"     判定: (a) 真實資料 gap 問題，forecast 正確拒絕含 NaN 窗口。")
    print(f"     Phase 3b 行為符合預期：長 gap 段不提供預測，非 bug。")
else:
    print(f"  None 分散在 {len(runs)} 段，可能為邊界或 off-by-one 問題，需細查。")

print()
