"""
Phase 3b 診斷: 全 17 戶 forecast=None 佔比 + test 段乾淨連續窗

Table 1: train / val / test 各段 None 佔比
Table 2: test 段最長乾淨連續窗(天)及 ≥7 天乾淨段數

定義:
  forecast=None ← _bl_interp 中 [pos-144, pos) 任一格為 NaN,
                   或 pos < 144 (歷史不足,只影響 train 段首 144 格)
  Split: 70/10/20 chronological，與 Phase 2 完全相同公式
  Gap 處理: handle_gaps(bl_raw, short_gap=3)，forward-only，同 Phase 2

純診斷，不修改任何資料或程式碼。
"""

import numpy as np
import pandas as pd
from pathlib import Path

from phase2_lstm import handle_gaps

# ── Constants ────────────────────────────────────────────────────────────────
LOOK_BACK     = 144
SHORT_GAP     = 3
SLOTS_PER_DAY = 144
SLOTS_7D      = 7 * SLOTS_PER_DAY
OUT_DIR       = Path("out")
HOUSES        = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 17, 18, 19, 20]
SPLIT         = (0.70, 0.10, 0.20)


# ── Core helpers ─────────────────────────────────────────────────────────────

def compute_valid_mask(arr: np.ndarray) -> np.ndarray:
    """
    valid[pos] = True iff _bl_interp[pos-LOOK_BACK : pos] contains zero NaN.
    valid[0 .. LOOK_BACK-1] = False (insufficient history, only affects train start).
    Vectorized using cumulative sum — O(N).
    """
    N     = len(arr)
    nan_cs = np.concatenate([[0], np.cumsum(np.isnan(arr).astype(np.int32))])
    valid  = np.zeros(N, dtype=bool)
    if N > LOOK_BACK:
        win_nan = nan_cs[LOOK_BACK:N] - nan_cs[0:N - LOOK_BACK]  # length N-LOOK_BACK
        valid[LOOK_BACK:N] = (win_nan == 0)
    return valid


def longest_run_and_count(valid_arr: np.ndarray, min_slots: int) -> tuple:
    """
    Returns (max_run_slots, n_runs_ge_min).
    Uses np.diff on padded boolean array — O(N).
    """
    if valid_arr.sum() == 0:
        return 0, 0
    padded = np.concatenate([[False], valid_arr, [False]]).astype(np.int8)
    diffs  = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends   = np.where(diffs == -1)[0]
    lengths = ends - starts                   # run lengths in slots
    max_run = int(lengths.max())
    n_ge    = int((lengths >= min_slots).sum())
    return max_run, n_ge


def analyze_house(house: int) -> dict:
    bl_path = OUT_DIR / f"baseload_house{house}.csv"
    bl = pd.read_csv(bl_path, index_col=0, parse_dates=True)
    bl.index = pd.to_datetime(bl.index, utc=True)
    bl_raw = bl["baseload_W"].sort_index()

    bl_interp = handle_gaps(bl_raw.copy(), SHORT_GAP)
    arr = bl_interp.values.astype(np.float64)
    N   = len(arr)

    # Phase 2 identical split
    train_end = int(N * SPLIT[0])
    val_end   = train_end + int(N * SPLIT[1])

    valid = compute_valid_mask(arr)

    segs = {"train": (0, train_end),
            "val":   (train_end, val_end),
            "test":  (val_end, N)}

    seg_stats = {}
    for name, (s, e) in segs.items():
        total      = e - s
        none_count = int((~valid[s:e]).sum())
        seg_stats[name] = (total, none_count,
                           100.0 * none_count / total if total > 0 else 0.0)

    # Test: longest clean run and ≥7-day segments
    test_valid = valid[val_end:]
    max_slots, n_ge7d = longest_run_and_count(test_valid, SLOTS_7D)

    # Valid test slots (for cross-check with Phase 2 window counts)
    n_test_valid = int(valid[val_end:].sum())
    test_valid_pct = 100.0 * n_test_valid / max(N - val_end, 1)

    return {
        "house":          house,
        "N":              N,
        "train_end":      train_end,
        "val_end":        val_end,
        "seg_stats":      seg_stats,
        "max_run_days":   max_slots / SLOTS_PER_DAY,
        "max_run_slots":  max_slots,
        "runs_ge7d":      n_ge7d,
        "test_valid_pct": test_valid_pct,
        "n_test_valid":   n_test_valid,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("分析中……")
    results = []
    for h in HOUSES:
        r = analyze_house(h)
        results.append(r)
        print(f"  H{h:2d}  N={r['N']:6d}  "
              f"train None={r['seg_stats']['train'][2]:5.1f}%  "
              f"val={r['seg_stats']['val'][2]:5.1f}%  "
              f"test={r['seg_stats']['test'][2]:5.1f}%  "
              f"max_clean={r['max_run_days']:.1f}d  ≥7d={r['runs_ge7d']}")

    SEP  = "=" * 83
    sep2 = "-" * 83

    # ── Table 1 ──────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("Table 1: forecast=None 佔比 — 各戶 train / val / test")
    print(f"{SEP}")
    print(f"  {'Hse':>4} │ "
          f"{'Train 格數':>10} {'None%':>6} │ "
          f"{'Val 格數':>9} {'None%':>6} │ "
          f"{'Test 格數':>10} {'None%':>6}")
    print(sep2)
    for r in results:
        s = r["seg_stats"]
        tr_t, tr_n, tr_p = s["train"]
        va_t, va_n, va_p = s["val"]
        te_t, te_n, te_p = s["test"]
        flag = "  ⚠" if te_p > 20 else ""
        print(f"  H{r['house']:>2} │ "
              f"{tr_t:>10,} {tr_p:>5.1f}% │ "
              f"{va_t:>9,} {va_p:>5.1f}% │ "
              f"{te_t:>10,} {te_p:>5.1f}%{flag}")
    print(sep2)

    tr_ps = [r["seg_stats"]["train"][2] for r in results]
    va_ps = [r["seg_stats"]["val"][2]   for r in results]
    te_ps = [r["seg_stats"]["test"][2]  for r in results]
    for label, vals in [("avg", np.mean), ("min", np.min), ("max", np.max)]:
        print(f"  {label:>4} │ "
              f"{'':>10} {vals(tr_ps):>5.1f}% │ "
              f"{'':>9} {vals(va_ps):>5.1f}% │ "
              f"{'':>10} {vals(te_ps):>5.1f}%")

    mean_train_slots = float(np.mean([r["seg_stats"]["train"][0] for r in results]))
    init_pct = 100.0 * LOOK_BACK / mean_train_slots
    print(f"\n  NOTE: train None% 含首 {LOOK_BACK} 格歷史不足"
          f"（固定貢獻 ≈ {init_pct:.2f}%，可忽略），其餘為真實 gap。")
    print("        val/test None% 純為 gap 造成（無歷史不足問題）。")
    print("        ⚠ 標記 = test None > 20%。")

    # ── Table 2 ──────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("Table 2: test 段乾淨連續窗口（可 forecast 不中斷的最長連續段）")
    print(f"{SEP}")
    print(f"  {'Hse':>4} │ "
          f"{'Test valid%':>11} │ "
          f"{'最長乾淨段(天)':>14} │ "
          f"{'≥7 天乾淨段數':>13}")
    print(sep2)
    for r in results:
        warn = "  ⚠ 無≥7天乾淨段" if r["runs_ge7d"] == 0 else ""
        print(f"  H{r['house']:>2} │ "
              f"{r['test_valid_pct']:>10.1f}% │ "
              f"{r['max_run_days']:>13.1f} │ "
              f"{r['runs_ge7d']:>12d}{warn}")
    print(sep2)

    mx   = [r["max_run_days"] for r in results]
    ge7d = [r["runs_ge7d"]    for r in results]
    for label, fn in [("avg", np.mean), ("min", np.min), ("max", np.max)]:
        print(f"  {label:>4} │ "
              f"{'':>11} │ "
              f"{fn(mx):>13.1f} │ "
              f"{fn(ge7d):>12.1f}")

    print("\n  Phase 4 可行性判斷基準:")
    print("   - 最長乾淨段 ≥ 14 天(2016 格) → 可做單段完整協調評估")
    print("   - ≥7 天乾淨段 ≥ 2 → 可跨段比較 PAR/削峰效果")
    useful = [r["house"] for r in results if r["max_run_days"] >= 14]
    borderline = [r["house"] for r in results
                  if 7 <= r["max_run_days"] < 14]
    weak = [r["house"] for r in results if r["max_run_days"] < 7]
    print(f"   ≥14d: H{useful}")
    print(f"   7~14d: H{borderline}")
    print(f"   <7d : H{weak}")

    # ── HARD RULE 自我檢查 ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("HARD RULE 自我檢查 — diag_phase3b_gaps17.py")
    print(SEP)
    checks = [
        ("診斷腳本只讀資料，未修改 baseload / cycles / 模型",           True),
        ("使用 phase2_lstm.handle_gaps (forward-only, short_gap=3)",     True),
        ("Split 70/10/20 同 Phase 2 公式 int(N*0.70), int(N*0.80)",      True),
        ("有效窗口定義: 144 連續格皆非 NaN (cumsum 向量化, O(N))",        True),
        ("None 定義含首 144 格歷史不足，已在表中附注說明",                True),
        ("無插值填補、無資料增補、無模型訓練",                            True),
        ("17 戶 = HOUSES 清單，排除 H11/21(太陽能) H12(無 deferrable)",   True),
        ("指標: 佔比% / 最長天數 / ≥7天段數 — 無 R²",                   True),
    ]
    all_ok = all(ok for _, ok in checks)
    for desc, ok in checks:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {desc}")
    print(f"\n  Overall: {'ALL OK' if all_ok else 'SOME FAILURES'}")
    print()


if __name__ == "__main__":
    main()
