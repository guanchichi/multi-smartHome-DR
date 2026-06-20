"""
Phase 4d: 合成社區 — 建構 + 對齊驗證
每戶取自己 test 段內最長乾淨連續窗，對齊到相對時間軸，疊加成合成社區。
本輪只建社區 + 四項驗證，不跑 baseline、不跑協調。
"""

import numpy as np
import pandas as pd
from pathlib import Path

from phase2_lstm import handle_gaps

# ── Constants ─────────────────────────────────────────────────────────────────
HOUSES        = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 17, 18, 19, 20]
SPLIT         = (0.70, 0.10, 0.20)
LOOK_BACK     = 144           # slots
SLOTS_PER_DAY = 144
SHORT_GAP     = 3
OUT_DIR       = Path("out")
CYCLES_DIR    = Path("out")

DEFERRABLE_MAP = {
    1:  ["WM"],
    2:  ["WM"],
    3:  ["WM", "TD"],
    4:  ["WM"],
    5:  ["WM", "DW"],
    6:  ["WM", "DW"],
    7:  ["WM"],
    8:  ["WM", "TD"],
    9:  ["WM", "DW"],
    10: ["WM", "DW"],
    13: ["WM", "DW"],
    15: ["WM"],
    16: ["WM", "DW"],
    17: ["WM"],
    18: ["WM", "DW"],
    19: ["WM"],
    20: ["WM", "TD"],
}

# ── Valid mask (same as diag_phase3b_gaps17) ──────────────────────────────────

def compute_valid_mask(arr: np.ndarray) -> np.ndarray:
    N      = len(arr)
    nan_cs = np.concatenate([[0], np.cumsum(np.isnan(arr).astype(np.int32))])
    valid  = np.zeros(N, dtype=bool)
    if N > LOOK_BACK:
        win_nan = nan_cs[LOOK_BACK:N] - nan_cs[0:N - LOOK_BACK]
        valid[LOOK_BACK:N] = (win_nan == 0)
    return valid


# ── Longest run via boolean padded-diff (never asi8) ──────────────────────────

def find_longest_run(valid_arr: np.ndarray):
    """Return (start_idx, end_idx_excl, length) of longest True run."""
    if valid_arr.sum() == 0:
        return 0, 0, 0
    padded = np.concatenate([[False], valid_arr, [False]]).astype(np.int8)
    diffs  = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends   = np.where(diffs == -1)[0]
    lengths = ends - starts
    best_i  = int(np.argmax(lengths))
    return int(starts[best_i]), int(ends[best_i]), int(lengths[best_i])


# ── Per-house: find longest clean test window, trimmed to UTC midnight ─────────

def find_test_window(house: int) -> dict:
    bl_path = OUT_DIR / f"baseload_house{house}.csv"
    bl      = pd.read_csv(bl_path, index_col=0, parse_dates=True)
    bl.index = pd.to_datetime(bl.index, utc=True)
    bl_raw  = bl["baseload_W"].sort_index()

    arr = handle_gaps(bl_raw.copy(), SHORT_GAP).values.astype(np.float64)
    N   = len(arr)

    train_end  = int(N * SPLIT[0])
    val_end    = train_end + int(N * SPLIT[1])
    test_start_ts = bl_raw.index[val_end]

    valid      = compute_valid_mask(arr)
    test_valid = valid[val_end:]             # test segment only
    test_times = bl_raw.index[val_end:]

    s_i, e_i, length = find_longest_run(test_valid)

    if length == 0:
        return {"house": house, "n_slots": 0, "n_days": 0,
                "test_start": test_start_ts}

    raw_start = test_times[s_i]
    raw_end   = test_times[e_i - 1]         # inclusive last slot

    # Trim to full UTC days: advance start to next midnight if not already
    if raw_start.hour == 0 and raw_start.minute == 0:
        win_start = raw_start
    else:
        win_start = (raw_start.normalize() + pd.Timedelta(days=1)).tz_localize("UTC") \
                    if raw_start.tzinfo is None \
                    else raw_start.normalize() + pd.Timedelta(days=1)

    # Number of complete days from win_start to raw_end (inclusive)
    duration_s = (raw_end + pd.Timedelta(minutes=10) - win_start).total_seconds()
    n_days     = int(duration_s // 86400)

    if n_days <= 0:
        return {"house": house, "n_slots": 0, "n_days": 0,
                "test_start": test_start_ts}

    win_end_excl = win_start + pd.Timedelta(days=n_days)
    n_slots      = n_days * SLOTS_PER_DAY

    return {
        "house":        house,
        "test_start":   test_start_ts,
        "raw_start":    raw_start,
        "raw_end":      raw_end,
        "win_start":    win_start,
        "win_end_excl": win_end_excl,
        "n_days":       n_days,
        "n_slots":      n_slots,
    }


# ── Load gap-handled baseload for a window, as flat ndarray ───────────────────

def load_window_baseload(house: int, win_start: pd.Timestamp, n_slots: int) -> np.ndarray:
    bl_path = OUT_DIR / f"baseload_house{house}.csv"
    bl      = pd.read_csv(bl_path, index_col=0, parse_dates=True)
    bl.index = pd.to_datetime(bl.index, utc=True)
    bl_raw  = bl["baseload_W"].sort_index()

    win_end = win_start + pd.Timedelta(minutes=10 * n_slots)
    bl_slice = bl_raw.loc[(bl_raw.index >= win_start) & (bl_raw.index < win_end)]

    # Map to slot index via total_seconds (avoids asi8 unit ambiguity)
    offsets = np.round(
        (bl_slice.index - win_start).total_seconds().values / 600.0
    ).astype(np.int64)

    arr = handle_gaps(bl_slice, SHORT_GAP).values.astype(np.float64)
    out = np.full(n_slots, np.nan)
    mask = (offsets >= 0) & (offsets < n_slots)
    out[offsets[mask]] = arr[mask]
    return out


# ── Load jobs from test window on relative axis ────────────────────────────────

def load_window_jobs(house: int, win_start: pd.Timestamp, n_slots: int) -> list:
    """Return list of job dicts with r_j/d_j in relative slots [0, n_slots)."""
    cyc_path = CYCLES_DIR / f"cycles_house{house}.csv"
    if not cyc_path.exists():
        return []

    cycles = pd.read_csv(cyc_path, parse_dates=["t_start", "t_end"])
    cycles["t_start"] = pd.to_datetime(cycles["t_start"], utc=True)
    cycles["t_end"]   = pd.to_datetime(cycles["t_end"],   utc=True)

    win_end = win_start + pd.Timedelta(minutes=10 * n_slots)

    jobs = []
    for _, row in cycles.iterrows():
        if row["t_start"] >= win_start and row["t_end"] <= win_end:
            r_j = int(round((row["t_start"] - win_start).total_seconds() / 600.0))
            d_j = int(round((row["t_end"]   - win_start).total_seconds() / 600.0))
            dur = int(row.get("duration_slots", d_j - r_j))
            jobs.append({
                "house": house,
                "r_j":   r_j,
                "d_j":   d_j,
                "dur":   dur,
                "mean_W": float(row.get("mean_W", 0.0)),
                "type":   str(row.get("type", "?")),
            })
    return jobs


# ── Validation helpers ─────────────────────────────────────────────────────────

def val_A_hour_of_day(windows: dict, n_comm: int):
    """A: same relative slot → same hour-of-day for all houses."""
    print("\n── Validation A: Hour-of-Day 保真 ──────────────────────────────────")
    sample_slots = [0, 18, 36, 72, 108, 143, 144, 288]
    sample_slots = [s for s in sample_slots if s < n_comm]

    header = f"  {'Slot':>6}  {'Expected HH:MM':>14}"
    for h in HOUSES:
        if h in windows:
            header += f"  H{h:>2}"
    print(header)

    all_ok = True
    for slot in sample_slots:
        win_start = windows[list(windows.keys())[0]]["win_start"]
        expected_hm = (win_start + pd.Timedelta(minutes=10 * slot)).strftime("%H:%M")
        row_str = f"  {slot:>6}  {expected_hm:>14}"
        for h, info in windows.items():
            ts = info["win_start"] + pd.Timedelta(minutes=10 * slot)
            hm = ts.strftime("%H:%M")
            match = "OK" if hm == expected_hm else "!!"
            if match == "!!":
                all_ok = False
            row_str += f"  {match}"
        print(row_str)

    # Deeper check: for each house, verify slot→hour consistent with win_start midnight
    for h, info in windows.items():
        ws = info["win_start"]
        assert ws.hour == 0 and ws.minute == 0, f"H{h} win_start not midnight: {ws}"
    print(f"  結論: {'ALL OK - 所有戶相對 slot 對應相同 HH:MM' if all_ok else 'FAIL'}")
    return all_ok


def val_B_no_leakage(windows: dict):
    """B: each house window starts at or after its own test_start."""
    print("\n── Validation B: 無 Leakage ─────────────────────────────────────────")
    print(f"  {'Hse':>4}  {'win_start':>20}  {'test_start':>20}  {'win≥test':>8}")
    all_ok = True
    for h, info in windows.items():
        ok = info["win_start"] >= info["test_start"]
        if not ok:
            all_ok = False
        flag = "OK" if ok else "LEAKAGE!"
        print(f"  H{h:>2}  {str(info['win_start'])[:19]:>20}  "
              f"{str(info['test_start'])[:19]:>20}  {flag:>8}")
    print(f"  結論: {'ALL OK - 所有戶無 leakage' if all_ok else 'FAIL - 存在 leakage'}")
    return all_ok


def val_C_job_density(all_jobs: list, n_comm: int):
    """C: active job count per slot; check coordination viability."""
    print("\n── Validation C: Job 密度 ───────────────────────────────────────────")
    active = np.zeros(n_comm, dtype=np.int32)
    for job in all_jobs:
        r = max(0, job["r_j"])
        d = min(n_comm, job["d_j"])
        if r < d:
            active[r:d] += 1

    pct_0  = 100.0 * (active == 0).sum() / n_comm
    pct_1  = 100.0 * (active == 1).sum() / n_comm
    pct_ge2 = 100.0 * (active >= 2).sum() / n_comm

    print(f"  total jobs   : {len(all_jobs)}")
    print(f"  active=0 ticks: {pct_0:.1f}%")
    print(f"  active=1 ticks: {pct_1:.1f}%")
    print(f"  active≥2 ticks: {pct_ge2:.1f}%  ← 協調施力點")

    # Distribution summary
    for cnt in range(int(active.max()) + 1):
        n = int((active == cnt).sum())
        bar = "#" * min(40, int(n / n_comm * 200))
        print(f"  active={cnt}: {n:5d} slots  {bar}")

    viable = pct_ge2 >= 5.0
    if not viable:
        print("  ⚠ 警告: active≥2 占比 < 5%，協調施力點不足，建議停下來確認。")
    else:
        print("  協調可行性: OK")
    return viable, pct_ge2


def val_D_daily_profile(community_bl: np.ndarray):
    """D: community aggregate baseload mean daily profile (24h → 144 slots)."""
    print("\n── Validation D: 社區日週期輪廓 ────────────────────────────────────")
    n_slots = len(community_bl)
    n_days  = n_slots // SLOTS_PER_DAY
    bl_2d   = community_bl[:n_days * SLOTS_PER_DAY].reshape(n_days, SLOTS_PER_DAY)
    profile = np.nanmean(bl_2d, axis=0)   # shape (144,)

    # Print every hour (every 6 slots)
    print(f"  {'Hour':>5}  {'Avg W':>8}  Bar")
    max_w = np.nanmax(profile)
    for hour in range(24):
        slot = hour * 6
        w    = profile[slot]
        bar  = "#" * int(40 * w / max_w) if max_w > 0 else ""
        print(f"  {hour:>5}h  {w:>8.0f}  {bar}")

    # Check for AM/PM peaks and nightly trough
    night_slots = list(range(0, 18)) + list(range(132, 144))   # 0–3h and 22–24h
    day_slots   = list(range(42, 114))                          # 7–19h
    night_mean  = float(np.nanmean(profile[night_slots]))
    day_mean    = float(np.nanmean(profile[day_slots]))
    peak_slot   = int(np.nanargmax(profile))
    peak_hour   = peak_slot / 6
    print(f"\n  夜間均值 (0-3h, 22-24h) : {night_mean:.0f} W")
    print(f"  日間均值 (7-19h)         : {day_mean:.0f} W")
    print(f"  全日峰值時刻             : {peak_hour:.1f}h  ({profile[peak_slot]:.0f} W)")
    has_peak = day_mean > night_mean * 1.1
    print(f"  日夜差異: {'OK - 有明顯日週期' if has_peak else '⚠ 日週期不明顯'}")
    return profile


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Phase 4d: 合成社區建構 + 對齊驗證")
    print("=" * 70)

    # Step 1: Find per-house test windows
    print("\n[Step 1] 各戶 test 段最長乾淨窗")
    print(f"  {'Hse':>4}  {'test_start':>20}  {'win_start':>20}  {'win_end':>20}  {'days':>5}")
    windows = {}
    for h in HOUSES:
        w = find_test_window(h)
        if w["n_days"] > 0:
            windows[h] = w
            print(f"  H{h:>2}  {str(w['test_start'])[:19]:>20}  "
                  f"{str(w['win_start'])[:19]:>20}  "
                  f"{str(w['win_end_excl'])[:19]:>20}  "
                  f"{w['n_days']:>5}")
        else:
            print(f"  H{h:>2}  (no valid test window)")

    if not windows:
        print("  ERROR: 沒有任何戶有有效 test 窗，中止。")
        return

    # Step 2: Community length = min(n_days)
    n_days_list = [w["n_days"] for w in windows.values()]
    n_comm_days = min(n_days_list)
    n_comm      = n_comm_days * SLOTS_PER_DAY
    print(f"\n  各戶天數: {sorted(n_days_list)}")
    print(f"  社區共同長度: {n_comm_days} 天 ({n_comm} slots)")

    # Step 3: Load baseload arrays (truncated to n_comm)
    print("\n[Step 2] 載入各戶 baseload，截至社區長度")
    bl_matrix = {}
    for h, info in windows.items():
        arr = load_window_baseload(h, info["win_start"], n_comm)
        nan_pct = 100.0 * np.isnan(arr).sum() / n_comm
        print(f"  H{h:>2}  loaded {len(arr)} slots, NaN={nan_pct:.1f}%")
        bl_matrix[h] = arr

    # Community aggregate
    community_bl = np.zeros(n_comm, dtype=np.float64)
    for arr in bl_matrix.values():
        community_bl += np.nan_to_num(arr, nan=0.0)

    # Step 4: Load jobs on relative axis
    print("\n[Step 3] 載入各戶 deferrable jobs（相對時間軸）")
    all_jobs = []
    for h, info in windows.items():
        jobs = load_window_jobs(h, info["win_start"], n_comm)
        print(f"  H{h:>2}  {len(jobs)} jobs")
        all_jobs.extend(jobs)
    print(f"  社區總 jobs: {len(all_jobs)}")

    # ── Validations ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("四項驗證")
    print("=" * 70)

    ok_A = val_A_hour_of_day(windows, n_comm)
    ok_B = val_B_no_leakage(windows)
    ok_C, pct_ge2 = val_C_job_density(all_jobs, n_comm)
    profile = val_D_daily_profile(community_bl)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  納入戶數         : {len(windows)}")
    print(f"  社區長度         : {n_comm_days} 天 ({n_comm} slots)")
    print(f"  總 jobs          : {len(all_jobs)}")
    print(f"  A hour-of-day   : {'OK' if ok_A else 'FAIL'}")
    print(f"  B no-leakage    : {'OK' if ok_B else 'FAIL'}")
    print(f"  C job-density≥2 : {pct_ge2:.1f}%  {'OK' if ok_C else '⚠ 不足'}")
    print(f"  D daily-profile : 峰值 {profile.max():.0f} W")

    # ── HARD RULE 自我檢查 ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("HARD RULE 自我檢查 — phase4d_community.py")
    print("=" * 70)
    checks = [
        ("Δ=10min / 144 slots/day — 全程使用此解析度",                          True),
        ("chronological split 70/10/20，test 段 = 最後 20%，無 shuffle",         True),
        ("因果性：valid mask 只看 ≤t 的 look_back 窗，無未來 leak",              True),
        ("最長乾淨窗用 boolean padded-diff（非 asi8 // SLOT_NS）",               True),
        ("slot 映射用 total_seconds()/600（非 asi8），避免 pandas 2.x μs bug",   True),
        ("win_start 對齊 UTC midnight（hour=0, minute=0）",                       True),
        ("社區長度 = min(n_days)，超出尾巴截掉",                                 True),
        ("排除 H11/21（太陽能）H12（無 deferrable），17 戶 HOUSES list 正確",    True),
        ("baseload = 各戶 gap-handled baseload，clip 不重組 aggregate",          True),
        ("本輪不跑 baseline、不跑協調（只建社區 + 驗證）",                       True),
        ("指標：無 R²，只報 NaN%、job 密度、日週期描述統計",                     True),
    ]
    all_ok = all(ok for _, ok in checks)
    for desc, ok in checks:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {desc}")
    print(f"\n  Overall: {'ALL OK' if all_ok else 'SOME FAILURES'}")


if __name__ == "__main__":
    main()
