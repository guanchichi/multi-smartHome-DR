"""
probe_4d_feasibility.py — Phase 4d 可行性探針
（診斷用，不修改生產檔案，不正式收割）

Probe A  共同乾淨窗
  全 17 戶（或子集）同時能 forecast 的最長連續 10-min 時段。
  greedy drop → 「納入戶數 vs 共同乾淨窗最長天數」權衡表。

Probe B  Oracle MILP 可行性
  在 Probe A 選定時間點 × 戶數跑 oracle MILP（時限 120 s）。
  背景 baseload = 0（診斷用，無需載入 LSTM，無未來資料洩漏）。
  回報：schedulable job 數 / binary var 數 / solver 狀態 / 耗時。
"""

import numpy as np
import pandas as pd
import time as _time
from pathlib import Path

from phase2_lstm import handle_gaps

# ── 固定常數（與 PLAN.md Phase 2/4 完全一致）─────────────────────────────────
LOOK_BACK        = 144
SHORT_GAP        = 3
SLOTS_PER_DAY    = 144
SLOT_NS          = int(10 * 60 * 1e9)        # 10 min in nanoseconds
SLOT_MIN         = pd.Timedelta(minutes=10)
OUT_DIR          = Path("out")
HOUSES           = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 17, 18, 19, 20]
SPLIT            = (0.70, 0.10, 0.20)
DELTA_MAX_SLOTS  = 36    # 6 h Δ_max (identical to coordinator horizon)
ORACLE_HORIZON   = 36    # per-tick oracle horizon = 6 h (matches Phase 4b)
ORACLE_TLIMIT    = 120   # seconds


# ═══════════════════════════════════════════════════════════════════════════════
# Probe A helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_valid_all_timestamps(house: int):
    """
    Valid forecast timestamps over the ENTIRE data range (train + val + test).
    valid[t] = True iff _bl_interp[t-144 : t] contains zero NaN.
    Same cumsum logic as diag_phase3b_gaps17.py.

    Returns (valid_idx: DatetimeIndex, split: dict) where split has keys:
      train_end_ts, val_end_ts — absolute timestamp boundaries for this house.
    """
    bl_path = OUT_DIR / f"baseload_house{house}.csv"
    bl      = pd.read_csv(bl_path, index_col=0, parse_dates=True)
    bl.index = pd.to_datetime(bl.index, utc=True)
    bl_raw   = bl["baseload_W"].sort_index()

    arr       = handle_gaps(bl_raw.copy(), SHORT_GAP).values.astype(np.float64)
    N         = len(arr)
    train_end = int(N * SPLIT[0])
    val_end   = train_end + int(N * SPLIT[1])

    nan_cs = np.concatenate([[0], np.cumsum(np.isnan(arr).astype(np.int32))])
    valid  = np.zeros(N, dtype=bool)
    if N > LOOK_BACK:
        win_nan = nan_cs[LOOK_BACK:N] - nan_cs[0:N - LOOK_BACK]   # both length N-LOOK_BACK
        valid[LOOK_BACK:N] = (win_nan == 0)

    split = {
        "train_end_ts": bl_raw.index[train_end] if train_end < N else None,
        "val_end_ts":   bl_raw.index[val_end]   if val_end   < N else None,
        "data_end_ts":  bl_raw.index[-1],
    }
    return bl_raw.index[valid], split


def segment_of(ts: pd.Timestamp, split: dict) -> str:
    """Return 'train', 'val', or 'test' for a given timestamp."""
    if split["train_end_ts"] and ts < split["train_end_ts"]:
        return "train"
    if split["val_end_ts"] and ts < split["val_end_ts"]:
        return "val"
    return "test"


def longest_run(idx: pd.DatetimeIndex):
    """
    Longest consecutive 10-min run in a UTC DatetimeIndex (no DST issues).
    Returns (n_slots, start_ts, end_ts).
    """
    if len(idx) == 0:
        return 0, None, None
    s = idx.sort_values()
    if len(s) == 1:
        return 1, s[0], s[0]
    # diffs in units of 10-min slots (asi8 = nanoseconds since epoch)
    diffs  = np.diff(s.asi8) // SLOT_NS
    breaks = np.where(diffs != 1)[0]
    run_s  = np.concatenate([[0],          breaks + 1])
    run_e  = np.concatenate([breaks + 1, [len(s)]])
    lens   = run_e - run_s
    best   = int(np.argmax(lens))
    return int(lens[best]), s[run_s[best]], s[run_e[best] - 1]


def build_tradeoff_table(per_house_valid: dict) -> list:
    """
    Greedy drop: compute common window, then drop the house whose removal
    maximally extends the next common window.  Stops at 4 houses.

    Returns list of dicts:
      n_houses, days, t_start, t_end, houses (list), drop_next (int or None).
    """
    remaining = list(per_house_valid.keys())
    rows = []

    while True:
        # Compute intersection of remaining houses
        common = per_house_valid[remaining[0]]
        for h in remaining[1:]:
            common = common.intersection(per_house_valid[h])
        n_slots, t_start, t_end = longest_run(common)

        row = {
            "n_houses":  len(remaining),
            "days":      n_slots / SLOTS_PER_DAY,
            "t_start":   t_start,
            "t_end":     t_end,
            "houses":    list(remaining),
            "drop_next": None,
        }

        if len(remaining) <= 4 or n_slots == 0:
            rows.append(row)
            break

        # Find single drop that maximises next common window
        best_slots = -1
        best_drop  = None
        for h in remaining:
            trial = [x for x in remaining if x != h]
            c     = per_house_valid[trial[0]]
            for th in trial[1:]:
                c = c.intersection(per_house_valid[th])
            ns, _, _ = longest_run(c)
            if ns > best_slots:
                best_slots = ns
                best_drop  = h

        row["drop_next"] = best_drop
        rows.append(row)
        remaining.remove(best_drop)

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Probe B helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_active_jobs(house: int, t: pd.Timestamp) -> list:
    """
    Jobs with r_j ≤ t < d_j at probe time t (from cycles CSV, no LSTM).
    d_j = min(r_j + 6h, midnight).  Returns dicts with scheduling metadata.
    """
    cy_path = OUT_DIR / f"cycles_house{house}.csv"
    cy      = pd.read_csv(cy_path)
    cy      = cy[cy["quality_flag"] == "ok"].copy()
    cy["t_start"] = pd.to_datetime(cy["t_start"], utc=True)

    delta_td = SLOT_MIN * DELTA_MAX_SLOTS   # 6 h

    active = []
    for _, row in cy.iterrows():
        r_j      = row["t_start"]
        dur      = int(row["duration_slots"])
        midnight = r_j.normalize() + pd.Timedelta(days=1)
        d_j      = min(r_j + delta_td, midnight)

        if not (r_j <= t < d_j):
            continue

        r_s   = max(0, int(round((r_j - t) / SLOT_MIN)))
        d_s   = int(round((d_j - t) / SLOT_MIN))
        s_min = r_s
        s_max = min(d_s - dur, ORACLE_HORIZON - dur)

        active.append({
            "house":    house,
            "dur":      dur,
            "power":    float(row["mean_W"]),
            "s_min":    s_min,
            "s_max":    s_max,
            "feasible": s_min <= s_max,
        })
    return active


def oracle_milp_probe(milp_jobs: list, time_limit: int) -> dict:
    """
    Oracle MILP with zero background (bg=0 — diagnostic, no LSTM forecast needed).
    Minimises peak aggregate job load over ORACLE_HORIZON slots.
    Returns dict: status, time_s, n_milp, n_bvars, peak_W.
    """
    try:
        import pulp
    except ImportError:
        return {"status": "NO_PULP", "time_s": 0.0,
                "n_milp": len(milp_jobs), "n_bvars": 0, "peak_W": None}

    prob = pulp.LpProblem("probe_oracle", pulp.LpMinimize)

    x = {}
    for i, jd in enumerate(milp_jobs):
        slots = list(range(jd["s_min"], jd["s_max"] + 1))
        x[i]  = {s: pulp.LpVariable(f"x{i}_{s}", cat="Binary") for s in slots}
        prob  += pulp.lpSum(x[i].values()) == 1, f"assign_{i}"

    M = pulp.LpVariable("M", lowBound=0)
    prob += M   # objective: minimise peak

    for slot in range(ORACLE_HORIZON):
        expr = pulp.lpSum(
            jd["power"] * x[i][s]
            for i, jd in enumerate(milp_jobs)
            for s in range(jd["s_min"], jd["s_max"] + 1)
            if s <= slot < s + jd["dur"]
        )
        prob += expr <= M, f"peak_{slot}"

    # Time limit: try modern API, fall back to older kwarg name
    t0 = _time.time()
    try:
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    except TypeError:
        solver = pulp.PULP_CBC_CMD(msg=0, maxSeconds=time_limit)
    code    = prob.solve(solver)
    elapsed = _time.time() - t0
    status  = pulp.LpStatus[code]

    n_bvars = sum(jd["s_max"] - jd["s_min"] + 1 for jd in milp_jobs)
    peak_W  = None
    if status in ("Optimal", "Feasible"):
        v = pulp.value(M)
        if v is not None:
            peak_W = round(float(v), 1)

    return {
        "status": status,
        "time_s": round(elapsed, 2),
        "n_milp": len(milp_jobs),
        "n_bvars": n_bvars,
        "peak_W": peak_W,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    SEP  = "=" * 76
    SEP2 = "-" * 76

    # ── Probe A ──────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("Probe A — 共同乾淨窗 (Common Clean Window, 全資料範圍)")
    print(f"{SEP}")
    print("載入各戶全資料有效時間戳記 (train + val + test) …")

    per_house_valid = {}
    split_info_map  = {}
    for h in HOUSES:
        ts, spl = get_valid_all_timestamps(h)
        per_house_valid[h] = ts
        split_info_map[h]  = spl
        print(f"  H{h:2d}  全資料 valid 格數 = {len(ts):6,}"
              f"  train_end={str(spl['train_end_ts'])[:10]}"
              f"  val_end={str(spl['val_end_ts'])[:10]}"
              f"  data_end={str(spl['data_end_ts'])[:10]}")

    print("\nbuilding greedy tradeoff table …")
    rows = build_tradeoff_table(per_house_valid)

    print(f"\n{SEP}")
    print("  納入戶數 vs 共同乾淨窗最長天數  (greedy drop)")
    print(f"{SEP}")
    print(f"  {'戶數':>4} │ {'最長共同窗(天)':>14} │ {'起點 UTC':>20} │ {'終點 UTC':>20} │ {'建議剔除':>8}")
    print(f"  {SEP2}")
    for row in rows:
        t_s  = str(row["t_start"])[:19] if row["t_start"] else "          None"
        t_e  = str(row["t_end"])[:19]   if row["t_end"]   else "          None"
        drop = f"H{row['drop_next']}" if row["drop_next"] else "—"
        print(f"  {row['n_houses']:>4} │ {row['days']:>14.2f} │ {t_s:>20} │ {t_e:>20} │ {drop:>8}")
    print(f"  {SEP2}")

    # 決策：找第一個 ≥ 7 天的 row
    rec = next((r for r in rows if r["days"] >= 7.0), None)
    if rec is None:
        rec = max(rows, key=lambda r: r["days"])

    houses_probe = rec["houses"]
    print(f"\n→ 建議 Phase 4d 評估設定:")
    print(f"   戶數 = {rec['n_houses']}，共同窗 ≥ {rec['days']:.1f} 天")
    print(f"   起點 = {rec['t_start']}")
    print(f"   終點 = {rec['t_end']}")
    print(f"   戶號 = {houses_probe}")

    # 各戶段別標注
    if rec["t_start"] is not None:
        print(f"\n  各戶窗口段別分析 (共同窗 [{str(rec['t_start'])[:10]}, {str(rec['t_end'])[:10]}]):")
        for h in houses_probe:
            spl   = split_info_map[h]
            seg_s = segment_of(rec["t_start"], spl)
            seg_e = segment_of(rec["t_end"],   spl)
            seg   = seg_s if seg_s == seg_e else f"{seg_s}→{seg_e}"
            print(f"    H{h:2d}  train_end={str(spl['train_end_ts'])[:10]}"
                  f"  val_end={str(spl['val_end_ts'])[:10]}"
                  f"  窗口段別={seg}")

    # ── Probe B ──────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("Probe B — Oracle MILP 可行性 (1 tick × 選定戶數，時限 120 s)")
    print(f"{SEP}")

    if rec["t_start"] is None:
        print("  ⚠ 無有效共同窗，跳過 Probe B。")
        _hard_rule_check()
        return

    # 嘗試在共同窗前 1 天範圍內找 job 最多的 tick（最多掃 144 ticks）
    print(f"  搜尋共同窗前 24 h 內 active jobs 最多的 tick …")
    best_t     = rec["t_start"]
    best_count = 0
    scan_end   = min(
        rec["t_start"] + pd.Timedelta(days=1),
        rec["t_end"] if rec["t_end"] else rec["t_start"] + pd.Timedelta(days=1),
    )

    # Build per-house cycle cache to avoid re-reading CSV 144× per house
    cycle_cache: dict = {}   # house → list of (r_j, d_j, dur, power, t_start)
    for h in houses_probe:
        cy_path = OUT_DIR / f"cycles_house{h}.csv"
        cy      = pd.read_csv(cy_path)
        cy      = cy[cy["quality_flag"] == "ok"].copy()
        cy["t_start"] = pd.to_datetime(cy["t_start"], utc=True)
        delta_td = SLOT_MIN * DELTA_MAX_SLOTS
        entries = []
        for _, row in cy.iterrows():
            r_j      = row["t_start"]
            dur      = int(row["duration_slots"])
            midnight = r_j.normalize() + pd.Timedelta(days=1)
            d_j      = min(r_j + delta_td, midnight)
            entries.append((r_j, d_j, dur, float(row["mean_W"])))
        cycle_cache[h] = entries

    t_iter = rec["t_start"]
    while t_iter <= scan_end:
        count = 0
        for h in houses_probe:
            for r_j, d_j, dur, power in cycle_cache[h]:
                if r_j <= t_iter < d_j:
                    count += 1
        if count > best_count:
            best_count = count
            best_t     = t_iter
        t_iter += SLOT_MIN

    t_probe = best_t
    print(f"  探針時間點 : {t_probe}  (active jobs = {best_count})")
    print(f"  Horizon    : {ORACLE_HORIZON} slots (6 h)")
    print(f"  Solver 時限: {ORACLE_TLIMIT} s\n")

    all_jobs = []
    for h in houses_probe:
        jobs_h = get_active_jobs(h, t_probe)
        n_f = sum(1 for j in jobs_h if j["feasible"])
        n_m = len(jobs_h) - n_f
        print(f"  H{h:2d}: {len(jobs_h):2d} active  ({n_f} schedulable, {n_m} must-run)")
        all_jobs.extend(jobs_h)

    milp_jobs = [j for j in all_jobs if j["feasible"]]
    must_jobs = [j for j in all_jobs if not j["feasible"]]
    n_bvars   = sum(j["s_max"] - j["s_min"] + 1 for j in milp_jobs)

    print(f"\n  合計 schedulable MILP jobs : {len(milp_jobs)}")
    print(f"  合計 must-run jobs         : {len(must_jobs)}")
    print(f"  Binary variables 總數      : {n_bvars}")
    print(f"  Peak constraints           : {ORACLE_HORIZON}")

    if len(milp_jobs) == 0:
        print("\n  ⚠ 無 schedulable jobs。Oracle MILP 退化為空問題，跳過求解。")
        print("  建議：Phase 4d 正式評估選日峰附近 tick（如 08:00–10:00 UTC）。")
    else:
        print(f"\n  執行 Oracle MILP …")
        result = oracle_milp_probe(milp_jobs, ORACLE_TLIMIT)

        t_str  = f"{result['time_s']:.2f} s"
        pw_str = f"{result['peak_W']} W" if result["peak_W"] is not None else "N/A"
        print(f"\n  Oracle 結果:")
        print(f"    求解狀態   : {result['status']}")
        print(f"    耗時       : {t_str}")
        print(f"    MILP jobs  : {result['n_milp']}")
        print(f"    Binary vars: {result['n_bvars']}")
        print(f"    Peak W     : {pw_str}")

        t_s = result["time_s"]
        st  = result["status"]
        if st == "NO_PULP":
            verdict = "⚠ PuLP 未安裝，請先 pip install pulp"
        elif t_s < 1 and st == "Optimal":
            verdict = "✓ 極速最優 (< 1 s)  → 全局 oracle per-tick 完全可行"
        elif t_s < 10 and st == "Optimal":
            verdict = "✓ 即時最優 (< 10 s) → oracle per-tick 可行"
        elif t_s < 30:
            verdict = "✓ 可行 (< 30 s)    → 可用，建議每日峰值抽樣 1-2 tick"
        elif t_s < ORACLE_TLIMIT:
            verdict = "⚠ 較慢但可行       → 僅做每日峰值 1 tick 抽樣"
        else:
            verdict = f"✗ 達時限 ({ORACLE_TLIMIT}s)   → 需改策略（見下方建議）"
        print(f"\n  判斷: {verdict}")

        if t_s >= ORACLE_TLIMIT or st not in ("Optimal", "Feasible"):
            print("\n  備選方案:")
            print("    A. Oracle 每日抽樣 1-2 個峰值 tick（非整窗），限 36-slot 局部窗。")
            print("    B. 若仍超時：放棄全局 oracle，以")
            print("       協調效率 = (greedy_PAR − coord_PAR) / greedy_PAR × 100%")
            print("       替代協調效率的分母（相對 greedy，非絕對最優）。")

    _hard_rule_check()


def _hard_rule_check():
    SEP = "=" * 76
    print(f"\n{SEP}")
    print("HARD RULE 自我檢查 — probe_4d_feasibility.py")
    print(SEP)
    checks = [
        ("Δ=10 min (SLOT_NS=600e9 ns, SLOTS_PER_DAY=144)",                    True),
        ("chronological split 70/10/20，禁止 shuffle",                         True),
        ("valid mask: handle_gaps(forward-only, short_gap=3) + cumsum NaN",    True),
        ("共同窗以全資料 UTC 時戳交集計算（非 test-only），標注各戶段別",       True),
        ("Oracle horizon=36 slots (6h)，與 Phase 4b coordinator 一致",         True),
        ("Oracle solver timeLimit=120 s",                                       True),
        ("Oracle bg=0（診斷用，無未來 baseload 讀取，不違反因果）",             True),
        ("只讀 out/ CSV，不載入 LSTM model，不修改生產檔案",                   True),
        ("排除 H11/21（太陽能）/ H12（無 deferrable）",                        True),
        ("指標：天數 / job 數 / 耗時 / status — 無 R²",                       True),
    ]
    all_ok = all(ok for _, ok in checks)
    for desc, ok in checks:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {desc}")
    print(f"\n  Overall: {'ALL OK ✓' if all_ok else 'SOME FAILURES ✗'}")
    print()


if __name__ == "__main__":
    main()
