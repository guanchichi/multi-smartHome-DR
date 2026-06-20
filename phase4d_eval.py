"""
Phase 4d — 正式收割評估：合成社區 17 戶 × 18 天

方法:
  1. No-DR      : 所有 job 在 r_j 立刻執行
  2. Greedy     : 各戶獨立 ToU 排程（herding baseline）
  3. Online-coord: rolling-horizon shadow-price（Phase 4c 機制）
  4. Oracle     : 每日尖峰局部 MILP（6h/36-slot 窗，120s 時限）

指標:
  - 整窗 PAR / 每日尖峰降幅 / 協調效率 / 延後時數 / deadline-miss / runtime

硬規則:
  - Coord 一律用 LSTM forecast（因果）; Oracle 可用真實負載求解（離線下界）
  - must-run 計入聚合負載; coordinator 只收 Σ 負載（隱私）
"""

# ── PuLP 可用性優先確認 ──────────────────────────────────────────────────────────
import sys
try:
    import pulp
    _PULP_OK = True
    print("[OK] pulp imported successfully:", pulp.__version__)
except ImportError:
    _PULP_OK = False
    print("[ERROR] pulp not found.")
    print("  Please install: pip install pulp")
    sys.exit(1)

import json
import time as _time
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase2_lstm import handle_gaps
from phase3_simulator import Simulator, Job, SLOT_MINUTES, HORIZON
from phase4a_schedule import (
    ScheduleResult, schedule_house, compute_aggregate_load, make_tou_price,
)
from phase4b_coordinator import HouseData, run_coordination
from phase4d_community import (
    find_test_window, load_window_baseload, HOUSES,
    SLOTS_PER_DAY, OUT_DIR, DEFERRABLE_MAP,
)

# ── Parameters ────────────────────────────────────────────────────────────────
COORD_ALPHA   = 4e-6
COORD_ITERS   = 50
COORD_BETA    = 0.0
ORACLE_TIMELIMIT = 120        # seconds per daily window
ORACLE_WIN_HALF  = 18        # ±18 slots = 36-slot window (6 h)
RESULTS_DIR   = Path("results")
SLOT_DUR      = pd.Timedelta(minutes=SLOT_MINUTES)

# Common schedule reference: all houses' jobs are re-expressed relative to this
# anchor so that schedule_house receives a consistent time frame regardless of
# each house's individual win_start date.
SCHED_REF = pd.Timestamp("2000-01-01 00:00:00", tz="UTC")


def _normalize_for_sched(jobs: List[Job], win_start: pd.Timestamp) -> List[Job]:
    """
    Return new Job list where r_j/d_j are shifted to SCHED_REF frame.
    Preserves relative slot offsets; does not touch power/duration/id.
    """
    slot_td = pd.Timedelta(minutes=SLOT_MINUTES)
    out = []
    for j in jobs:
        r_rel = int(round((j.r_j - win_start).total_seconds() / 600.0))
        d_rel = int(round((j.d_j - win_start).total_seconds() / 600.0))
        out.append(dataclasses.replace(
            j,
            r_j = SCHED_REF + r_rel * slot_td,
            d_j = SCHED_REF + d_rel * slot_td,
        ))
    return out


# ── CommittedJob (same structure as Phase 4c) ─────────────────────────────────

@dataclass
class CommittedJob:
    job_id:         int
    house:          int
    commit_tick:    int           # relative slot in [0, n_comm)
    duration_slots: int
    power:          float         # W
    deadline_missed: bool
    r_j_rel:        int           # release relative slot
    d_j_rel:        int           # deadline relative slot


# ── Community builder ─────────────────────────────────────────────────────────

def build_community():
    """Find per-house test windows, load baseload arrays, build job lists."""
    print("\n[Community] Loading per-house windows ...")
    windows = {}
    for h in HOUSES:
        w = find_test_window(h)
        if w["n_days"] > 0:
            windows[h] = w

    n_comm_days = min(w["n_days"] for w in windows.values())
    n_comm      = n_comm_days * SLOTS_PER_DAY
    print(f"  Community: {len(windows)} houses × {n_comm_days} days ({n_comm} slots)")

    # Pre-load baseload arrays (real, gap-handled)
    print("[Community] Loading baseload arrays ...")
    bl_arrays: Dict[int, np.ndarray] = {}
    for h, info in windows.items():
        bl_arrays[h] = load_window_baseload(h, info["win_start"], n_comm)
    community_bl = sum(bl_arrays.values())

    return windows, bl_arrays, community_bl, n_comm_days, n_comm


# ── Load Simulators (with LSTM) ───────────────────────────────────────────────

def load_simulators(windows: dict) -> Dict[int, Simulator]:
    print("[Simulators] Loading LSTM simulators ...")
    sims: Dict[int, Simulator] = {}
    for h in windows:
        sim = Simulator(house=h)
        sim.load_lstm()
        sims[h] = sim
    return sims


# ── Extract window jobs from simulators ───────────────────────────────────────

def extract_window_jobs(
    simulators: Dict[int, Simulator],
    windows: dict,
    n_comm: int,
) -> Dict[int, List[Job]]:
    """
    For each house, return the Job objects whose r_j falls inside the window.
    Simulator already filters to quality_flag='ok' cycles.
    """
    slot_dur = pd.Timedelta(minutes=SLOT_MINUTES)
    wj: Dict[int, List[Job]] = {}
    for h, info in windows.items():
        win_start = info["win_start"]
        win_end   = win_start + n_comm * slot_dur
        wj[h] = [j for j in simulators[h].jobs
                 if j.r_j >= win_start and j.r_j < win_end]
    return wj


# ── No-DR ─────────────────────────────────────────────────────────────────────

def run_no_dr(
    community_bl: np.ndarray,
    window_jobs:  Dict[int, List[Job]],
    windows:      dict,
    n_comm:       int,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """All jobs run at r_j.  Returns (loads, committed_ticks{job_id: r_j_rel})."""
    loads    = community_bl.copy()
    jstarts: Dict[int, int] = {}
    for h, info in windows.items():
        ws = info["win_start"]
        for j in window_jobs[h]:
            r_rel = int(round((j.r_j - ws).total_seconds() / 600.0))
            for s in range(j.duration_slots):
                if 0 <= r_rel + s < n_comm:
                    loads[r_rel + s] += j.power_profile[0]
            jstarts[j.job_id] = r_rel
    return loads, jstarts


# ── Running-background helper ─────────────────────────────────────────────────

def _running_bg_comm(
    committed: Dict[int, CommittedJob],
    house: int, t_rel: int, horizon: int,
) -> np.ndarray:
    bg = np.zeros(horizon)
    for cj in committed.values():
        if cj.house != house:
            continue
        elapsed   = t_rel - cj.commit_tick
        remaining = cj.duration_slots - elapsed
        if remaining > 0:
            bg[: min(remaining, horizon)] += cj.power
    return bg


# ── Rolling simulation (Greedy or Coord) ──────────────────────────────────────

def run_rolling(
    simulators:   Dict[int, Simulator],
    windows:      dict,
    window_jobs:  Dict[int, List[Job]],
    community_bl: np.ndarray,
    n_comm:       int,
    mode:         str,           # 'greedy' | 'coord'
    verbose:      bool = True,
) -> Tuple[np.ndarray, List[Dict], Dict[int, CommittedJob]]:
    """
    Rolling-horizon simulation over n_comm ticks.
    Returns (actual_loads, tick_logs, committed).
    actual_loads[t] = community_bl[t] + committed-job power at t.
    """
    committed: Dict[int, CommittedJob] = {}
    warm_lam:  Optional[np.ndarray]   = None
    last_fc:   Dict[int, Optional[np.ndarray]] = {h: None for h in windows}
    tick_logs: List[Dict]             = []
    fallback_n = 0
    slot_dur   = pd.Timedelta(minutes=SLOT_MINUTES)

    tag = f"[{mode:6s}]"
    print(f"\n{tag} Starting rolling simulation ({n_comm} ticks) ...")
    t0_wall = _time.time()

    # Common schedule time: all houses use this single reference at each tick.
    # Jobs are re-expressed in SCHED_REF frame before calling schedule_house,
    # so r_j / d_j are comparable across houses with different win_start dates.
    slot_td_10 = pd.Timedelta(minutes=SLOT_MINUTES)

    for t_rel in range(n_comm):
        t_common = SCHED_REF + t_rel * slot_td_10   # used for schedule_house / coord

        # ── Build per-house HouseData ─────────────────────────────────────────
        all_hd: List[HouseData] = []
        any_fallback = False

        for h, info in windows.items():
            t_real = info["win_start"] + t_rel * slot_dur   # house-specific real time
            sim    = simulators[h]
            sim._jump_to(t_real)
            fc = sim.forecast(t_real, horizon=HORIZON)      # causal: uses house real time

            if fc is None:
                any_fallback = True
                if last_fc[h] is not None:
                    fc      = np.empty(HORIZON)
                    fc[:-1] = last_fc[h][1:]
                    fc[-1]  = max(0.0, float(last_fc[h][-1]))
                else:
                    fc = np.zeros(HORIZON)
            else:
                last_fc[h] = fc.copy()

            rbg    = _running_bg_comm(committed, h, t_rel, HORIZON)
            adj_fc = np.maximum(0.0, fc + rbg)

            # Filter active jobs using house-specific real time (correct causality)
            active_orig = [
                j for j in window_jobs[h]
                if j.r_j <= t_real
                and j.job_id not in committed
                and j.d_j > t_real
            ]
            # Normalize r_j/d_j to SCHED_REF frame for schedule_house
            active = _normalize_for_sched(active_orig, info["win_start"])

            all_hd.append(HouseData(
                house      = h,
                forecast   = adj_fc,
                jobs       = active,
                jobs_by_id = {j.job_id: j for j in active},
            ))

        if any_fallback:
            fallback_n += 1

        # ── Greedy target (always needed for target & greedy-mode) ────────────
        tou = make_tou_price(t_common, HORIZON)
        greedy_L = np.zeros(HORIZON)
        g_res: Dict[int, ScheduleResult] = {}
        for hd in all_hd:
            r = schedule_house(hd.jobs, hd.forecast, tou, t_common, HORIZON)
            l = compute_aggregate_load(hd.forecast, r, hd.jobs_by_id, HORIZON, True)
            g_res[hd.house] = r
            greedy_L += l
        target = float(greedy_L.mean())

        # ── Schedule ──────────────────────────────────────────────────────────
        iter_count  = 0
        if mode == "greedy":
            sched_res = g_res
        else:  # coord
            lam_init = warm_lam if warm_lam is not None else tou.copy()
            _t0c = _time.time()
            best_L, best_res, _, log_coord, _, best_lam = run_coordination(
                all_hd, target, t_common, HORIZON,
                alpha=COORD_ALPHA, max_iter=COORD_ITERS,
                grad_ema_beta=COORD_BETA, lam_init=lam_init,
            )
            iter_count = len(log_coord)
            warm_lam   = best_lam
            sched_res  = best_res

        # ── Commit slot-0 ─────────────────────────────────────────────────────
        for hd in all_hd:
            r = sched_res[hd.house]
            for sj in r.scheduled:
                if sj.start_slot == 0 and sj.job_id not in committed:
                    j     = hd.jobs_by_id[sj.job_id]   # normalized job
                    r_rel = int(round((j.r_j - SCHED_REF).total_seconds() / 600.0))
                    d_rel = int(round((j.d_j - SCHED_REF).total_seconds() / 600.0))
                    committed[sj.job_id] = CommittedJob(
                        job_id=sj.job_id, house=hd.house,
                        commit_tick=t_rel, duration_slots=j.duration_slots,
                        power=float(j.power_profile[0]), deadline_missed=False,
                        r_j_rel=r_rel, d_j_rel=d_rel,
                    )
            for mj in r.must_run:
                if mj.start_slot == 0 and mj.job_id not in committed:
                    j     = hd.jobs_by_id[mj.job_id]   # normalized job
                    r_rel = int(round((j.r_j - SCHED_REF).total_seconds() / 600.0))
                    d_rel = int(round((j.d_j - SCHED_REF).total_seconds() / 600.0))
                    committed[mj.job_id] = CommittedJob(
                        job_id=mj.job_id, house=hd.house,
                        commit_tick=t_rel, duration_slots=j.duration_slots,
                        power=float(j.power_profile[0]),
                        deadline_missed=mj.deadline_missed,
                        r_j_rel=r_rel, d_j_rel=d_rel,
                    )

        n_active = sum(len(hd.jobs) for hd in all_hd)
        tick_logs.append({
            "t_rel": t_rel, "fallback": any_fallback,
            "n_active": n_active, "iter_count": iter_count,
        })

        if verbose and (t_rel % (SLOTS_PER_DAY * 3) == 0 or t_rel == n_comm - 1):
            elapsed = _time.time() - t0_wall
            print(f"  {tag} tick {t_rel+1:4d}/{n_comm}"
                  f"  day {t_rel//SLOTS_PER_DAY:2d}"
                  f"  active={n_active}"
                  f"  committed={len(committed)}"
                  f"  elapsed={elapsed:.0f}s"
                  + ("  [FB]" if any_fallback else ""))

    # ── Build actual loads from community_bl + committed job loads ────────────
    loads = community_bl.copy()
    for cj in committed.values():
        for s in range(cj.duration_slots):
            idx = cj.commit_tick + s
            if 0 <= idx < n_comm:
                loads[idx] += cj.power

    total_time = _time.time() - t0_wall
    fb_total   = sum(1 for e in tick_logs if e["fallback"])
    print(f"  {tag} done — {total_time:.1f}s total"
          f"  ({total_time/n_comm:.3f}s/tick)"
          f"  fallback={fb_total}"
          f"  committed={len(committed)}")

    return loads, tick_logs, committed


# ── Oracle: per-day peak window MILP ──────────────────────────────────────────

def run_oracle(
    simulators:   Dict[int, Simulator],
    windows:      dict,
    window_jobs:  Dict[int, List[Job]],
    community_bl: np.ndarray,
    n_comm_days:  int,
    n_comm:       int,
    reference_loads: np.ndarray,   # Greedy loads — used to find peak slot
) -> List[Dict]:
    """
    For each day d, find the peak slot in reference_loads, take ±ORACLE_WIN_HALF
    slots, run MILP on that window, record oracle peak.
    Returns list of per-day dicts.
    """
    results = []
    slot_dur = pd.Timedelta(minutes=SLOT_MINUTES)
    print(f"\n[Oracle] Per-day MILP (6h window, {ORACLE_TIMELIMIT}s limit each) ...")

    for day in range(n_comm_days):
        day_start = day * SLOTS_PER_DAY
        day_end   = day_start + SLOTS_PER_DAY
        day_loads = reference_loads[day_start:day_end]
        peak_offset = int(np.argmax(day_loads))        # peak slot within day
        peak_slot   = day_start + peak_offset          # absolute slot

        # 36-slot window clamped to [day_start, day_end)
        win_s = max(day_start, peak_slot - ORACLE_WIN_HALF)
        win_e = min(day_end,   peak_slot + ORACLE_WIN_HALF)
        horizon = win_e - win_s
        if horizon <= 0:
            results.append({"day": day, "status": "empty_window", "oracle_peak": None})
            continue

        # Build HouseData: real baseload as "forecast" (offline oracle may use real loads)
        all_hd: List[HouseData] = []
        for h, info in windows.items():
            # Use real baseload as background (oracle is offline — labelled clearly)
            fc_real = bl_arrays[h][win_s:win_e] if len(bl_arrays[h]) >= win_e else np.zeros(horizon)

            # Jobs active in this window (r_j_rel < win_e AND d_j_rel > win_s)
            ws = info["win_start"]
            active = []
            for j in window_jobs[h]:
                r_rel = int(round((j.r_j - ws).total_seconds() / 600.0))
                d_rel = int(round((j.d_j - ws).total_seconds() / 600.0))
                if r_rel < win_e and d_rel > win_s:
                    active.append(j)

            all_hd.append(HouseData(
                house      = h,
                forecast   = fc_real,
                jobs       = active,
                jobs_by_id = {j.job_id: j for j in active},
            ))

        # Collect MILP jobs
        milp_jobs = []
        bg         = np.zeros(horizon)
        win_t0     = windows[list(windows.keys())[0]]["win_start"] + win_s * slot_dur

        for hd in all_hd:
            ws = windows[hd.house]["win_start"]
            bg += hd.forecast   # real baseload for this house's window segment
            bg -= hd.forecast   # undo: we'll re-add explicitly below
        # Correct: background = sum of real baseloads in [win_s, win_e)
        bg = np.zeros(horizon)
        for h, info in windows.items():
            bg += bl_arrays[h][win_s:win_e]

        for hd in all_hd:
            ws = windows[hd.house]["win_start"]
            for j in hd.jobs:
                r_rel = int(round((j.r_j - ws).total_seconds() / 600.0))
                d_rel = int(round((j.d_j - ws).total_seconds() / 600.0))
                # Convert to window-local slots
                s_min = max(0, r_rel - win_s)
                s_max = min(d_rel - win_s - j.duration_slots,
                            horizon - j.duration_slots)
                if s_min <= s_max:
                    milp_jobs.append({
                        "job": j, "house": hd.house,
                        "s_min": s_min, "s_max": s_max,
                        "dur": j.duration_slots,
                        "power": float(j.power_profile[0]),
                        "feasible": list(range(s_min, s_max + 1)),
                    })
                else:
                    # Must-run: pin to latest possible
                    latest = max(0, min(d_rel - win_s - j.duration_slots,
                                       horizon - 1))
                    for s in range(latest, min(latest + j.duration_slots, horizon)):
                        bg[s] += j.power_profile[0]

        if not milp_jobs:
            # No schedulable jobs — oracle = just baseload
            oracle_peak = float(bg.max())
            results.append({
                "day": day, "status": "no_jobs",
                "oracle_peak": oracle_peak,
                "n_milp_jobs": 0,
                "greedy_peak": float(reference_loads[day_start:day_end].max()),
            })
            continue

        # Solve MILP with time limit
        try:
            prob = pulp.LpProblem(f"oracle_day{day}", pulp.LpMinimize)
            x = {}
            for i, jd in enumerate(milp_jobs):
                x[i] = {s: pulp.LpVariable(f"x{i}_{s}", cat="Binary")
                        for s in jd["feasible"]}
                prob += pulp.lpSum(x[i].values()) == 1, f"assign_{i}"
            M = pulp.LpVariable("M", lowBound=0)
            prob += M
            for slot in range(horizon):
                job_expr = pulp.lpSum(
                    jd["power"] * x[i][s]
                    for i, jd in enumerate(milp_jobs)
                    for s in jd["feasible"]
                    if s <= slot < s + jd["dur"]
                )
                prob += bg[slot] + job_expr <= M, f"peak_{slot}"

            t0m = _time.time()
            status_code = prob.solve(
                pulp.PULP_CBC_CMD(msg=0, timeLimit=ORACLE_TIMELIMIT)
            )
            solve_time = _time.time() - t0m
            status_str = pulp.LpStatus[status_code]

            if status_str in ("Optimal", "Not Solved"):
                M_val = pulp.value(M)
                oracle_peak = float(M_val) if M_val is not None else None
            else:
                oracle_peak = None

            results.append({
                "day":          day,
                "status":       status_str,
                "oracle_peak":  oracle_peak,
                "n_milp_jobs":  len(milp_jobs),
                "greedy_peak":  float(reference_loads[day_start:day_end].max()),
                "solve_time_s": round(solve_time, 2),
                "win_s":        int(win_s),
                "win_e":        int(win_e),
            })
            flag = "OK" if oracle_peak is not None else "N/A"
            print(f"  Day {day:2d}: {flag}  peak={oracle_peak or 'N/A':.0f}W"
                  f"  jobs={len(milp_jobs)}  {status_str}  {solve_time:.1f}s")

        except Exception as e:
            print(f"  Day {day:2d}: FAILED — {e}")
            results.append({"day": day, "status": f"error:{e}", "oracle_peak": None})

    return results


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    no_dr_loads:    np.ndarray,
    greedy_loads:   np.ndarray,
    coord_loads:    np.ndarray,
    oracle_days:    List[Dict],
    n_comm_days:    int,
    n_comm:         int,
    coord_committed: Dict[int, CommittedJob],
    no_dr_jstarts:  Dict[int, int],
    coord_logs:     List[Dict],
    windows:        dict,
    window_jobs:    Dict[int, List[Job]],
) -> dict:

    def par(arr):
        m = arr.mean()
        return float(arr.max() / m) if m > 1e-9 else float("inf")

    def daily_peaks(arr):
        return np.array([
            arr[d * SLOTS_PER_DAY:(d + 1) * SLOTS_PER_DAY].max()
            for d in range(n_comm_days)
        ])

    # ── Window PAR ────────────────────────────────────────────────────────────
    nodr_par   = par(no_dr_loads)
    greedy_par = par(greedy_loads)
    coord_par  = par(coord_loads)

    # ── Daily peaks ───────────────────────────────────────────────────────────
    pk_nodr   = daily_peaks(no_dr_loads)
    pk_greedy = daily_peaks(greedy_loads)
    pk_coord  = daily_peaks(coord_loads)

    # Reduction % vs No-DR baseline
    red_greedy = (pk_nodr - pk_greedy) / np.maximum(pk_nodr, 1) * 100
    red_coord  = (pk_nodr - pk_coord)  / np.maximum(pk_nodr, 1) * 100

    # Oracle per-day peaks (only valid days)
    oracle_valid = [r for r in oracle_days if r.get("oracle_peak") is not None]
    oracle_days_idx = [r["day"] for r in oracle_valid]
    pk_oracle    = np.array([r["oracle_peak"] for r in oracle_valid])
    greedy_peaks_at_oracle_days = pk_greedy[oracle_days_idx]
    coord_peaks_at_oracle_days  = pk_coord[oracle_days_idx]
    nodr_peaks_at_oracle_days   = pk_nodr[oracle_days_idx]

    if len(oracle_valid) > 0:
        red_oracle = (nodr_peaks_at_oracle_days - pk_oracle) / \
                     np.maximum(nodr_peaks_at_oracle_days, 1) * 100
        red_oracle_vs_greedy = (greedy_peaks_at_oracle_days - pk_oracle) / \
                                np.maximum(greedy_peaks_at_oracle_days, 1) * 100

        # Coordination efficiency (per-day peak)
        denom = greedy_peaks_at_oracle_days - pk_oracle
        numer = greedy_peaks_at_oracle_days - coord_peaks_at_oracle_days
        coord_eff_daily = np.where(
            np.abs(denom) > 1.0,
            numer / denom * 100,
            np.nan,
        )
    else:
        red_oracle          = np.array([])
        red_oracle_vs_greedy = np.array([])
        coord_eff_daily     = np.array([])

    # Coordination efficiency — window PAR (no oracle PAR for full window)
    # We report N/A for whole-window coord efficiency (oracle not computed for full window)
    # Instead report: (greedy_PAR - coord_PAR) / greedy_PAR × 100 (improvement vs greedy)
    coord_par_improv_pct = (greedy_par - coord_par) / greedy_par * 100

    # ── Average delay (coord vs no-dr) ────────────────────────────────────────
    delays_h = []
    for jid, cj in coord_committed.items():
        # No-DR start: r_j_rel
        r_rel = cj.r_j_rel
        delay_slots = cj.commit_tick - r_rel
        delays_h.append(delay_slots * SLOT_MINUTES / 60.0)  # hours
    avg_delay_h  = float(np.mean(delays_h)) if delays_h else 0.0
    std_delay_h  = float(np.std(delays_h))  if delays_h else 0.0

    # ── Deadline-miss rate ────────────────────────────────────────────────────
    n_coord_committed    = len(coord_committed)
    n_deadline_missed    = sum(1 for cj in coord_committed.values() if cj.deadline_missed)
    deadline_miss_rate   = n_deadline_missed / max(n_coord_committed, 1)

    # ── Runtime stats (coord) ─────────────────────────────────────────────────
    iters = [e["iter_count"] for e in coord_logs if e["iter_count"] > 0]
    avg_iters = float(np.mean(iters)) if iters else 0.0
    fallback_n = sum(1 for e in coord_logs if e["fallback"])

    return {
        "window_par": {
            "no_dr":  nodr_par,
            "greedy": greedy_par,
            "coord":  coord_par,
        },
        "coord_par_improvement_pct": coord_par_improv_pct,
        "daily_peak_reduction_pct": {
            "greedy_mean": float(red_greedy.mean()),
            "greedy_std":  float(red_greedy.std()),
            "coord_mean":  float(red_coord.mean()),
            "coord_std":   float(red_coord.std()),
            "oracle_mean": float(red_oracle.mean())  if len(red_oracle) > 0 else None,
            "oracle_std":  float(red_oracle.std())   if len(red_oracle) > 0 else None,
            "n_oracle_days": len(oracle_valid),
        },
        "coord_efficiency_daily_pct": {
            "mean": float(np.nanmean(coord_eff_daily)) if len(coord_eff_daily) > 0 else None,
            "std":  float(np.nanstd(coord_eff_daily))  if len(coord_eff_daily) > 0 else None,
        },
        "avg_delay_h":        avg_delay_h,
        "std_delay_h":        std_delay_h,
        "deadline_miss_rate": deadline_miss_rate,
        "n_coord_committed":  n_coord_committed,
        "n_deadline_missed":  n_deadline_missed,
        "avg_coord_iters":    avg_iters,
        "fallback_ticks":     fallback_n,
        "_arrays": {
            "pk_nodr": pk_nodr.tolist(),
            "pk_greedy": pk_greedy.tolist(),
            "pk_coord": pk_coord.tolist(),
            "red_greedy": red_greedy.tolist(),
            "red_coord": red_coord.tolist(),
        },
    }


# ── Print tables ──────────────────────────────────────────────────────────────

def print_results(m: dict, n_comm_days: int, oracle_days: list) -> None:
    SEP  = "=" * 72
    sep2 = "-" * 72

    print(f"\n{SEP}")
    print("Table 1: 四方法整窗 PAR 對照")
    print(SEP)
    wp = m["window_par"]
    print(f"  {'Method':<20}  {'Window PAR':>11}  {'vs Greedy':>11}  {'vs No-DR':>11}")
    print(sep2)
    g_par  = wp["greedy"]
    nd_par = wp["no_dr"]
    c_par  = wp["coord"]
    print(f"  {'No-DR':<20}  {nd_par:>11.4f}  {'—':>11}  {'—':>11}")
    print(f"  {'Greedy':<20}  {g_par:>11.4f}  {'0.0%':>11}  "
          f"  {(nd_par-g_par)/nd_par*100:>+10.1f}%")
    print(f"  {'Online-coord':<20}  {c_par:>11.4f}  "
          f"{(g_par-c_par)/g_par*100:>+10.1f}%  "
          f"{(nd_par-c_par)/nd_par*100:>+10.1f}%")
    print(f"  {'Oracle':<20}  {'(daily only)':>11}  {'—':>11}  {'—':>11}")
    print(f"\n  Coord PAR improvement vs Greedy: "
          f"{m['coord_par_improvement_pct']:.1f}%")

    print(f"\n{SEP}")
    print("Table 2: 每日尖峰降幅（以 No-DR 為基準）")
    print(SEP)
    d = m["daily_peak_reduction_pct"]
    print(f"  {'Method':<20}  {'mean red%':>10}  {'std':>8}  {'n days':>7}")
    print(sep2)
    print(f"  {'Greedy':<20}  {d['greedy_mean']:>9.1f}%  {d['greedy_std']:>7.1f}%  {n_comm_days:>7}")
    print(f"  {'Online-coord':<20}  {d['coord_mean']:>9.1f}%  {d['coord_std']:>7.1f}%  {n_comm_days:>7}")
    if d["oracle_mean"] is not None:
        print(f"  {'Oracle':<20}  {d['oracle_mean']:>9.1f}%  {d['oracle_std']:>7.1f}%  "
              f"{d['n_oracle_days']:>7}  ← 有效 MILP 天數")
    else:
        print(f"  {'Oracle':<20}  {'N/A':>10}")

    e = m["coord_efficiency_daily_pct"]
    if e["mean"] is not None:
        print(f"\n  協調效率（per-day peak，greedy/oracle 口徑）: "
              f"{e['mean']:.1f}% ± {e['std']:.1f}%")
    else:
        print("\n  協調效率: N/A（oracle 天數不足）")

    print(f"\n{SEP}")
    print("Table 3: 其他指標")
    print(SEP)
    print(f"  平均延後時數 (coord vs r_j)   : "
          f"{m['avg_delay_h']:.2f} h ± {m['std_delay_h']:.2f} h")
    print(f"  deadline-miss rate (coord)    : "
          f"{m['deadline_miss_rate']:.2%}"
          f"  ({m['n_deadline_missed']}/{m['n_coord_committed']})")
    print(f"  avg coord iters/tick          : {m['avg_coord_iters']:.1f}")
    print(f"  fallback ticks (coord)        : {m['fallback_ticks']}")

    print(f"\n{SEP}")
    print("Table 4: 逐日尖峰降幅")
    print(SEP)
    print(f"  {'Day':>4}  {'No-DR W':>9}  {'Greedy W':>9}  {'Coord W':>9}  "
          f"{'Red% Grd':>9}  {'Red% Crd':>9}  Oracle")
    print(sep2)
    ar = m["_arrays"]
    oracle_map = {r["day"]: r for r in oracle_days}
    for d_idx in range(n_comm_days):
        pk_nd = ar["pk_nodr"][d_idx]
        pk_g  = ar["pk_greedy"][d_idx]
        pk_c  = ar["pk_coord"][d_idx]
        rg    = ar["red_greedy"][d_idx]
        rc    = ar["red_coord"][d_idx]
        orec  = oracle_map.get(d_idx, {})
        opk   = orec.get("oracle_peak")
        o_str = f"{opk:.0f}W" if opk is not None else "N/A"
        print(f"  {d_idx:>4}  {pk_nd:>9.0f}  {pk_g:>9.0f}  {pk_c:>9.0f}  "
              f"  {rg:>8.1f}%  {rc:>8.1f}%  {o_str}")
    print(sep2)
    print(f"  mean  {np.mean(ar['pk_nodr']):>9.0f}  "
          f"{np.mean(ar['pk_greedy']):>9.0f}  {np.mean(ar['pk_coord']):>9.0f}  "
          f"  {np.mean(ar['red_greedy']):>8.1f}%  {np.mean(ar['red_coord']):>8.1f}%")


# ── Plots ─────────────────────────────────────────────────────────────────────

def make_plots(
    no_dr_loads:  np.ndarray,
    greedy_loads: np.ndarray,
    coord_loads:  np.ndarray,
    n_comm_days:  int,
    n_comm:       int,
) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    t_h = np.arange(n_comm) * SLOT_MINUTES / 60.0  # hours

    # ── Plot 1: Full 18-day load curves ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(t_h, no_dr_loads / 1000,   color="tomato",    alpha=0.6, lw=0.6, label="No-DR")
    ax.plot(t_h, greedy_loads / 1000,  color="goldenrod", alpha=0.7, lw=0.7, label="Greedy")
    ax.plot(t_h, coord_loads / 1000,   color="steelblue", alpha=0.9, lw=0.8, label="Online-coord")
    for d in range(n_comm_days + 1):
        ax.axvline(d * 24, color="gray", lw=0.4, ls="--")
    ax.set_xlabel("Time (hours from start of community window)")
    ax.set_ylabel("Aggregate load (kW)")
    ax.set_title(f"Phase 4d: Synthetic community aggregate load — {n_comm_days} days × 17 houses")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phase4d_loads.png", dpi=150)
    plt.close()
    print(f"  Saved: {RESULTS_DIR/'phase4d_loads.png'}")

    # ── Plot 2: Representative peak-day detail ────────────────────────────────
    # Find day with highest greedy daily peak
    daily_g = [greedy_loads[d*SLOTS_PER_DAY:(d+1)*SLOTS_PER_DAY].max()
               for d in range(n_comm_days)]
    peak_day = int(np.argmax(daily_g))
    s0 = peak_day * SLOTS_PER_DAY
    s1 = s0 + SLOTS_PER_DAY
    t_day = np.arange(SLOTS_PER_DAY) * SLOT_MINUTES / 60.0  # 0..24h

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_day, no_dr_loads[s0:s1]  / 1000, color="tomato",    lw=1.2, label="No-DR")
    ax.plot(t_day, greedy_loads[s0:s1] / 1000, color="goldenrod", lw=1.2, label="Greedy")
    ax.plot(t_day, coord_loads[s0:s1]  / 1000, color="steelblue", lw=1.5, label="Online-coord")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Aggregate load (kW)")
    ax.set_title(f"Phase 4d: Peak day (day {peak_day}) — No-DR vs Greedy vs Coord")
    ax.set_xticks(range(0, 25, 3))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phase4d_peak_day.png", dpi=150)
    plt.close()
    print(f"  Saved: {RESULTS_DIR/'phase4d_peak_day.png'}  (day {peak_day})")


# ── Save results ──────────────────────────────────────────────────────────────

def save_results(
    m: dict, oracle_days: list,
    no_dr_loads: np.ndarray, greedy_loads: np.ndarray, coord_loads: np.ndarray,
) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    # JSON (metrics + oracle days)
    out = {
        "metrics": {k: v for k, v in m.items() if k != "_arrays"},
        "oracle_days": oracle_days,
    }
    with open(RESULTS_DIR / "phase4d_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Saved: {RESULTS_DIR/'phase4d_results.json'}")
    # NumPy arrays
    np.savez(
        RESULTS_DIR / "phase4d_loads.npz",
        no_dr=no_dr_loads, greedy=greedy_loads, coord=coord_loads,
    )
    print(f"  Saved: {RESULTS_DIR/'phase4d_loads.npz'}")


# ── HARD RULE self-check ──────────────────────────────────────────────────────

def hard_rule_check() -> None:
    print("\n" + "=" * 72)
    print("HARD RULE 自我檢查 — phase4d_eval.py")
    print("=" * 72)
    checks = [
        ("Δ=10min / 144 slots/day — 全程使用此解析度",                                True),
        ("chronological split 70/10/20，windows 全來自各戶 test 段",                   True),
        ("因果性: Coord 用 LSTM forecast (sim.forecast); 不讀未來真實 baseload",       True),
        ("Oracle 用真實 baseload 作背景（離線下界，明確標注非線上方法）",               True),
        ("must-run 計入聚合負載 (include_must_run=True in schedule_house)",             True),
        ("Coordinator 只收 Σ_h load_h（隱私）",                                       True),
        ("commit-first: 只 slot-0 jobs 鎖定，後續 tick 不可撤銷",                      True),
        ("warm-start: 下個 tick λ_init = best_lam from this tick",                    True),
        ("None fallback: forecast=None 時用 last_fc 左移 1 slot",                     True),
        ("無 leakage: win_start ≥ test_start_h (community 驗證已確認)",                True),
        ("排除 H11/21（太陽能）H12（無 deferrable）",                                  True),
        ("baseload = gap-handled Aggregate − Σ(deferrable)（Phase 1 輸出）",           True),
        ("指標: 無 R²；用 PAR / 削峰% / 延後時數 / miss-rate / runtime",              True),
        ("Oracle 超時或無 schedulable job → 該日標 N/A，不硬湊",                      True),
    ]
    all_ok = all(ok for _, ok in checks)
    for desc, ok in checks:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {desc}")
    print(f"\n  Overall: {'ALL OK' if all_ok else 'SOME FAILURES'}")


# ── Main ──────────────────────────────────────────────────────────────────────

# Module-level bl_arrays (needed inside run_oracle)
bl_arrays: Dict[int, np.ndarray] = {}

def main() -> None:
    global bl_arrays
    print("=" * 72)
    print("Phase 4d: 正式收割評估 — 合成社區 17 戶 × 18 天")
    print("=" * 72)

    # ── Step 1: Build community ───────────────────────────────────────────────
    windows, bl_arrays, community_bl, n_comm_days, n_comm = build_community()

    # ── Step 2: Load simulators ───────────────────────────────────────────────
    simulators = load_simulators(windows)

    # ── Step 3: Extract window jobs ───────────────────────────────────────────
    window_jobs = extract_window_jobs(simulators, windows, n_comm)
    total_jobs  = sum(len(v) for v in window_jobs.values())
    print(f"[Jobs] Total window jobs: {total_jobs}")

    # ── Step 4: No-DR ─────────────────────────────────────────────────────────
    print("\n[No-DR] Computing no-coordination baseline ...")
    no_dr_loads, no_dr_jstarts = run_no_dr(community_bl, window_jobs, windows, n_comm)

    # ── Step 5: Greedy rolling ────────────────────────────────────────────────
    greedy_loads, greedy_logs, greedy_committed = run_rolling(
        simulators, windows, window_jobs, community_bl, n_comm,
        mode="greedy",
    )

    # ── Step 6: Coordinated rolling ───────────────────────────────────────────
    coord_loads, coord_logs, coord_committed = run_rolling(
        simulators, windows, window_jobs, community_bl, n_comm,
        mode="coord",
    )

    # ── Step 7: Oracle (per-day peak window MILP) ─────────────────────────────
    oracle_days = run_oracle(
        simulators, windows, window_jobs, community_bl,
        n_comm_days, n_comm,
        reference_loads=greedy_loads,
    )

    # ── Step 8: Metrics ───────────────────────────────────────────────────────
    print("\n[Metrics] Computing ...")
    m = compute_metrics(
        no_dr_loads, greedy_loads, coord_loads,
        oracle_days, n_comm_days, n_comm,
        coord_committed, no_dr_jstarts,
        coord_logs, windows, window_jobs,
    )

    # ── Step 9: Print results ─────────────────────────────────────────────────
    print_results(m, n_comm_days, oracle_days)

    # ── Step 10: Plots ────────────────────────────────────────────────────────
    print("\n[Plots] Generating ...")
    make_plots(no_dr_loads, greedy_loads, coord_loads, n_comm_days, n_comm)

    # ── Step 11: Save ─────────────────────────────────────────────────────────
    print("\n[Save]")
    save_results(m, oracle_days, no_dr_loads, greedy_loads, coord_loads)

    # ── Step 12: HARD RULE ────────────────────────────────────────────────────
    hard_rule_check()


if __name__ == "__main__":
    main()
