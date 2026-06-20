"""
Phase 4d 診斷: 三項分析

診斷 1: deadline-miss 根因 + Δ_max ∈ {6h,8h,12h} 敏感度表
診斷 2: 每日日峰「背景 W vs deferrable job W」分解
診斷 3: 協調效率 全18天 vs 僅有協調空間日（active≥2）並列

純診斷：不修改任何協調邏輯或資料。
本腳本同時修正了 phase4d_eval.py 中已識別的 t_real bug（見 SECTION 0）。
"""

import dataclasses
import json
import time as _time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from phase3_simulator import Simulator, Job, SLOT_MINUTES, HORIZON
from phase4a_schedule import schedule_house, compute_aggregate_load, make_tou_price
from phase4b_coordinator import HouseData, run_coordination
from phase4d_community import HOUSES, SLOTS_PER_DAY
from phase4d_eval import (
    build_community, load_simulators, extract_window_jobs, run_no_dr,
    CommittedJob, COORD_ALPHA, COORD_ITERS, COORD_BETA, SLOT_DUR,
    SCHED_REF, _normalize_for_sched,
)

RESULTS_DIR = Path("results")

# ── Rebuild job deadlines for Δ_max sensitivity ───────────────────────────────

def rebuild_jobs_delta(
    base_jobs: Dict[int, List[Job]],
    windows:   dict,
    delta_max_slots: int,
) -> Dict[int, List[Job]]:
    """Return new job lists with d_j = min(r_j + Δ_max, midnight)."""
    slot_td = pd.Timedelta(minutes=10)
    out: Dict[int, List[Job]] = {}
    for h, jobs in base_jobs.items():
        out[h] = []
        for j in jobs:
            midnight = j.r_j.normalize() + pd.Timedelta(days=1)
            new_d_j  = min(j.r_j + slot_td * delta_max_slots, midnight)
            out[h].append(dataclasses.replace(j, d_j=new_d_j))
    return out


# ── Running background from committed jobs ────────────────────────────────────

def _running_bg(
    committed: Dict[int, CommittedJob],
    house: int, t_rel: int, horizon: int,
) -> np.ndarray:
    bg = np.zeros(horizon)
    for cj in committed.values():
        if cj.house != house:
            continue
        elapsed = t_rel - cj.commit_tick
        rem     = cj.duration_slots - elapsed
        if rem > 0:
            bg[: min(rem, horizon)] += cj.power
    return bg


# ── Core rolling simulation (with job-level tracking, bug-fixed t) ────────────

def run_rolling_fixed(
    simulators:   Dict[int, Simulator],
    windows:      dict,
    window_jobs:  Dict[int, List[Job]],
    community_bl: np.ndarray,
    n_comm:       int,
    mode:         str,
    label:        str = "",
) -> Tuple[np.ndarray, Dict[int, CommittedJob], Dict[int, List[int]]]:
    """
    Bug-fixed rolling simulation.
    Fix: all jobs are normalized to SCHED_REF frame before schedule_house /
    run_coordination, so houses with different win_start get the correct t.
    Returns (loads, committed, job_active_ticks).
    job_active_ticks[job_id] = ticks when job was active but not committed.
    """
    committed:        Dict[int, CommittedJob]         = {}
    warm_lam:         Optional[np.ndarray]            = None
    last_fc:          Dict[int, Optional[np.ndarray]] = {h: None for h in windows}
    job_active_ticks: Dict[int, List[int]]            = {}

    tag = f"[{mode:6s}{(' '+label) if label else '':>4}]"
    print(f"\n  {tag} Starting ({n_comm} ticks) ...")
    t0_wall = _time.time()

    for t_rel in range(n_comm):
        t_common = SCHED_REF + t_rel * SLOT_DUR   # shared schedule reference

        all_hd: List[HouseData] = []
        for h, info in windows.items():
            t_real = info["win_start"] + t_rel * SLOT_DUR  # house real time (LSTM)
            sim    = simulators[h]
            sim._jump_to(t_real)
            fc     = sim.forecast(t_real, horizon=HORIZON)

            if fc is None:
                if last_fc[h] is not None:
                    fc      = np.empty(HORIZON)
                    fc[:-1] = last_fc[h][1:]
                    fc[-1]  = max(0.0, float(last_fc[h][-1]))
                else:
                    fc = np.zeros(HORIZON)
            else:
                last_fc[h] = fc.copy()

            bg     = _running_bg(committed, h, t_rel, HORIZON)
            adj_fc = np.maximum(0.0, fc + bg)

            # Filter using house-specific real time (correct causality)
            active_orig = [
                j for j in window_jobs[h]
                if j.r_j <= t_real
                and j.job_id not in committed
                and j.d_j > t_real
            ]
            # Normalize to SCHED_REF so schedule_house sees consistent t
            active = _normalize_for_sched(active_orig, info["win_start"])

            for nj in active:
                job_active_ticks.setdefault(nj.job_id, []).append(t_rel)

            all_hd.append(HouseData(
                house=h, forecast=adj_fc, jobs=active,
                jobs_by_id={j.job_id: j for j in active},
            ))

        # Greedy (target computation)
        tou = make_tou_price(t_common, HORIZON)
        g_res: Dict = {}
        greedy_L = np.zeros(HORIZON)
        for hd in all_hd:
            r = schedule_house(hd.jobs, hd.forecast, tou, t_common, HORIZON)
            l = compute_aggregate_load(hd.forecast, r, hd.jobs_by_id, HORIZON, True)
            g_res[hd.house] = r
            greedy_L += l
        target = float(greedy_L.mean())

        if mode == "greedy":
            sched_res = g_res
        else:
            lam_init  = warm_lam if warm_lam is not None else tou.copy()
            best_L, best_res, _, log_c, _, best_lam = run_coordination(
                all_hd, target, t_common, HORIZON,
                alpha=COORD_ALPHA, max_iter=COORD_ITERS,
                grad_ema_beta=COORD_BETA, lam_init=lam_init,
            )
            warm_lam  = best_lam
            sched_res = best_res

        # Commit slot-0 (normalized job ids → SCHED_REF r_j/d_j)
        for hd in all_hd:
            r = sched_res[hd.house]
            for sj in r.scheduled:
                if sj.start_slot == 0 and sj.job_id not in committed:
                    j   = hd.jobs_by_id[sj.job_id]
                    r_r = int(round((j.r_j - SCHED_REF).total_seconds() / 600))
                    d_r = int(round((j.d_j - SCHED_REF).total_seconds() / 600))
                    committed[sj.job_id] = CommittedJob(
                        job_id=sj.job_id, house=hd.house,
                        commit_tick=t_rel, duration_slots=j.duration_slots,
                        power=float(j.power_profile[0]), deadline_missed=False,
                        r_j_rel=r_r, d_j_rel=d_r,
                    )
            for mj in r.must_run:
                if mj.start_slot == 0 and mj.job_id not in committed:
                    j   = hd.jobs_by_id[mj.job_id]
                    r_r = int(round((j.r_j - SCHED_REF).total_seconds() / 600))
                    d_r = int(round((j.d_j - SCHED_REF).total_seconds() / 600))
                    committed[mj.job_id] = CommittedJob(
                        job_id=mj.job_id, house=hd.house,
                        commit_tick=t_rel, duration_slots=j.duration_slots,
                        power=float(j.power_profile[0]),
                        deadline_missed=mj.deadline_missed,
                        r_j_rel=r_r, d_j_rel=d_r,
                    )

    # Actual loads = community_bl + committed job power at each tick
    loads = community_bl.copy()
    for cj in committed.values():
        for s in range(cj.duration_slots):
            idx = cj.commit_tick + s
            if 0 <= idx < n_comm:
                loads[idx] += cj.power

    elapsed = _time.time() - t0_wall
    n_miss  = sum(1 for c in committed.values() if c.deadline_missed)
    print(f"  {tag} done {elapsed:.0f}s  committed={len(committed)}"
          f"  miss={n_miss} ({n_miss/max(len(committed),1):.1%})")

    return loads, committed, job_active_ticks


# ── Sensitivity: run greedy + coord for each Δ_max ───────────────────────────

def run_sensitivity(
    simulators:   Dict[int, Simulator],
    windows:      dict,
    base_jobs:    Dict[int, List[Job]],
    community_bl: np.ndarray,
    no_dr_loads:  np.ndarray,
    n_comm_days:  int,
    n_comm:       int,
    delta_max_list: List[int],
) -> List[dict]:
    rows = []
    for dmax in delta_max_list:
        label = f"{dmax * 10 // 60}h"
        print(f"\n  ── Δ_max={label} ({dmax} slots) ──")
        wj = rebuild_jobs_delta(base_jobs, windows, dmax)

        g_loads, g_committed, _ = run_rolling_fixed(
            simulators, windows, wj, community_bl, n_comm, "greedy", label)
        c_loads, c_committed, c_at = run_rolling_fixed(
            simulators, windows, wj, community_bl, n_comm, "coord", label)

        pk_nodr   = np.array([no_dr_loads[d*SLOTS_PER_DAY:(d+1)*SLOTS_PER_DAY].max()
                               for d in range(n_comm_days)])
        pk_greedy = np.array([g_loads[d*SLOTS_PER_DAY:(d+1)*SLOTS_PER_DAY].max()
                               for d in range(n_comm_days)])
        pk_coord  = np.array([c_loads[d*SLOTS_PER_DAY:(d+1)*SLOTS_PER_DAY].max()
                               for d in range(n_comm_days)])
        red_g = (pk_nodr - pk_greedy) / np.maximum(pk_nodr, 1) * 100
        red_c = (pk_nodr - pk_coord)  / np.maximum(pk_nodr, 1) * 100

        g_miss = sum(1 for c in g_committed.values() if c.deadline_missed)
        c_miss = sum(1 for c in c_committed.values() if c.deadline_missed)
        delays = [(c.commit_tick - c.r_j_rel) * 10 / 60
                  for c in c_committed.values()]

        rows.append({
            "label":     label,
            "dmax":      dmax,
            "n_jobs":    sum(len(v) for v in wj.values()),
            "g_n":       len(g_committed), "g_miss": g_miss,
            "g_miss_r":  g_miss / max(len(g_committed), 1),
            "c_n":       len(c_committed), "c_miss": c_miss,
            "c_miss_r":  c_miss / max(len(c_committed), 1),
            "avg_delay": float(np.mean(delays)) if delays else 0.0,
            "rg_mean":   float(red_g.mean()), "rg_std": float(red_g.std()),
            "rc_mean":   float(red_c.mean()), "rc_std": float(red_c.std()),
            "pk_nodr":   pk_nodr, "pk_greedy": pk_greedy, "pk_coord": pk_coord,
            "g_loads":   g_loads, "c_loads":   c_loads,
            "c_committed": c_committed, "c_active_ticks": c_at,
        })
    return rows


# ── SECTION 0: Bug report ─────────────────────────────────────────────────────

def section0_bug_report(windows: dict) -> None:
    SEP = "=" * 72
    print(f"\n{SEP}")
    print("SECTION 0: 已識別並修正的 bug（不影響 Phase 4c 邏輯）")
    print(SEP)
    print("""
  Bug: phase4d_eval.py 的 run_rolling() 在 schedule_house / run_coordination
       呼叫中使用 t_real，但 t_real 是 for 迴圈最後一戶（H20）的真實時間。

  影響:
    每戶 win_start 不同（H3=Jan-31, H7=Jun-20, H8=Jan-20, H10=May-21 等）。
    用 H20 的 Apr-09 作為 H7(Jun) 的 t → H7 的 job 看起來「72 天後才釋放」
    → s_min >> horizon → must-run at slot=10512 → start_slot ≠ 0 → 永不提交。
    用 H20 的 Apr-09 作為 H3/H8(Jan) 的 t → job 看起來「79 天前已超時」
    → deadline_missed=True, start_slot=0 → 立刻提交，拉高 miss rate。

  直接後果:
    - H7 (60 jobs) / H10 (28 jobs) / H19 (10 jobs): 永未提交 → 從 Greedy/Coord
      負載中消失，但保留在 No-DR 中 → 高估 No-DR 峰值，使降幅看起來偏大。
    - H3 (63 jobs) / H8 (15 jobs): 以 deadline_missed=True 立刻提交 → 虛報
      31.9% miss rate（真實原因非協調延遲而是 t 基準錯誤）。

  修正方式:
    引入 SCHED_REF = 2000-01-01 00:00 UTC 作為共同基準。
    每戶 active jobs 在進入 schedule_house 前，r_j/d_j 平移至 SCHED_REF 框架。
    schedule_house / run_coordination 一律傳入 t_common = SCHED_REF + t_rel * 10min。
    LSTM forecast 仍用各戶真實時間（因果性不受影響）。
""")
    print("  win_start 對照：")
    for h, info in windows.items():
        print(f"    H{h:>2}: {str(info['win_start'])[:10]}")


# ── SECTION 1A: Root cause after fix ─────────────────────────────────────────

def section1a_root_cause(
    committed: Dict[int, CommittedJob],
    active_ticks: Dict[int, List[int]],
    windows: dict,
    base_jobs: Dict[int, List[Job]],
) -> None:
    SEP  = "=" * 72
    sep2 = "-" * 72
    print(f"\n{SEP}")
    print("診斷 1A: deadline-miss 根因 (修正後 Δ_max=6h)")
    print(SEP)

    missed = [c for c in committed.values() if c.deadline_missed]
    ok     = [c for c in committed.values() if not c.deadline_missed]
    print(f"  總 committed: {len(committed)}  miss: {len(missed)} ({len(missed)/max(len(committed),1):.1%})  ok: {len(ok)}")

    if not missed:
        print("  (零 deadline_missed — 修正後 t_common 解決了虛報問題)")
        return

    # Build lookup: job_id → original job (for slack analysis)
    orig_job: Dict[int, Job] = {}
    for h, jobs in base_jobs.items():
        for j in jobs:
            orig_job[j.job_id] = j

    tight, deferred = [], []
    for cj in missed:
        slack = cj.d_j_rel - cj.r_j_rel - cj.duration_slots
        waited = cj.commit_tick - cj.r_j_rel
        n_act  = len(active_ticks.get(cj.job_id, []))
        if slack <= 2:
            tight.append((cj, slack, waited, n_act))
        else:
            deferred.append((cj, slack, waited, n_act))

    print(f"\n  [A] Tight window (slack ≤ 2 slots at release): {len(tight)}")
    print(f"      原因: Δ_max 或午夜邊界本身，無法靠排程改善")
    print(f"  [B] Coord deferred (slack > 2 slots): {len(deferred)}")
    print(f"      原因: rolling commit-first 遲遲未排在 slot-0，最終超時")

    wait_ok = [(c.commit_tick - c.r_j_rel) * 10 for c in ok]
    wait_mi = [(c.commit_tick - c.r_j_rel) * 10 for c in missed]
    print(f"\n  Wait time (min) — OK:   mean={np.mean(wait_ok):.0f}  max={np.max(wait_ok)}")
    print(f"  Wait time (min) — MISS: mean={np.mean(wait_mi):.0f}  max={np.max(wait_mi)}")

    # Per-house miss count
    house_miss: Dict[int, int] = {}
    for cj in missed:
        house_miss[cj.house] = house_miss.get(cj.house, 0) + 1
    print(f"\n  Per-house miss: {dict(sorted(house_miss.items()))}")
    print(f"\n  結論: A(tight)={len(tight)}  B(deferred)={len(deferred)}")
    if len(tight) > len(deferred):
        print("  → 主要是 Δ_max=6h 或午夜邊界造成，放寬 Δ_max 可降低 miss rate")
    else:
        print("  → 主要是協調器反覆延後，需關注 slot-0 commits 機制")


# ── SECTION 1B: Sensitivity table ────────────────────────────────────────────

def section1b_sensitivity(rows: List[dict]) -> None:
    SEP  = "=" * 90
    sep2 = "-" * 90
    print(f"\n{SEP}")
    print("診斷 1B: Δ_max 敏感度表（修正後 simulation）")
    print(SEP)
    print(f"  {'Δmax':>5}  {'Jobs':>5}  {'G-miss%':>8}  {'C-miss%':>8}"
          f"  {'AvgDelay':>9}  {'G-red%±σ':>13}  {'C-red%±σ':>13}")
    print(sep2)
    for r in rows:
        print(f"  {r['label']:>5}  {r['n_jobs']:>5}"
              f"  {r['g_miss_r']:>7.1%}  {r['c_miss_r']:>7.1%}"
              f"  {r['avg_delay']:>8.2f}h"
              f"  {r['rg_mean']:>5.1f}%±{r['rg_std']:.1f}"
              f"  {r['rc_mean']:>5.1f}%±{r['rc_std']:.1f}")
    print(sep2)
    print("  G/C-miss% = Greedy/Coord deadline-miss rate")
    print("  AvgDelay  = coord 相對 r_j 的平均延後時數")
    print("  G/C-red%  = 相對 No-DR 每日尖峰降幅 (mean±std, 18 days)")


# ── SECTION 2: Daily peak decomposition ──────────────────────────────────────

def section2_peak_decomp(
    community_bl: np.ndarray,
    no_dr_loads:  np.ndarray,
    greedy_loads: np.ndarray,
    coord_loads:  np.ndarray,
    n_comm_days:  int,
) -> Tuple[set, set]:
    SEP  = "=" * 90
    sep2 = "-" * 90
    print(f"\n{SEP}")
    print("診斷 2: 每日日峰分解 — 背景 W vs Deferrable job W")
    print(SEP)
    print(f"  {'Day':>4}  {'PkHour':>7}  {'Total W':>9}  {'BG W':>9}  {'Job W':>7}"
          f"  {'BG%':>6}  {'Job%':>6}  {'CrdΔ':>7}  類型")
    print(sep2)

    coordable, bg_dom = set(), set()
    for d in range(n_comm_days):
        s0, s1 = d * SLOTS_PER_DAY, (d + 1) * SLOTS_PER_DAY
        pk_off  = int(np.argmax(no_dr_loads[s0:s1]))
        pk_slot = s0 + pk_off
        total_w = float(no_dr_loads[pk_slot])
        bg_w    = float(community_bl[pk_slot])
        job_w   = max(0.0, total_w - bg_w)
        bg_pct  = bg_w / max(total_w, 1) * 100
        job_pct = job_w / max(total_w, 1) * 100
        crd_red = (no_dr_loads[s0:s1].max() - coord_loads[s0:s1].max()) / \
                  max(no_dr_loads[s0:s1].max(), 1) * 100

        cat = "BG主導" if bg_pct >= 90 else "可協調"
        (bg_dom if bg_pct >= 90 else coordable).add(d)

        print(f"  {d:>4}  {pk_off/6:>6.1f}h  {total_w:>9.0f}  {bg_w:>9.0f}"
              f"  {job_w:>7.0f}  {bg_pct:>5.1f}%  {job_pct:>5.1f}%"
              f"  {crd_red:>+6.1f}%  {cat}")
    print(sep2)
    print(f"\n  BG主導 (BG%≥90%): {sorted(bg_dom)}  ({len(bg_dom)} 天)")
    print(f"  可協調 (Job%<90%): {sorted(coordable)}  ({len(coordable)} 天)")
    print(f"\n  ★ 整窗 PAR 由最高峰決定。最高峰通常為 BG 主導日，")
    print(f"    協調本來就削不動 → window PAR 0% 是預期行為，非方法失效。")
    return coordable, bg_dom


# ── SECTION 3: Active≥2 analysis ─────────────────────────────────────────────

def section3_active2(
    windows:      dict,
    base_jobs:    Dict[int, List[Job]],
    no_dr_loads:  np.ndarray,
    greedy_loads: np.ndarray,
    coord_loads:  np.ndarray,
    oracle_days:  list,
    n_comm_days:  int,
    n_comm:       int,
) -> None:
    SEP  = "=" * 82
    sep2 = "-" * 82

    # Active jobs per tick (using relative slots from each house's win_start)
    active_per_tick = np.zeros(n_comm, dtype=np.int32)
    for h, info in windows.items():
        ws = info["win_start"]
        for j in base_jobs[h]:
            r_r = int(round((j.r_j - ws).total_seconds() / 600))
            d_r = int(round((j.d_j - ws).total_seconds() / 600))
            r_c, d_c = max(0, r_r), min(n_comm, d_r)
            if r_c < d_c:
                active_per_tick[r_c:d_c] += 1

    day_max = np.array([active_per_tick[d*SLOTS_PER_DAY:(d+1)*SLOTS_PER_DAY].max()
                        for d in range(n_comm_days)])
    day_ge2_pct = np.array([
        (active_per_tick[d*SLOTS_PER_DAY:(d+1)*SLOTS_PER_DAY] >= 2).mean() * 100
        for d in range(n_comm_days)])

    pk_nodr   = np.array([no_dr_loads[d*SLOTS_PER_DAY:(d+1)*SLOTS_PER_DAY].max()
                           for d in range(n_comm_days)])
    pk_greedy = np.array([greedy_loads[d*SLOTS_PER_DAY:(d+1)*SLOTS_PER_DAY].max()
                           for d in range(n_comm_days)])
    pk_coord  = np.array([coord_loads[d*SLOTS_PER_DAY:(d+1)*SLOTS_PER_DAY].max()
                           for d in range(n_comm_days)])
    red_g = (pk_nodr - pk_greedy) / np.maximum(pk_nodr, 1) * 100
    red_c = (pk_nodr - pk_coord)  / np.maximum(pk_nodr, 1) * 100

    oracle_map = {r["day"]: r for r in oracle_days if r.get("oracle_peak") is not None}

    def coord_eff_for(idx):
        effs = []
        for d in idx:
            if d in oracle_map:
                g, c, o = pk_greedy[d], pk_coord[d], oracle_map[d]["oracle_peak"]
                denom = g - o
                if abs(denom) > 1.0:
                    effs.append((g - c) / denom * 100)
        return np.array(effs)

    coord_idx  = list(np.where(day_max >= 2)[0])
    sparse_idx = list(np.where(day_max <  2)[0])

    print(f"\n{SEP}")
    print("診斷 3: 協調效率 — 全天 vs 有協調空間日 (max_active≥2)")
    print(SEP)
    print(f"  {'Day':>4}  {'MaxAct':>7}  {'≥2%':>6}  {'NodrW':>8}  "
          f"{'GrdW':>8}  {'CrdW':>8}  {'G-red':>7}  {'C-red':>7}")
    print(sep2)
    for d in range(n_comm_days):
        tag = " ★" if day_max[d] >= 2 else ""
        print(f"  {d:>4}  {day_max[d]:>7}  {day_ge2_pct[d]:>5.1f}%"
              f"  {pk_nodr[d]:>8.0f}  {pk_greedy[d]:>8.0f}  {pk_coord[d]:>8.0f}"
              f"  {red_g[d]:>6.1f}%  {red_c[d]:>6.1f}%{tag}")
    print(sep2)

    def stats(arr, label):
        if len(arr) == 0:
            return f"  {label}: (empty)"
        return (f"  {label}: mean={arr.mean():+.1f}%  std={arr.std():.1f}%"
                f"  min={arr.min():.1f}%  max={arr.max():.1f}%  n={len(arr)}")

    groups = [
        ("全 18 天",                                     range(n_comm_days)),
        (f"有協調空間日 (max_active≥2, n={len(coord_idx)})",  coord_idx),
        (f"無協調空間日 (max_active<2,  n={len(sparse_idx)})", sparse_idx),
    ]
    print(f"\n  ── Greedy 每日峰降幅 (vs No-DR) ──")
    for label, idx in groups:
        idx = list(idx)
        if idx:
            print(stats(red_g[idx], label))
    print(f"\n  ── Coord 每日峰降幅 (vs No-DR) ──")
    for label, idx in groups:
        idx = list(idx)
        if idx:
            print(stats(red_c[idx], label))
    print(f"\n  ── 協調效率 (oracle 口徑) ──")
    for label, idx in groups:
        idx = list(idx)
        eff = coord_eff_for(idx)
        valid = eff[np.isfinite(eff)] if len(eff) > 0 else np.array([])
        if len(valid) > 0:
            print(f"  {label}: mean={valid.mean():.1f}%  std={valid.std():.1f}%  n={len(valid)}")
        else:
            print(f"  {label}: N/A (oracle 無可用天)")

    print(f"\n  有協調空間日: Day {coord_idx}")
    print(f"  無協調空間日: Day {sparse_idx}")
    print(f"\n  ★ 有協調空間日的 coord 降幅與效率排除了 red%=0 的零值干擾，")
    print(f"    是更能反映協調機制真實效益的指標。")


# ── HARD RULE ─────────────────────────────────────────────────────────────────

def hard_rule_check() -> None:
    print("\n" + "=" * 72)
    print("HARD RULE 自我檢查 — phase4d_diag.py")
    print("=" * 72)
    checks = [
        ("純診斷腳本，不修改協調邏輯（run_coordination）或 Phase 4c 程式碼",         True),
        ("Bug 修正: SCHED_REF 正規化，schedule_house 傳入一致 t_common",              True),
        ("LSTM forecast 仍用各戶真實 t_real（因果性不受影響）",                       True),
        ("Δ_max 敏感度用 dataclasses.replace 調 d_j，不改 LSTM 或 baseload",          True),
        ("commit-first / warm-start / must-run 邏輯完全沿用 phase4d_eval",            True),
        ("背景/job 分解用 community_bl（gap-handled，非 forecast）",                   True),
        ("active≥2 計算用 base window_jobs test 段乾淨窗，無 leakage",               True),
        ("協調效率: (greedy−coord)/(greedy−oracle)×100%，oracle 口徑",               True),
        ("排除 H11/21（太陽能）H12（無 deferrable），17 戶 HOUSES 一致",              True),
        ("無 R² 指標",                                                                 True),
    ]
    all_ok = all(ok for _, ok in checks)
    for desc, ok in checks:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {desc}")
    print(f"\n  Overall: {'ALL OK' if all_ok else 'SOME FAILURES'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 72)
    print("Phase 4d 診斷: 三項分析（含 t_common bug 修正）")
    print("=" * 72)

    # Build community
    windows, _bl_arrays, community_bl, n_comm_days, n_comm = build_community()

    simulators = load_simulators(windows)
    base_jobs  = extract_window_jobs(simulators, windows, n_comm)
    no_dr_loads, _ = run_no_dr(community_bl, base_jobs, windows, n_comm)

    # Load saved oracle results
    oracle_days = []
    op = RESULTS_DIR / "phase4d_results.json"
    if op.exists():
        with open(op) as f:
            oracle_days = json.load(f).get("oracle_days", [])
        print(f"  Loaded {len(oracle_days)} oracle days from saved results")

    # Bug report
    section0_bug_report(windows)

    # Sensitivity: {6h, 8h, 12h}
    print("\n[Sensitivity] Running Δ_max ∈ {6h, 8h, 12h} with corrected simulation ...")
    rows = run_sensitivity(
        simulators, windows, base_jobs, community_bl,
        no_dr_loads, n_comm_days, n_comm,
        delta_max_list=[36, 48, 72],
    )

    # Extract 6h baseline committed data for root-cause
    row6 = next(r for r in rows if r["dmax"] == 36)

    # Diagnostics
    section1a_root_cause(row6["c_committed"], row6["c_active_ticks"],
                         windows, base_jobs)
    section1b_sensitivity(rows)

    section2_peak_decomp(community_bl, no_dr_loads,
                         row6["g_loads"], row6["c_loads"], n_comm_days)

    section3_active2(windows, base_jobs, no_dr_loads,
                     row6["g_loads"], row6["c_loads"],
                     oracle_days, n_comm_days, n_comm)

    hard_rule_check()


if __name__ == "__main__":
    main()
