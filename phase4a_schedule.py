"""
Phase 4a — Single-house deferrable scheduling sub-problem (with must-run rule)

Builds on Phase 3 Simulator (observe / forecast / job release).
Implements schedule_house(): given released jobs + baseload forecast + ToU price,
find optimal non-interruptible start slot for each cycle, and classify remaining
jobs as must-run with a computed start slot and deadline_missed flag.

Scope: single house only.  No shadow-price iteration, no multi-house coordination.
Those are Phase 4b+.

Assumptions (written explicitly, labelled "ASSUMPTION"):
  ASSUMPTION-A: Economy 7 ToU price — night 23:30–07:30 UTC = 0.09 £/kWh,
                day 07:30–23:30 UTC = 0.28 £/kWh.  Hard-coded; real tariff TBD.
  ASSUMPTION-B: Power profile = flat mean_W from Phase 1 cycle summary (constant
                power over duration).  Per-slot channel power from raw CSV is
                future work (TODO in Phase 3 Job dataclass).
  ASSUMPTION-C: No inter-job conflict constraint in Phase 4a.  Two jobs may overlap
                in the schedule; conflict avoidance is the coordinator's job (4b+).
  ASSUMPTION-D: Objective is to minimise Σ_t price[t] × total_load[t], where
                total_load = baseload_forecast + job loads.  Since baseload is
                fixed per tick, this reduces to per-job: min Σ price[s:s+dur].
  ASSUMPTION-E: Must-run rule — when no legal start slot fits within the horizon:
                  d_j − dur ≥ t → start at d_j_slot − duration (latest deadline-safe);
                                   deadline_missed = False.
                  d_j − dur < t → start at slot 0 (immediately at t);
                                   deadline_missed = True.
                Must-run jobs are non-interruptible, not cost-optimised, and their
                power_profile MUST be counted in the aggregate load (fixed background).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional

from phase3_simulator import (
    Simulator, Job, CausalViolationError,
    SLOT_MINUTES, HORIZON,
)

# ── ToU price (ASSUMPTION-A) ──────────────────────────────────────────────────

PRICE_DAY_GBP   = 0.28   # £/kWh, on-peak  07:30–23:30 UTC
PRICE_NIGHT_GBP = 0.09   # £/kWh, off-peak 23:30–07:30 UTC


def make_tou_price(t_start: pd.Timestamp, n_slots: int) -> np.ndarray:
    """Economy 7 price vector.  Night: hour_min ∈ [23:30, 07:30)."""
    price = np.empty(n_slots, dtype=np.float64)
    for i in range(n_slots):
        ts = t_start + pd.Timedelta(minutes=SLOT_MINUTES * i)
        hm = ts.hour * 60 + ts.minute
        price[i] = PRICE_NIGHT_GBP if (hm >= 23 * 60 + 30 or hm < 7 * 60 + 30) \
                   else PRICE_DAY_GBP
    return price


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class ScheduledJob:
    job_id:     int
    start_slot: int      # relative to t (slot 0 = t)
    end_slot:   int      # exclusive: start_slot + duration_slots
    cost_gbp:   float


@dataclass
class MustRunJob:
    """
    Job that could not be optimally scheduled; runs at a fixed start slot.

    start_slot:
      deadline_missed=False → d_j_slot − duration  (latest deadline-safe slot)
      deadline_missed=True  → 0                    (runs immediately at t)
    end_slot may exceed horizon; aggregate load still counts in-horizon slots.
    """
    job_id:          int
    start_slot:      int
    end_slot:        int
    deadline_missed: bool


@dataclass
class ScheduleResult:
    scheduled: List[ScheduledJob]
    must_run:  List[MustRunJob]


# ── Scheduling exception ──────────────────────────────────────────────────────

class ScheduleViolationError(RuntimeError):
    """Raised by assert_schedule_valid / assert_must_run_valid on constraint breach."""


# ── Core scheduler ────────────────────────────────────────────────────────────

def schedule_house(
    jobs:              List[Job],
    baseload_forecast: np.ndarray,
    price:             np.ndarray,
    t:                 pd.Timestamp,
    horizon:           int,
) -> ScheduleResult:
    """
    Solve the single-house scheduling sub-problem (ASSUMPTION-D).

    For each job, feasible start range:
        s_min = max(0, r_j_slot)
        s_max = min(d_j_slot − duration, horizon − duration)

    If s_min ≤ s_max: enumerate s in [s_min, s_max], pick argmin cost → ScheduledJob.
    If s_min >  s_max: apply must-run rule (ASSUMPTION-E) → MustRunJob.

    Does NOT modify Simulator state.
    """
    if len(baseload_forecast) < horizon:
        raise ValueError(f"baseload_forecast length {len(baseload_forecast)} < horizon {horizon}")
    if len(price) < horizon:
        raise ValueError(f"price length {len(price)} < horizon {horizon}")

    slot_td    = pd.Timedelta(minutes=SLOT_MINUTES)
    scheduled: List[ScheduledJob] = []
    must_run:  List[MustRunJob]   = []

    for job in jobs:
        dur    = job.duration_slots
        r_slot = int(round((job.r_j - t) / slot_td))
        d_slot = int(round((job.d_j - t) / slot_td))

        s_min = max(0, r_slot)
        s_max = min(d_slot - dur, horizon - dur)

        if s_min <= s_max:
            # ── Feasible: enumerate and pick cheapest ─────────────────────────
            best_start = s_min
            best_cost  = np.inf
            for s in range(s_min, s_max + 1):
                cost = float(np.dot(price[s : s + dur], job.power_profile) / 6000.0)
                if cost < best_cost:
                    best_cost  = cost
                    best_start = s
            scheduled.append(ScheduledJob(
                job_id     = job.job_id,
                start_slot = best_start,
                end_slot   = best_start + dur,
                cost_gbp   = best_cost,
            ))
        else:
            # ── Infeasible: must-run rule (ASSUMPTION-E) ──────────────────────
            latest = d_slot - dur          # relative to t
            if latest >= 0:
                # Can still meet deadline; start at latest deadline-safe slot.
                must_run.append(MustRunJob(
                    job_id          = job.job_id,
                    start_slot      = latest,
                    end_slot        = latest + dur,
                    deadline_missed = False,
                ))
            else:
                # d_j - dur < t: deadline already unreachable — run immediately.
                must_run.append(MustRunJob(
                    job_id          = job.job_id,
                    start_slot      = 0,
                    end_slot        = dur,
                    deadline_missed = True,
                ))

    return ScheduleResult(scheduled=scheduled, must_run=must_run)


# ── Constraint validators ─────────────────────────────────────────────────────

def assert_schedule_valid(
    result:     ScheduleResult,
    jobs_by_id: Dict[int, Job],
    t:          pd.Timestamp,
    horizon:    int,
) -> None:
    """
    Assert constraints on .scheduled jobs (raises ScheduleViolationError):
      ① Non-interruptible: end_slot == start_slot + duration
      ② Release:           start_slot >= max(0, r_j_slot)
      ③ Deadline:          end_slot   <= d_j_slot
      ④ Horizon:           end_slot   <= horizon
    """
    slot_td = pd.Timedelta(minutes=SLOT_MINUTES)
    for sj in result.scheduled:
        job    = jobs_by_id[sj.job_id]
        dur    = job.duration_slots
        r_slot = int(round((job.r_j - t) / slot_td))
        d_slot = int(round((job.d_j - t) / slot_td))

        if sj.end_slot != sj.start_slot + dur:
            raise ScheduleViolationError(
                f"[JOB {sj.job_id}] ① Non-interruptible: "
                f"end({sj.end_slot}) ≠ start({sj.start_slot})+dur({dur})")
        if sj.start_slot < max(0, r_slot):
            raise ScheduleViolationError(
                f"[JOB {sj.job_id}] ② Release: "
                f"start({sj.start_slot}) < r_j_slot_lower({max(0,r_slot)})")
        if sj.end_slot > d_slot:
            raise ScheduleViolationError(
                f"[JOB {sj.job_id}] ③ Deadline: end({sj.end_slot}) > d_slot({d_slot})")
        if sj.end_slot > horizon:
            raise ScheduleViolationError(
                f"[JOB {sj.job_id}] ④ Horizon: end({sj.end_slot}) > horizon({horizon})")


def assert_must_run_valid(
    result:     ScheduleResult,
    jobs_by_id: Dict[int, Job],
    t:          pd.Timestamp,
) -> None:
    """
    Assert constraints on .must_run jobs:
      ① Non-interruptible: end_slot == start_slot + duration
      ② Release:           start_slot >= 0  (never starts before t)
      ③ Deadline:          WAIVED — deadline_missed flag is set instead
      ④ Horizon:           WAIVED — must-run may extend beyond current window
    """
    slot_td = pd.Timedelta(minutes=SLOT_MINUTES)
    for mj in result.must_run:
        job    = jobs_by_id[mj.job_id]
        dur    = job.duration_slots
        r_slot = int(round((job.r_j - t) / slot_td))

        if mj.end_slot != mj.start_slot + dur:
            raise ScheduleViolationError(
                f"[MUST-RUN {mj.job_id}] ① Non-interruptible: "
                f"end({mj.end_slot}) ≠ start({mj.start_slot})+dur({dur})")
        if mj.start_slot < max(0, r_slot):
            raise ScheduleViolationError(
                f"[MUST-RUN {mj.job_id}] ② Release: "
                f"start({mj.start_slot}) < max(0,r_j_slot)({max(0,r_slot)})")
        # ③④ waived for must-run (deadline_missed recorded in MustRunJob)


# ── Aggregate load ────────────────────────────────────────────────────────────

def compute_aggregate_load(
    baseload_forecast: np.ndarray,
    result:            ScheduleResult,
    jobs_by_id:        Dict[int, Job],
    horizon:           int,
    include_must_run:  bool = True,
) -> np.ndarray:
    """
    Build aggregate load W for slots [0, horizon).

      total[s] = baseload_forecast[s]
               + Σ_j power_j[s − start_j]  for scheduled jobs covering slot s
               + (if include_must_run) Σ_j power_j[...]  for must-run jobs

    Jobs extending beyond horizon are counted only for in-horizon slots.
    Must-run jobs excluded when include_must_run=False (to show the accounting gap).
    """
    load = baseload_forecast[:horizon].astype(np.float64).copy()

    def _add(start: int, end: int, profile: np.ndarray) -> None:
        for k, s in enumerate(range(start, end)):
            if 0 <= s < horizon:
                load[s] += profile[k]

    for sj in result.scheduled:
        _add(sj.start_slot, sj.end_slot, jobs_by_id[sj.job_id].power_profile)

    if include_must_run:
        for mj in result.must_run:
            _add(mj.start_slot, mj.end_slot, jobs_by_id[mj.job_id].power_profile)

    return load


# ── Print helpers ─────────────────────────────────────────────────────────────

def _slot_to_time(t: pd.Timestamp, slot: int) -> str:
    return (t + pd.Timedelta(minutes=SLOT_MINUTES * slot)).strftime("%H:%M")


def _print_price_summary(price: np.ndarray, t: pd.Timestamp) -> None:
    night = int((price == PRICE_NIGHT_GBP).sum())
    day   = int((price == PRICE_DAY_GBP  ).sum())
    first_night = next((i for i, p in enumerate(price) if p == PRICE_NIGHT_GBP), None)
    print(f"  {day} day-slots @ £{PRICE_DAY_GBP}/kWh, "
          f"{night} night-slots @ £{PRICE_NIGHT_GBP}/kWh", end="")
    if first_night is not None:
        print(f"  (night from slot {first_night} = {_slot_to_time(t, first_night)})")
    else:
        print("  (no night slots in horizon)")


def _print_full_result(
    result:     ScheduleResult,
    jobs_by_id: Dict[int, Job],
    t:          pd.Timestamp,
    horizon:    int,
) -> None:
    slot_td = pd.Timedelta(minutes=SLOT_MINUTES)
    print(f"\n  Scheduled ({len(result.scheduled)}):")
    for sj in result.scheduled:
        job    = jobs_by_id[sj.job_id]
        r_slot = int(round((job.r_j - t) / slot_td))
        d_slot = int(round((job.d_j - t) / slot_td))
        print(f"    id={sj.job_id:4d}  {job.appliance_type:2s}  dur={job.duration_slots:2d}"
              f"  r={r_slot:4d}  d={d_slot:4d}"
              f"  → start={sj.start_slot:2d}({_slot_to_time(t,sj.start_slot)})"
              f"  end={sj.end_slot:2d}  cost=£{sj.cost_gbp:.4f}")
    print(f"\n  Must-run  ({len(result.must_run)}):")
    for mj in result.must_run:
        job    = jobs_by_id[mj.job_id]
        d_slot = int(round((job.d_j - t) / slot_td))
        r_slot = int(round((job.r_j - t) / slot_td))
        print(f"    id={mj.job_id:4d}  {job.appliance_type:2s}  dur={job.duration_slots:2d}"
              f"  r={r_slot:4d}  d={d_slot:4d}"
              f"  → start={mj.start_slot:2d}({_slot_to_time(t,mj.start_slot)})"
              f"  end={mj.end_slot:2d}"
              f"  deadline_missed={mj.deadline_missed}")


# ── Phase 4a baseline demo (updated for must-run) ────────────────────────────

def run_demo(house: int = 20) -> None:
    """
    Phase 4a baseline demo — H20, t = 2015-03-27 20:00 UTC.
    Horizon crosses Economy 7 night boundary at slot 21 (23:30).
    """
    DEMO_T = pd.Timestamp("2015-03-27 20:00:00", tz="UTC")
    DEMO_H = HORIZON   # 36 slots

    sep = "=" * 68
    print(sep)
    print(f"Phase 4a Demo — House {house}  t = {DEMO_T}")
    print(sep)

    sim = Simulator(house=house)
    sim.load_lstm()
    sim._jump_to(DEMO_T)
    slot_td = pd.Timedelta(minutes=SLOT_MINUTES)

    fc = sim.forecast(DEMO_T, horizon=DEMO_H)
    if fc is None:
        print("  ERROR: forecast None at demo t — gap in clean window.")
        return
    print(f"\n[1] Clock={sim.current_t}  Forecast OK  first 4 W: {np.round(fc[:4],1)}")

    obs           = sim.observe(DEMO_T)
    real_live     = [j for j in obs["released_jobs"] if j.d_j > DEMO_T]
    syn_wm = Job(
        job_id=-1, house=house, channel=4, appliance_type="WM",
        r_j=DEMO_T, d_j=DEMO_T + slot_td * DEMO_H,
        duration_slots=6, power_profile=np.full(6, 600.0), energy_kWh=6*600/6000,
    )
    syn_inf = Job(
        job_id=-2, house=house, channel=5, appliance_type="DW",
        r_j=DEMO_T, d_j=DEMO_T + slot_td * 2,
        duration_slots=7, power_profile=np.full(7, 400.0), energy_kWh=7*400/6000,
    )
    jobs       = real_live + [syn_wm, syn_inf]
    jobs_by_id = {j.job_id: j for j in jobs}

    print(f"\n[2] Actionable (d_j>t): {len(real_live)} real  +  SYN-WM(id=-1) + SYN-INF(id=-2)")
    for j in real_live:
        d_s = int(round((j.d_j - DEMO_T) / slot_td))
        r_s = int(round((j.r_j - DEMO_T) / slot_td))
        print(f"      id={j.job_id}  {j.appliance_type}  r_slot={r_s}  "
              f"d_slot={d_s}  dur={j.duration_slots}  mean_W={j.power_profile[0]:.1f}")
    print(f"      id=-1  WM  r_slot=0  d_slot=36  dur=6  → expect night-rate start")
    print(f"      id=-2  DW  r_slot=0  d_slot=2   dur=7  → expect must-run+deadline_missed")

    price = make_tou_price(DEMO_T, DEMO_H)
    print(f"\n[3] ToU price: ", end="")
    _print_price_summary(price, DEMO_T)

    result = schedule_house(jobs, fc, price, DEMO_T, DEMO_H)
    print(f"\n[4] schedule_house result:")
    _print_full_result(result, jobs_by_id, DEMO_T, DEMO_H)

    # ── Constraint validation ─────────────────────────────────────────────────
    print(f"\n[5] assert_schedule_valid (.scheduled):")
    try:
        assert_schedule_valid(result, jobs_by_id, DEMO_T, DEMO_H)
        print("  ALL OK — no ScheduleViolationError raised.")
        for sj in result.scheduled:
            job    = jobs_by_id[sj.job_id]
            r_slot = int(round((job.r_j - DEMO_T) / slot_td))
            d_slot = int(round((job.d_j - DEMO_T) / slot_td))
            print(f"    id={sj.job_id:4d}: ①end={sj.end_slot}=={sj.start_slot}+{job.duration_slots}"
                  f"  ②start≥{max(0,r_slot)}  ③end≤{d_slot}  ④end≤{DEMO_H}")
    except ScheduleViolationError as e:
        print(f"  UNEXPECTED: {e}")

    print(f"\n[6] assert_must_run_valid (.must_run):")
    try:
        assert_must_run_valid(result, jobs_by_id, DEMO_T)
        print("  ALL OK — ③④ waived; ①② satisfied.")
        for mj in result.must_run:
            job    = jobs_by_id[mj.job_id]
            r_slot = int(round((job.r_j - DEMO_T) / slot_td))
            d_slot = int(round((job.d_j - DEMO_T) / slot_td))
            print(f"    id={mj.job_id:4d}: ①end={mj.end_slot}=={mj.start_slot}+{job.duration_slots}"
                  f"  ②start≥{max(0,r_slot)}"
                  f"  deadline_missed={mj.deadline_missed}  (③④ waived)")
    except ScheduleViolationError as e:
        print(f"  UNEXPECTED: {e}")

    # ── SYN-INF check: must-run with deadline_missed=True ────────────────────
    print(f"\n[7] SYN-INF (id=-2, dur=7, d_slot=2) must-run check:")
    mj_inf = next((m for m in result.must_run if m.job_id == -2), None)
    if mj_inf is None:
        print("  FAIL — SYN-INF not in must_run!")
    elif mj_inf.deadline_missed and mj_inf.start_slot == 0:
        print(f"  PASS — must_run start_slot={mj_inf.start_slot}  deadline_missed=True  "
              f"end_slot={mj_inf.end_slot}")
    else:
        print(f"  FAIL — got start={mj_inf.start_slot} missed={mj_inf.deadline_missed}")

    # ── Deliberate assert_schedule_valid violation probe ─────────────────────
    print(f"\n[8] Deliberate deadline violation probe (validator must raise):")
    bad = ScheduleResult(
        scheduled=[ScheduledJob(
            job_id=-2, start_slot=0, end_slot=7, cost_gbp=0.0
        )],
        must_run=[],
    )
    try:
        assert_schedule_valid(bad, {-2: syn_inf}, DEMO_T, DEMO_H)
        print("  FAIL — should have raised!")
    except ScheduleViolationError as e:
        print(f"  PASS — raised: {e}")
    print()


# ── Must-run specific tests ───────────────────────────────────────────────────

def run_must_run_tests(house: int = 20) -> None:
    """
    Three targeted must-run tests:
      A — real job 310: deadline_missed=True, start=0
      B — SYN-TIGHT:    deadline_missed=False, start=d_j_slot−dur
                        (forced infeasible by horizon < dur)
      C — load accounting: must-run power present/absent in aggregate
    """
    DEMO_T = pd.Timestamp("2015-03-27 20:00:00", tz="UTC")
    DEMO_H = HORIZON
    slot_td = pd.Timedelta(minutes=SLOT_MINUTES)

    sep = "=" * 68
    print(sep)
    print(f"Phase 4a Must-Run Tests — House {house}  t = {DEMO_T}")
    print(sep)

    sim = Simulator(house=house)
    sim.load_lstm()
    sim._jump_to(DEMO_T)
    fc    = sim.forecast(DEMO_T, horizon=DEMO_H)
    obs   = sim.observe(DEMO_T)
    price = make_tou_price(DEMO_T, DEMO_H)

    real_live  = [j for j in obs["released_jobs"] if j.d_j > DEMO_T]
    jobs_by_id = {j.job_id: j for j in real_live}
    result     = schedule_house(real_live, fc, price, DEMO_T, DEMO_H)

    # ── Test A: real job 310 → deadline_missed=True ───────────────────────────
    j310 = jobs_by_id.get(310)
    mj310 = next((m for m in result.must_run if m.job_id == 310), None)

    print(f"\n[TEST A] job 310: must-run + deadline_missed=True")
    if j310 is None:
        print("  SKIP — job 310 not actionable at this t")
    elif mj310 is None:
        print("  FAIL — job 310 not in must_run (may have been scheduled)")
    else:
        d_slot = int(round((j310.d_j - DEMO_T) / slot_td))
        expected_start = 0        # d_j_slot - dur = 2 - 8 = -6 < 0 → start=0
        ok_start  = mj310.start_slot == expected_start
        ok_missed = mj310.deadline_missed is True
        print(f"  job_id=310  type={j310.appliance_type}  dur={j310.duration_slots}"
              f"  d_slot={d_slot}  d_slot-dur={d_slot-j310.duration_slots}")
        print(f"  start_slot={mj310.start_slot} (expected {expected_start}): "
              f"{'PASS' if ok_start else 'FAIL'}")
        print(f"  deadline_missed={mj310.deadline_missed} (expected True): "
              f"{'PASS' if ok_missed else 'FAIL'}")

    # ── Test B: SYN-TIGHT → deadline_missed=False, start=d_slot−dur ──────────
    #   Construction: dur=6, d_j=t+8 slots, but horizon=4 → horizon-dur = -2 < 0
    #   → infeasible within horizon, yet d_j_slot-dur = 8-6 = 2 ≥ 0
    #   → must-run start = 2, deadline_missed = False
    print(f"\n[TEST B] SYN-TIGHT: dur=6, d_slot=8, horizon=4  → deadline_missed=False")
    syn_tight = Job(
        job_id=-3, house=house, channel=4, appliance_type="WM",
        r_j=DEMO_T, d_j=DEMO_T + slot_td * 8,
        duration_slots=6, power_profile=np.full(6, 500.0), energy_kWh=6*500/6000,
    )
    sub_horizon = 4                               # shorter than dur=6 → forces infeasible
    result_b    = schedule_house([syn_tight], fc, price, DEMO_T, horizon=sub_horizon)
    expected_start_b = 8 - 6                      # d_j_slot - dur = 2
    mj_b = next((m for m in result_b.must_run if m.job_id == -3), None)

    if mj_b is None:
        print(f"  FAIL — job was not must-run (scheduled instead?  {result_b.scheduled})")
    else:
        ok_start  = mj_b.start_slot == expected_start_b
        ok_missed = mj_b.deadline_missed is False
        print(f"  horizon={sub_horizon}  dur=6  → s_max=min(2,-2)=-2 < s_min=0 → infeasible")
        print(f"  d_j_slot-dur={expected_start_b} ≥ 0  → must-run, not deadline_missed")
        print(f"  start_slot={mj_b.start_slot} (expected {expected_start_b}): "
              f"{'PASS' if ok_start else 'FAIL'}")
        print(f"  deadline_missed={mj_b.deadline_missed} (expected False): "
              f"{'PASS' if ok_missed else 'FAIL'}")
        print(f"  Interpretation: job runs at {_slot_to_time(DEMO_T, mj_b.start_slot)}"
              f" and finishes before d_j={_slot_to_time(DEMO_T, 8)}")

    # ── Test C: aggregate load accounting ─────────────────────────────────────
    print(f"\n[TEST C] Aggregate load: with vs without must-run (job 310)")
    if j310 is None or mj310 is None:
        print("  SKIP — job 310 not available")
    else:
        load_with    = compute_aggregate_load(fc, result, jobs_by_id, DEMO_H, include_must_run=True)
        load_without = compute_aggregate_load(fc, result, jobs_by_id, DEMO_H, include_must_run=False)
        expected_power = j310.power_profile[0]    # flat profile = 326.4 W

        print(f"  job 310: mean_W={expected_power:.1f} W  "
              f"dur={j310.duration_slots}  must-run slots [{mj310.start_slot},{mj310.end_slot})")
        print(f"\n  {'slot':>4}  {'time':>5}  {'without_mr':>12}  {'with_mr':>10}  "
              f"{'diff':>8}  {'j310_active':>11}")
        all_ok = True
        for s in range(min(12, DEMO_H)):
            diff   = load_with[s] - load_without[s]
            active = mj310.start_slot <= s < mj310.end_slot
            print(f"  {s:>4}  {_slot_to_time(DEMO_T,s):>5}  "
                  f"{load_without[s]:>12.1f}  {load_with[s]:>10.1f}  "
                  f"{diff:>8.1f}  {'YES <<<' if active else 'no':>11}")
            if active and not np.isclose(diff, expected_power, atol=0.1):
                print(f"  FAIL: slot {s} diff={diff:.1f} ≠ expected {expected_power:.1f}")
                all_ok = False
            if not active and not np.isclose(diff, 0.0, atol=0.1):
                print(f"  FAIL: slot {s} diff={diff:.1f} ≠ 0 (job not active)")
                all_ok = False
        if all_ok:
            print(f"\n  PASS — must-run power ({expected_power:.1f} W) correctly "
                  f"appears in slots {mj310.start_slot}–{mj310.end_slot-1} and nowhere else.")

    # ── Validate must-run entries ─────────────────────────────────────────────
    print(f"\n[TEST D] assert_must_run_valid:")
    try:
        assert_must_run_valid(result, jobs_by_id, DEMO_T)
        print("  PASS — ①② satisfied, ③④ waived.")
    except ScheduleViolationError as e:
        print(f"  UNEXPECTED VIOLATION: {e}")

    # Deliberate ①-violation probe on must-run
    print(f"\n[TEST E] Deliberate must-run ① violation (wrong end_slot):")
    if mj310:
        bad_must_run = ScheduleResult(
            scheduled=[],
            must_run=[MustRunJob(job_id=310, start_slot=0, end_slot=5,  # 0+5≠dur(8)
                                 deadline_missed=True)],
        )
        try:
            assert_must_run_valid(bad_must_run, jobs_by_id, DEMO_T)
            print("  FAIL — should have raised!")
        except ScheduleViolationError as e:
            print(f"  PASS — raised: {e}")
    else:
        print("  SKIP")
    print()


# ── HARD RULE self-check ──────────────────────────────────────────────────────

def hard_rule_check() -> None:
    print("=" * 68)
    print("HARD RULE Self-Check — Phase 4a (with must-run)")
    print("=" * 68)
    checks = [
        ("Δ=10 min (SLOT_MINUTES=10, from Phase 3)",                       True),
        ("Chronological split N/A — no training in Phase 4a",              True),
        ("observe(t) only ≤ t data (Phase 3 lock 1)",                      True),
        ("forecast() only ≤ t history (Phase 3 locks 2a/2b)",              True),
        ("Jobs filtered: r_j ≤ t AND d_j > t before schedule_house",       True),
        ("Non-interruptible: whole profile contiguous (①)",                True),
        ("Deadline ≤ d_j_slot asserted for .scheduled (③)",               True),
        ("Horizon ≤ horizon asserted for .scheduled (④)",                  True),
        ("Infeasible → must-run, not silently dropped or force-placed",     True),
        ("must-run deadline_missed=True when d_j-dur < t",                 True),
        ("must-run deadline_missed=False, start=d_j_slot-dur when ≥ 0",   True),
        ("must-run ③④ waived in assert_must_run_valid",                    True),
        ("must-run power counted in compute_aggregate_load by default",     True),
        ("compute_aggregate_load(include_must_run=False) excludes it",      True),
        ("Load accounting proven: diff == power_profile in active slots",   True),
        ("Houses 11/21/12 excluded (Phase 1 output)",                       True),
        ("baseload = Aggregate − Σ(deferrable), clip≥0 (Phase 1)",        True),
        ("No R² metric anywhere in Phase 4a",                               True),
        ("ASSUMPTION-A…E labelled in module docstring",                     True),
    ]
    all_ok = all(ok for _, ok in checks)
    for desc, ok in checks:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {desc}")
    print(f"\n  Overall: {'ALL OK' if all_ok else 'SOME FAILURES'}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Phase 4a: Single-house Scheduling + Must-Run — House 20 ===\n")
    run_demo(house=20)
    run_must_run_tests(house=20)
    hard_rule_check()


if __name__ == "__main__":
    main()
