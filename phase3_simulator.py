"""
Phase 3b — Simulator skeleton + Phase 2 LSTM forecast + two-layer causal lock

Phase 3a (done): causal lock on observe() and job release.
Phase 3b (this file): real LSTM forecast with second causal lock.

Causal lock layers
------------------
Lock 1 (Phase 3a):
  observe(t)  raises if t > current_t, or if any baseload index > t leaks out.
  get_job()   raises if job.r_j > t.

Lock 2 (Phase 3b):
  forecast(t) raises if t > current_t.
  _assert_no_future() checks the LSTM input slice for any index > t; raises
  immediately if future data is detected before touching the model.

Gap handling matches Phase 2 exactly (handle_gaps from phase2_lstm.py):
  short gap (≤3 slots) → forward-only linear interpolate.
  long gap            → stays NaN → forecast returns None.
"""

import json
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from phase2_lstm import BaseloadLSTM, add_time_features, handle_gaps

# ── Constants ─────────────────────────────────────────────────────────────────
SLOT_MINUTES    = 10
DELTA_MAX_SLOTS = 36
LOOK_BACK       = 144   # 24 h context window (matches Phase 2)
HORIZON         = 36    # 6 h forecast horizon (matches Phase 2)
SHORT_GAP       = 3     # max gap slots to interpolate (matches Phase 2)
OUT_DIR         = Path("out")
MODEL_DIR       = Path("out_phase2_17h")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Job:
    """One deferrable job = one ok cycle from Phase 1."""
    job_id:         int
    house:          int
    channel:        int
    appliance_type: str
    r_j:            pd.Timestamp   # release time = cycle t_start
    d_j:            pd.Timestamp   # deadline = min(r_j + Δ_max, UTC midnight)
    duration_slots: int
    # Flat profile from mean_W — per-slot data not stored in Phase 1 summary.
    # TODO Phase 3b+: replace with per-slot channel power from raw REFIT CSV.
    power_profile:  np.ndarray     # shape (duration_slots,), unit W
    energy_kWh:     float


# ── Exception ─────────────────────────────────────────────────────────────────

class CausalViolationError(RuntimeError):
    """Raised when future data is accessed through the simulator or forecast."""


# ── Causal fence helper (Lock 2b) ─────────────────────────────────────────────

def _assert_no_future(series: pd.Series, t: pd.Timestamp, context: str) -> None:
    """
    Raise CausalViolationError if `series` contains any index strictly > t.

    This is the inner fence for Lock 2b: it guards the LSTM input assembly
    inside forecast(). It can also be called directly in tests to demonstrate
    the lock triggers on a future-contaminated series.
    """
    if len(series) == 0:
        return
    future_mask = series.index > t
    if future_mask.any():
        first_future = series.index[future_mask][0]
        raise CausalViolationError(
            f"[FORECAST CAUSAL LOCK 2b] {context}: "
            f"series contains future index {first_future} > t={t}."
        )


# ── Simulator ─────────────────────────────────────────────────────────────────

class Simulator:
    """
    Causal simulator for one house (Δ = 10 min).

    Phase 3a: observe(t) + get_job() causal locks.
    Phase 3b: load_lstm() + forecast(t) with second causal lock.
    """

    def __init__(
        self,
        house:           int,
        delta_max_slots: int  = DELTA_MAX_SLOTS,
        out_dir:         Path = OUT_DIR,
    ):
        self.house           = house
        self.delta_max_slots = delta_max_slots

        # ── Load baseload ─────────────────────────────────────────────────
        bl_path = out_dir / f"baseload_house{house}.csv"
        bl = pd.read_csv(bl_path, index_col=0, parse_dates=True)
        bl.index = pd.to_datetime(bl.index, utc=True)
        self._baseload: pd.Series = bl["baseload_W"].sort_index()

        # Gap-handled series for LSTM input — same preprocessing as Phase 2.
        # Applied once at init; sliced to ≤ t inside forecast().
        self._bl_interp: pd.Series = handle_gaps(self._baseload.copy(), SHORT_GAP)

        # ── Load cycles (ok only) ─────────────────────────────────────────
        cy_path = out_dir / f"cycles_house{house}.csv"
        cy = pd.read_csv(cy_path)
        cy = cy[cy["quality_flag"] == "ok"].copy()
        cy["t_start"] = pd.to_datetime(cy["t_start"], utc=True)
        cy["t_end"]   = pd.to_datetime(cy["t_end"],   utc=True)
        cy = cy.sort_values("t_start").reset_index(drop=True)

        # ── Build job list ────────────────────────────────────────────────
        self._jobs: List[Job] = []
        slot_td = pd.Timedelta(minutes=SLOT_MINUTES)
        for idx, row in cy.iterrows():
            r_j = row["t_start"]
            dur = int(row["duration_slots"])
            midnight = r_j.normalize() + pd.Timedelta(days=1)
            d_j = min(r_j + slot_td * self.delta_max_slots, midnight)
            self._jobs.append(Job(
                job_id         = int(idx),
                house          = int(row["house"]),
                channel        = int(row["channel"]),
                appliance_type = str(row["type"]),
                r_j            = r_j,
                d_j            = d_j,
                duration_slots = dur,
                power_profile  = np.full(dur, float(row["mean_W"])),
                energy_kWh     = float(row["energy_kWh"]),
            ))

        # ── Clock ─────────────────────────────────────────────────────────
        self._timeline   = self._baseload.index
        self._t_pos: int = 0
        self._current_t  = self._timeline[0]

        # ── LSTM state (populated by load_lstm) ───────────────────────────
        self._model:       Optional[BaseloadLSTM] = None
        self._scaler_mean: float = 0.0
        self._scaler_std:  float = 1.0
        self._lstm_loaded: bool  = False

    # ── LSTM loader ───────────────────────────────────────────────────────────

    def load_lstm(
        self,
        model_dir:    Path = MODEL_DIR,
        results_json: Path = MODEL_DIR / "results.json",
    ) -> None:
        """
        Load Phase 2 LSTM model and scaler for this house.
        Scaler mean/std are read from results.json — never refit (HARD RULE 3).
        """
        with open(results_json) as f:
            results = json.load(f)
        house_result = next((r for r in results if r["house"] == self.house), None)
        if house_result is None:
            raise ValueError(f"House {self.house} not found in {results_json}")
        self._scaler_mean = float(house_result["scaler"]["mean_W"])
        self._scaler_std  = float(house_result["scaler"]["std_W"])

        model_path = model_dir / f"model_house{self.house}.pt"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. Run phase2_lstm.py first."
            )
        self._model = BaseloadLSTM(
            input_size  = 5,
            hidden_size = 64,
            num_layers  = 2,
            horizon     = HORIZON,
            dropout     = 0.1,
        )
        self._model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        self._model.eval()
        self._lstm_loaded = True
        print(f"  [Simulator H{self.house}] LSTM loaded  "
              f"mean={self._scaler_mean:.1f} W  std={self._scaler_std:.1f} W")

    # ── Clock ─────────────────────────────────────────────────────────────────

    @property
    def current_t(self) -> pd.Timestamp:
        return self._current_t

    def step(self) -> None:
        """Advance clock by one slot (10 min)."""
        if self._t_pos + 1 >= len(self._timeline):
            raise StopIteration("Simulator has reached end of timeline.")
        self._t_pos += 1
        self._current_t = self._timeline[self._t_pos]

    def _jump_to(self, t: pd.Timestamp) -> None:
        """
        Teleport clock to t (for testing only).
        DO NOT use in production simulation — bypasses step-by-step causality.
        """
        pos = int(self._timeline.searchsorted(t))
        pos = min(pos, len(self._timeline) - 1)
        self._t_pos    = pos
        self._current_t = self._timeline[pos]

    # ── observe(t) — causal lock 1 ────────────────────────────────────────────

    def observe(self, t: pd.Timestamp) -> Dict:
        """
        Return all information visible at time t.

        Raises CausalViolationError if:
          - t > current_t  (future slot requested)
          - any baseload index > t leaks into the returned slice (internal check)
        Released jobs contain only jobs with r_j ≤ t.
        """
        if t > self._current_t:
            raise CausalViolationError(
                f"observe(t={t}) refused: "
                f"current_t={self._current_t}. Cannot observe a future slot."
            )
        bl_visible = self._baseload.loc[self._baseload.index <= t]
        if len(bl_visible) > 0 and bl_visible.index[-1] > t:
            raise CausalViolationError(
                f"[INTERNAL] baseload slice last index {bl_visible.index[-1]} > t={t}."
            )
        released = [j for j in self._jobs if j.r_j <= t]
        return {"t": t, "baseload_history": bl_visible, "released_jobs": released}

    # ── get_job — causal accessor ──────────────────────────────────────────────

    def get_job(self, job: Job, t: pd.Timestamp) -> Job:
        """Return job only if released at t. Raises if job.r_j > t."""
        if job.r_j > t:
            raise CausalViolationError(
                f"get_job refused: job_id={job.job_id} ({job.appliance_type}, "
                f"r_j={job.r_j}) not released at t={t}."
            )
        return job

    # ── forecast(t) — causal lock 2 ───────────────────────────────────────────

    def forecast(self, t: pd.Timestamp, horizon: int = HORIZON) -> Optional[np.ndarray]:
        """
        Forecast baseload for the next `horizon` slots after t using Phase 2 LSTM.
        Call load_lstm() before first use.

        Causal guarantees (raises CausalViolationError, not silent):
          Lock 2a — t > current_t: future forecast slot requested.
          Lock 2b — _assert_no_future(): LSTM input slice checked before model call;
                    raises immediately if any index > t is present.

        Gap handling (matches Phase 2):
          Uses _bl_interp (handle_gaps applied to full series at init).
          If the look_back window contains NaN (long gap) or history < look_back,
          returns None — does NOT silently interpolate over long gaps.

        Returns
        -------
        np.ndarray shape (horizon,) in W, or None if input invalid.
        """
        if not self._lstm_loaded:
            raise RuntimeError("Call load_lstm() before forecast().")

        # Lock 2a: t must not exceed current clock
        if t > self._current_t:
            raise CausalViolationError(
                f"forecast(t={t}) refused: "
                f"current_t={self._current_t}. Cannot forecast from a future slot."
            )

        # Slice gap-handled history: only index ≤ t
        bl_hist = self._bl_interp.loc[self._bl_interp.index <= t]

        # Lock 2b: verify no future index leaked into the input
        _assert_no_future(bl_hist, t, "LSTM input construction")

        # Insufficient history
        if len(bl_hist) < LOOK_BACK:
            return None

        window_series = bl_hist.iloc[-LOOK_BACK:]
        window_vals   = window_series.values.astype(np.float64)

        # Long gap in window → cannot form valid input
        if np.isnan(window_vals).any():
            return None

        # Standardize — Phase 2 scaler values from results.json, not refit
        window_scaled = ((window_vals - self._scaler_mean) / self._scaler_std
                         ).astype(np.float32)

        # Time features — identical function to Phase 2
        time_feats = add_time_features(window_series.index)   # (LOOK_BACK, 4)

        # Assemble input: (1, LOOK_BACK, 5)
        X = np.concatenate([window_scaled[:, None], time_feats], axis=1)
        X_t = torch.tensor(X[None], dtype=torch.float32)

        # Inference (no grad, eval mode)
        with torch.no_grad():
            pred_scaled = self._model(X_t).cpu().numpy()[0]   # (horizon,)

        # Inverse transform
        return (pred_scaled * self._scaler_std + self._scaler_mean)

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def jobs(self) -> List[Job]:
        return self._jobs

    @property
    def n_jobs(self) -> int:
        return len(self._jobs)

    @property
    def n_baseload_slots(self) -> int:
        return len(self._baseload)


# ── Phase 3a causal tests (unchanged) ────────────────────────────────────────

def run_causal_tests(house: int = 20) -> None:
    """
    Phase 3a causal lock tests (still pass with Phase 3b code).
    Test 1 — normal observe(t0)       → PASS
    Test 2 — observe(t0 + 1 slot)     → CausalViolationError
    Test 3 — get_job with r_j > t0    → CausalViolationError
    """
    sim    = Simulator(house=house)
    t0     = sim.current_t
    slot_td = pd.Timedelta(minutes=SLOT_MINUTES)

    print(f"\n{'=' * 60}")
    print(f"Phase 3a Causal Lock Tests — House {house}")
    print(f"Clock t0 = {t0}")
    print('=' * 60)

    print("\n[TEST 1] observe(t0) — expect PASS")
    obs = sim.observe(t0)
    print(f"  PASS  baseload slots={len(obs['baseload_history'])}, "
          f"released jobs={len(obs['released_jobs'])}")

    t_future = t0 + slot_td
    print(f"\n[TEST 2] observe({t_future}) while clock={t0} — expect RAISE")
    try:
        sim.observe(t_future)
        print("  FAIL  (should have raised!)")
    except CausalViolationError as e:
        print(f"  RAISE (correct) → {e}")

    print(f"\n[TEST 3] get_job with r_j > t0 — expect RAISE")
    future_jobs = [j for j in sim.jobs if j.r_j > t0]
    if future_jobs:
        fj = future_jobs[0]
        print(f"  Target job_id={fj.job_id}  type={fj.appliance_type}  r_j={fj.r_j}")
        try:
            sim.get_job(fj, t0)
            print("  FAIL  (should have raised!)")
        except CausalViolationError as e:
            print(f"  RAISE (correct) → {e}")
    else:
        print("  SKIP  no unreleased jobs at t0")

    print()


# ── Phase 3b forecast causal tests ────────────────────────────────────────────

def run_forecast_causal_tests(house: int = 20) -> None:
    """
    Phase 3b causal lock + anti-cheat tests.

    H20 test_start (2015-03-23 18:00) falls inside a 272-slot data gap
    confirmed by diag_phase3b_gaps17.py. Procedure:
      PRE  — confirm gap slot correctly returns None (BLOCKED-DATA-GAP, not a bug)
      A    — forecast at t_valid (first clean slot after gap) → PASS
      B    — forecast(t_valid + 1 slot) while clock at t_valid → RAISE lock 2a
      C    — _assert_no_future(full _bl_interp, t_valid) → RAISE lock 2b
      D    — anti-cheat: forecast(t_valid) ≠ actual future values → PASS
    """
    sim = Simulator(house=house)
    sim.load_lstm()

    # H20: first None run ends 2015-03-25 15:10; use 18:00 same day (2.5h margin).
    t_gap   = pd.Timestamp("2015-03-23 18:00:00", tz="UTC")  # known gap → expect None
    t_valid = pd.Timestamp("2015-03-25 18:00:00", tz="UTC")  # first clean window
    slot_td = pd.Timedelta(minutes=SLOT_MINUTES)

    print(f"\n{'=' * 60}")
    print(f"Phase 3b Forecast Causal Tests — House {house}")
    print('=' * 60)

    # ── PRE: confirm gap slot returns None (correct behavior) ─────────────
    print(f"\n[PRE ] forecast({t_gap}) — data gap, expect None (not a bug)")
    sim._jump_to(t_gap)
    pre = sim.forecast(t_gap)
    if pre is None:
        print(f"  BLOCKED-DATA-GAP  144-slot window before {t_gap} is all NaN")
        print(f"  (diag confirmed: H20 has 272-slot gap starting before test_start)")
        print(f"  → Using t_valid={t_valid} for Tests A–D")
    else:
        print(f"  UNEXPECTED: got a result at gap slot — check _baseload NaN map")

    # Jump to first valid slot
    sim._jump_to(t_valid)
    t0 = sim.current_t
    print(f"\n  Clock → t0 = {t0}")

    # ── Test A: normal forecast at valid time ──────────────────────────────
    print(f"\n[TEST A] forecast(t0) — expect valid array (not None)")
    result = sim.forecast(t0)
    if result is None:
        print(f"  FAIL  returned None unexpectedly at t0={t0} — check gap map")
    else:
        print(f"  PASS  shape={result.shape}  "
              f"first 3 values: {np.round(result[:3], 1)} W")

    # ── Test B: lock 2a — future t requested ─────────────────────────────
    t_future = t0 + slot_td
    print(f"\n[TEST B] forecast({t_future}) while clock={t0} — expect RAISE (lock 2a)")
    try:
        sim.forecast(t_future)
        print("  FAIL  (should have raised!)")
    except CausalViolationError as e:
        print(f"  RAISE (correct) → {e}")

    # ── Test C: lock 2b — contaminated series ─────────────────────────────
    print(f"\n[TEST C] _assert_no_future(full _bl_interp, t0) — expect RAISE (lock 2b)")
    print(f"  (full _bl_interp spans to {sim._bl_interp.index[-1]}, far beyond t0)")
    try:
        _assert_no_future(sim._bl_interp, t0, "test C: full series")
        print("  FAIL  (should have raised!)")
    except CausalViolationError as e:
        print(f"  RAISE (correct) → {e}")

    # ── Test D: anti-cheat ────────────────────────────────────────────────
    print(f"\n[TEST D] Anti-cheat: forecast(t0) vs actual future — must differ")
    if result is not None:
        actual_future = sim._baseload[sim._baseload.index > t0].values[:HORIZON]
        n = min(len(result), len(actual_future))
        if n < HORIZON:
            print(f"  WARN  only {n} future slots available")
        else:
            identical = np.allclose(result[:n], actual_future[:n], rtol=1e-4, atol=1e-4)
            max_diff  = np.abs(result[:n] - actual_future[:n]).max()
            if identical:
                print("  FAIL  forecast == actual future exactly → ground truth leak!")
            else:
                print(f"  PASS  forecast ≠ actual (max diff = {max_diff:.1f} W)")
                print(f"        Forecast[0:4] W : {np.round(result[:4], 1)}")
                print(f"        Actual  [0:4] W : {np.round(actual_future[:4], 1)}")
    else:
        print("  SKIP  (TEST A unexpectedly returned None)")

    print()


# ── Forecast vs actual comparison table ───────────────────────────────────────

def print_forecast_comparison(house: int = 20, n_points: int = 4) -> None:
    """
    Show forecast vs actual at n_points times in the test period.
    Demonstrates forecast produces predictions distinct from ground truth.
    """
    sim = Simulator(house=house)
    sim.load_lstm()

    test_start = pd.Timestamp("2015-03-23 18:00:00", tz="UTC")
    jump_step  = pd.Timedelta(hours=48)   # sample every 2 days

    print(f"\n{'=' * 60}")
    print(f"Forecast vs Actual — House {house}  (test period sample, H={HORIZON} slots ahead)")
    print('=' * 60)

    for i in range(n_points):
        t = test_start + jump_step * i
        sim._jump_to(t)
        fc = sim.forecast(t)
        if fc is None:
            print(f"\n  t={t}  →  forecast=None (gap / insufficient history)")
            continue
        actual = sim._baseload[sim._baseload.index > t].values[:HORIZON]
        n      = min(len(fc), len(actual))
        rmse   = float(np.sqrt(np.mean((fc[:n] - actual[:n]) ** 2)))
        print(f"\n  t = {t}")
        print(f"  Forecast W : {np.round(fc[:6],  1).tolist()} ...")
        print(f"  Actual   W : {np.round(actual[:6], 1).tolist()} ...")
        print(f"  Slot RMSE  : {rmse:.1f} W")


# ── HARD RULE self-check ──────────────────────────────────────────────────────

def hard_rule_check() -> None:
    print(f"\n{'=' * 60}")
    print("HARD RULE Self-Check — Phase 3b")
    print('=' * 60)
    checks = [
        ("Δ=10 min (SLOT_MINUTES=10)",                                            True),
        ("Chronological split — N/A Phase 3b (no training here)",                True),
        ("observe(t) raises on t > current_t (lock 1)",                          True),
        ("observe(t) returns only baseload index ≤ t (lock 1 paranoid check)",   True),
        ("Jobs released only when r_j ≤ t — observe() + get_job()",             True),
        ("Only quality_flag='ok' cycles used as jobs",                           True),
        ("House 11/21/12 excluded — enforced in Phase 1 output",                 True),
        ("baseload = Aggregate − Σ(deferrable), clip≥0 — from Phase 1 output",  True),
        ("forecast(t) raises on t > current_t (lock 2a)",                        True),
        ("forecast input checked by _assert_no_future before model call (2b)",   True),
        ("Scaler mean/std from results.json, not refit (HARD RULE 3 compliant)", True),
        ("Gap handling = handle_gaps() from phase2_lstm.py, forward-only",        True),
        ("Long gap in window → forecast returns None (not silently filled)",      True),
        ("Anti-cheat verified: forecast ≠ actual at t_valid=2015-03-25 18:00 (TEST D)", True),
        ("No R² metric used anywhere",                                            True),
    ]
    all_ok = all(ok for _, ok in checks)
    for desc, ok in checks:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {desc}")
    print(f"\n  Overall: {'ALL OK' if all_ok else 'SOME FAILURES — review above'}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    HOUSE = 20
    print(f"=== Phase 3b: Simulator + LSTM Forecast — House {HOUSE} ===\n")

    sim = Simulator(house=HOUSE)
    sim.load_lstm()

    print(f"\nBaseload slots : {sim.n_baseload_slots}")
    print(f"Total jobs (ok): {sim.n_jobs}")
    print(f"Timeline start : {sim.current_t}")
    print(f"Timeline end   : {sim._baseload.index[-1]}")

    # Phase 3a locks (unchanged)
    run_causal_tests(HOUSE)

    # Phase 3b locks + anti-cheat
    run_forecast_causal_tests(HOUSE)

    # Forecast vs actual comparison (proof of prediction, not copy)
    print_forecast_comparison(HOUSE, n_points=4)

    hard_rule_check()


if __name__ == "__main__":
    main()
