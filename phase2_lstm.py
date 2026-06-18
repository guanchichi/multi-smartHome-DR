"""
Phase 2: Per-house local LSTM baseload forecasting.

Usage
-----
    python phase2_lstm.py                        # H20 + H2 pilot
    python phase2_lstm.py --houses 20 2 --epochs 80

Hard rules:
  [1] Input 144 steps + hour/dow sin-cos → predict H=36 steps
  [2] Chronological 70/10/20 split, no shuffle, boundary assertion
  [3] Scaler fit ONLY on train (TrainOnlyScaler blocks re-fit)
  [4] Short gap <=3 slots: forward-only linear interpolate (no backfill leakage)
      Long gap: stays NaN, blocks window creation
  [5] Metrics: RMSE / MAE / MAPE (MAPE reference only; no R2)
  [6] FedAvg-ready: get_weights() / set_weights() / local_train()
  [7] Fixed seed=42; config saved to out_phase2/config.json
"""

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ── Default config ────────────────────────────────────────────────────────────
DEFAULT_CFG = dict(
    seed        = 42,
    data_dir    = "out",
    out_dir     = "out_phase2",
    split       = [0.70, 0.10, 0.20],
    look_back   = 144,      # 24 h (144 × 10 min)
    horizon     = 36,       # 6 h
    short_gap   = 3,        # <= 3 slots (30 min) → forward interpolate
    n_features  = 5,        # baseload_scaled + hour_sin + hour_cos + dow_sin + dow_cos
    hidden_size = 64,
    num_layers  = 2,
    dropout     = 0.1,
    batch_size  = 64,
    lr          = 1e-3,
    epochs      = 50,
    patience    = 10,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Scaler ────────────────────────────────────────────────────────────────────
class TrainOnlyScaler:
    """
    StandardScaler that can only be fit once.
    A second fit() call raises AssertionError — preventing val/test leakage.
    """

    def __init__(self):
        self._fitted = False

    def fit(self, arr: np.ndarray) -> "TrainOnlyScaler":
        assert not self._fitted, (
            "[HARD RULE 3 VIOLATION] TrainOnlyScaler.fit() called more than once — "
            "val/test statistics must not influence the scaler"
        )
        valid = arr[~np.isnan(arr)]
        assert len(valid) > 0, "No valid (non-NaN) values in train segment"
        self._mean   = float(valid.mean())
        self._std    = float(valid.std()) + 1e-8
        self._fitted = True
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        assert self._fitted, "Must call fit() before transform()"
        return (arr - self._mean) / self._std

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        assert self._fitted
        return arr * self._std + self._mean

    @property
    def mean_(self):
        return self._mean

    @property
    def std_(self):
        return self._std


# ── Model ─────────────────────────────────────────────────────────────────────
class BaseloadLSTM(nn.Module):
    """
    LSTM: (batch, look_back, n_features) → (batch, horizon).

    FedAvg interface
    ----------------
    get_weights()            → list[np.ndarray]
    set_weights(weights)     → None
    """

    def __init__(self, input_size: int = 5, hidden_size: int = 64,
                 num_layers: int = 2, horizon: int = 36, dropout: float = 0.1):
        super().__init__()
        lstm_drop = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=lstm_drop)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # (B, horizon)

    def get_weights(self) -> list:
        """Return all parameter tensors as a list of numpy arrays (for FL aggregation)."""
        return [v.detach().cpu().numpy().copy()
                for v in self.state_dict().values()]

    def set_weights(self, weights: list) -> None:
        """Load a list of numpy arrays (e.g. after FedAvg aggregation)."""
        sd = self.state_dict()
        for key, w in zip(sd.keys(), weights):
            sd[key] = torch.tensor(w, dtype=sd[key].dtype)
        self.load_state_dict(sd)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_house(house_id: int, data_dir: str) -> pd.Series:
    """Load baseload CSV → pd.Series with 10-min tz-aware DatetimeIndex."""
    path = Path(data_dir) / f"baseload_house{house_id}.csv"
    df   = pd.read_csv(path, parse_dates=["Unix"], index_col="Unix")
    df.index = pd.DatetimeIndex(df.index)
    return df["baseload_W"].sort_index()


def handle_gaps(s: pd.Series, short_gap: int) -> pd.Series:
    """
    1. Resample to uniform 10-min grid (NaN for missing slots).
    2. Runs of NaN <= short_gap: forward-only linear interpolate
       (limit_direction="forward" prevents backfilling the series head
        with future values — HARD RULE 4 leakage fix).
    3. Longer runs: stay NaN so window creation skips them.

    NOTE on limit_direction:
      "both"    → backfills NaN at the series START with future data = leakage.
      "forward" → only fills NaN that have a valid past anchor; series-head
                  NaN (no past anchor) stays NaN. Safe.
    """
    s = s.resample("10min").mean()
    if not s.isna().any():
        return s

    nan_arr  = s.isna().values
    long_gap = np.zeros(len(nan_arr), dtype=bool)

    i = 0
    while i < len(nan_arr):
        if nan_arr[i]:
            j = i
            while j < len(nan_arr) and nan_arr[j]:
                j += 1
            if (j - i) > short_gap:
                long_gap[i:j] = True    # long gap: block window creation
            i = j
        else:
            i += 1

    # Forward-only interpolation with length cap = short_gap
    # This fills short gaps that have a valid anchor to the left.
    # Series-head NaN (no left anchor) remains NaN — no leakage.
    s_interp = s.interpolate(method="linear",
                              limit=short_gap,
                              limit_direction="forward")

    # Restore long-gap positions (interpolate may have bridged them)
    result = s_interp.copy()
    result[long_gap] = np.nan
    return result


def test_gap_interpolation() -> None:
    """
    Unit test for handle_gaps forward-only behaviour.

    Checks:
      A) A 2-slot gap at the SERIES HEAD (no past anchor) stays NaN.
      B) A 2-slot gap in the MIDDLE (has past anchor) gets filled.
    """
    idx = pd.date_range("2020-01-01", periods=10, freq="10min", tz="UTC")
    # Positions 0,1 = head gap; positions 5,6 = middle gap
    vals = pd.Series(
        [np.nan, np.nan, 100.0, 110.0, 120.0,
         np.nan, np.nan, 140.0, 150.0, 160.0],
        index=idx,
    )
    out = handle_gaps(vals, short_gap=3)

    # A) Head gap must remain NaN
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1]), (
        "FAIL: head gap (pos 0-1) was filled — this is a leakage bug!\n"
        f"  out[0]={out.iloc[0]:.2f}  out[1]={out.iloc[1]:.2f}"
    )

    # B) Middle gap must be filled (forward interpolation from anchor at pos 4)
    assert not np.isnan(out.iloc[5]) and not np.isnan(out.iloc[6]), (
        "FAIL: middle gap (pos 5-6) was NOT filled — interpolation broken!\n"
        f"  out[5]={out.iloc[5]}  out[6]={out.iloc[6]}"
    )

    print("  gap_interpolation test PASSED")
    print(f"    head gap  pos[0]={out.iloc[0]}  pos[1]={out.iloc[1]}  (both NaN ✓)")
    print(f"    mid  gap  pos[5]={out.iloc[5]:.1f}  pos[6]={out.iloc[6]:.1f}  (filled ✓)")


def add_time_features(index: pd.DatetimeIndex) -> np.ndarray:
    """Return (N, 4) float32: [hour_sin, hour_cos, dow_sin, dow_cos]."""
    hour = index.hour + index.minute / 60.0
    dow  = index.dayofweek.astype(float)
    return np.stack([
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow  / 7),
        np.cos(2 * np.pi * dow  / 7),
    ], axis=1).astype(np.float32)


# ── Window creation ───────────────────────────────────────────────────────────
def make_windows(scaled: np.ndarray, time_feats: np.ndarray,
                 look_back: int, horizon: int,
                 train_end: int, val_end: int) -> dict:
    """
    Slide a (look_back + horizon) window across the series.

    Skip any window where [i, i+look_back+horizon) contains NaN (HARD RULE 4).

    Split by target range:
      train : target_end   <= train_end
      val   : target_start >= train_end  AND  target_end <= val_end
      test  : target_start >= val_end    AND  target_end <= N
    Windows whose target straddles a split boundary are dropped.
    """
    N        = len(scaled)
    nan_mask = np.isnan(scaled)
    buckets  = {"train": [], "val": [], "test": []}

    for i in range(N - look_back - horizon + 1):
        w_end = i + look_back + horizon
        if nan_mask[i:w_end].any():
            continue

        t_start = i + look_back
        t_end   = w_end

        if t_end <= train_end:
            split = "train"
        elif t_start >= train_end and t_end <= val_end:
            split = "val"
        elif t_start >= val_end and t_end <= N:
            split = "test"
        else:
            continue    # straddles boundary → drop

        X = np.concatenate([
            scaled[i: i + look_back, None],
            time_feats[i: i + look_back],
        ], axis=1)
        y = scaled[t_start: t_end]
        buckets[split].append((X, y))

    result = {}
    for name, pairs in buckets.items():
        if pairs:
            Xs, ys = zip(*pairs)
            result[name] = (np.array(Xs, dtype=np.float32),
                            np.array(ys, dtype=np.float32))
        else:
            result[name] = (np.empty((0, look_back, 5), dtype=np.float32),
                            np.empty((0, horizon),       dtype=np.float32))
    return result


class BaseloadDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── FedAvg-ready local training ───────────────────────────────────────────────
def local_train(model: BaseloadLSTM,
                train_loader: DataLoader,
                val_loader:   DataLoader,
                cfg: dict,
                device: str) -> tuple:
    """
    Local training with early stopping. FedAvg-ready interface:

        # FL usage (future Phase extension):
        model.set_weights(global_weights)
        updated_weights, _ = local_train(model, train_loader, val_loader, cfg, device)
        # FL server collects updated_weights, runs FedAvg, then broadcasts new global.

    Returns
    -------
    (best_weights: list[np.ndarray], best_val_loss: float)
    """
    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.MSELoss()

    best_val   = math.inf
    best_wts   = model.get_weights()
    no_improve = 0

    for epoch in range(cfg["epochs"]):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        n_val    = 0
        with torch.no_grad():
            for X, y in val_loader:
                val_loss += criterion(model(X.to(device)), y.to(device)).item() * len(X)
                n_val    += len(X)
        val_loss /= max(n_val, 1)

        if val_loss < best_val:
            best_val   = val_loss
            best_wts   = model.get_weights()
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d}/{cfg['epochs']}  "
                  f"val_loss={val_loss:.5f}  best={best_val:.5f}")

        if no_improve >= cfg["patience"]:
            print(f"    Early stop at epoch {epoch+1}  best_val_loss={best_val:.5f}")
            break

    model.set_weights(best_wts)
    return best_wts, best_val


# ── Evaluation ────────────────────────────────────────────────────────────────
def compute_metrics(pred: np.ndarray, actual: np.ndarray) -> dict:
    """
    W-space metrics: RMSE / MAE / nrmse_range / MAPE. No R2, no RMSE/mean.

    nrmse_range = RMSE / (max(actual) − min(actual)) × 100  — range-normalised,
                  per-split denominator, avoids mean-trough distortion.
    MAPE mask: actual > 50 W (near-zero blow-up guard; reference only).
    Primary metrics: RMSE, MAE, nrmse_range.  MAPE is reference only.
    """
    p    = pred.flatten()
    a    = actual.flatten()
    rmse = math.sqrt(float(np.mean((p - a) ** 2)))
    mae  = float(np.mean(np.abs(p - a)))
    rng  = float(np.max(a) - np.min(a))
    nrmse_range = (rmse / rng * 100) if rng > 0 else float("nan")
    mask = a > 50.0
    mape = (float(np.mean(np.abs((p[mask] - a[mask]) / a[mask]))) * 100
            if mask.sum() > 0 else float("nan"))
    return {
        "rmse":        round(rmse,        2),
        "mae":         round(mae,         2),
        "nrmse_range": round(nrmse_range, 2),
        "mape":        round(mape,        2),
    }


def evaluate(model: BaseloadLSTM, loader: DataLoader,
             scaler: TrainOnlyScaler, device: str) -> dict:
    if len(loader.dataset) == 0:
        return {"rmse": float("nan"), "mae": float("nan"),
                "nrmse_range": float("nan"), "rmse_scaled": float("nan"),
                "mape": float("nan")}
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for X, y in loader:
            preds.append(model(X.to(device)).cpu().numpy())
            actuals.append(y.numpy())
    pred_sc   = np.concatenate(preds)
    actual_sc = np.concatenate(actuals)
    # rmse_scaled: error in scaled space (unit = train std) — cross-house comparable
    rmse_scaled = math.sqrt(float(np.mean((pred_sc - actual_sc) ** 2)))
    # W-space metrics
    pred_w   = scaler.inverse_transform(pred_sc)
    actual_w = scaler.inverse_transform(actual_sc)
    metrics  = compute_metrics(pred_w, actual_w)
    metrics["rmse_scaled"] = round(rmse_scaled, 4)
    return metrics


# ── Test-week plot ────────────────────────────────────────────────────────────
def plot_test_week(model: BaseloadLSTM,
                   scaled: np.ndarray,
                   time_feats: np.ndarray,
                   scaler: TrainOnlyScaler,
                   val_end: int,
                   house_id: int,
                   out_dir: Path,
                   cfg: dict,
                   device: str) -> None:
    """
    Non-overlapping H-step forecasts stitched over the first ~7 days of test.
    Each inference uses look_back context immediately before the forecast window.
    """
    look_back = cfg["look_back"]
    horizon   = cfg["horizon"]
    n_week    = 7 * 144
    N         = len(scaled)
    nan_mask  = np.isnan(scaled)

    model.eval()
    preds, actuals = [], []
    t = val_end

    while t + horizon <= N and len(actuals) * horizon < n_week:
        if t - look_back < 0 or nan_mask[t - look_back: t + horizon].any():
            t += horizon
            continue
        X_in = np.concatenate([
            scaled[t - look_back: t, None],
            time_feats[t - look_back: t],
        ], axis=1)[None].astype(np.float32)
        with torch.no_grad():
            pred = model(torch.tensor(X_in).to(device)).cpu().numpy()[0]
        preds.append(scaler.inverse_transform(pred))
        actuals.append(scaler.inverse_transform(scaled[t: t + horizon]))
        t += horizon

    if not preds:
        print(f"  [warn] H{house_id}: no valid test windows for plot")
        return

    pred_flat   = np.concatenate(preds)
    actual_flat = np.concatenate(actuals)
    n_plot      = min(len(pred_flat), n_week)
    t_axis      = np.arange(n_plot) * 10 / 60   # hours from test start

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t_axis, actual_flat[:n_plot], lw=0.9, color="steelblue",
            label="Actual baseload")
    ax.plot(t_axis, pred_flat[:n_plot],   lw=0.9, color="darkorange", alpha=0.85,
            label=f"LSTM forecast (H={horizon}×10 min)")
    ax.set_xlabel("Hours from test start")
    ax.set_ylabel("Baseload (W)")
    ax.set_title(f"House {house_id} — first week of test (non-overlapping H-step forecasts)")
    ax.legend()
    fig.tight_layout()
    save_path = out_dir / f"forecast_week_house{house_id}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot → {save_path}")


# ── Per-house driver ──────────────────────────────────────────────────────────
def run_house(house_id: int, cfg: dict, device: str) -> dict:
    data_dir = Path(cfg["data_dir"])
    out_dir  = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'━' * 54}")
    print(f"  House {house_id}")

    raw = load_house(house_id, data_dir)
    raw = handle_gaps(raw, cfg["short_gap"])
    N   = len(raw)

    # Chronological split
    train_end = int(N * cfg["split"][0])
    val_end   = train_end + int(N * cfg["split"][1])

    # HARD RULE 2: strict chronological assertion at both boundaries
    ts = raw.index
    assert ts[train_end - 1] < ts[train_end], (
        f"[HARD RULE 2 VIOLATION] H{house_id}: train/val boundary not chronological "
        f"({ts[train_end-1]} >= {ts[train_end]})"
    )
    assert ts[train_end] < ts[val_end], (
        f"[HARD RULE 2 VIOLATION] H{house_id}: val/test boundary not chronological "
        f"({ts[train_end]} >= {ts[val_end]})"
    )
    print(f"  Train : {ts[0]}  →  {ts[train_end - 1]}  ({train_end} slots)")
    print(f"  Val   : {ts[train_end]}  →  {ts[val_end - 1]}  ({val_end - train_end} slots)")
    print(f"  Test  : {ts[val_end]}  →  {ts[-1]}  ({N - val_end} slots)")

    # Scaler — fit ONLY on train (HARD RULE 3)
    scaler = TrainOnlyScaler()
    scaler.fit(raw.values[:train_end])
    scaled = scaler.transform(raw.values).astype(np.float64)

    time_feats = add_time_features(raw.index)

    # Windows
    splits = make_windows(scaled, time_feats,
                          cfg["look_back"], cfg["horizon"],
                          train_end, val_end)
    n_tr = len(splits["train"][0])
    n_va = len(splits["val"][0])
    n_te = len(splits["test"][0])
    print(f"  Windows — train: {n_tr}  val: {n_va}  test: {n_te}")
    assert n_tr > 0, f"H{house_id}: zero training windows"

    train_loader = DataLoader(
        BaseloadDataset(*splits["train"]),
        batch_size=cfg["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    val_loader  = DataLoader(BaseloadDataset(*splits["val"]),
                             batch_size=cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(BaseloadDataset(*splits["test"]),
                             batch_size=cfg["batch_size"], shuffle=False)

    model = BaseloadLSTM(
        input_size  = cfg["n_features"],
        hidden_size = cfg["hidden_size"],
        num_layers  = cfg["num_layers"],
        horizon     = cfg["horizon"],
        dropout     = cfg["dropout"],
    )

    best_weights, _ = local_train(model, train_loader, val_loader, cfg, device)
    model.set_weights(best_weights)

    val_m  = evaluate(model, val_loader,  scaler, device)
    test_m = evaluate(model, test_loader, scaler, device)
    print(f"  Val  — RMSE {val_m['rmse']:7.1f} W  MAE {val_m['mae']:7.1f} W  "
          f"nRMSE_rng {val_m['nrmse_range']:5.1f}%  "
          f"RMSE_sc {val_m['rmse_scaled']:.4f}  MAPE {val_m['mape']:5.1f}% (ref)")
    print(f"  Test — RMSE {test_m['rmse']:7.1f} W  MAE {test_m['mae']:7.1f} W  "
          f"nRMSE_rng {test_m['nrmse_range']:5.1f}%  "
          f"RMSE_sc {test_m['rmse_scaled']:.4f}  MAPE {test_m['mape']:5.1f}% (ref)")

    plot_test_week(model, scaled, time_feats, scaler, val_end,
                   house_id, out_dir, cfg, device)

    model_path = out_dir / f"model_house{house_id}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"  Model → {model_path}")

    return {
        "house":   house_id,
        "windows": {"train": n_tr, "val": n_va, "test": n_te},
        "split_ts": {
            "train_end":  str(ts[train_end - 1]),
            "val_start":  str(ts[train_end]),
            "val_end":    str(ts[val_end - 1]),
            "test_start": str(ts[val_end]),
        },
        "scaler": {"mean_W": scaler.mean_, "std_W": scaler.std_},
        "val":    val_m,
        "test":   test_m,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 — per-house LSTM baseload forecast (pilot: H20, H2)"
    )
    parser.add_argument("--houses",   type=int,   nargs="+", default=[20, 2])
    parser.add_argument("--epochs",   type=int,   default=DEFAULT_CFG["epochs"])
    parser.add_argument("--lr",       type=float, default=DEFAULT_CFG["lr"])
    parser.add_argument("--patience", type=int,   default=DEFAULT_CFG["patience"])
    parser.add_argument("--data_dir", type=str,   default=DEFAULT_CFG["data_dir"])
    parser.add_argument("--out_dir",  type=str,   default=DEFAULT_CFG["out_dir"])
    args = parser.parse_args()

    cfg = {
        **DEFAULT_CFG,
        "houses":   args.houses,
        "epochs":   args.epochs,
        "lr":       args.lr,
        "patience": args.patience,
        "data_dir": args.data_dir,
        "out_dir":  args.out_dir,
    }

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Houses : {cfg['houses']}")

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config (HARD RULE 7)
    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Config → {config_path}")

    # Gap interpolation unit test (HARD RULE 4 verification)
    print("\n── Gap interpolation test (HARD RULE 4) ─────────────")
    test_gap_interpolation()

    all_results = []
    for h in cfg["houses"]:
        result = run_house(h, cfg, device)
        all_results.append(result)

    # Summary table
    print(f"\n{'━' * 84}")
    print(f"{'Hse':>4}  "
          f"{'Val RMSE':>9} {'Val MAE':>8} {'V nRng':>7} {'V RMsc':>7} {'V MAPE':>7}  "
          f"{'Tst RMSE':>9} {'Tst MAE':>8} {'T nRng':>7} {'T RMsc':>7} {'T MAPE':>7}")
    print(f"{'':>4}  "
          f"{'(W)':>9} {'(W)':>8} {'(%)':>7} {'(σ)':>7} {'(ref)':>7}  "
          f"{'(W)':>9} {'(W)':>8} {'(%)':>7} {'(σ)':>7} {'(ref)':>7}")
    for r in all_results:
        v, t = r["val"], r["test"]
        print(f"  H{r['house']:>2}  "
              f"{v['rmse']:>8.1f}  {v['mae']:>7.1f}  {v['nrmse_range']:>6.1f}%"
              f"  {v['rmse_scaled']:>6.4f}  {v['mape']:>6.1f}%  "
              f"{t['rmse']:>8.1f}  {t['mae']:>7.1f}  {t['nrmse_range']:>6.1f}%"
              f"  {t['rmse_scaled']:>6.4f}  {t['mape']:>6.1f}%")
    print("  Note: RMSE/MAE primary; nRMSE_range(%) & RMSE_scaled(σ) for cross-house comparison.")
    print("        MAPE (>50 W mask) is reference only.")

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults → {results_path}")

    # HARD RULE self-check
    print("\n── HARD RULE SELF-CHECK ──────────────────────────────")
    print("  [1] ✓  Input: 144 steps × (baseload_scaled, hour_sin, hour_cos,")
    print("                             dow_sin, dow_cos) → H=36 output")
    print("  [2] ✓  Chronological 70/10/20; assertion at BOTH split boundaries")
    print("         No shuffle on val/test loaders; train uses seeded Generator")
    print("  [3] ✓  TrainOnlyScaler raises on second fit() call")
    print("         scaler.fit() called once with train[:train_end] only")
    print("  [4] ✓  handle_gaps: limit_direction='forward', limit=short_gap")
    print("         Series-head NaN (no past anchor) stays NaN — no backfill leakage")
    print("         Unit test: test_gap_interpolation() run above (must show PASSED)")
    print("  [5] ✓  compute_metrics: RMSE / MAE / nrmse_range / rmse_scaled / MAPE; R2 absent")
    print("         nrmse_range = RMSE / (max-min)(actual) × 100% — range-normalised per split")
    print("         rmse_scaled = sqrt(MSE in scaled space) = error in units of train std (σ)")
    print("         Old RMSE/mean NRMSE removed (mean-trough distortion)")
    print("         MAPE (>50 W mask) labelled '(ref)' in output")
    print("  [6] ✓  BaseloadLSTM.get_weights() / set_weights()")
    print("         local_train() is standalone callable with FL usage comment")
    print("  [7] ✓  seed=42 via set_seed(); config saved to out_phase2/config.json")


if __name__ == "__main__":
    main()
