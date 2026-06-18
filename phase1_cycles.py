"""
Phase 1 — REFIT (官方 CLEAN 版) deferrable-cycle 抽取
====================================================
load_refit() -> resample_10min() -> extract_cycles() -> 每戶 cycle table + 健全性圖

控制層解析度 Δ = 10 min (144 slots/day)。
功率單位全程 Watt;能量換算 energy_kWh = Σ(slot_avg_W) / 6000
  (每格 = avg_W * (10/60)h, 再 /1000 -> kWh)

抽取順序:**先合併短 gap，再過濾長度**。順序反了會把洗碗/泡水段切碎。

用法:
  python phase1_cycles.py --data-dir /path/to/CLEAN_REFIT --out ./out
  python phase1_cycles.py --selftest          # 用合成資料跑通,不需真實檔案
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DELTA_MIN = 10                      # 控制層解析度 (分鐘)
SLOTS_PER_DAY = 24 * 60 // DELTA_MIN

# channel 0 = Aggregate;1..9 對應 CSV 欄位 Appliance1..Appliance9
# type: WM 洗衣 / DW 洗碗 / TD 滾筒烘衣 / WD 洗烘一體 / DR 烘衣(同 TD)
DEFERRABLE_MAP: dict[int, dict[int, str]] = {
    1:  {4: "TD", 5: "WM", 6: "DW"},
    2:  {2: "WM", 3: "DW"},
    3:  {4: "TD", 5: "DW", 6: "WM"},
    4:  {4: "WM", 5: "WM"},
    5:  {3: "WM", 4: "DW"},             # ch2 confirmed dehumidifier -> removed
    6:  {2: "WM", 3: "DW"},
    7:  {4: "TD", 5: "WM", 6: "DW"},   # ch6 簽章變更 @2014-05-20
    8:  {3: "DR", 4: "WM"},
    9:  {2: "WD", 3: "WM", 4: "DW"},
    10: {5: "WM", 6: "DW"},
    11: {3: "WM", 4: "DW"},            # SOLAR -> 主實驗排除
    13: {3: "WM", 4: "DW", 5: "TD"},   # ch5 簽章不穩
    15: {2: "TD", 3: "WM", 4: "DW"},
    16: {5: "WM", 6: "DW"},
    17: {3: "TD", 4: "WM"},
    18: {4: "WD", 5: "WM", 6: "DW"},
    19: {2: "WM"},
    20: {3: "TD", 4: "WM", 5: "DW"},
    21: {2: "TD", 3: "WM", 4: "DW"},   # SOLAR -> 主實驗排除
}
# House 12: 零 deferrable -> 不在 map。11/21: Aggregate 受太陽能污染。
EXCLUDE_HOUSES = {12, 11, 21}

# 五類 cycle 抽取參數 (10-min 平均功率)
#   threshold[W], min_len[格], max_len[格,超過標 anomaly], merge_gap[格]
PARAMS = {
    "WM": dict(threshold=40,  min_len=3,  max_len=22, merge_gap=1),
    "DW": dict(threshold=40,  min_len=4,  max_len=20, merge_gap=2),
    "TD": dict(threshold=80,  min_len=3,  max_len=16, merge_gap=1),
    "WD": dict(threshold=40,  min_len=6,  max_len=30, merge_gap=3),
    "DR": dict(threshold=80,  min_len=3,  max_len=16, merge_gap=1),
}

# 污染期處理:cut_after=丟棄該日後 cycle;flag_after=保留但標記;flag_all=全標記
CONTAMINATION = {
    (7, 6):  {"action": "flag_after", "date": "2014-05-20", "reason": "sig_change"},
    (13, 5): {"action": "flag_all",                          "reason": "unstable_sig"},
}

# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
def load_refit(house: int, data_dir: str) -> pd.DataFrame:
    """讀 CLEAN_House{N}.csv,回傳 W、DatetimeIndex、欄位 Aggregate + Appliance1..9。"""
    path = os.path.join(data_dir, f"CLEAN_House{house}.csv")
    cols = ["Unix", "Aggregate"] + [f"Appliance{i}" for i in range(1, 10)]
    df = pd.read_csv(path, usecols=cols)
    df.index = pd.to_datetime(df["Unix"], unit="s", utc=True)
    df = df.drop(columns=["Unix"]).sort_index()
    return df


def resample_10min(df: pd.DataFrame) -> pd.DataFrame:
    """8s -> 10-min 平均功率 (W)。空 bin 為 NaN (後續視為 off)。"""
    return df.resample(f"{DELTA_MIN}min").mean()


# ----------------------------------------------------------------------------
# CYCLE EXTRACTION
# ----------------------------------------------------------------------------
def merge_short_gaps(on: np.ndarray, max_gap: int) -> np.ndarray:
    """把被 True 夾住、長度 <= max_gap 的 False 段補成 True(洗衣泡水 / 洗碗段間)。"""
    on = on.copy()
    n = len(on)
    i = 0
    while i < n:
        if not on[i]:
            j = i
            while j < n and not on[j]:
                j += 1
            if i > 0 and j < n and on[i - 1] and on[j] and (j - i) <= max_gap:
                on[i:j] = True
            i = j
        else:
            i += 1
    return on


def runs_of_true(on: np.ndarray):
    """回傳所有連續 True 區段 (start, end_exclusive)。"""
    runs, n, i = [], len(on), 0
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def extract_cycles(series: pd.Series, house: int, channel: int, ctype: str) -> list[dict]:
    """單一 deferrable channel 的 10-min 序列 -> cycle 清單。"""
    p = series.to_numpy(dtype=float)
    p_on = np.nan_to_num(p, nan=0.0)            # NaN bin 視為 off
    par = PARAMS[ctype]
    on = p_on >= par["threshold"]
    on = merge_short_gaps(on, par["merge_gap"]) # 先合併
    idx = series.index

    cycles = []
    for s, e in runs_of_true(on):               # 再過濾
        L = e - s
        if L < par["min_len"]:
            continue
        prof = p_on[s:e]
        flag = "anomaly_long" if L > par["max_len"] else "ok"
        cycles.append(dict(
            house=house, channel=channel, type=ctype,
            t_start=idx[s], t_end=idx[e - 1],
            duration_slots=int(L), duration_min=int(L * DELTA_MIN),
            energy_kWh=float(prof.sum() / 6000.0),
            peak_W=float(prof.max()), mean_W=float(prof.mean()),
            quality_flag=flag,
        ))
    return cycles


def apply_contamination(cycles: list[dict], house: int, channel: int) -> list[dict]:
    rule = CONTAMINATION.get((house, channel))
    if not rule:
        return cycles
    action = rule["action"]
    if action == "flag_all":
        for c in cycles:
            c["quality_flag"] = rule["reason"]
        return cycles
    cutoff = pd.Timestamp(rule["date"], tz="UTC")
    if action == "cut_after":
        return [c for c in cycles if c["t_start"] < cutoff]
    if action == "flag_after":
        for c in cycles:
            if c["t_start"] >= cutoff and c["quality_flag"] == "ok":
                c["quality_flag"] = rule["reason"]
        return cycles
    return cycles


# ----------------------------------------------------------------------------
# PER-HOUSE DRIVER
# ----------------------------------------------------------------------------
def process_house(house: int, data_dir: str) -> tuple[pd.DataFrame, pd.Series]:
    """回傳 (該戶 cycle table, baseload 序列 W)。"""
    df = load_refit(house, data_dir)
    p10 = resample_10min(df)

    defmap = DEFERRABLE_MAP[house]
    deferr_cols = [f"Appliance{ch}" for ch in defmap]
    # baseload = 量測 aggregate - Σ(deferrable);clip>=0 (REFIT aggregate != 子表加總)
    baseload = (p10["Aggregate"] - p10[deferr_cols].sum(axis=1)).clip(lower=0)
    baseload.name = "baseload_W"

    rows = []
    for ch, ctype in defmap.items():
        cyc = extract_cycles(p10[f"Appliance{ch}"], house, ch, ctype)
        cyc = apply_contamination(cyc, house, ch)
        rows.extend(cyc)
    table = pd.DataFrame(rows)
    return table, baseload


# ----------------------------------------------------------------------------
# SANITY / OUTPUT
# ----------------------------------------------------------------------------
def summarize(all_cycles: pd.DataFrame) -> pd.DataFrame:
    if all_cycles.empty:
        return pd.DataFrame()
    g = all_cycles.groupby(["house", "type"])
    summ = g.agg(
        n_cycles=("duration_min", "size"),
        median_dur_min=("duration_min", "median"),
        median_kWh=("energy_kWh", "median"),
        anomaly_pct=("quality_flag", lambda s: 100.0 * (s == "anomaly_long").mean()),
    ).reset_index()
    return summ


def plot_hists(all_cycles: pd.DataFrame, out_dir: str):
    if all_cycles.empty:
        return
    types = [t for t in ["WM", "DW", "TD", "WD", "DR"] if t in set(all_cycles["type"])]
    for metric, fname, xlabel in [
        ("duration_min", "hist_duration_by_type.png", "cycle duration (min)"),
        ("energy_kWh",   "hist_energy_by_type.png",   "cycle energy (kWh)"),
    ]:
        fig, axes = plt.subplots(1, len(types), figsize=(3.2 * len(types), 3), squeeze=False)
        for ax, t in zip(axes[0], types):
            ax.hist(all_cycles.loc[all_cycles["type"] == t, metric].dropna(), bins=25)
            ax.set_title(t); ax.set_xlabel(xlabel); ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=110)
        plt.close(fig)


def run(houses: list[int], data_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    all_tables = []
    for h in houses:
        if h in EXCLUDE_HOUSES or h not in DEFERRABLE_MAP:
            print(f"[skip] House {h}")
            continue
        table, baseload = process_house(h, data_dir)
        table.to_csv(os.path.join(out_dir, f"cycles_house{h}.csv"), index=False)
        baseload.to_frame().to_csv(os.path.join(out_dir, f"baseload_house{h}.csv"))
        all_tables.append(table)
        print(f"[ok]   House {h}: {len(table)} cycles")
    all_cycles = pd.concat(all_tables, ignore_index=True) if all_tables else pd.DataFrame()
    all_cycles.to_csv(os.path.join(out_dir, "cycles_all.csv"), index=False)
    summ = summarize(all_cycles)
    summ.to_csv(os.path.join(out_dir, "summary_by_type.csv"), index=False)
    plot_hists(all_cycles, out_dir)
    print("\n=== summary_by_type ===")
    print(summ.to_string(index=False) if not summ.empty else "(empty)")
    return all_cycles, summ


# ----------------------------------------------------------------------------
# SELF-TEST (合成資料,驗證 pipeline 可跑;非真實流程)
# ----------------------------------------------------------------------------
def _synth_house(house: int, defmap: dict[int, str], out_dir: str, days: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed + house)
    step = 8  # 秒
    n = days * 86400 // step
    t0 = pd.Timestamp("2014-10-01", tz="UTC")
    unix = (t0.value // 10**9) + np.arange(n) * step

    # 連續 baseload (fridge 循環 + 雜訊),非 deferrable
    base = 80 + 30 * np.sin(np.arange(n) / 450.0) + rng.normal(0, 8, n)
    chans = {f"Appliance{i}": np.full(n, 2.0) + rng.normal(0, 0.5, n) for i in range(1, 10)}

    # 注入 cycle:每類給 days*0.7 次,長度/功率隨型別
    spec = {"WM": (45, 1800), "DW": (90, 2000), "TD": (60, 2200),
            "WD": (140, 2000), "DR": (60, 2200)}
    injected = {}
    for ch, ctype in defmap.items():
        dur_min, pk = spec[ctype]
        col = f"Appliance{ch}"; cnt = 0
        for d in range(days):
            if rng.random() < 0.7:
                start_min = int(rng.integers(8 * 60, 21 * 60))
                s = (d * 86400 + start_min * 60) // step
                L = int((dur_min + rng.integers(-10, 10)) * 60 // step)
                if s + L >= n:
                    continue
                # 高原 + 兩個加熱尖峰,中間穿插低耗(模擬泡水/段間)
                prof = np.full(L, 120.0)
                prof[: L // 5] = pk                      # 初段加熱
                prof[L // 2 : L // 2 + L // 8] = pk*0.9  # 中段加熱
                soak = slice(L // 5, L // 5 + max(1, L // 12))
                prof[soak] = 5.0                          # 低耗泡水段 (測 merge_gap)
                chans[col][s : s + L] = prof + rng.normal(0, 15, L)
                cnt += 1
        injected[(ch, ctype)] = cnt

    agg = base + sum(chans.values())
    df = pd.DataFrame({"Unix": unix, "Aggregate": agg, **chans})
    df.to_csv(os.path.join(out_dir, f"CLEAN_House{house}.csv"), index=False)
    return injected


def selftest():
    import tempfile
    tmp = tempfile.mkdtemp(prefix="refit_synth_")
    out = os.path.join(tmp, "out")
    test_houses = {20: DEFERRABLE_MAP[20], 2: DEFERRABLE_MAP[2], 18: DEFERRABLE_MAP[18]}
    inj_total = {}
    for h, dm in test_houses.items():
        inj = _synth_house(h, dm, tmp)
        for (ch, ct), c in inj.items():
            inj_total[(h, ch, ct)] = c
    print("synthetic dir:", tmp)
    all_cycles, summ = run(list(test_houses), tmp, out)
    # 對照注入數 vs 抽出數 (含 anomaly)
    print("\n=== injected vs extracted ===")
    det = all_cycles.groupby(["house", "channel", "type"]).size().to_dict()
    for (h, ch, ct), inj in sorted(inj_total.items()):
        got = det.get((h, ch, ct), 0)
        print(f"  House{h:>2} ch{ch} {ct}: injected {inj} -> extracted {got}")
    print("\noutputs in:", out)
    return tmp, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None, help="CLEAN_REFIT 目錄")
    ap.add_argument("--out", default="./out")
    ap.add_argument("--houses", default=None, help="逗號分隔,如 1,2,3;預設全部 map")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
    else:
        if not args.data_dir:
            raise SystemExit("需 --data-dir 或 --selftest")
        houses = ([int(x) for x in args.houses.split(",")]
                  if args.houses else sorted(DEFERRABLE_MAP))
        run(houses, args.data_dir, args.out)
