"""
diag_raw_power.py — H5 ch2/ch3、H8 ch3 原始功率曲線診斷
  目的:
    (a) 判斷 H5 整戶壞掉的日期範圍(ch2=TD/dehumidifier, ch3=WM)
    (b) H8 DR(ch3) 是否被 threshold=80W 截掉冷卻段
  輸出: out_diag/ 下若干 PNG
用法:
  python diag_raw_power.py --data-dir data/raw
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 重點觀察日期範圍 ──────────────────────────────────────────
# H5: 月份全覽(每月抽一週)、加上報告點名的異常時段
H5_WINDOWS = [
    # label,                  start,              end
    ("H5 2013-10 overview",   "2013-10-01",       "2013-10-31"),  # 正常期參考
    ("H5 2014-07-18~22",      "2014-07-18",       "2014-07-23"),  # 88-slot TD anomaly
    ("H5 2014-08-09~13",      "2014-08-09",       "2014-08-13"),  # 72-slot TD anomaly
    ("H5 2014-08-14~18",      "2014-08-14",       "2014-08-18"),  # 48-slot TD anomaly
    ("H5 2014-08-29~09-03",   "2014-08-29",       "2014-09-03"),  # 278-slot TD + 280-slot WM
    ("H5 2014-09-14~18",      "2014-09-14",       "2014-09-18"),  # 70-slot TD, 69/76-slot WM
    ("H5 2014-10-12 early",   "2013-10-10",       "2013-10-15"),  # 17-slot TD 最早出現
    ("H5 2014-11-15~25",      "2014-11-15",       "2014-11-25"),  # 跨 cut_after 前後
]

# H8: 找幾個完整的 DR cycle 看冷卻段
H8_WINDOWS = [
    ("H8 2013-10",            "2013-10-01",       "2013-10-31"),
    ("H8 2014-03",            "2014-03-01",       "2014-03-31"),
    ("H8 2014-06",            "2014-06-01",       "2014-06-30"),
]

THRESHOLDS = {"TD": 80, "WM": 40, "DR": 80}
COLORS     = {"ch2": "#d62728", "ch3": "#1f77b4", "ch4": "#2ca02c"}


def load_10min(house: int, data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, f"CLEAN_House{house}.csv")
    cols = ["Unix", "Aggregate"] + [f"Appliance{i}" for i in range(1, 10)]
    df = pd.read_csv(path, usecols=cols)
    df.index = pd.to_datetime(df["Unix"], unit="s", utc=True)
    df = df.drop(columns=["Unix"]).sort_index()
    return df.resample("10min").mean()


def load_raw(house: int, data_dir: str) -> pd.DataFrame:
    """Load raw 8-second resolution for zoom-in plots."""
    path = os.path.join(data_dir, f"CLEAN_House{house}.csv")
    cols = ["Unix"] + [f"Appliance{i}" for i in range(1, 10)]
    df = pd.read_csv(path, usecols=cols)
    df.index = pd.to_datetime(df["Unix"], unit="s", utc=True)
    return df.drop(columns=["Unix"]).sort_index()


def plot_window(ax, series_10min, label: str, threshold: float, color: str):
    ax.plot(series_10min.index, series_10min.values, lw=0.8, color=color, label=label)
    ax.axhline(threshold, color=color, lw=0.6, ls="--", alpha=0.7,
               label=f"threshold={threshold}W")
    ax.set_ylabel("W")
    ax.legend(fontsize=7, loc="upper right")


def save_fig(fig, out_dir: str, fname: str):
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


# ── H5 診斷 ───────────────────────────────────────────────────
def diag_h5(data_dir: str, out_dir: str):
    print("[H5] loading 10-min resampled data...")
    p = load_10min(5, data_dir)
    ch2 = p["Appliance2"]   # TD / dehumidifier
    ch3 = p["Appliance3"]   # WM

    for title, t0, t1 in H5_WINDOWS:
        s = pd.Timestamp(t0, tz="UTC")
        e = pd.Timestamp(t1, tz="UTC")
        seg2 = ch2.loc[s:e]
        seg3 = ch3.loc[s:e]
        if len(seg2) == 0:
            print(f"  [skip] {title}: no data")
            continue

        fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
        fig.suptitle(f"House 5  {title}", fontsize=10)

        plot_window(axes[0], seg2, "ch2 (TD/dehumidifier)", THRESHOLDS["TD"], COLORS["ch2"])
        axes[0].set_title("ch2 — classified as TD (切掉點 2014-11-21 後)", fontsize=8)

        plot_window(axes[1], seg3, "ch3 (WM)", THRESHOLDS["WM"], COLORS["ch3"])
        axes[1].set_title("ch3 — WM", fontsize=8)

        # 標 cut_after 線
        cut = pd.Timestamp("2014-11-21", tz="UTC")
        if s <= cut <= e:
            for ax in axes:
                ax.axvline(cut, color="red", lw=1.2, ls=":", label="cut_after 2014-11-21")

        fig.autofmt_xdate()
        fig.tight_layout()
        fname = "h5_" + title.replace(" ", "_").replace("/", "-") + ".png"
        save_fig(fig, out_dir, fname)

    # ── 月度全覽: 每月最大功率 + 異常次數 ────────────────────
    print("[H5] monthly anomaly summary (slots > 16 for ch2, > 18 for ch3)...")
    for col, thr, max_len, tag in [
        ("Appliance2", 80, 16, "TD"),
        ("Appliance3", 40, 22, "WM-new"),
    ]:
        series = p[col]
        on = (series.fillna(0) >= thr).to_numpy()
        # simple run-length
        runs = []
        n, i = len(on), 0
        while i < n:
            if on[i]:
                j = i
                while j < n and on[j]:
                    j += 1
                runs.append(j - i)
                i = j
            else:
                i += 1
        long_runs = [r for r in runs if r > max_len]
        months = series.resample("ME").apply(lambda x: (x.fillna(0) >= thr).sum())
        print(f"  {col}({tag}): total runs={len(runs)}, >max_len={len(long_runs)}, "
              f"longest={max(runs) if runs else 0} slots")
        # print monthly on-slots as a quick proxy for activity level
        print(f"    on-slots/month (first 24): {months.values[:24].tolist()}")


# ── H8 DR 診斷 ────────────────────────────────────────────────
def diag_h8(data_dir: str, out_dir: str):
    print("[H8] loading 10-min + raw data...")
    p10  = load_10min(8, data_dir)
    raw  = load_raw(8, data_dir)
    ch3_10  = p10["Appliance3"]   # DR
    ch3_raw = raw["Appliance3"]

    # ── 全期 10-min 覽圖 ──────────────────────────────────────
    for title, t0, t1 in H8_WINDOWS:
        s = pd.Timestamp(t0, tz="UTC")
        e = pd.Timestamp(t1, tz="UTC")
        seg = ch3_10.loc[s:e]
        if len(seg) == 0:
            print(f"  [skip] {title}: no data")
            continue
        fig, ax = plt.subplots(figsize=(14, 3))
        fig.suptitle(f"House 8 ch3 (DR)  {title}", fontsize=10)
        plot_window(ax, seg, "ch3 DR (10-min avg)", THRESHOLDS["DR"], COLORS["ch3"])
        fig.autofmt_xdate()
        fig.tight_layout()
        fname = "h8_" + title.replace(" ", "_").replace("/", "-") + ".png"
        save_fig(fig, out_dir, fname)

    # ── 每個 DR cycle 的細部原始曲線 (raw 8s) ─────────────────
    # 找 10-min 段裡 >= 80W 的 run，再往前後延 30min 抓 raw 顯示
    on = (ch3_10.fillna(0) >= 80).to_numpy()
    idx10 = ch3_10.index
    n, i = len(on), 0
    cycle_windows = []
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            cycle_windows.append((idx10[i], idx10[j - 1]))
            i = j
        else:
            i += 1

    print(f"[H8] found {len(cycle_windows)} DR candidate runs (10-min, thr=80W)")
    # Plot first 12 individual cycles at raw resolution
    plotted = 0
    for k, (t_s, t_e) in enumerate(cycle_windows):
        if plotted >= 12:
            break
        margin = pd.Timedelta("30min")
        seg_raw = ch3_raw.loc[t_s - margin: t_e + margin]
        if len(seg_raw) == 0:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        fig.suptitle(f"H8 DR cycle #{k+1}  {t_s.date()} "
                     f"({int((t_e-t_s).total_seconds()//60)} min on-10min)", fontsize=9)

        # Raw 8s
        axes[0].plot(seg_raw.index, seg_raw.values, lw=0.5, color="steelblue")
        axes[0].axhline(80, color="red", lw=0.8, ls="--", label="threshold=80W")
        axes[0].axhline(40, color="orange", lw=0.8, ls="--", label="alt threshold=40W")
        axes[0].set_ylabel("W (raw 8s)")
        axes[0].legend(fontsize=7)

        # 10-min avg
        seg10 = ch3_10.loc[t_s - margin: t_e + margin]
        axes[1].plot(seg10.index, seg10.values, lw=1.2, color="steelblue",
                     marker="o", ms=3, label="10-min avg")
        axes[1].axhline(80, color="red", lw=0.8, ls="--", label="threshold=80W")
        axes[1].axhline(40, color="orange", lw=0.8, ls="--", label="alt threshold=40W")
        axes[1].set_ylabel("W (10-min avg)")
        axes[1].legend(fontsize=7)

        # 標 on/off 邊界
        for ax in axes:
            ax.axvline(t_s, color="green",  lw=0.8, ls=":", alpha=0.8, label="on-start")
            ax.axvline(t_e, color="purple", lw=0.8, ls=":", alpha=0.8, label="on-end")

        fig.autofmt_xdate()
        fig.tight_layout()
        fname = f"h8_dr_cycle_{k+1:02d}_{t_s.strftime('%Y%m%d_%H%M')}.png"
        save_fig(fig, out_dir, fname)
        plotted += 1

    # ── 列出每個 DR cycle 的 slots 和 mean/peak ───────────────
    print("\n[H8] DR cycle table (10-min, thr=80W):")
    rows = []
    for k, (t_s, t_e) in enumerate(cycle_windows):
        seg = ch3_10.loc[t_s:t_e].fillna(0)
        L = len(seg)
        rows.append(dict(
            cycle=k+1, t_start=t_s, slots=L,
            min_W=round(float(seg.min()), 1),
            mean_W=round(float(seg.mean()), 1),
            peak_W=round(float(seg.max()), 1),
            slots_below_80=int((seg < 80).sum()),
            slots_below_40=int((seg < 40).sum()),
        ))
    tbl = pd.DataFrame(rows)
    if len(tbl):
        print(tbl.to_string(index=False))
        # 重跑 threshold=40 看多少 cycle
        on40 = (ch3_10.fillna(0) >= 40).to_numpy()
        n40_runs = 0; i = 0
        while i < len(on40):
            if on40[i]:
                j = i
                while j < len(on40) and on40[j]:
                    j += 1
                if j - i >= 3:  # min_len=3
                    n40_runs += 1
                i = j
            else:
                i += 1
        print(f"\n  threshold=80W -> {len(cycle_windows)} runs (min_len filter不含)")
        print(f"  threshold=40W -> {n40_runs} runs (min_len>=3 filtered)")
        print(f"\n  slots_below_80 > 0 的 cycle 數: {(tbl['slots_below_80']>0).sum()} / {len(tbl)}")
        print(f"  slots_below_40 > 0 的 cycle 數: {(tbl['slots_below_40']>0).sum()} / {len(tbl)}")


# ── main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="out_diag")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    diag_h5(args.data_dir, args.out)
    diag_h8(args.data_dir, args.out)
    print(f"\n全部完成，圖片在 {args.out}/")
