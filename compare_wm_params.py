"""
compare_wm_params.py — 對比 WM 參數修改前後的 cycle 統計
  OLD: out/cycles_all.csv   (merge_gap=2, max_len=18)
  NEW: out_new/cycles_all.csv (merge_gap=1, max_len=22)
"""
import pandas as pd
import numpy as np

OLD = "out/cycles_all.csv"
NEW = "out_new/cycles_all.csv"

old = pd.read_csv(OLD)
new = pd.read_csv(NEW)

SEP = "=" * 62

# ── ① 每類 anomaly_long pct ──────────────────────────────────
print(SEP)
print("① anomaly_long pct  (anomaly_long / total per type)")
print(SEP)
rows = []
for t in ["WM", "DW", "TD", "WD", "DR"]:
    for label, df in [("OLD", old), ("NEW", new)]:
        sub = df[df["type"] == t]
        if len(sub) == 0:
            continue
        pct = 100.0 * (sub["quality_flag"] == "anomaly_long").mean()
        rows.append(dict(type=t, ver=label, n=len(sub), anomaly_long_pct=round(pct, 2)))
tbl = pd.DataFrame(rows).pivot(index="type", columns="ver", values=["n", "anomaly_long_pct"])
print(tbl.to_string())

# ── ② 各 type 剛好等於 min_len 的比例 ───────────────────────
print()
print(SEP)
print("② min_len 比例  (min_len: WM=3, DW=4, TD=3, WD=6, DR=3)")
print(SEP)
MIN_LEN = {"WM": 3, "DW": 4, "TD": 3, "WD": 6, "DR": 3}
rows2 = []
for t, ml in MIN_LEN.items():
    for label, df in [("OLD", old), ("NEW", new)]:
        sub = df[df["type"] == t]
        if len(sub) == 0:
            continue
        pct = 100.0 * (sub["duration_slots"] == ml).mean()
        rows2.append(dict(type=t, ver=label, at_min_pct=round(pct, 2)))
tbl2 = pd.DataFrame(rows2).pivot(index="type", columns="ver", values="at_min_pct")
print(tbl2.to_string())

# ── ③ 每戶 WM 中位長度 (duration_min) ───────────────────────
print()
print(SEP)
print("③ per-house WM 中位 duration_min")
print(SEP)
wm_old = old[old["type"] == "WM"].groupby("house")["duration_min"].median().rename("OLD")
wm_new = new[new["type"] == "WM"].groupby("house")["duration_min"].median().rename("NEW")
wm_cnt_old = old[old["type"] == "WM"].groupby("house").size().rename("n_OLD")
wm_cnt_new = new[new["type"] == "WM"].groupby("house").size().rename("n_NEW")
wm_tbl = pd.concat([wm_old, wm_new, wm_cnt_old, wm_cnt_new], axis=1)
wm_tbl["Δmedian"] = wm_tbl["NEW"] - wm_tbl["OLD"]
wm_tbl["Δn"] = wm_tbl["n_NEW"] - wm_tbl["n_OLD"]
print(wm_tbl.to_string())

# ── ④ 總 cycle 數 ────────────────────────────────────────────
print()
print(SEP)
print("④ 總 cycle 數 (含所有 quality_flag)")
print(SEP)
for label, df in [("OLD", old), ("NEW", new)]:
    n_total = len(df)
    by_flag = df["quality_flag"].value_counts().to_dict()
    print(f"  {label}: total={n_total}  |  {by_flag}")

# ── 驗證 H2 WM: 原 anomaly_long 有沒有裂開 ───────────────────
print()
print(SEP)
print("驗 H2 WM — anomaly_long 變化 & 新增 ok 週期")
print(SEP)
for label, df in [("OLD", old), ("NEW", new)]:
    h2 = df[(df["house"] == 2) & (df["type"] == "WM")]
    n_anom = (h2["quality_flag"] == "anomaly_long").sum()
    n_ok   = (h2["quality_flag"] == "ok").sum()
    med    = h2["duration_min"].median()
    print(f"  {label}: total={len(h2)}  ok={n_ok}  anomaly_long={n_anom}  median={med:.0f} min")
print("  (anomaly_long 應大幅減少; total 或增加因原大 cycle 裂成兩筆)")

# ── 驗證 H6 WM: 真實長週期沒被誤刪 ──────────────────────────
print()
print(SEP)
print("驗 H6 WM — 真實長週期 (原 anomaly_long) 變化")
print(SEP)
for label, df in [("OLD", old), ("NEW", new)]:
    h6 = df[(df["house"] == 6) & (df["type"] == "WM")]
    n_anom = (h6["quality_flag"] == "anomaly_long").sum()
    n_ok   = (h6["quality_flag"] == "ok").sum()
    med    = h6["duration_min"].median()
    p90    = h6["duration_min"].quantile(0.9)
    print(f"  {label}: total={len(h6)}  ok={n_ok}  anomaly_long={n_anom}  "
          f"median={med:.0f} min  p90={p90:.0f} min")
print("  (21-22 slot 週期應轉為 ok; >22 slot 仍 anomaly_long; 總數不變)")

# ── H5 WM anomaly_long slots 分布(新) ────────────────────────
print()
print(SEP)
print("H5 WM anomaly_long slots 分布 (NEW only)")
print(SEP)
h5wm_new = new[(new["house"] == 5) & (new["type"] == "WM") & (new["quality_flag"] == "anomaly_long")]
if len(h5wm_new):
    print(h5wm_new["duration_slots"].value_counts().sort_index().to_string())
else:
    print("  (none)")
