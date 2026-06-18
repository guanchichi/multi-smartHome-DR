"""
verify_phase1_final.py — Phase 1 收尾驗證
  比對 out/(新正式) vs out_new/(前次 WM 參數調整後)
  預期:只有 H5 不同(移除 TD),其餘所有戶完全一致。
"""
import pandas as pd
import sys

NEW = "out/cycles_all.csv"       # 本次跑出(H5 ch2 移除後)
REF = "out_new/cycles_all.csv"   # 上次跑出(WM 參數調整後,H5 仍有 TD)

new = pd.read_csv(NEW)
ref = pd.read_csv(REF)

PASS = True

# ── ① H5 只剩 WM + DW ────────────────────────────────────────
print("=" * 60)
print("① H5 cycle 類型檢查")
print("=" * 60)
h5_new = new[new["house"] == 5]
h5_ref = ref[ref["house"] == 5]
h5_types_new = sorted(h5_new["type"].unique())
h5_types_ref = sorted(h5_ref["type"].unique())
print(f"  REF H5 types : {h5_types_ref}")
print(f"  NEW H5 types : {h5_types_new}")
if set(h5_types_new) == {"WM", "DW"}:
    print("  ✅ H5 只剩 WM + DW,TD 已移除")
else:
    print("  ❌ H5 類型不符預期!")
    PASS = False

# ── ② H5 cycle 數前後對照 ────────────────────────────────────
print()
print("=" * 60)
print("② H5 改動前後 cycle 數對照")
print("=" * 60)
print("  REF (有 TD):")
print(h5_ref.groupby(["type", "quality_flag"]).size().to_string())
print()
print("  NEW (無 TD):")
print(h5_new.groupby(["type", "quality_flag"]).size().to_string())
print()
# WM / DW 應一致
for t in ["WM", "DW"]:
    n_ref = len(h5_ref[h5_ref["type"] == t])
    n_new = len(h5_new[h5_new["type"] == t])
    ok = "✅" if n_ref == n_new else "❌"
    print(f"  H5 {t}: REF={n_ref}  NEW={n_new}  {ok}")
n_td_new = len(h5_new[h5_new["type"] == "TD"])
print(f"  H5 TD: NEW={n_td_new}  {'✅ (0)' if n_td_new == 0 else '❌ 仍有 TD!'}")

# ── ③ 其他所有戶完全一致 ─────────────────────────────────────
print()
print("=" * 60)
print("③ 非 H5 各戶 summary 比對 (應全部一致)")
print("=" * 60)
other_new = new[new["house"] != 5]
other_ref = ref[ref["house"] != 5]

# group by house + type
def make_summary(df):
    return (df.groupby(["house", "type"])
              .agg(n=("duration_slots", "size"),
                   med_slots=("duration_slots", "median"),
                   anom_long=("quality_flag", lambda s: (s == "anomaly_long").sum()))
              .reset_index())

s_new = make_summary(other_new).set_index(["house", "type"])
s_ref = make_summary(other_ref).set_index(["house", "type"])

diff = s_new.compare(s_ref)
if diff.empty:
    print("  ✅ 全部非 H5 的戶數 / type / 中位 / anomaly_long 數完全一致")
else:
    print("  ❌ 發現差異:")
    print(diff.to_string())
    PASS = False

# ── ④ 總 cycle 數 ────────────────────────────────────────────
print()
print("=" * 60)
print("④ 總 cycle 數比對")
print("=" * 60)
td_removed = len(h5_ref[h5_ref["type"] == "TD"])
print(f"  REF total: {len(ref)}")
print(f"  NEW total: {len(new)}")
print(f"  差值: {len(new) - len(ref)}  (預期 = −{td_removed}, 即 H5 TD 全數移除)")
expected_diff = -td_removed
actual_diff = len(new) - len(ref)
if actual_diff == expected_diff:
    print(f"  ✅ 差值符合預期 (−{td_removed})")
else:
    print(f"  ❌ 差值 {actual_diff} ≠ 預期 {expected_diff}")
    PASS = False

# ── ⑤ CONTAMINATION 規則無殘留 ──────────────────────────────
print()
print("=" * 60)
print("⑤ H5 無 cut_after 殘留 (quality_flag 不應出現 dehumidifier)")
print("=" * 60)
bad = new[new["quality_flag"] == "dehumidifier"]
if len(bad) == 0:
    print("  ✅ 無 dehumidifier flag")
else:
    print(f"  ❌ 仍有 {len(bad)} 筆 dehumidifier flag")
    PASS = False

# ── ⑥ baseload 影響說明 ──────────────────────────────────────
print()
print("=" * 60)
print("⑥ H5 baseload 變化說明")
print("=" * 60)
print("  ch2 從 deferrable 移除後,H5 baseload = Aggregate - (ch3_WM + ch4_DW)")
print("  ch2(除濕機)功率不再被扣除 → H5 baseload 將上升(更接近真實電器基底)。")
print("  這是 Phase 2 的 note,不影響 Phase 1 驗收。")

# ── 最終判定 ─────────────────────────────────────────────────
print()
print("=" * 60)
if PASS:
    print("✅  Phase 1 收尾驗證全部通過。out/ 可作為正式輸出。")
else:
    print("❌  有項目未通過,請檢查上方錯誤訊息。")
print("=" * 60)
sys.exit(0 if PASS else 1)
