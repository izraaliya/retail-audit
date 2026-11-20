"""Measure the detector against the planted ground truth (Label_Anomali.csv). Run: python scripts/evaluate.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.audit import detect_kind, run_audit  # noqa: E402

S = Path(__file__).resolve().parent.parent / "sample_data"
tables = {}
for f in ("Data_Order_Lengkap", "Data_Master"):
    for n, df in pd.read_excel(S / f"{f}.xlsx", sheet_name=None).items():
        if n != "Petunjuk":
            df.columns = [c.lower() for c in df.columns]
            tables[detect_kind(df)] = df
res = run_audit(tables)
truth = pd.read_csv(S / "Label_Anomali.csv")
truth = truth[truth.skema != "total_tidak_cocok"]
bad = set(truth[truth.kasir.notna()].kasir)
flag_rules = {c for c in res["risk"][res["risk"].level == "Tinggi"].cashier_id}
ml = res["ml"]["cashiers"]
ml_rank = sorted(ml, key=lambda c: -ml[c]["flag_share"])[:len(bad)]
tp = len(bad & flag_rules)
print(f"Kasir bermasalah yang ditanam : {sorted(bad)}")
print(f"Ditandai Tinggi (aturan+model): {sorted(flag_rules)}  -> recall {tp}/{len(bad)}, precision {tp}/{len(flag_rules)}")
print(f"Model saja, {len(bad)} teratas  : {ml_rank} -> kena {len(bad & set(ml_rank))}/{len(bad)}")
print("Susut stok ditanam di ST05      :", any(f["rule"] == "susut_stok" and "Malioboro" in f["subject"] for f in res["findings"]))
