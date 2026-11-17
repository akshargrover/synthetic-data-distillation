"""
Statistical checks used in the paper.

* 95% bootstrap confidence intervals (2,000 resamples of the 400 test items)
  for the macro-F1 of the main systems.
* Exact McNemar tests on paired per-item correctness between the proposed
  student (seed 0) and each baseline.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

import config as C

R = C.RESULTS_DIR
SYSTEMS = {
    "Teacher zero-shot": R / "preds_teacher_zero_shot.csv",
    "Teacher few-shot": R / "preds_teacher_few_shot.csv",
    "Student seeds only": R / "preds_student_Seeds_only.csv",
    "Student + raw synthetic": R / "preds_student_Seeds__raw_synthetic.csv",
    "Student proposed": R / "preds_student_Seeds__full_filter_proposed.csv",
    "Student all real data": R / "preds_student_Oracle.csv",
    "MiniLM + LR seeds only": R / "preds_classical_MiniLM_Seeds.csv",
    "MiniLM + LR proposed data": R / "preds_classical_MiniLM_Seeds_proposed.csv",
}


def bootstrap_ci(y, p, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        vals.append(f1_score(y[i], p[i], average="macro"))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar(a_correct, b_correct):
    b = int((a_correct & ~b_correct).sum())     # a right, b wrong
    c = int((~a_correct & b_correct).sum())     # a wrong, b right
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return dict(a_only=b, b_only=c, p_value=float(p))


def main():
    preds = {k: pd.read_csv(v) for k, v in SYSTEMS.items() if v.exists()}
    out = {"bootstrap_ci": {}, "mcnemar_vs_proposed": {}}
    for k, d in preds.items():
        lo, hi = bootstrap_ci(d.label, d.pred)
        out["bootstrap_ci"][k] = dict(f1=f1_score(d.label, d.pred, average="macro"), lo=lo, hi=hi)
    prop = preds["Student proposed"]
    pc = (prop.label == prop.pred).values
    for k, d in preds.items():
        if k == "Student proposed":
            continue
        out["mcnemar_vs_proposed"][k] = mcnemar(pc, (d.label == d.pred).values)
    (R / "stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
