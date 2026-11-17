"""
Baseline - the teacher LLM used directly as a classifier on the hidden test set.

  zero-shot : label list + descriptions only
  few-shot  : + 2 human seed examples per class (20 in-context examples)

Reports macro-F1 / accuracy, invalid-output rate, per-query latency (batch = 1,
GPU) and peak GPU memory.
"""
import difflib
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

import config as C
from pipeline.teacher import load_teacher


def build_messages(text, shots=None):
    labels = "\n".join(f"- {l}: {d}" for l, d in C.CLASSES.items())
    sys_msg = ("You are an intent classifier for a bank's customer-support chat. "
               "Choose exactly one intent from this list:\n" + labels +
               "\nAnswer with the intent label only, exactly as written.")
    msgs = [{"role": "system", "content": sys_msg}]
    if shots is not None:
        for t, l in shots:
            msgs += [{"role": "user", "content": t}, {"role": "assistant", "content": l}]
    msgs.append({"role": "user", "content": text})
    return msgs


def parse_label(out):
    o = out.strip().lower().strip("`'\". ")
    for l in sorted(C.LABELS, key=len, reverse=True):      # longest first
        if l in o:
            return l, True
    close = difflib.get_close_matches(o.split("\n")[0], C.LABELS, n=1, cutoff=0.0)
    return close[0], False


def evaluate(teacher, test, shots, tag):
    preds, valid = [], []
    for b in range(0, len(test), 16):
        batch = test.text.iloc[b:b + 16].tolist()
        outs = teacher.chat([build_messages(t, shots) for t in batch], max_new_tokens=16)
        for o in outs:
            p, ok = parse_label(o)
            preds.append(p)
            valid.append(ok)
    y = test.label.tolist()
    res = dict(model=f"Teacher {tag}", macro_f1=f1_score(y, preds, average="macro"),
               accuracy=accuracy_score(y, preds), invalid_rate=1 - float(np.mean(valid)),
               per_class=classification_report(y, preds, labels=C.LABELS, output_dict=True,
                                               zero_division=0))
    pd.DataFrame({"text": test.text, "label": y, "pred": preds}).to_csv(
        C.RESULTS_DIR / f"preds_teacher_{tag}.csv", index=False)

    # latency at batch size 1
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    lat = []
    for t in test.text.iloc[:C.TEACHER_LATENCY_QUERIES]:
        s = time.perf_counter()
        teacher.chat([build_messages(t, shots)], max_new_tokens=16)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - s) * 1000)
    res["latency_ms_mean"] = float(np.mean(lat))
    res["latency_ms_p95"] = float(np.percentile(lat, 95))
    res["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20
    res["device"] = torch.cuda.get_device_name(0)
    print(tag, {k: v for k, v in res.items() if k != "per_class"})
    return res


def main():
    test = pd.read_csv(C.DATA_DIR / "test.csv")
    seeds = pd.read_csv(C.DATA_DIR / "seeds.csv")
    shots = []
    for l in C.LABELS:
        shots += [(t, l) for t in seeds.loc[seeds.label == l, "text"].head(2)]
    rng = np.random.RandomState(0)
    shots = [shots[i] for i in rng.permutation(len(shots))]

    teacher = load_teacher()
    params = sum(p.numel() for p in teacher.model.parameters()) if hasattr(teacher, "model") else None
    out = {"teacher_name": teacher.name, "teacher_params": params,
           "zero_shot": evaluate(teacher, test, None, "zero_shot"),
           "few_shot": evaluate(teacher, test, shots, "few_shot")}
    (C.RESULTS_DIR / "teacher_eval.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
