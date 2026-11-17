"""
Phase 3/4 driver - train every student configuration on 3 random seeds and
evaluate on the hidden human test set.

Experiment groups
  main      : seeds only / + raw / + heuristic / + dedup / + full filter /
              synthetic only / oracle (all real training data)
  prompt    : diverse (persona+style) vs plain prompting at equal size
  threshold : semantic-dedup threshold sweep tau in {0.80 .. 0.95, none}
  scaling   : 25/50/100/200 samples per class, synthetic vs real
  classical : TF-IDF + LR and MiniLM-embedding + LR (deterministic)

Results are appended to results/student_runs.csv, so the script can be
re-started and will skip finished runs.
"""
import json
import time

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

import config as C
from pipeline.filtering import Embedder, heuristic_filter
from pipeline.student import embed_lr, predict, tfidf_lr, train_student

RUNS = C.RESULTS_DIR / "student_runs.csv"


def per_class_sample(df, n, seed=0):
    return (df.groupby("label", group_keys=False)
              .apply(lambda g: g.sample(min(n, len(g)), random_state=seed))
              .reset_index(drop=True))


def match_counts(df, counts, seed=0):
    return (df.groupby("label", group_keys=False)
              .apply(lambda g: g.sample(min(counts.get(g.name, 0), len(g)), random_state=seed))
              .reset_index(drop=True))


def build_experiments():
    d = C.DATA_DIR
    seeds = pd.read_csv(d / "seeds.csv")
    oracle = pd.read_csv(d / "oracle.csv")
    raw = pd.read_csv(d / "synthetic_raw.csv").dropna(subset=["text"])[["text", "label"]]
    heur, _ = heuristic_filter(raw, seeds)
    tau90 = pd.read_csv(d / "synthetic_tau0.9.csv")[["text", "label"]]
    full = pd.read_csv(d / "synthetic_filtered.csv")[["text", "label"]]
    plain = pd.read_csv(d / "synthetic_plain_filtered.csv")[["text", "label"]]
    plain_counts = plain.label.value_counts().to_dict()
    diverse_matched = match_counts(full, plain_counts)

    cat = lambda *xs: pd.concat(xs, ignore_index=True)
    E = [
        ("main", "Seeds only", seeds),
        ("main", "Seeds + raw synthetic", cat(seeds, raw)),
        ("main", "Seeds + heuristic-filtered", cat(seeds, heur[["text", "label"]])),
        ("main", "Seeds + dedup (tau=0.90)", cat(seeds, tau90)),
        ("main", "Seeds + full filter (proposed)", cat(seeds, full)),
        ("main", "Synthetic only (full filter)", full),
        ("main", "Oracle: all real training data", cat(seeds, oracle)),
        ("prompt", "Plain prompting (size-matched)", cat(seeds, plain)),
        ("prompt", "Persona+style prompting (size-matched)", cat(seeds, diverse_matched)),
    ]
    for tau in C.THRESHOLD_SWEEP:
        if tau == 0.90:
            continue
        df = pd.read_csv(d / f"synthetic_tau{tau}.csv")[["text", "label"]]
        E.append(("threshold", f"tau={tau}", cat(seeds, df)))
    for n in C.SCALING_SIZES:
        E.append(("scaling", f"synthetic n={n}", cat(seeds, per_class_sample(full, n))))
        E.append(("scaling", f"real n={n}", cat(seeds, per_class_sample(oracle, n))))
    return E, seeds, oracle, full, raw


def main():
    test = pd.read_csv(C.DATA_DIR / "test.csv")
    E, seeds, oracle, full, raw = build_experiments()
    done = set()
    if RUNS.exists():
        prev = pd.read_csv(RUNS)
        done = set(zip(prev.config, prev.seed))

    sizes = {name: len(df) for _, name, df in E}
    (C.RESULTS_DIR / "experiment_sizes.json").write_text(json.dumps(sizes, indent=2))

    for group, name, train_df in E:
        for seed in C.TRAIN_SEEDS:
            if (name, seed) in done:
                continue
            t0 = time.time()
            model = train_student(train_df, seed)
            train_s = time.time() - t0
            preds = predict(model, test.text.tolist())
            row = dict(group=group, config=name, seed=seed, n_train=len(train_df),
                       epochs=model.train_info["epochs"], steps=model.train_info["steps"],
                       train_seconds=round(train_s, 1),
                       macro_f1=f1_score(test.label, preds, average="macro"),
                       accuracy=accuracy_score(test.label, preds))
            print(row, flush=True)
            pd.DataFrame([row]).to_csv(RUNS, mode="a", header=not RUNS.exists(), index=False)
            if seed == 0 and group == "main":
                tag = name.split(":")[0].replace(" ", "_").replace("+", "").replace("(", "").replace(")", "").replace("=", "")
                pd.DataFrame({"text": test.text, "label": test.label, "pred": preds}).to_csv(
                    C.RESULTS_DIR / f"preds_student_{tag}.csv", index=False)
                if name == "Seeds + full filter (proposed)":
                    model.save_pretrained(C.MODEL_DIR / "student_main")
            del model
            torch.cuda.empty_cache()

    # classical baselines (deterministic -> single run)
    emb = Embedder()
    cat = lambda *xs: pd.concat(xs, ignore_index=True)
    rows = []
    for name, df in [("Seeds only", seeds), ("Seeds + full filter (proposed)", cat(seeds, full)),
                     ("Oracle: all real training data", cat(seeds, oracle))]:
        for method, fn in [("TF-IDF + LR", lambda a, b: tfidf_lr(a, b)),
                           ("MiniLM + LR", lambda a, b: embed_lr(a, b, emb))]:
            p = fn(df, test.text)
            tag = f"{method.split()[0]}_{name.split()[0].rstrip(':')}_{'proposed' if 'proposed' in name else ''}".strip("_")
            pd.DataFrame({"text": test.text, "label": test.label, "pred": p}).to_csv(
                C.RESULTS_DIR / f"preds_classical_{tag}.csv", index=False)
            rows.append(dict(method=method, data=name, n_train=len(df),
                             macro_f1=f1_score(test.label, p, average="macro"),
                             accuracy=accuracy_score(test.label, p)))
            print(rows[-1])
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "classical_baselines.csv", index=False)


if __name__ == "__main__":
    main()
