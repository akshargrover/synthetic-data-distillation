"""
Phase 2 driver - filter the raw synthetic data and report data-quality metrics.

Outputs
  data/synthetic_filtered.csv          main training set (tau = 0.90, + centroid)
  data/synthetic_plain_filtered.csv    same pipeline on the plain-prompt data
  results/filter_stats.json            per-stage counts, diversity, contamination
"""
import json

import numpy as np
import pandas as pd

import config as C
from pipeline.filtering import (Embedder, distinct_n, mean_intra_class_similarity,
                                normalise, run_filter)


def quality_report(df, emb, test, seeds):
    """Diversity + contamination statistics for one dataset."""
    rep = dict(n=len(df), distinct_1=distinct_n(df.text, 1), distinct_2=distinct_n(df.text, 2),
               mean_words=float(df.text.str.split().str.len().mean()),
               intra_class_cos=mean_intra_class_similarity(df, emb))
    # contamination: does the synthetic data reproduce hidden test sentences?
    test_norm = set(test.text.map(normalise))
    rep["exact_test_matches"] = int(df.text.map(normalise).isin(test_norm).sum())
    X = emb.encode(df.text.tolist())
    T = emb.encode(test.text.tolist())
    nn = (T @ X.T).max(1)                       # nearest synthetic neighbour of each test item
    rep["test_nn_cos_mean"] = float(nn.mean())
    rep["test_with_nn_cos_ge_0.95"] = int((nn >= 0.95).sum())
    Sd = emb.encode(seeds.text.tolist())
    rep["test_nn_cos_mean_seeds_only"] = float((T @ Sd.T).max(1).mean())
    return rep


def main():
    seeds = pd.read_csv(C.DATA_DIR / "seeds.csv")
    test = pd.read_csv(C.DATA_DIR / "test.csv")
    emb = Embedder()
    report = {}

    for name in ["synthetic", "synthetic_plain"]:
        raw_path = C.DATA_DIR / f"{name}_raw.csv"
        if not raw_path.exists():
            continue
        raw = pd.read_csv(raw_path).dropna(subset=["text"])
        filt, stats = run_filter(raw, seeds, emb, C.DEDUP_THRESHOLD, use_centroid=True)
        filt.to_csv(C.DATA_DIR / f"{name}_filtered.csv", index=False)
        stats["quality_raw"] = quality_report(raw.drop_duplicates("text"), emb, test, seeds)
        stats["quality_filtered"] = quality_report(filt, emb, test, seeds)
        report[name] = stats
        print(name, json.dumps({k: v for k, v in stats.items() if not k.startswith("quality")}, indent=1))

    # threshold sweep (diverse data): how many samples survive each tau
    raw = pd.read_csv(C.DATA_DIR / "synthetic_raw.csv").dropna(subset=["text"])
    sweep = {}
    for tau in C.THRESHOLD_SWEEP:
        f, s = run_filter(raw, seeds, emb, tau, use_centroid=False)
        sweep[str(tau)] = dict(after_semantic=s["after_semantic"],
                               intra_class_cos=mean_intra_class_similarity(f, emb),
                               distinct_2=distinct_n(f.text, 2))
        f.to_csv(C.DATA_DIR / f"synthetic_tau{tau}.csv", index=False)
    report["threshold_sweep"] = sweep
    report["seeds_quality"] = dict(distinct_2=distinct_n(seeds.text, 2),
                                   intra_class_cos=mean_intra_class_similarity(seeds, emb))
    report["test_quality"] = dict(distinct_2=distinct_n(test.text, 2),
                                  intra_class_cos=mean_intra_class_similarity(test, emb))
    (C.RESULTS_DIR / "filter_stats.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["threshold_sweep"], indent=1))


if __name__ == "__main__":
    main()
