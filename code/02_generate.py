"""
Phase 1 - Seed ingestion & prompting strategy.

For every class the teacher LLM is called repeatedly. Each call receives
  * the class name and its one-line description,
  * GEN_SHOTS randomly sampled human seed examples of that class,
  * (diverse mode) a random customer PERSONA and writing STYLE,
and is asked for GEN_PER_CALL new, different customer messages.

Two prompting modes are produced:
  diverse : persona + style injection           -> data/synthetic_raw.csv
  plain   : same prompt without persona/style   -> data/synthetic_plain_raw.csv
            (used for the prompting ablation)

Usage:  python 02_generate.py            (both modes)
        python 02_generate.py diverse    (one mode)
"""
import json
import random
import re
import sys
import time

import pandas as pd

import config as C
from pipeline.teacher import load_teacher

SYSTEM = ("You generate realistic training data for a bank's customer-support "
          "intent classifier. You only output the requested list, nothing else.")


def build_prompt(label, examples, persona=None, style=None, n=C.GEN_PER_CALL):
    ex = "\n".join(f"- {e}" for e in examples)
    who = ""
    if persona:
        who = (f"\nWrite every message as if it comes from {persona}. "
               f"Style of the messages: {style}.")
    user = (
        f"Intent: {label}\n"
        f"Meaning: {C.CLASSES[label]}\n\n"
        f"Real customer messages with this intent:\n{ex}\n\n"
        f"Write {n} NEW customer messages that a bank's support chat could receive "
        f"and that clearly have this intent.{who}\n"
        "Rules: every message is written to the support team of a digital bank and is about "
        "the customer's own bank account, bank card, payments or transfers at this bank "
        "(never about passports, IDs, insurance or other companies); vary the wording and "
        "sentence structure; do not copy the examples; do not mention the intent name; "
        "each message is one line.\n"
        f"Output format: a numbered list 1. to {n}. with one message per line."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


_LINE = re.compile(r"^\s*(?:\d+[\.\)]|[-*•])\s*(.+?)\s*$")


def parse_list(text):
    out = []
    for line in text.splitlines():
        m = _LINE.match(line)
        if m:
            s = m.group(1).strip().strip('"').strip("“”").strip()
            if s:
                out.append(s)
    return out


def run(mode, teacher, seeds):
    rng = random.Random(C.GEN_SEED + (0 if mode == "diverse" else 1))
    calls_per_class = C.GEN_CALLS_PER_CLASS if mode == "diverse" else C.GEN_PLAIN_CALLS_PER_CLASS
    jobs = []
    for label in C.LABELS:
        pool = seeds.loc[seeds.label == label, "text"].tolist()
        for k in range(calls_per_class):
            shots = rng.sample(pool, C.GEN_SHOTS)
            persona = rng.choice(C.PERSONAS) if mode == "diverse" else None
            style = rng.choice(C.STYLES) if mode == "diverse" else None
            jobs.append(dict(label=label, call=k, persona=persona, style=style,
                             messages=build_prompt(label, shots, persona, style)))

    rows, t0, n_tokens_out = [], time.time(), 0
    for b in range(0, len(jobs), C.GEN_BATCH):
        batch = jobs[b:b + C.GEN_BATCH]
        outs = teacher.chat([j["messages"] for j in batch],
                            max_new_tokens=C.GEN_MAX_NEW_TOKENS,
                            temperature=C.GEN_TEMPERATURE, top_p=C.GEN_TOP_P,
                            seed=C.GEN_SEED + b)
        for j, o in zip(batch, outs):
            n_tokens_out += len(o.split())
            for s in parse_list(o):
                rows.append(dict(text=s, label=j["label"], mode=mode, call=j["call"],
                                 persona=j["persona"], style=j["style"]))
        done = min(b + C.GEN_BATCH, len(jobs))
        print(f"[{mode}] {done}/{len(jobs)} calls, {len(rows)} samples, "
              f"{time.time() - t0:.0f}s", flush=True)

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)
    out = C.DATA_DIR / ("synthetic_raw.csv" if mode == "diverse" else "synthetic_plain_raw.csv")
    df.to_csv(out, index=False)
    stats = dict(mode=mode, teacher=teacher.name, calls=len(jobs), samples=len(df),
                 seconds=round(elapsed, 1), samples_per_sec=round(len(df) / elapsed, 2),
                 approx_words_out=n_tokens_out)
    print(stats)
    return stats


def main():
    modes = sys.argv[1:] or ["diverse", "plain"]
    seeds = pd.read_csv(C.DATA_DIR / "seeds.csv")
    teacher = load_teacher()
    stats_path = C.RESULTS_DIR / "generation_stats.json"
    all_stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    for mode in modes:
        all_stats[mode] = run(mode, teacher, seeds)
        stats_path.write_text(json.dumps(all_stats, indent=2))


if __name__ == "__main__":
    main()
