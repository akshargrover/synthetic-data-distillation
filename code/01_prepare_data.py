"""
Phase 0 - Build the three data splits used in every experiment.

  seeds.csv  : 8 human examples per class (80 total)  -> the ONLY human data the
               generation pipeline and the student are allowed to see.
  test.csv   : the official Banking77 *test* split for the 10 classes (400 rows),
               purely human-annotated and never shown to the teacher or student.
  oracle.csv : all remaining human training examples of the 10 classes. Used
               only for an upper-bound "fully supervised" reference model.
"""
import pandas as pd
from datasets import load_dataset

import config as C


def main():
    ds = load_dataset(C.DATASET_ID, revision=C.DATASET_REVISION)
    names = ds["train"].features["label"].names
    keep = {names.index(l) for l in C.LABELS}

    def to_df(split):
        df = ds[split].to_pandas()
        df = df[df["label"].isin(keep)].copy()
        df["label"] = df["label"].map(lambda i: names[i])
        df["text"] = df["text"].str.strip()
        return df[["text", "label"]].reset_index(drop=True)

    train, test = to_df("train"), to_df("test")

    seeds = (train.groupby("label", group_keys=False)
                  .apply(lambda g: g.sample(C.SEEDS_PER_CLASS, random_state=C.DATA_SEED)))
    oracle = train.drop(seeds.index)

    # Guard against any text that occurs in both train and test.
    test_texts = set(test["text"].str.lower())
    assert not (set(seeds["text"].str.lower()) & test_texts), "seed/test overlap"

    seeds.reset_index(drop=True).to_csv(C.DATA_DIR / "seeds.csv", index=False)
    test.to_csv(C.DATA_DIR / "test.csv", index=False)
    oracle.reset_index(drop=True).to_csv(C.DATA_DIR / "oracle.csv", index=False)

    print(f"seeds={len(seeds)}  test={len(test)}  oracle={len(oracle)}")
    print(test["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
