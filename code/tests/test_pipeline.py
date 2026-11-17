"""
Automated tests for the synthetic-data pipeline.

Run from the code/ directory:   python -m pytest tests -v
Unit tests use small hand-made inputs; integration tests check the data and
result files produced by a full pipeline run.
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C                                                    # noqa: E402
from pipeline.filtering import (Embedder, centroid_filter, distinct_n,  # noqa: E402
                                heuristic_filter, normalise, semantic_dedup,
                                vocab_coverage)

gen = importlib.import_module("02_generate")
tev = importlib.import_module("04_teacher_eval")
D, R = C.DATA_DIR, C.RESULTS_DIR


@pytest.fixture(scope="module")
def seeds():
    return pd.read_csv(D / "seeds.csv")


@pytest.fixture(scope="module")
def emb():
    return Embedder(device="cpu")


# ----------------------------------------------------------------- Phase 1
def test_parse_numbered_list():
    out = 'Sure!\n1. "Where is my card?"\n2) It still has not arrived\n- bullet item\n\nnot a list line'
    assert gen.parse_list(out) == ["Where is my card?", "It still has not arrived", "bullet item"]


def test_prompt_contains_persona_style_and_seeds():
    msgs = gen.build_prompt("card_arrival", ["ex one", "ex two"], "a student", "very short")
    user = msgs[1]["content"]
    assert "ex one" in user and "a student" in user and "very short" in user
    assert C.CLASSES["card_arrival"] in user


def test_plain_prompt_has_no_persona():
    user = gen.build_prompt("card_arrival", ["ex"], None, None)[1]["content"]
    assert "as if it comes from" not in user


# ----------------------------------------------------------------- Phase 2
def test_vocab_coverage_detects_gibberish():
    assert vocab_coverage("Where is my new card, it has not arrived yet") > 0.9
    assert vocab_coverage("nfw tkks fr tske fr mcry hnsnd cddd") < C.VOCAB_COVERAGE_MIN


def test_heuristic_filter_reasons(seeds):
    df = pd.DataFrame({"label": ["card_arrival"] * 6, "text": [
        "Here are 10 messages for you",                 # meta text
        "hi",                                            # too short
        "nfw tkks fr tske fr mcry hnsnd",                # gibberish
        "is this a card_arrival issue?",                 # leaked label
        seeds.text.iloc[0],                              # copy of a seed
        "My card has still not been delivered to me",    # valid
    ]})
    kept, reasons = heuristic_filter(df, seeds)
    assert kept.text.tolist() == ["My card has still not been delivered to me"]
    assert set(reasons) == {"meta_text", "length", "gibberish_or_non_english",
                            "label_leak", "seed_copy"}


def test_semantic_dedup_removes_paraphrase(seeds, emb):
    df = pd.DataFrame({"label": ["card_arrival"] * 3, "text": [
        "My new card has not arrived yet",
        "My new card hasn't arrived yet",                # near-duplicate
        "How much is the fee for taking cash out at an ATM?"]})
    kept, removed = semantic_dedup(df, seeds.iloc[:0], emb, threshold=0.90)
    assert removed == 1 and len(kept) == 2


def test_dedup_disabled_above_one(seeds, emb):
    df = pd.DataFrame({"label": ["card_arrival"] * 2, "text": ["same text", "same text"]})
    kept, removed = semantic_dedup(df, seeds, emb, threshold=1.01)
    assert removed == 0 and len(kept) == 2


def test_centroid_filter_rejects_mislabelled(seeds, emb):
    df = pd.DataFrame({"label": ["cash_withdrawal_charge", "cash_withdrawal_charge"], "text": [
        "Why was I charged a fee for withdrawing cash from the ATM?",
        "I think my card was stolen yesterday, please block it"]})
    kept, removed = centroid_filter(df, seeds, emb)
    assert removed == 1 and "ATM" in kept.text.iloc[0]


def test_distinct_n():
    assert distinct_n(["a b c", "a b c"], 2) == 0.5
    assert distinct_n(["a b c d"], 2) == 1.0


# ----------------------------------------------------------------- teacher parsing
@pytest.mark.parametrize("raw,expected,valid", [
    ("card_arrival", "card_arrival", True),
    ("The intent is: `top_up_reverted`.", "top_up_reverted", True),
    ("card_payment_wrong_exchange_rate", "card_payment_wrong_exchange_rate", True),
    ("exchange rate", "exchange_rate", False),
])
def test_parse_teacher_label(raw, expected, valid):
    assert tev.parse_label(raw) == (expected, valid)


# ----------------------------------------------------------------- data integrity
def test_split_sizes(seeds):
    test, oracle = pd.read_csv(D / "test.csv"), pd.read_csv(D / "oracle.csv")
    assert len(seeds) == 80 and seeds.label.value_counts().eq(8).all()
    assert len(test) == 400 and test.label.value_counts().eq(40).all()
    assert set(seeds.label) == set(test.label) == set(C.LABELS)
    assert len(oracle) == 1271


def test_no_leakage_between_splits(seeds):
    test, oracle = pd.read_csv(D / "test.csv"), pd.read_csv(D / "oracle.csv")
    t = set(test.text.map(normalise))
    assert not t & set(seeds.text.map(normalise))
    assert not set(seeds.text) & set(oracle.text)


def test_synthetic_has_no_test_sentences():
    test = set(pd.read_csv(D / "test.csv").text.map(normalise))
    syn = pd.read_csv(D / "synthetic_filtered.csv")
    assert not set(syn.text.map(normalise)) & test


def test_filtered_data_is_clean():
    syn = pd.read_csv(D / "synthetic_filtered.csv")
    assert set(syn.label) <= set(C.LABELS)
    assert syn.text.map(vocab_coverage).ge(C.VOCAB_COVERAGE_MIN).all()
    assert not syn.text.map(normalise).duplicated().any()


# ----------------------------------------------------------------- results
def test_every_student_config_has_three_seeds():
    runs = pd.read_csv(R / "student_runs.csv")
    counts = runs.groupby("config").seed.nunique()
    assert len(counts) == 21 and counts.eq(3).all()


def test_metrics_in_valid_range():
    runs = pd.read_csv(R / "student_runs.csv")
    assert runs.macro_f1.between(0, 1).all() and runs.accuracy.between(0, 1).all()
    t = json.loads((R / "teacher_eval.json").read_text())
    assert 0 <= t["zero_shot"]["invalid_rate"] < 0.05


def test_int8_student_keeps_accuracy():
    b = json.loads((R / "benchmark.json").read_text())
    assert abs(b["student_cpu_int8"]["macro_f1"] - b["student_cpu_fp32"]["macro_f1"]) < 0.02
    assert b["student_cpu_int8"]["latency_ms_mean"] < b["student_cpu_fp32"]["latency_ms_mean"]
