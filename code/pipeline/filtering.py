"""
Phase 2 - Semantic filtering & quality control.

Stage 1  heuristic  : length bounds, LLM meta-text / refusals, gibberish or
                      non-English text (vocabulary coverage < 0.70), leaked intent
                      names, exact (normalised) duplicates, verbatim seed copies.
Stage 2  semantic   : greedy near-duplicate removal. Every sample is embedded with
                      all-MiniLM-L6-v2; a sample is kept only if its cosine
                      similarity to every already-kept sample (and every seed)
                      is below the threshold tau (default 0.90).
Stage 3  centroid   : label-consistency check. A sample is kept only if it is
                      closer to the seed centroid of its own class than to the
                      centroid of any other class (margin > 0).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import config as C

_META = re.compile(
    r"(here are|here is|sure[,!]|as an ai|i cannot|i can't help|intent:|meaning:|"
    r"customer message|numbered list|\bexample\b)", re.I)


def normalise(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


class Embedder:
    def __init__(self, model_id=C.EMBED_MODEL, device=None):
        from sentence_transformers import SentenceTransformer
        import torch
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_id, device=device)
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, texts):
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vecs = self.model.encode(missing, batch_size=128, normalize_embeddings=True,
                                     show_progress_bar=False, convert_to_numpy=True)
            self._cache.update(zip(missing, vecs))
        return np.stack([self._cache[t] for t in texts])


_VOCAB = None


def vocab_coverage(text: str) -> float:
    """Share of alphabetic tokens that are whole words in the DistilBERT vocabulary."""
    global _VOCAB
    if _VOCAB is None:
        from transformers import AutoTokenizer
        _VOCAB = {w for w in AutoTokenizer.from_pretrained(C.STUDENT_MODEL).vocab if w.isalpha()}
    words = re.findall(r"[a-z]+", text.lower())
    return sum(w in _VOCAB for w in words) / max(len(words), 1)


def heuristic_filter(df: pd.DataFrame, seeds: pd.DataFrame):
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    n_words = df["text"].str.split().str.len()
    seed_norm = set(seeds["text"].map(normalise))
    label_tokens = re.compile("|".join(re.escape(l) for l in C.LABELS), re.I)

    reasons = pd.Series("", index=df.index)
    reasons[(n_words < C.MIN_WORDS) | (n_words > C.MAX_WORDS)] = "length"
    reasons[(reasons == "") & df["text"].str.contains(_META)] = "meta_text"
    cov = df["text"].map(vocab_coverage)
    reasons[(reasons == "") & (cov < C.VOCAB_COVERAGE_MIN)] = "gibberish_or_non_english"
    reasons[(reasons == "") & df["text"].str.contains(label_tokens)] = "label_leak"
    norm = df["text"].map(normalise)
    reasons[(reasons == "") & norm.isin(seed_norm)] = "seed_copy"
    dup = norm.duplicated(keep="first")
    reasons[(reasons == "") & dup] = "exact_duplicate"
    keep = df[reasons == ""].reset_index(drop=True)
    return keep, reasons[reasons != ""].value_counts().to_dict()


def semantic_dedup(df: pd.DataFrame, seeds: pd.DataFrame, emb: Embedder,
                   threshold: float, rng_seed: int = 0):
    if threshold > 1.0 or len(df) == 0:
        return df.reset_index(drop=True), 0
    order = np.random.RandomState(rng_seed).permutation(len(df))
    X = emb.encode(df["text"].tolist())[order]
    S = emb.encode(seeds["text"].tolist())
    # pre-allocated matrix of everything kept so far (seeds first)
    kept_mat = np.zeros((len(S) + len(X), X.shape[1]), dtype=X.dtype)
    kept_mat[:len(S)] = S
    n_kept = len(S)
    kept_idx = []
    for pos, i in enumerate(order):
        v = X[pos]
        if (kept_mat[:n_kept] @ v).max() < threshold:
            kept_idx.append(i)
            kept_mat[n_kept] = v
            n_kept += 1
    out = df.iloc[sorted(kept_idx)].reset_index(drop=True)
    return out, len(df) - len(out)


def seed_centroids(seeds: pd.DataFrame, emb: Embedder):
    cents = []
    for l in C.LABELS:
        v = emb.encode(seeds.loc[seeds.label == l, "text"].tolist()).mean(0)
        cents.append(v / np.linalg.norm(v))
    return np.stack(cents)


def centroid_filter(df: pd.DataFrame, seeds: pd.DataFrame, emb: Embedder,
                    margin: float = C.CENTROID_MARGIN):
    if len(df) == 0:
        return df, 0
    cents = seed_centroids(seeds, emb)
    sims = emb.encode(df["text"].tolist()) @ cents.T            # (n, K)
    own = np.array([C.LABEL2ID[l] for l in df["label"]])
    own_sim = sims[np.arange(len(df)), own]
    sims_other = sims.copy()
    sims_other[np.arange(len(df)), own] = -np.inf
    keep_mask = (own_sim - sims_other.max(1)) > margin
    out = df[keep_mask].reset_index(drop=True)
    return out, int((~keep_mask).sum())


def run_filter(raw: pd.DataFrame, seeds: pd.DataFrame, emb: Embedder,
               threshold: float = C.DEDUP_THRESHOLD, use_centroid: bool = True):
    stats = {"raw": len(raw)}
    df, reasons = heuristic_filter(raw, seeds)
    stats["after_heuristic"] = len(df)
    stats["heuristic_reasons"] = reasons
    df, n = semantic_dedup(df, seeds, emb, threshold)
    stats["after_semantic"] = len(df)
    if use_centroid:
        df, n = centroid_filter(df, seeds, emb)
    stats["after_centroid"] = len(df)
    stats["per_class"] = df["label"].value_counts().reindex(C.LABELS).fillna(0).astype(int).to_dict()
    return df, stats


# --------------------------------------------------------------------------
# Diversity metrics used in the paper
# --------------------------------------------------------------------------
def distinct_n(texts, n=2):
    grams, total = set(), 0
    for t in texts:
        toks = normalise(t).split()
        g = list(zip(*[toks[i:] for i in range(n)]))
        grams.update(g)
        total += len(g)
    return len(grams) / max(total, 1)


def mean_intra_class_similarity(df, emb):
    vals = []
    for l in C.LABELS:
        t = df.loc[df.label == l, "text"].tolist()
        if len(t) < 2:
            continue
        X = emb.encode(t)
        S = X @ X.T
        n = len(t)
        vals.append((S.sum() - n) / (n * (n - 1)))
    return float(np.mean(vals))
