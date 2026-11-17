"""
Phase 3 - Student model distillation (fine-tuning DistilBERT) and light-weight
classical baselines.
"""
from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

import config as C


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


_TOK = None


def tokenizer():
    global _TOK
    if _TOK is None:
        _TOK = AutoTokenizer.from_pretrained(C.STUDENT_MODEL)
    return _TOK


def train_student(train_df: pd.DataFrame, seed: int, device: str = "cuda"):
    """Fine-tune DistilBERT with cross-entropy. Returns the trained model (eval mode)."""
    set_seed(seed)
    tok = tokenizer()
    model = AutoModelForSequenceClassification.from_pretrained(
        C.STUDENT_MODEL, num_labels=len(C.LABELS), id2label=C.ID2LABEL, label2id=C.LABEL2ID
    ).to(device)

    texts = train_df["text"].astype(str).tolist()
    labels = [C.LABEL2ID[l] for l in train_df["label"]]
    data = list(zip(texts, labels))

    def collate(batch):
        t, y = zip(*batch)
        enc = tok(list(t), padding=True, truncation=True, max_length=C.MAX_LEN, return_tensors="pt")
        enc["labels"] = torch.tensor(y)
        return enc

    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(data, batch_size=C.BATCH_SIZE, shuffle=True, collate_fn=collate, generator=g)
    steps_per_epoch = len(loader)
    epochs = max(C.EPOCHS, math.ceil(C.MIN_STEPS / steps_per_epoch))
    total = epochs * steps_per_epoch

    no_decay = ["bias", "LayerNorm.weight"]
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": C.WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=C.LR)
    sched = get_linear_schedule_with_warmup(opt, int(C.WARMUP_RATIO * total), total)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    model.train()
    for _ in range(epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = model(**batch).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
    model.eval()
    model.train_info = dict(epochs=epochs, steps=total, n_train=len(data))
    return model


@torch.no_grad()
def predict(model, texts, device="cuda", batch_size=128):
    tok = tokenizer()
    model.to(device)
    preds = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], padding=True, truncation=True,
                  max_length=C.MAX_LEN, return_tensors="pt").to(device)
        preds += model(**enc).logits.argmax(-1).tolist()
    return [C.ID2LABEL[p] for p in preds]


# --------------------------------------------------------------------------
# Classical CPU baselines
# --------------------------------------------------------------------------
def tfidf_lr(train_df, test_texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
                        LogisticRegression(max_iter=2000, C=10))
    clf.fit(train_df.text, train_df.label)
    return clf.predict(test_texts).tolist()


def embed_lr(train_df, test_texts, emb):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=10)
    clf.fit(emb.encode(train_df.text.tolist()), train_df.label)
    return clf.predict(emb.encode(list(test_texts))).tolist()
