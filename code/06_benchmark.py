"""
Phase 4 - Deployment benchmark of the student on a standard CPU.

Measures, for the proposed student (seed 0):
  * macro-F1 on the test set when run on CPU (FP32 and dynamic INT8),
  * per-query latency at batch size 1 (mean / p95) with CPU_THREADS threads,
  * throughput at batch size 32,
  * model size on disk and process memory (RSS) increase,
  * GPU latency for reference.
"""
import json
import os
import tempfile
import time

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification

import config as C
from pipeline.student import predict, tokenizer


def size_mb(model):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.pt")
        torch.save(model.state_dict(), p)
        return os.path.getsize(p) / 2**20


def latency(model, texts, device, bs=1):
    tok = tokenizer()
    model.to(device)
    lat = []
    with torch.no_grad():
        for t in texts[:20]:                                   # warm-up
            model(**tok([t], return_tensors="pt").to(device))
        for i in range(0, len(texts), bs):
            s = time.perf_counter()
            enc = tok(texts[i:i + bs], padding=True, truncation=True,
                      max_length=C.MAX_LEN, return_tensors="pt").to(device)
            model(**enc).logits.argmax(-1).tolist()
            if device == "cuda":
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - s) * 1000)
    return np.array(lat)


def main():
    torch.set_num_threads(C.CPU_THREADS)
    test = pd.read_csv(C.DATA_DIR / "test.csv")
    texts = test.text.tolist()[:C.LATENCY_QUERIES]
    proc = psutil.Process()
    out = {"cpu": f"{C.CPU_THREADS} threads, {os.cpu_count()} logical cores available",
           "torch": torch.__version__}

    rss0 = proc.memory_info().rss
    fp32 = AutoModelForSequenceClassification.from_pretrained(C.MODEL_DIR / "student_main").eval()
    int8 = torch.quantization.quantize_dynamic(
        AutoModelForSequenceClassification.from_pretrained(C.MODEL_DIR / "student_main").eval(),
        {torch.nn.Linear}, dtype=torch.qint8)

    for name, m in [("fp32", fp32), ("int8", int8)]:
        preds = predict(m, test.text.tolist(), device="cpu", batch_size=64)
        l1 = latency(m, texts, "cpu", bs=1)
        l32 = latency(m, texts, "cpu", bs=32)
        out[f"student_cpu_{name}"] = dict(
            macro_f1=f1_score(test.label, preds, average="macro"),
            latency_ms_mean=float(l1.mean()), latency_ms_p95=float(np.percentile(l1, 95)),
            throughput_qps_bs32=float(len(texts) / (l32.sum() / 1000)),
            size_mb=size_mb(m), params=sum(p.numel() for p in m.parameters()))
        print(name, out[f"student_cpu_{name}"])
    out["process_rss_increase_mb"] = (proc.memory_info().rss - rss0) / 2**20

    # Frozen MiniLM sentence embeddings + logistic regression (trained on the 80 seeds),
    # the strongest classical baseline in Table III, timed under the same CPU budget.
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    seeds = pd.read_csv(C.DATA_DIR / "seeds.csv")
    enc = SentenceTransformer(C.EMBED_MODEL, device="cpu")
    clf = LogisticRegression(max_iter=2000, C=10).fit(
        enc.encode(seeds.text.tolist(), normalize_embeddings=True), seeds.label)
    preds = clf.predict(enc.encode(test.text.tolist(), normalize_embeddings=True))
    for t in texts[:20]:
        clf.predict(enc.encode([t], normalize_embeddings=True))
    lat = []
    for t in texts:
        s = time.perf_counter()
        clf.predict(enc.encode([t], normalize_embeddings=True))
        lat.append((time.perf_counter() - s) * 1000)
    lat = np.array(lat)
    out["minilm_lr_cpu"] = dict(
        macro_f1=f1_score(test.label, preds, average="macro"),
        latency_ms_mean=float(lat.mean()), latency_ms_p95=float(np.percentile(lat, 95)),
        size_mb=size_mb(enc[0].auto_model), params=sum(p.numel() for p in enc.parameters()))
    print("minilm+lr", out["minilm_lr_cpu"])

    if torch.cuda.is_available():
        g = AutoModelForSequenceClassification.from_pretrained(C.MODEL_DIR / "student_main").eval()
        torch.cuda.reset_peak_memory_stats()
        lg = latency(g, texts, "cuda", bs=1)
        out["student_gpu_fp32"] = dict(latency_ms_mean=float(lg.mean()),
                                       latency_ms_p95=float(np.percentile(lg, 95)),
                                       peak_gpu_mb=torch.cuda.max_memory_allocated() / 2**20)
        print("gpu", out["student_gpu_fp32"])

    (C.RESULTS_DIR / "benchmark.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
