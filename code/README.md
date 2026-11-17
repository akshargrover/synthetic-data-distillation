# Synthetic Data Generation Pipeline for Low-Compute Domain-Specific Text Classification

Minor Project (AIDS451), Dept. of AI & DS, ADGIPS. A large "Teacher" LLM takes
80 human seed examples and generates synthetic training data. The data is filtered
on semantic similarity, then used to fine-tune a small "Student" model
(DistilBERT) that runs on a normal CPU.

## Task
10 intents from **Banking77**, chosen as 5 pairs that are easy to confuse
(e.g. `card_arrival` vs `card_delivery_estimate`, `top_up_failed` vs `top_up_reverted`).

| Split | Size | Use |
|---|---|---|
| `seeds.csv` | 80 (8 / class) | the only human data the pipeline sees |
| `test.csv` | 400 (40 / class) | official Banking77 test set, hidden until evaluation |
| `oracle.csv` | 1,271 | remaining real training data, used only for the upper-bound model |

## Pipeline (run in order)

| Step | Script | What it does |
|---|---|---|
| 0 | `01_prepare_data.py` | builds the seed / test / oracle splits |
| 1 | `02_generate.py` | Teacher LLM (Qwen2.5-3B-Instruct, 4-bit) writes synthetic messages using persona × style prompts, plus a plain-prompt set for the ablation |
| 2 | `03_filter.py` | heuristic filter → MiniLM semantic de-duplication (cosine ≥ 0.90) → seed-centroid label check, plus diversity and contamination statistics |
| 3 | `04_teacher_eval.py` | Teacher zero-shot and few-shot accuracy, latency and GPU memory on the test set |
| 4 | `05_train_students.py` | trains every DistilBERT configuration on 3 random seeds (main, prompting, threshold and scaling experiments), plus TF-IDF and MiniLM baselines |
| 5 | `06_benchmark.py` | CPU latency, throughput, size and memory for the student (FP32 and INT8) |
| 6 | `07_figures.py` | all figures and summary tables used in the paper |
| 7 | `08_stats.py` | bootstrap confidence intervals and McNemar significance tests |
| 8 | `../paper/build_paper.py` | builds the IEEE-format paper (.docx, then .pdf through Microsoft Word) from `results/` |

```bash
uv venv .venv --python 3.12
uv pip install --python .venv torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv -r code/requirements.txt
cd code   # then run each script with ../.venv/Scripts/python
python 01_prepare_data.py && python 02_generate.py && python 03_filter.py
python 04_teacher_eval.py && python 05_train_students.py && python 06_benchmark.py && python 07_figures.py
python 08_stats.py && cd ../paper && python build_paper.py
```

Run time on an RTX 4050 Laptop GPU: generation takes about 22 min, teacher evaluation 10 min, and the 63 student training runs 28 min.

## Headline results (hidden 400-query human test set, macro-F1)
| System | Macro-F1 | Latency / query |
|---|---|---|
| Teacher Qwen2.5-3B zero-shot (GPU) | 0.823 | 523 ms |
| Teacher few-shot, 20 examples (GPU) | 0.872 | 705 ms |
| DistilBERT, 80 seeds only | 0.821 ± 0.021 | |
| **DistilBERT, seeds + filtered synthetic (proposed)** | **0.850 ± 0.004** | 22 ms CPU / 9.5 ms INT8 |
| MiniLM + LR, 80 seeds only | 0.893 | 10 ms CPU |
| DistilBERT, all 1,351 real examples | 0.948 ± 0.006 | |

All settings (classes, seeds, model names, thresholds, hyper-parameters) are in
`config.py`. To use Llama-3 or Mistral through Ollama, LM Studio, vLLM or Groq, set
`TEACHER_BACKEND = "openai"` and fill in `OPENAI_BASE_URL` / `OPENAI_MODEL`
(set the `TEACHER_API_KEY` environment variable if the endpoint needs one).

## Outputs
* `data/`: splits, raw synthetic data, filtered synthetic data
* `results/`: JSON and CSV metrics (one row per training run in `student_runs.csv`)
* `figures/`: the PNG figures used in the paper
* `models/student_main/`: the fine-tuned student checkpoint (not needed to read the results)
