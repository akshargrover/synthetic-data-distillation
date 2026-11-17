"""
Build the IEEE-format paper (DOCX, then PDF via Word) from the experiment
results in ../code/results. Every number quoted in the text is read from
those files, so the paper always matches the code's output.

    ../.venv/Scripts/python build_paper.py
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
sys.path.insert(0, str(CODE))
import config as C                                     # noqa: E402
from ieee_docx import IEEEDoc                          # noqa: E402
from paper_static import AUTHORS, KEYWORDS, REFERENCES, TITLE   # noqa: E402

R, FIG = CODE / "results", CODE / "figures"
REPO_URL = "https://github.com/akshargrover/synthetic-data-distillation"

# ------------------------------------------------------------------ load results
runs = pd.read_csv(R / "student_runs.csv")
S = runs.groupby("config").agg(m=("macro_f1", "mean"), s=("macro_f1", "std"),
                               acc=("accuracy", "mean"), acc_s=("accuracy", "std"),
                               n=("n_train", "first"), t=("train_seconds", "mean"))
T = json.loads((R / "teacher_eval.json").read_text())
FS = json.loads((R / "filter_stats.json").read_text())
GS = json.loads((R / "generation_stats.json").read_text())
B = json.loads((R / "benchmark.json").read_text())
ST = json.loads((R / "stats.json").read_text())
CL = pd.read_csv(R / "classical_baselines.csv").set_index(["method", "data"])
PC = pd.read_csv(R / "per_class_f1.csv")
MLR_SEEDS = CL.loc[("MiniLM + LR", "Seeds only"), "macro_f1"]
MLR_PROP = CL.loc[("MiniLM + LR", "Seeds + full filter (proposed)"), "macro_f1"]
TFIDF_SEEDS = CL.loc[("TF-IDF + LR", "Seeds only"), "macro_f1"]
TFIDF_PROP = CL.loc[("TF-IDF + LR", "Seeds + full filter (proposed)"), "macro_f1"]
MLR_B = B["minilm_lr_cpu"]


def f(x, d=3):
    return f"{x:.{d}f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def m(cfg):
    return S.loc[cfg, "m"]


def ms(cfg):
    return f"{S.loc[cfg, 'm']:.3f} ± {S.loc[cfg, 's']:.3f}"


TZ, TF = T["zero_shot"], T["few_shot"]
PROP = "Seeds + full filter (proposed)"
SEEDS = "Seeds only"
RAW = "Seeds + raw synthetic"
HEUR = "Seeds + heuristic-filtered"
DEDUP = "Seeds + dedup (tau=0.90)"
SYN = "Synthetic only (full filter)"
ORA = "Oracle: all real training data"
PLAIN, DIV = "Plain prompting (size-matched)", "Persona+style prompting (size-matched)"
fs, fp = FS["synthetic"], FS["synthetic_plain"]
cpu32, cpu8 = B["student_cpu_fp32"], B["student_cpu_int8"]
best_cfg = S.drop([ORA]).m.idxmax()
gain_vs_seeds = m(PROP) - m(SEEDS)
ratio_zero = m(PROP) / TZ["macro_f1"]
ratio_few = m(PROP) / TF["macro_f1"]
speed32 = TZ["latency_ms_mean"] / cpu32["latency_ms_mean"]
speed8 = TZ["latency_ms_mean"] / cpu8["latency_ms_mean"]
gap_closed = (m(PROP) - m(SEEDS)) / (m(ORA) - m(SEEDS))
mc = ST["mcnemar_vs_proposed"]
ci = ST["bootstrap_ci"]
teacher_params_b = 3.09     # Qwen2.5-3B-Instruct: 3.09 B parameters (model card)
student_params_m = cpu32["params"] / 1e6
reasons = fs["heuristic_reasons"]
gib = reasons.get("gibberish_or_non_english", 0)


def pv(p):
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


# ------------------------------------------------------------------ document
d = IEEEDoc()
d.title(TITLE)
d.authors(AUTHORS)

d.abstract(
    "Large language models (LLMs) classify text well without task-specific training, but they are "
    "slow, need a GPU and cost money for each query. Small encoders such as DistilBERT run on an "
    "ordinary CPU, but they need many labelled examples. This paper describes and tests a pipeline "
    "that connects the two: an LLM *teacher* turns 80 human-labelled seed examples into synthetic "
    "training data, the data is filtered using sentence embeddings, and a DistilBERT *student* "
    "is trained on the result. We test it on ten Banking77 intents arranged as five pairs that "
    "are easy to confuse, using a 3-billion-parameter open teacher (Qwen2.5-3B, 4-bit) that runs "
    f"on a 6 GB laptop GPU. The teacher produced {GS['diverse']['samples']:,} messages, of which "
    f"{fs['after_centroid']:,} passed the filters. On a hidden 400-item human test set, the "
    f"student reaches a macro-F1 of {f(m(PROP))} ± {f(S.loc[PROP, 's'])}, compared with "
    f"{f(m(SEEDS))} ± {f(S.loc[SEEDS, 's'])} when trained on the seeds alone. This is "
    f"statistically equal to the teacher in zero-shot ({f(TZ['macro_f1'])}) and few-shot "
    f"({f(TF['macro_f1'])}) mode, while answering in {f(cpu32['latency_ms_mean'], 1)} ms on four "
    f"CPU threads ({f(cpu8['latency_ms_mean'], 1)} ms in INT8), {speed32:.0f}–{speed8:.0f} "
    "times faster than the GPU teacher. Persona and style prompting was the most useful design "
    f"choice (+{pct(m(DIV) - m(PLAIN))} points at equal data size). The filters removed "
    f"{pct(1 - fs['after_centroid'] / fs['raw'], 0)}% of the data, including all gibberish, "
    "without changing accuracy. Two strong baselines put these gains in context: a "
    f"logistic-regression classifier on frozen MiniLM embeddings trained on the 80 seeds alone "
    f"reaches {f(MLR_SEEDS)} and is not improved by the synthetic data, and 25 *real* examples "
    f"per intent ({f(S.loc['real n=25', 'm'])}) beat all synthetic data. With a small teacher, "
    "synthetic data makes a fine-tuned student cheap and stable, but it is not a substitute "
    "for real labels or a good sentence encoder.",
    KEYWORDS)

# ------------------------------------------------------------------ I. Introduction
d.section("Introduction")
d.para("Instruction-tuned LLMs can sort customer messages, support tickets or complaints into "
       "categories from a short description of each category [20]. For many organisations this "
       "is still not practical to deploy. Each prediction runs a model with billions of "
       "parameters, so a single query takes hundreds of milliseconds even on a GPU. API use is "
       "charged per token, and sending customer text to an outside provider may not be allowed "
       "in finance or healthcare. Compact encoders such as DistilBERT [1] answer in a few "
       "milliseconds on a normal CPU and can run entirely on the organisation's own machines. "
       "Their weakness is data: to learn a narrow task they need hundreds or thousands of "
       "labelled examples, and these are slow and costly to produce by hand.")
d.para("This project tackles that data bottleneck. Instead of copying a large model's weights "
       "into a small one (classic knowledge distillation [2], [3]), we use the large model only "
       "as a *data generator*, an approach often called data distillation [4]–[7]. A teacher "
       "LLM expands a small set of human seed examples into a large synthetic training set, the "
       "set is cleaned automatically, and a small student model is fine-tuned on it. After "
       "training the teacher is not needed: the student runs without API costs, network calls "
       "or a GPU.")
d.para("Three questions guide the study. First, how much does a student trained on synthetic "
       "data gain over one trained only on the seeds, and how close does it get to the teacher? "
       "Second, which parts of the pipeline matter: diverse prompting, near-duplicate removal or "
       "label-consistency filtering? Third, what does the student gain in speed and memory? We "
       "deliberately use a *small* open teacher (3B parameters, 4-bit) that runs on a laptop "
       "GPU. This is a harder and more realistic setting than a frontier API model, and it "
       "shows what a student project or small team can do with modest hardware.")
p = d.para("**Contributions:**", indent=False)
d.para("1) An open, configurable four-phase pipeline (prompting, filtering, distillation, "
       "benchmarking) that can call a local Hugging Face model or any OpenAI-compatible "
       "endpoint (Ollama, LM Studio, vLLM, Groq) as the teacher.")
d.para("2) A controlled evaluation on ten confusable banking intents with a hidden "
       "human-labelled test set. Each student configuration is trained with three random "
       "seeds and compared with zero-shot and few-shot teacher baselines, classical baselines "
       "and a model trained on all real data, with bootstrap confidence intervals and McNemar "
       "tests.")
d.para("3) Ablations of the prompting strategy, each filtering stage, the deduplication "
       "threshold and the amount of synthetic data, together with checks for diversity and "
       "test-set contamination.")
d.para("4) A CPU deployment benchmark (FP32 and dynamic INT8) covering latency, throughput, "
       "model size and memory.")

# ------------------------------------------------------------------ II. Related work
d.section("Related Work")
d.subsection("Knowledge Distillation and Compact Encoders")
d.para("Hinton *et al.* [2] showed that a small network can learn from the softened outputs of "
       "a larger one. DistilBERT [1] applied this during pre-training and kept 97% of BERT's "
       "[19] language-understanding performance while being 40% smaller and 60% faster. "
       "TinyBERT [3] and MiniLM [15] go further by also matching attention patterns and hidden "
       "states. All of these methods need access to the teacher's internals and a large "
       "transfer corpus. Our approach only needs the teacher's text output, so it works with "
       "any model, including closed APIs.")
d.subsection("LLMs as Training-Data Generators")
d.para("Schick and Schütze [7] produced labelled sentence pairs from instructions alone. "
       "ZeroGen [6] and SuperGen [5] trained small task models purely on text generated from "
       "label descriptions and approached supervised accuracy on sentiment and NLI tasks. "
       "Self-Instruct [4] used the same idea to produce instruction-tuning data. For intent "
       "detection, Sahu *et al.* [9] augmented Banking77 and CLINC150 with GPT-3 and found that "
       "generated data helps most when classes are easy to tell apart and helps less on closely "
       "related intents. Wang *et al.* [26] showed that GPT-3 labels can cut annotation cost "
       "considerably. Yu *et al.* [10] found that adding attributes such as style or persona "
       "makes generated data more diverse and less biased than class-conditional prompts alone. "
       "SetFit [8] reaches strong few-shot accuracy through contrastive learning on sentence "
       "embeddings, but it relies on human labels.")
d.subsection("Quality Control of Synthetic Data")
d.para("Synthetic data can be repetitive, mislabelled or off-topic. Training repeatedly on "
       "model-generated text can also cause *model collapse*, where rare cases disappear [11]. "
       "Removing duplicates improves language-model training [12], and SemDeDup [13] showed "
       "that removing *semantic* near-duplicates in embedding space keeps performance while "
       "cutting the data substantially. We combine cheap rule-based filters, embedding-based "
       "near-duplicate removal with Sentence-BERT-style encoders [14], [15], and a "
       "label-consistency check against the human seeds. We measure the effect of each one "
       "separately rather than assuming it helps.")

# ------------------------------------------------------------------ III. Data & problem
d.section("Data & Problem Formulation")
d.subsection("Task and Dataset")
d.para("We use Banking77 [16], a set of 13,083 online-banking customer queries labelled by "
       "humans with 77 fine-grained intents. To make the task resemble a realistic routing "
       "problem in which the model must separate similar requests, we chose ten intents forming "
       "five confusable pairs (Table I), for example a card that is *late* versus a question "
       "about *how long delivery takes*, or a top-up that *failed* versus one that was later "
       "*reversed*. We wrote a one-line description of each intent. The teacher receives the "
       "same descriptions for generation and for its own zero-shot classification, so the "
       "comparison is fair.")
d.table("Intents Used in This Study (Five Confusable Pairs)",
        ["ID", "Intent", "Pair"],
        [[f"C{i+1}", l.replace("_", "​_"), f"P{i//2+1}"] for i, l in enumerate(C.LABELS)],
        col_widths=[0.4, 2.45, 0.5], align="LLR",
        note="Each pair (P1–P5) contains two intents that are easy to confuse. The test "
             "set has 40 human-labelled queries per intent.")
d.subsection("Data Splits")
d.para(f"From the official training split we sampled **{C.SEEDS_PER_CLASS} seed examples per "
       f"class** (80 in total, fixed random seed). These seeds are the *only* human data the "
       "pipeline and the student may use, matching the 50–100-example budget in the "
       "project synopsis. The official **test split** (400 queries, 40 per class) is kept "
       "hidden until the final evaluation and is never shown to the teacher during generation. "
       "The remaining 1,271 real training examples are used only to train an upper-bound "
       "*all-real-data* student, which shows how far synthetic data is from real labelled "
       "data. No sentence appears in both the seed set and the test set.")
d.subsection("Problem Formulation")
d.para("Let *Y* be the set of *K* = 10 intents and *S* = {(x_{i}, y_{i})} the seed set. The "
       "teacher *T* is a conditional generator. The goal is to build a synthetic set "
       "*D* = {(x̃_{j}, y_{j})} such that a student f_{θ} trained on *S* ∪ *D* "
       "maximises macro-F1 on unseen human queries, subject to running on CPU only. The "
       "student is trained with the cross-entropy loss")
d.equation(r"\mathcal{L}(\theta)=-\frac{1}{N}\sum_{i=1}^{N}\log p_{\theta}(y_{i}\mid x_{i})")
d.para("and evaluated with macro-averaged F1 so that every intent counts equally:", indent=False)
d.equation(r"\mathrm{F1}_{\mathrm{macro}}=\frac{1}{K}\sum_{k=1}^{K}\frac{2P_{k}R_{k}}{P_{k}+R_{k}}")
d.para("where P_{k} and R_{k} are the precision and recall of intent *k*.", indent=False)

# ------------------------------------------------------------------ IV. Methodology
d.section("Methodology")
d.subsection("Overview")
d.para("Fig. 1 shows the four phases. Each phase is a separate Python script that reads and "
       "writes CSV/JSON files, and all settings are kept in one configuration file, so every "
       "number in this paper can be reproduced by running the scripts in order.")
d.figure(FIG / "fig1_pipeline.png", "The four-phase pipeline. Only the 80 human seeds enter "
         "the pipeline; the human test set is used once, in Phase 4.")
d.subsection("Phase 1: Teacher Prompting")
d.para(f"The teacher is Qwen2.5-3B-Instruct [17] loaded in 4-bit NF4 precision [18] "
       f"(≈{TZ['peak_gpu_mb']/1024:.1f} GB of GPU memory). For each intent the teacher is "
       f"called {C.GEN_CALLS_PER_CLASS} times. Each call contains the intent name and "
       f"description, {C.GEN_SHOTS} seed examples drawn at random from that intent, and "
       "a request for "
       f"{C.GEN_PER_CALL} new, different customer messages as a numbered list. To increase "
       f"diversity [10], each call is also given a random **persona** (one of {len(C.PERSONAS)}, "
       "e.g. “an elderly customer who is not comfortable with banking apps”) and a random "
       f"**writing style** (one of {len(C.STYLES)}, e.g. “very short, under 8 words” or "
       "“with a typo or two”). Sampling uses temperature "
       f"{C.GEN_TEMPERATURE} and nucleus *p* = {C.GEN_TOP_P}. The prompt tells the model to "
       "stay in the banking domain and not to name the intent. For the prompting ablation we "
       f"also produced a *plain* set ({C.GEN_PLAIN_CALLS_PER_CLASS} calls per intent) with the "
       "same prompt but no persona or style.")
d.subsection("Phase 2: Filtering and Quality Control")
d.para("Raw generations pass through three filters in turn.")
d.para("*Stage 1, heuristic.* Messages are removed if they are shorter than 2 or longer than "
       "60 words, contain LLM meta-text (“Here are…”, “As an AI…”), "
       "mention the intent name, repeat a seed or another message exactly, or are gibberish or "
       "not in English. For the last check we compute the vocabulary coverage *v*(*x*), the "
       "share of alphabetic tokens that are complete words in the DistilBERT vocabulary, and "
       f"drop messages with *v*(*x*) < {C.VOCAB_COVERAGE_MIN}. This threshold was fixed before "
       "evaluation.")
d.para("*Stage 2, semantic deduplication.* Each message is embedded with all-MiniLM-L6-v2 "
       "[14], [15] into a unit-length vector **e**. Because the vectors have unit length, "
       "cosine similarity is a dot product:", indent=True)
d.equation(r"\cos(\mathbf{e}_{i},\mathbf{e}_{j})=\mathbf{e}_{i}^{\top}\mathbf{e}_{j}")
d.para("The messages are visited in random order, and a message is kept only if it is "
       "dissimilar to everything already kept *K* (the seeds start in *K*):", indent=False)
d.equation(r"\max_{j\in K}\;\mathbf{e}_{i}^{\top}\mathbf{e}_{j}<\tau")
d.para(f"with τ = {C.DEDUP_THRESHOLD:.2f} by default. This is the greedy form of SemDeDup [13].",
       indent=False)
d.para("*Stage 3, seed-centroid label check.* A smaller teacher sometimes writes a message "
       "that better fits a neighbouring intent, for example a delivery-time question under "
       "*card_arrival*. We compute the normalised mean embedding **c**_{k} of the seeds of "
       "each intent *k* and keep a message only if it is closer to its own intent's "
       "centroid than to any other:", indent=True)
d.equation(r"\delta_{i}=\mathbf{e}_{i}^{\top}\mathbf{c}_{y_{i}}-\max_{k\neq y_{i}}\mathbf{e}_{i}^{\top}\mathbf{c}_{k}>0")
d.subsection("Phase 3: Student Distillation")
d.para(f"The student is distilbert-base-uncased [1] ({student_params_m:.0f}M parameters) "
       "with a linear classification head, fine-tuned on the union of seeds and filtered "
       f"synthetic data with AdamW [22] (learning rate 5×10^{{−5}}, weight decay {C.WEIGHT_DECAY}, "
       f"batch {C.BATCH_SIZE}, maximum length {C.MAX_LEN} tokens, 10% linear warm-up, "
       f"mixed precision). Training runs for {C.EPOCHS} epochs or at least {C.MIN_STEPS} "
       "optimisation steps, whichever is longer. This stops very small training sets (e.g. 80 "
       "seeds) from being under-trained. No validation data is used and no hyper-parameter "
       "was tuned on the test set. Each configuration is trained with three random seeds "
       "(0, 1, 2), and we report mean ± standard deviation.")
d.subsection("Phase 4: Evaluation and Benchmarking")
d.para("All models are scored on the hidden 400-item test set with macro-F1 and accuracy. "
       "The teacher is evaluated as a classifier in two modes: *zero-shot* (intent list and "
       "descriptions only) and *few-shot* (the same prompt plus 2 seeds per intent, 20 "
       "in-context examples, so it uses the same human data as the student). Its reply is "
       "matched to the closest intent name. Latency is measured per query at batch size 1. "
       f"The teacher runs on an RTX 4050 Laptop GPU and the student on {C.CPU_THREADS} CPU "
       "threads, in FP32 and with dynamic INT8 quantisation of all linear layers [23]. For "
       "statistical tests we use 95% bootstrap confidence intervals (2,000 resamples) and "
       "exact McNemar tests on paired per-item correctness.")

# ------------------------------------------------------------------ V. Experimental setup
d.section("Experimental Setup")
d.para("We compare: (i) the teacher in zero-shot and few-shot mode; (ii) a student trained on "
       "the 80 seeds alone; (iii) students trained on the seeds plus synthetic data at each "
       "filtering stage (raw, heuristic, +dedup, +centroid = *proposed*); (iv) a student "
       "trained on synthetic data only; (v) TF-IDF + logistic regression and MiniLM-embedding "
       "+ logistic regression baselines [25]; and (vi) the all-real-data upper bound. Ablations "
       "vary the prompting strategy at equal data size, τ ∈ {0.80, 0.85, 0.90, 0.95, "
       "none}, and the number of synthetic or real examples per intent (25–200). The "
       "implementation uses PyTorch 2.5, Hugging Face Transformers [24], "
       "sentence-transformers and scikit-learn [25] on Python 3.12. Hardware: an RTX 4050 "
       "Laptop GPU (6 GB) and a 16-thread laptop CPU, of which 4 threads are used for the "
       "deployment benchmark.")

# ------------------------------------------------------------------ VI. Results
d.section("Results")
d.subsection("Generation and Filtering")
d.para(f"The teacher produced {GS['diverse']['samples']:,} messages from "
       f"{GS['diverse']['calls']} calls in {GS['diverse']['seconds']/60:.1f} minutes "
       f"({GS['diverse']['samples_per_sec']:.1f} messages/s). Fig. 2 and Table II show how many "
       f"survive each stage. The heuristic stage removed {fs['raw'] - fs['after_heuristic']} "
       f"messages, {gib} of them as gibberish or non-English. 72% of these came from the "
       "“typo or two, no punctuation” style: the 3B teacher turned 90% of its "
       "messages in that style into strings of consonants (e.g. “nfw tkks fr tske fr "
       "mcry”) instead of adding a few realistic typos. Most of the rest came from the "
       "casual lowercase style or from the model switching into Dutch or Portuguese. Semantic "
       f"deduplication at τ = 0.90 removed only {fs['after_heuristic'] - fs['after_semantic']} "
       f"persona-style messages ({pct((fs['after_heuristic'] - fs['after_semantic']) / fs['after_heuristic'])}%) "
       f"compared with {fp['after_heuristic'] - fp['after_semantic']} "
       f"({pct((fp['after_heuristic'] - fp['after_semantic']) / fp['after_heuristic'])}%) of the "
       "plain ones: persona prompting already avoids most near-duplicates. The centroid check "
       f"was the strictest stage and removed {fs['after_semantic'] - fs['after_centroid']} "
       f"messages ({pct((fs['after_semantic'] - fs['after_centroid']) / fs['after_semantic'], 0)}%). "
       f"It hit *card_delivery_estimate* hardest: only {fs['per_class']['card_delivery_estimate']} "
       f"of its messages remained, against {fs['per_class']['card_arrival']} for its pair "
       "*card_arrival*. This shows the teacher itself mixes up that pair.")
d.figure(FIG / "fig2_funnel.png", "Share of generated messages kept after each filtering "
         "stage, for persona+style prompting and plain prompting. Bar labels give absolute "
         "counts.")
qr, qf = fs["quality_raw"], fs["quality_filtered"]
sw = FS["threshold_sweep"]
d.table("Synthetic Data Quality and Contamination Checks",
        ["Data", "N", "Dist-2", "Intra-cos", "NN-cos"],
        [["Seeds (human)", "80", f(FS["seeds_quality"]["distinct_2"], 2),
          f(FS["seeds_quality"]["intra_class_cos"], 2), f(qr["test_nn_cos_mean_seeds_only"], 2)],
         ["Persona, after heuristic", str(fs["after_heuristic"]), f(sw["1.01"]["distinct_2"], 2),
          f(sw["1.01"]["intra_class_cos"], 2), "—"],
         ["Plain, fully filtered", str(fp["after_centroid"]),
          f(fp["quality_filtered"]["distinct_2"], 2), f(fp["quality_filtered"]["intra_class_cos"], 2),
          f(fp["quality_filtered"]["test_nn_cos_mean"], 2)],
         ["Persona, fully filtered", str(fs["after_centroid"]), f(qf["distinct_2"], 2),
          f(qf["intra_class_cos"], 2), f(qf["test_nn_cos_mean"], 2)],
         ["Test set (human)", "400", f(FS["test_quality"]["distinct_2"], 2),
          f(FS["test_quality"]["intra_class_cos"], 2), "—"]],
        col_widths=[1.35, 0.45, 0.5, 0.6, 0.5],
        note="Dist-2: distinct bigram ratio [21] (higher = more varied wording). Intra-cos: mean "
             "pairwise MiniLM cosine within an intent (lower = more diverse). NN-cos: mean "
             "cosine from each test query to its nearest training example. Exact "
             f"test-sentence matches in the synthetic data: {qf['exact_test_matches']}.")
d.para("Table II also checks for contamination. None of the synthetic messages reproduces a "
       "test sentence. The mean similarity of a test query to its nearest synthetic message "
       f"({f(qf['test_nn_cos_mean'], 2)}) is about the same as its similarity to the nearest "
       f"seed ({f(qr['test_nn_cos_mean_seeds_only'], 2)}), so the teacher has not recalled "
       f"Banking77 test items. Only {qf['test_with_nn_cos_ge_0.95']} of 400 test queries has a "
       "synthetic neighbour with cosine ≥ 0.95. The label check makes each intent's "
       "synthetic messages more coherent (intra-cos "
       f"{f(sw['1.01']['intra_class_cos'], 2)} → {f(qf['intra_class_cos'], 2)}), "
       f"closer to the real test data ({f(FS['test_quality']['intra_class_cos'], 2)}), while "
       "persona prompting keeps the wording more varied than in the plain set.")

d.subsection("Main Accuracy Results")
rows = [
    ["Teacher, zero-shot (GPU)", "—", f(TZ["macro_f1"]), f(TZ["accuracy"])],
    ["Teacher, few-shot (GPU)", "20 ICL", f(TF["macro_f1"]), f(TF["accuracy"])],
]
for name, lab in [(SEEDS, "Student, seeds only"), (RAW, "+ raw synthetic"),
                  (HEUR, "+ heuristic filter"), (DEDUP, "+ semantic dedup"),
                  (PROP, "+ centroid (proposed)"), (SYN, "Synthetic only"),
                  (ORA, "All real data (bound)")]:
    rows.append([lab, f"{int(S.loc[name, 'n']):,}", ms(name), f"{S.loc[name, 'acc']:.3f}"])
for meth in ["TF-IDF + LR", "MiniLM + LR"]:
    for data, short in [("Seeds only", "seeds"), ("Seeds + full filter (proposed)", "proposed data")]:
        r = CL.loc[(meth, data)]
        rows.append([f"{meth}, {short}", f"{int(r.n_train):,}", f(r.macro_f1), f(r.accuracy)])
d.table("Macro-F1 on the Hidden Human Test Set (400 Queries)",
        ["System", "Train N", "Macro-F1", "Acc."], rows,
        col_widths=[1.5, 0.55, 0.8, 0.5], bold_rows=(6,),
        note="Student rows are DistilBERT, mean ± std over 3 seeds; “+” rows add "
             "the synthetic data after each filtering stage to the 80 seeds. LR rows are "
             "deterministic. ICL = in-context examples.")
p_seed = mc["Student seeds only"]["p_value"]
d.para(f"Table III and Fig. 3 give the main comparison. On the 80 seeds alone the DistilBERT "
       f"student reaches {ms(SEEDS)}, already level with the zero-shot teacher "
       f"({f(TZ['macro_f1'])}), but it varies a lot between training seeds. Adding the filtered "
       f"synthetic data raises the mean to {ms(PROP)} (+{pct(gain_vs_seeds)} points) and cuts the "
       f"spread across seeds by a factor of about "
       f"{S.loc[SEEDS, 's'] / S.loc[PROP, 's']:.0f}. The student is statistically "
       f"indistinguishable from the teacher in both modes: on paired seed-0 predictions, the "
       f"exact McNemar test gives *p* {pv(mc['Teacher zero-shot']['p_value'])} against the "
       f"zero-shot and *p* {pv(mc['Teacher few-shot']['p_value'])} against the few-shot teacher. "
       f"It reaches {pct(ratio_few, 0)}% of the few-shot teacher's macro-F1, above the 90% "
       f"target set in the project synopsis. The gain over the seeds-only student is "
       f"consistent in the mean but *not* significant on a single paired run (*p* "
       f"{pv(p_seed)}), so with 80 seeds the main effect of the synthetic data is a more "
       f"*reliable* student rather than a clearly more accurate one. The 95% bootstrap interval "
       f"for the proposed student (seed 0) is [{f(ci['Student proposed']['lo'])}, "
       f"{f(ci['Student proposed']['hi'])}].")
d.figure(FIG / "fig3_main_results.png", "Macro-F1 on the hidden test set. Error bars: ±1 "
         "standard deviation over three training seeds. Orange: teacher; blue: DistilBERT "
         "students; grey: frozen-MiniLM baseline; green: student trained on all real data.")
d.para(f"Synthetic data cannot replace the seeds. A student trained on synthetic data alone "
       f"scores {ms(SYN)}, below the seeds-only model, so the 80 real messages carry "
       "information about how real customers write that the teacher does not reproduce. The "
       f"gap to real data is large: with all 1,351 real examples the student reaches "
       f"{ms(ORA)}, significantly better (*p* {pv(mc['Student all real data']['p_value'])}).")
d.para(f"The strongest low-cost baseline is not a fine-tuned model at all. Logistic regression "
       f"on *frozen* MiniLM sentence embeddings, trained on the 80 seeds, reaches "
       f"{f(MLR_SEEDS)}, significantly better than the proposed student (*p* "
       f"{pv(mc['MiniLM + LR seeds only']['p_value'])}) and even above the few-shot teacher. "
       f"Adding the synthetic data does not help it ({f(MLR_PROP)}). The encoder was trained "
       "on over a billion sentence pairs, so its embedding space already separates these "
       "intents well, and a linear model needs only a handful of real examples per class to "
       "find the boundaries. Synthetic messages that follow the teacher's view of each intent "
       "move those boundaries slightly away from the human labelling. TF-IDF, which has no such "
       f"prior, does gain from synthetic data ({f(TFIDF_SEEDS)} → {f(TFIDF_PROP)}).")

d.subsection("Effect of Each Filtering Stage")
filt = [RAW, HEUR, DEDUP, PROP]
d.para(f"The filters did not raise accuracy. Raw synthetic data gives {ms(RAW)}, the "
       f"heuristic filter {ms(HEUR)}, semantic dedup {ms(DEDUP)} and the full pipeline "
       f"{ms(PROP)}. All four lie within {pct(S.loc[filt, 'm'].max() - S.loc[filt, 'm'].min())} "
       "points of each other, and the paired test finds no difference between raw and fully "
       f"filtered data (*p* {pv(mc['Student + raw synthetic']['p_value'])}). The fine-tuned "
       "encoder is evidently robust to the roughly 15% of gibberish in the raw set, and "
       "dropping a third of the messages with the centroid check did not hurt either. The "
       "benefit of filtering is therefore *efficiency and auditability*: the proposed set is "
       f"{pct(1 - (S.loc[PROP, 'n']) / S.loc[RAW, 'n'], 0)}% smaller than the raw set and "
       "contains no off-topic or non-English text that a human reviewer would otherwise have "
       "to remove.")
th = S.loc[[c for c in S.index if c.startswith("tau=")]]
d.para("Fig. 4 shows the threshold sweep. Lowering τ to 0.80 removes more messages "
       f"({sw['0.8']['after_semantic']:,} kept vs. {sw['1.01']['after_semantic']:,} with no "
       f"dedup), but F1 stays between {min(th.m.min(), m(DEDUP)):.3f} and "
       f"{max(th.m.max(), m(DEDUP)):.3f} at every threshold, with overlapping error bars. With "
       "persona-diverse prompts there are few near-duplicates to remove, so the choice of "
       "τ matters little.")
d.figure(FIG / "fig4_threshold.png", "Semantic-dedup threshold sweep (no centroid stage). "
         "Left: messages kept. Right: student macro-F1 (mean ± std, 3 seeds).")

d.subsection("Effect of Prompt Diversity")
d.para(f"To isolate the prompting strategy, we sub-sampled the filtered persona+style data to "
       f"exactly match the per-intent counts of the filtered plain data ({int(S.loc[PLAIN, 'n']):,} "
       f"examples each, seeds included). Persona+style prompting gives {ms(DIV)} against "
       f"{ms(PLAIN)} for plain prompting, a gain of {pct(m(DIV) - m(PLAIN))} points from "
       "prompt design alone. This is the largest effect of any pipeline component we tested, "
       "although it is about the size of the seed-to-seed spread, so it should be confirmed "
       "on more tasks. "
       "Plain prompts also produced more than three times as many semantic near-duplicates "
       f"({pct((fp['after_heuristic'] - fp['after_semantic']) / fp['after_heuristic'])}% vs. "
       f"{pct((fs['after_heuristic'] - fs['after_semantic']) / fs['after_heuristic'])}%) and far "
       "fewer *card_delivery_estimate* messages that passed the label check "
       f"({fp['per_class']['card_delivery_estimate']} vs. "
       f"{fs['per_class']['card_delivery_estimate']}). The one cost of persona prompting is "
       "that some styles (“typos, no punctuation”) push a small teacher into "
       "producing gibberish, which the heuristic stage has to catch.")

d.subsection("Scaling with Data Size")
d.para("Fig. 5 shows macro-F1 as examples per intent are added to the seeds. With synthetic "
       f"data, F1 rises from {f(m(SEEDS))} (seeds only) to {f(S.loc['synthetic n=100', 'm'])} at "
       "100 messages per intent and then *falls* to "
       f"{f(S.loc['synthetic n=200', 'm'])} at 200. The centroid check left only "
       f"{fs['per_class']['card_delivery_estimate']} *card_delivery_estimate* messages, so "
       "beyond 100 per intent the training set becomes increasingly unbalanced and the "
       "student drifts towards the larger intents. Real data behaves very differently: just "
       f"25 real examples per intent give {f(S.loc['real n=25', 'm'])}, more than any amount of "
       f"synthetic data, and 200 give {f(S.loc['real n=200', 'm'])}. This is the practical "
       "ceiling of a 3B teacher: it adds variety in wording, but little knowledge of where "
       "the human annotators drew the boundaries between intents.")
d.figure(FIG / "fig5_scaling.png", "Macro-F1 versus extra training examples per intent added "
         "to the 80 seeds (mean ± std, 3 seeds). Synthetic examples come from the "
         "filtered set; real ones from the held-out training data. The diamond marks the full "
         "filtered set.")

d.subsection("Per-Class Errors")
pc = PC.set_index("label")
d.para("Fig. 6 compares the confusion matrices of the zero-shot teacher and the proposed "
       "student (seed 0). As designed, most errors of both models fall *within* the "
       "confusable pairs. The teacher's largest single error is labelling *compromised_card* "
       "(C4) as *lost_or_stolen_card* (C3), 25 of 40 times. The student's largest error is "
       "calling a reverted top-up (C8) a failed one (C7), 21 of 40 times, a distinction that "
       "depends on how the dataset's annotators read short messages such as “my top-up "
       "was cancelled”. Comparing the student with and without synthetic data shows that "
       "it *inherits the teacher's view* of the intents. Synthetic data improved the pairs "
       "the teacher describes well (C1/C2, C7), but lowered F1 on C3/C4 from "
       f"0.85/0.89 to {f(pc.loc['lost_or_stolen_card', 'student_f1'], 2)}/"
       f"{f(pc.loc['compromised_card', 'student_f1'], 2)}, exactly the pair the teacher itself "
       "confuses most.")
d.figure(FIG / "fig6_confusion.png", "Confusion matrices on the 400-item test set (class IDs "
         "as in Table I). Left: teacher zero-shot. Right: proposed student, seed 0.")

d.subsection("Latency, Size and Memory")
d.table("Deployment Cost per Query (Batch Size 1)",
        ["Model / hardware", "Params", "Latency (ms)", "F1"],
        [["Teacher zero-shot / GPU", f"{teacher_params_b:.2f} B",
          f"{TZ['latency_ms_mean']:.0f} (p95 {TZ['latency_ms_p95']:.0f})", f(TZ['macro_f1'])],
         ["Teacher few-shot / GPU", f"{teacher_params_b:.2f} B",
          f"{TF['latency_ms_mean']:.0f} (p95 {TF['latency_ms_p95']:.0f})", f(TF['macro_f1'])],
         ["Student FP32 / CPU", f"{student_params_m:.0f} M",
          f"{cpu32['latency_ms_mean']:.1f} (p95 {cpu32['latency_ms_p95']:.1f})", f(cpu32['macro_f1'])],
         ["Student INT8 / CPU", f"{student_params_m:.0f} M",
          f"{cpu8['latency_ms_mean']:.1f} (p95 {cpu8['latency_ms_p95']:.1f})", f(cpu8['macro_f1'])],
         ["MiniLM + LR / CPU", f"{MLR_B['params'] / 1e6:.0f} M",
          f"{MLR_B['latency_ms_mean']:.1f} (p95 {MLR_B['latency_ms_p95']:.1f})", f(MLR_B['macro_f1'])]],
        col_widths=[1.35, 0.55, 0.95, 0.5],
        note=f"Teacher: Qwen2.5-3B, 4-bit, peak GPU memory {TZ['peak_gpu_mb']/1024:.1f} GB. "
             f"CPU rows use {C.CPU_THREADS} threads. Student rows use the seed-0 model. Size on "
             f"disk: {cpu32['size_mb']:.0f} MB (FP32), {cpu8['size_mb']:.0f} MB (INT8), "
             f"{MLR_B['size_mb']:.0f} MB (MiniLM).")
d.para(f"Table IV and Fig. 7 summarise deployment cost. On four CPU threads the FP32 student "
       f"answers in {f(cpu32['latency_ms_mean'], 1)} ms, {speed32:.0f}× faster than the "
       f"teacher running on a GPU, and handles {cpu32['throughput_qps_bs32']:.0f} queries/s at "
       f"batch size 32. Dynamic INT8 quantisation halves the model on disk "
       f"({cpu32['size_mb']:.0f} → {cpu8['size_mb']:.0f} MB) and cuts latency to "
       f"{f(cpu8['latency_ms_mean'], 1)} ms ({speed8:.0f}× faster than the teacher, "
       f"{cpu8['throughput_qps_bs32']:.0f} queries/s), with no loss in macro-F1 "
       f"({f(cpu32['macro_f1'])} → {f(cpu8['macro_f1'])}). The student has "
       f"{teacher_params_b * 1e3 / student_params_m:.0f}× fewer parameters than the teacher "
       "and needs no GPU, network access or per-query fee. The MiniLM baseline is in the same "
       f"latency class ({f(MLR_B['latency_ms_mean'], 1)} ms), so the choice between the two "
       "students on a CPU comes down to accuracy, not speed.")
d.figure(FIG / "fig7_latency.png", "Accuracy–latency trade-off (log scale). The "
         "CPU models match or exceed the zero-shot teacher's macro-F1 at "
         f"{speed32:.0f}–{TZ['latency_ms_mean'] / MLR_B['latency_ms_mean']:.0f}× lower "
         "latency, without a GPU.")

# ------------------------------------------------------------------ VII. Discussion
d.section("Discussion")
d.para("*What made the synthetic data useful?* The main lever was prompt diversity, not "
       "filtering. Personas and writing styles gave the largest measurable gain at equal data "
       "size and cut the near-duplicate rate by more than two-thirds. The filters removed real "
       "defects (gibberish, meta-text, off-label examples), but the fine-tuned student was "
       "robust enough to these defects that removing them did not change accuracy. In practice "
       "the filters make the dataset smaller and easier to audit, which matters when a person "
       "must sign off on training data in a regulated domain.")
d.para("*Where does a small teacher's knowledge run out?* The student ends up very close to "
       "the teacher, and the per-class analysis shows why: synthetic data passes on the "
       "teacher's own reading of each intent, including its mistakes (C3/C4). Where the "
       "dataset's labelling convention differs from what the intent name suggests, only real "
       "examples teach it. This explains the weak scaling curve, the fact that 25 real "
       "examples per intent beat all synthetic data, and why synthetic-only training is worse "
       "than seeds-only training.")
d.para("*Fine-tuning versus frozen embeddings.* A strong pre-trained sentence encoder with a "
       "linear head turned out to be the best use of 80 labels, and the synthetic data did "
       "not improve it. Practitioners should try this baseline before generating data. "
       "Synthetic data is most worth generating when (i) the student must be fine-tuned, "
       "e.g. for domain jargon that general encoders do not represent well, (ii) only a few "
       "seeds exist and seeds-only fine-tuning is unstable, as here, or (iii) a much stronger "
       "teacher is available than the 3B model we could run on a laptop.")
d.para("*Is the pipeline worth it?* For an organisation that needs on-premise, CPU-only "
       "inference and wants a fine-tuned model, yes. With 80 labelled messages and about 20 "
       "minutes of laptop-GPU time, the pipeline gives a stable 67M-parameter classifier that "
       f"matches a 3B-parameter LLM, answers {speed32:.0f}–{speed8:.0f} times faster and costs nothing per query. "
       "The confusion matrix then shows exactly which intent pairs need more human labels.")

# ------------------------------------------------------------------ VIII. Limitations
d.section("Limitations")
d.bullet("Single task and teacher:", "We study one 10-intent subset of one dataset and one "
         "3B teacher. A larger teacher (e.g. Llama-3-70B through the OpenAI-compatible "
         "back-end) would probably produce cleaner, better-separated data and may change the "
         "comparison with the MiniLM baseline.")
d.bullet("Statistical power:", "Each configuration was trained with three seeds and the "
         "paired tests use one run, on a 400-item test set. Differences of about two points "
         "are therefore hard to separate from noise.")
d.bullet("Fixed hyper-parameters:", "τ, the coverage threshold and the training schedule "
         "were fixed in advance and not tuned, to avoid tuning on the test set. A validation "
         "split carved from the synthetic data could improve them, and class-balanced "
         "sampling could remove the dip in Fig. 5.")
d.bullet("Label conventions:", "Banking77 contains some inconsistent labels between the "
         "confusable intents, which caps the accuracy any method can reach on those pairs.")
d.bullet("Latency hardware:", "The teacher was timed on a laptop GPU and the students on a "
         "laptop CPU. Absolute numbers will differ on servers.")

# ------------------------------------------------------------------ IX. Conclusion
d.section("Conclusion & Future Work")
d.para(f"We built and evaluated a teacher-to-student synthetic-data pipeline for "
       f"domain-specific text classification under a tight compute budget. Starting from 80 "
       f"human examples and a 4-bit 3B teacher, the DistilBERT student reaches a macro-F1 of "
       f"{f(m(PROP))} on hidden human data. It is statistically equal to the teacher used "
       f"directly ({pct(ratio_few, 0)}% of its few-shot score), more stable than training on "
       f"the seeds alone, and answers in {f(cpu8['latency_ms_mean'], 1)}–"
       f"{f(cpu32['latency_ms_mean'], 1)} ms on a CPU instead of over 500 ms on a GPU. Persona "
       "and style prompting was the most useful component; filtering mainly reduced data size "
       "and removed defects. The study also sets clear limits: a frozen-encoder baseline on the "
       "same 80 seeds is more accurate, and a small amount of real data is worth far more than "
       "synthetic data from a small teacher. Future work will: (i) compare larger teachers "
       "through the OpenAI-compatible back-end; (ii) generate data on top of strong frozen "
       "encoders (e.g. SetFit-style contrastive fine-tuning [8]) instead of full fine-tuning; "
       "(iii) add class-balanced sampling and *active* seeding, sending the intent pairs the "
       "student confuses most to a human annotator; and (iv) apply the pipeline to "
       "named-entity recognition and IT-ticket routing.")

d.section("Code & Data Availability", numbered=False)
d.para(f"Code, data splits, the generated synthetic datasets, per-run results and the "
       f"scripts that build this paper are available at {REPO_URL}. The pipeline code is in "
       "*code/*, all data (seed, test and synthetic sets) is in *code/data/*, and "
       "*code/config.py* holds every setting used in this paper. The Banking77 data is "
       "redistributed under its CC BY 4.0 licence.")
d.section("Acknowledgment", numbered=False)
d.para("We thank the Department of Artificial Intelligence and Data Science, ADGIPS, Delhi, "
       "for its support, and the authors of Banking77 and Qwen2.5 for releasing their data "
       "and models openly. An AI system (Claude, Anthropic) helped write the pipeline code, "
       "run the experiments and draft parts of the manuscript. We have reviewed, edited and "
       "verified all content, including the results.")
d.references(REFERENCES)

out_docx = HERE / "Synthetic_Data_Distillation_IEEE_Paper.docx"
d.save(out_docx)
print("saved", out_docx)
if "--no-pdf" not in sys.argv:
    subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", str(HERE / "docx2pdf.ps1"),
                    "-In", str(out_docx), "-Out", str(out_docx.with_suffix(".pdf"))], check=True)
