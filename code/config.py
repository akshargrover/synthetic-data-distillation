"""
Central configuration for the Synthetic Data Generation Pipeline.

Everything that affects results (class list, seed counts, model names,
thresholds, hyper-parameters, random seeds) lives here so that every
script in the pipeline reads the same values.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
MODEL_DIR = ROOT / "models"
for _d in (DATA_DIR, RESULTS_DIR, FIG_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Task definition: 10 banking intents (5 deliberately confusable pairs)
# Source dataset: Banking77 (Casanueva et al., 2020), human-annotated.
# The descriptions are the "task specification" a practitioner would write;
# the SAME descriptions are given to the teacher for generation and for
# zero-shot classification, so the comparison is fair.
# --------------------------------------------------------------------------
DATASET_ID = "PolyAI/banking77"
DATASET_REVISION = "refs/convert/parquet"

CLASSES = {
    "card_arrival": "The customer ordered a card and it has not arrived yet; they ask where it is or report it is late.",
    "card_delivery_estimate": "The customer asks how long card delivery takes or when a new card will be delivered (before any delay).",
    "lost_or_stolen_card": "The customer's physical card was lost or stolen and they want to report or block it.",
    "compromised_card": "The customer suspects their card details were compromised, hacked or used fraudulently while they still have the card.",
    "exchange_rate": "The customer asks what exchange rate is used or how currency conversion rates are determined.",
    "card_payment_wrong_exchange_rate": "The customer complains that a card payment abroad was charged at a wrong or bad exchange rate.",
    "top_up_failed": "The customer's attempt to top up / add money to the account failed or was declined.",
    "top_up_reverted": "A top-up initially went through but was later reverted, cancelled or returned.",
    "transfer_not_received_by_recipient": "The customer sent a transfer but the recipient says they have not received the money.",
    "cash_withdrawal_charge": "The customer asks about or complains about a fee charged for an ATM cash withdrawal.",
}
LABELS = list(CLASSES.keys())
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# --------------------------------------------------------------------------
# Data splits
# --------------------------------------------------------------------------
SEEDS_PER_CLASS = 8          # 10 x 8 = 80 human seed examples (synopsis: 50-100)
DATA_SEED = 42

# --------------------------------------------------------------------------
# Phase 1: Teacher LLM generation
# --------------------------------------------------------------------------
TEACHER_BACKEND = "hf"       # "hf" (local transformers) or "openai" (any OpenAI-compatible server: Ollama, LM Studio, vLLM, Groq)
TEACHER_MODEL = "Qwen/Qwen2.5-3B-Instruct"
TEACHER_LOAD_4BIT = True     # NF4 quantisation so the 3B teacher fits a 6 GB laptop GPU
OPENAI_BASE_URL = "http://localhost:11434/v1"   # used only when TEACHER_BACKEND == "openai"
OPENAI_MODEL = "llama3.1:8b"

GEN_CALLS_PER_CLASS = 30     # each call asks for GEN_PER_CALL samples
GEN_PER_CALL = 10            # => up to 300 raw samples per class (3,000 total)
GEN_PLAIN_CALLS_PER_CLASS = 15   # ablation: plain prompting without persona/style => up to 150/class
GEN_SHOTS = 4                # seed examples shown in each generation prompt
GEN_TEMPERATURE = 0.9
GEN_TOP_P = 0.95
GEN_MAX_NEW_TOKENS = 420
GEN_BATCH = 25               # prompts generated in parallel on the GPU
GEN_SEED = 1234

PERSONAS = [
    "an elderly customer who is not comfortable with banking apps",
    "a busy small-business owner typing quickly on a phone",
    "a university student on a tight budget",
    "a non-native English speaker who writes simple sentences",
    "a frequent traveller currently abroad",
    "an anxious first-time customer",
    "an angry customer who has already contacted support twice",
    "a very polite and formal customer",
    "a tech-savvy young professional",
    "a parent managing money for the family",
]
STYLES = [
    "very short, under 8 words",
    "casual chat style with lowercase letters",
    "with a typo or two and no punctuation",
    "a full sentence giving some background context",
    "phrased as a direct question",
    "phrased as a complaint",
    "polite and formal",
    "mentioning a specific amount, date or country",
]

# --------------------------------------------------------------------------
# Phase 2: Filtering
# --------------------------------------------------------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEDUP_THRESHOLD = 0.90               # cosine >= threshold => near-duplicate, drop
THRESHOLD_SWEEP = [0.80, 0.85, 0.90, 0.95, 1.01]   # 1.01 == no semantic dedup
MIN_WORDS, MAX_WORDS = 2, 60
VOCAB_COVERAGE_MIN = 0.70            # share of alphabetic tokens that are real English words (DistilBERT vocab); removes gibberish / non-English
CENTROID_MARGIN = 0.0                # keep if sim(own seed centroid) - max sim(other centroid) > margin

# --------------------------------------------------------------------------
# Phase 3: Student
# --------------------------------------------------------------------------
STUDENT_MODEL = "distilbert-base-uncased"
MAX_LEN = 64
LR = 5e-5
BATCH_SIZE = 32
EPOCHS = 4
MIN_STEPS = 300            # small training sets are trained for at least this many steps
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
TRAIN_SEEDS = [0, 1, 2]    # every student configuration is trained 3x
SCALING_SIZES = [25, 50, 100, 200]   # synthetic samples per class (plus "all")

# --------------------------------------------------------------------------
# Phase 4: Benchmarking
# --------------------------------------------------------------------------
CPU_THREADS = 4            # simulate a standard 4-core CPU deployment
LATENCY_QUERIES = 400
TEACHER_LATENCY_QUERIES = 100
