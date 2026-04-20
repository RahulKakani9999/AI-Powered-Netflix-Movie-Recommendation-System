import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Base paths — work on both local and Google Colab
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = DATA_DIR / "processed"
MOVIELENS_DIR = DATA_DIR / "movielens"
TMDB_DIR = DATA_DIR / "tmdb"
SAVED_MODELS_DIR = MODELS_DIR / "saved"

# Auto-create directories so imports never fail on a fresh clone / Colab runtime
for _dir in (PROCESSED_DIR, MOVIELENS_DIR, TMDB_DIR, SAVED_MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ---------------------------------------------------------------------------
# SVD (collaborative filtering) hyperparameters
# ---------------------------------------------------------------------------
SVD_N_FACTORS = 100
SVD_N_EPOCHS = 20
SVD_LR = 0.005
SVD_REG = 0.02

# ---------------------------------------------------------------------------
# Recommendation settings
# ---------------------------------------------------------------------------
TOP_N_RECOMMENDATIONS = 10
FAISS_TOP_K = 20

# ---------------------------------------------------------------------------
# Embedding model (sentence-transformers)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
