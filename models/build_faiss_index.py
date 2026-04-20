import sys
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, SAVED_MODELS_DIR, EMBEDDING_MODEL


def build_index():
    # ── Load metadata ─────────────────────────────────────────
    print("Loading movies_metadata.csv ...")
    df = pd.read_csv(PROCESSED_DIR / "movies_metadata.csv")
    print(f"  {len(df):,} movies loaded")

    # ── Filter movies without usable combined_text ────────────
    before = len(df)
    mask = df["combined_text"].notna() & (df["combined_text"].str.len() >= 10)
    df = df[mask].reset_index(drop=True)
    skipped = before - len(df)
    print(f"  Filtered out : {skipped:,} (missing / too-short combined_text)")
    print(f"  To be indexed: {len(df):,}")

    # ── Load embedding model ──────────────────────────────────
    print(f"\nLoading embedding model: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # ── Encode ────────────────────────────────────────────────
    print("\nEncoding combined_text ...")
    embeddings = model.encode(
        df["combined_text"].tolist(),
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True,
    )
    # FAISS requires float32; normalize_L2 is in-place
    embeddings = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    print(f"  Shape: {embeddings.shape}  dtype: {embeddings.dtype}")

    # ── Build IndexFlatIP (cosine similarity after L2 norm) ───
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"\nFAISS index: {index.ntotal:,} vectors  |  dim={dim}")

    # ── Save artefacts ────────────────────────────────────────
    # Row i in indexed_movies.csv ↔ vector i in faiss_index.bin
    index_path = SAVED_MODELS_DIR / "faiss_index.bin"
    movies_path = SAVED_MODELS_DIR / "indexed_movies.csv"
    model_name_path = SAVED_MODELS_DIR / "embedding_model_name.txt"

    faiss.write_index(index, str(index_path))
    print(f"\nSaved → {index_path}")

    df.to_csv(movies_path, index=False)
    print(f"Saved → {movies_path}")

    model_name_path.write_text(EMBEDDING_MODEL, encoding="utf-8")
    print(f"Saved → {model_name_path}")

    sep = "─" * 52
    print(
        f"\n{sep}\n"
        "Next steps:\n"
        "  python utils/vector_search.py     # smoke-test FAISS\n"
        "  python utils/llm_explainer.py     # test LLM explanations (Layer 3)\n"
        f"{sep}"
    )


if __name__ == "__main__":
    build_index()
