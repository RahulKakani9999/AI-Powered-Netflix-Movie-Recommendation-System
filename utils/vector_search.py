import ast
import sys
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SAVED_MODELS_DIR, FAISS_TOP_K


def _parse_list_field(val) -> list[str]:
    """Parse a field that may be a list or a string representation of a list."""
    if isinstance(val, list):
        return [str(x) for x in val]
    try:
        result = ast.literal_eval(str(val))
        if isinstance(result, list):
            return [str(x) for x in result]
    except (ValueError, SyntaxError):
        pass
    # Last resort: comma-separated string
    return [s.strip() for s in str(val).split(",") if s.strip()]


class VectorSearch:
    def __init__(self):
        index_path = SAVED_MODELS_DIR / "faiss_index.bin"
        movies_path = SAVED_MODELS_DIR / "indexed_movies.csv"
        model_name_path = SAVED_MODELS_DIR / "embedding_model_name.txt"

        print("Loading FAISS index ...")
        self.index = faiss.read_index(str(index_path))

        print("Loading indexed movies ...")
        self.movies = pd.read_csv(movies_path)
        self.movies["movie_id"] = self.movies["movie_id"].astype(int)

        # Use the exact model that encoded the index vectors
        model_name = model_name_path.read_text(encoding="utf-8").strip()
        print(f"Loading embedding model: {model_name} ...")
        self.model = SentenceTransformer(model_name)

        print(f"VectorSearch ready — {self.index.ntotal:,} vectors indexed.\n")

    # ── Internal helpers ──────────────────────────────────────

    def _encode_query(self, query: str) -> np.ndarray:
        """Return a (1, dim) float32 L2-normalised embedding for query."""
        vec = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(vec)
        return vec

    # ── Public API ────────────────────────────────────────────

    def search(self, query: str, top_k: int = FAISS_TOP_K) -> pd.DataFrame:
        """
        Encode query, run FAISS inner-product search, and return a DataFrame
        of matched movies with a similarity_score column (0–1, higher = closer).
        """
        vec = self._encode_query(query)
        scores, indices = self.index.search(vec, top_k)

        # FAISS pads with -1 when the index has fewer entries than top_k
        valid = indices[0] >= 0
        valid_indices = indices[0][valid]
        valid_scores = scores[0][valid]

        hits = self.movies.iloc[valid_indices].copy().reset_index(drop=True)
        hits["similarity_score"] = valid_scores.round(4)
        return hits

    def search_by_movie_titles(self, titles: list[str], top_k: int = 5) -> pd.DataFrame:
        """
        Combine title strings into a single query and return similar movies,
        excluding any exact title matches from the input list.
        """
        query = " ".join(titles)
        # Fetch extra results so we still have top_k after filtering input titles
        raw = self.search(query, top_k=top_k + len(titles))

        lower_titles = {t.lower() for t in titles}
        mask = ~raw["title"].fillna("").str.lower().isin(lower_titles)
        return raw[mask].head(top_k).reset_index(drop=True)

    def get_movie_context(self, movie_ids: list[int]) -> str:
        """
        Build a structured text block for the given movie IDs, formatted for
        use as context in an LLM prompt.  Returns an empty string if no IDs match.
        """
        if not movie_ids:
            return ""

        rows = self.movies[self.movies["movie_id"].isin(movie_ids)]
        if rows.empty:
            return ""

        blocks = []
        for _, row in rows.iterrows():
            year = f" ({int(row['year'])})" if pd.notna(row.get("year")) else ""
            lines = [f"Title: {row.get('title', 'Unknown')}{year}"]

            overview = row.get("overview")
            if pd.notna(overview) and str(overview).strip():
                lines.append(f"Overview: {overview}")

            for col, label in [("genres", "Genres"), ("keywords", "Keywords")]:
                val = row.get(col)
                if pd.notna(val) and str(val).strip() not in ("", "[]", "nan"):
                    names = _parse_list_field(val)
                    if names:
                        lines.append(f"{label}: {', '.join(names)}")

            cast = row.get("cast")
            if pd.notna(cast) and str(cast).strip() not in ("", "[]", "nan"):
                names = _parse_list_field(cast)
                if names:
                    lines.append(f"Cast: {', '.join(names)}")

            director = row.get("director")
            if pd.notna(director) and str(director).strip():
                lines.append(f"Director: {director}")

            blocks.append("\n".join(lines))

        return "\n\n".join(blocks)


# ── Smoke test ────────────────────────────────────────────────

if __name__ == "__main__":
    vs = VectorSearch()

    # Free-text query
    query = "space adventure with robots and artificial intelligence"
    print(f"Query: '{query}'")
    results = vs.search(query, top_k=5)
    for _, row in results.iterrows():
        year = int(row["year"]) if pd.notna(row.get("year")) else "?"
        print(f"  {row['similarity_score']:.4f}  {row.get('title', 'Unknown')} ({year})")

    print()

    # Title-based search
    titles = ["The Matrix", "Inception"]
    print(f"Similar to: {titles}")
    results = vs.search_by_movie_titles(titles, top_k=5)
    for _, row in results.iterrows():
        print(f"  {row['similarity_score']:.4f}  {row.get('title', 'Unknown')}")

    print()

    # LLM context block
    sample_ids = results["movie_id"].head(2).tolist()
    ctx = vs.get_movie_context(sample_ids)
    print("LLM context sample:")
    print(ctx)
