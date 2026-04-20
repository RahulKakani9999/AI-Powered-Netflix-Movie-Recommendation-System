import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.recommender import SVDRecommender
from utils.vector_search import VectorSearch
from utils.llm_explainer import LLMExplainer
from config import TOP_N_RECOMMENDATIONS


class RecommendationPipeline:
    def __init__(self):
        print("─" * 52)
        print("Initialising RecommendationPipeline ...")
        print("─" * 52)
        self.svd = SVDRecommender()
        self.vs = VectorSearch()
        self.llm = LLMExplainer()
        print("─" * 52)
        print("Pipeline ready.")
        print("─" * 52)

    # ── Layer 1 + 3: SVD recommendations with LLM explanations ──

    def get_explained_recommendations(
        self, user_id: int, n: int = TOP_N_RECOMMENDATIONS
    ) -> dict:
        """
        Full Layer-1 + Layer-3 flow:
          SVD predictions → LLM explanations

        Returns:
            {
                "recommendations": DataFrame,   # top-N predicted movies
                "explanation":     str,         # LLM explanation text
                "user_history":    DataFrame,   # user's rated movies
            }
        """
        recommendations = self.svd.get_recommendations(user_id, n=n)
        user_history = self.svd.get_user_history(user_id)

        explanation = self.llm.explain_recommendations(
            recommendations_df=recommendations,
            user_history_df=user_history,
        )

        return {
            "recommendations": recommendations,
            "explanation": explanation,
            "user_history": user_history,
        }

    # ── Layer 2 + 3: RAG chat ────────────────────────────────────

    def chat(
        self,
        user_message: str,
        user_id: Optional[int] = None,
        conversation_history: Optional[list[dict]] = None,
    ) -> str:
        """
        Full Layer-2 + Layer-3 flow:
          FAISS semantic search → RAG context → LLM response

        user_id is accepted for future personalisation but not yet used —
        the vector search already retrieves contextually relevant movies.
        """
        relevant_movies = self.vs.search(user_message, top_k=10)
        movie_ids = relevant_movies["movie_id"].tolist()
        movie_context = self.vs.get_movie_context(movie_ids)

        return self.llm.chat(
            user_message=user_message,
            movie_context=movie_context,
            conversation_history=conversation_history,
        )

    # ── Convenience wrappers ─────────────────────────────────────

    def search_movies(self, query: str, top_k: int = 5):
        """Semantic movie search — thin wrapper around VectorSearch.search."""
        return self.vs.search(query, top_k=top_k)

    def get_valid_users(self, sample: int = 10) -> list:
        """Users with ≥ 50 ratings — thin wrapper for demo/testing."""
        return self.svd.get_valid_user_ids(sample=sample)


# ── Smoke test ───────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd

    pipeline = RecommendationPipeline()

    # ── Demo: explained recommendations ──────────────────────
    users = pipeline.get_valid_users(sample=1)
    if users:
        uid = users[0]
        print(f"\nExplained recommendations for user {uid}:")
        result = pipeline.get_explained_recommendations(uid, n=5)

        print("\nTop recommendations:")
        for _, row in result["recommendations"].iterrows():
            title = row.get("title") or "Unknown"
            rating = row.get("predicted_rating", "?")
            print(f"  {rating}  {title}")

        print(f"\nLLM explanation:\n{result['explanation']}")

    # ── Demo: RAG chat ────────────────────────────────────────
    print("\n" + "─" * 52)
    query = "I want a mind-bending thriller with a twist ending"
    print(f"Chat query: '{query}'")
    response = pipeline.chat(query)
    print(f"\nResponse:\n{response}")
