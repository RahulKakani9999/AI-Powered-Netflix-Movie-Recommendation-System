import sys
import pickle
import pandas as pd
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, SAVED_MODELS_DIR, TOP_N_RECOMMENDATIONS


class SVDRecommender:
    def __init__(self):
        print("Loading SVD model ...")
        with open(SAVED_MODELS_DIR / "svd_model.pkl", "rb") as f:
            self.model = pickle.load(f)

        print("Loading trainset ...")
        with open(SAVED_MODELS_DIR / "trainset.pkl", "rb") as f:
            self.trainset = pickle.load(f)

        print("Loading movies ...")
        self.movies = pd.read_csv(PROCESSED_DIR / "movies.csv")
        self.movies["movie_id"] = self.movies["movie_id"].astype(int)

        # Cache raw movie IDs as a set for O(1) exclusion checks
        self._all_movie_ids: set = {
            int(self.trainset.to_raw_iid(iid))
            for iid in self.trainset.all_items()
        }

        # Precompute popularity ranking for unknown-user fallback
        self._popular: pd.DataFrame = self._compute_popular_movies()

        print("SVDRecommender ready.\n")

    # ── Internal helpers ──────────────────────────────────────

    def _compute_popular_movies(self) -> pd.DataFrame:
        """Rank all movies by (n_ratings DESC, avg_rating DESC) using trainset.ir."""
        rows = []
        for inner_iid in self.trainset.all_items():
            ratings_list = [r for _, r in self.trainset.ir[inner_iid]]
            rows.append(
                {
                    "movie_id": int(self.trainset.to_raw_iid(inner_iid)),
                    "n_ratings": len(ratings_list),
                    "avg_rating": sum(ratings_list) / len(ratings_list),
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values(["n_ratings", "avg_rating"], ascending=False)
            .reset_index(drop=True)
        )

    def _unknown_user_fallback(self, n: int) -> pd.DataFrame:
        """Return top-N popular movies when the user is not in the trainset."""
        top = self._popular.head(n)[["movie_id", "avg_rating"]].copy()
        top = top.rename(columns={"avg_rating": "predicted_rating"})
        top["predicted_rating"] = top["predicted_rating"].round(3)
        return top.merge(self.movies, on="movie_id", how="left")

    # ── Public API ────────────────────────────────────────────

    def get_recommendations(
        self, user_id: int, n: int = TOP_N_RECOMMENDATIONS
    ) -> pd.DataFrame:
        """
        Top-N predicted movies for user_id, excluding already-rated titles.
        Unknown users receive popularity-based recommendations instead.
        """
        try:
            inner_uid = self.trainset.to_inner_uid(user_id)
        except ValueError:
            print(f"  User {user_id} not in trainset — returning popular movies.")
            return self._unknown_user_fallback(n)

        # Build set of raw movie IDs already rated by this user
        rated_raw: set = {
            int(self.trainset.to_raw_iid(iid))
            for iid, _ in self.trainset.ur[inner_uid]
        }

        # Predict on every unrated movie
        predictions = [
            (mid, self.model.predict(user_id, mid).est)
            for mid in self._all_movie_ids
            if mid not in rated_raw
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)

        result = pd.DataFrame(predictions[:n], columns=["movie_id", "predicted_rating"])
        result["predicted_rating"] = result["predicted_rating"].round(3)
        return result.merge(self.movies, on="movie_id", how="left")

    def get_user_history(self, user_id: int) -> pd.DataFrame:
        """All movies rated by user_id, sorted by rating descending."""
        try:
            inner_uid = self.trainset.to_inner_uid(user_id)
        except ValueError:
            return pd.DataFrame(columns=["movie_id", "title", "year", "rating"])

        rows = [
            (int(self.trainset.to_raw_iid(iid)), float(rating))
            for iid, rating in self.trainset.ur[inner_uid]
        ]
        df = (
            pd.DataFrame(rows, columns=["movie_id", "rating"])
            .sort_values("rating", ascending=False)
            .reset_index(drop=True)
        )
        return df.merge(
            self.movies[["movie_id", "title", "year"]], on="movie_id", how="left"
        )

    def get_valid_user_ids(self, sample: int = 10) -> list:
        """User IDs that have rated at least 50 movies (useful for demos)."""
        qualifying = [
            int(self.trainset.to_raw_uid(uid))
            for uid in self.trainset.all_users()
            if len(self.trainset.ur[uid]) >= 50
        ]
        return qualifying[:sample]


# ── Smoke test ────────────────────────────────────────────────

if __name__ == "__main__":
    rec = SVDRecommender()

    users = rec.get_valid_user_ids(sample=3)
    print(f"Sample users with 50+ ratings: {users}\n")

    for uid in users:
        print(f"{'─' * 50}")
        print(f"User {uid}")

        history = rec.get_user_history(uid)
        top3 = list(history["title"].head(3))
        print(f"  Top-3 rated  : {top3}")

        recs = rec.get_recommendations(uid, n=5)
        print("  Top-5 recs   :")
        for _, row in recs.iterrows():
            title = row.get("title") or "Unknown"
            print(f"    {row['predicted_rating']:.3f}  {title}")
        print()
