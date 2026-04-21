"""
Model evaluation for the SVD recommendation layer.

Note on methodology: the saved SVD model was trained on the full dataset
(build_full_trainset). The 80/20 split here creates a synthetic holdout so
we can audit the production model's prediction quality.  Because the model
has seen all users and items during training, these metrics represent
near-optimal performance rather than true generalisation — use them as a
sanity-check floor, not as held-out validation numbers.  For rigorous
offline evaluation, re-train a fresh SVD on only the 80 % split.
"""

import sys
import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, SAVED_MODELS_DIR


# ── Shared setup helpers ───────────────────────────────────────

def _build_split():
    """Return (trainset, testset) from ratings.csv with 80/20 split."""
    ratings = pd.read_csv(PROCESSED_DIR / "ratings.csv")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[["user_id", "movie_id", "rating"]], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    n_test = len(testset)
    n_users = len({uid for uid, _, _ in testset})
    print(f"  Test set : {n_test:,} ratings  |  {n_users:,} users")
    return trainset, testset


def _load_model():
    with open(SAVED_MODELS_DIR / "svd_model.pkl", "rb") as f:
        return pickle.load(f)


# ── Metrics ────────────────────────────────────────────────────

def compute_rmse() -> float:
    """
    Predict ratings for the 20 % testset with the saved model and return RMSE.
    Lower is better; MovieLens 1M SVD typically achieves ~0.87.
    """
    print("Computing RMSE …")
    _, testset = _build_split()
    model = _load_model()

    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    return rmse


def compute_precision_at_k(k: int = 10, threshold: float = 4.0) -> float:
    """
    Average Precision@K across all users in the testset.

    For each user:
      - Sort test predictions by estimated rating (descending)
      - Keep the top K
      - Precision = |{items in top-K with actual rating >= threshold}| / K

    Returns the mean precision across all users who have at least 1 test rating.
    """
    print(f"Computing Precision@{k}  (relevant threshold = {threshold}) …")
    _, testset = _build_split()
    model = _load_model()

    predictions = model.test(testset)

    # Group (estimated, actual) pairs by user
    user_preds: dict[str, list] = defaultdict(list)
    for pred in predictions:
        user_preds[pred.uid].append((pred.est, pred.r_ui))

    precisions = []
    for uid, preds in user_preds.items():
        preds.sort(key=lambda x: x[0], reverse=True)
        top_k = preds[:k]
        n_relevant = sum(1 for est, actual in top_k if actual >= threshold)
        precisions.append(n_relevant / k)

    return sum(precisions) / len(precisions) if precisions else 0.0


# ── Main ───────────────────────────────────────────────────────

def main():
    sep = "=" * 46
    print(f"\n{sep}")
    print("  SVD Recommendation Model — Evaluation")
    print(sep)
    print("  Split  : 80 % train / 20 % test  (seed=42)")
    print("  Model  : models/saved/svd_model.pkl")
    print(f"{sep}\n")

    rmse = compute_rmse()
    print()
    prec_10 = compute_precision_at_k(k=10, threshold=4.0)

    dash = "─" * 36
    print(f"\n{dash}")
    print(f"  {'Metric':<22} {'Value':>8}")
    print(dash)
    print(f"  {'RMSE':<22} {rmse:>8.4f}")
    print(f"  {'Precision@10 (≥4★)':<22} {prec_10:>8.4f}")
    print(dash)
    print("  Relevant = actual rating ≥ 4.0 / 5.0")
    print(f"{dash}\n")
    print("Next: run human evaluation — see evaluation/human_eval_template.md")


if __name__ == "__main__":
    main()
