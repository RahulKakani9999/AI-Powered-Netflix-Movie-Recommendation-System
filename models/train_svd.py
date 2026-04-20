import sys
import pickle
import pandas as pd
from pathlib import Path
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    PROCESSED_DIR,
    SAVED_MODELS_DIR,
    SVD_N_FACTORS,
    SVD_N_EPOCHS,
    SVD_LR,
    SVD_REG,
)


def train():
    # ── Load ratings ──────────────────────────────────────────
    print("Loading ratings ...")
    ratings = pd.read_csv(PROCESSED_DIR / "ratings.csv")
    n_users = ratings["user_id"].nunique()
    n_movies = ratings["movie_id"].nunique()
    print(f"  {len(ratings):,} ratings  |  {n_users:,} users  |  {n_movies:,} movies")

    # ── Build Surprise dataset ────────────────────────────────
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[["user_id", "movie_id", "rating"]], reader)

    # ── SVD configuration ─────────────────────────────────────
    algo = SVD(
        n_factors=SVD_N_FACTORS,
        n_epochs=SVD_N_EPOCHS,
        lr_all=SVD_LR,
        reg_all=SVD_REG,
        random_state=42,
        verbose=False,
    )
    print(
        f"\nSVD config: n_factors={SVD_N_FACTORS}, n_epochs={SVD_N_EPOCHS}, "
        f"lr={SVD_LR}, reg={SVD_REG}"
    )

    # ── 5-fold cross-validation ───────────────────────────────
    print("\nRunning 5-fold cross-validation ...")
    cv_results = cross_validate(
        algo, data, measures=["RMSE", "MAE"], cv=5, verbose=True
    )

    mean_rmse = cv_results["test_rmse"].mean()
    mean_mae = cv_results["test_mae"].mean()
    std_rmse = cv_results["test_rmse"].std()
    std_mae = cv_results["test_mae"].std()
    print(f"\nCV Results:")
    print(f"  RMSE : {mean_rmse:.4f} ± {std_rmse:.4f}")
    print(f"  MAE  : {mean_mae:.4f} ± {std_mae:.4f}")

    # ── Train on full dataset ─────────────────────────────────
    print("\nTraining final model on full dataset ...")
    trainset = data.build_full_trainset()
    algo.fit(trainset)
    print("  Done.")

    # ── Save artefacts ────────────────────────────────────────
    model_path = SAVED_MODELS_DIR / "svd_model.pkl"
    trainset_path = SAVED_MODELS_DIR / "trainset.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(algo, f)
    print(f"\nSaved → {model_path}")

    with open(trainset_path, "wb") as f:
        pickle.dump(trainset, f)
    print(f"Saved → {trainset_path}")

    sep = "─" * 52
    print(
        f"\n{sep}\n"
        "Next steps:\n"
        "  python utils/recommender.py       # smoke-test SVD\n"
        "  python models/train_faiss.py      # build FAISS index (Layer 2)\n"
        f"{sep}"
    )


if __name__ == "__main__":
    train()
