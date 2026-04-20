import sys
import zipfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MOVIELENS_DIR, TMDB_DIR

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MOVIELENS_FILES = ["ratings.dat", "movies.dat", "users.dat"]

KAGGLE_DATASET = "tmdb/tmdb-movie-metadata"
TMDB_FILES = ["tmdb_5000_movies.csv", "tmdb_5000_credits.csv"]


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb = downloaded / 1_048_576
        total_mb = total_size / 1_048_576
        print(f"\r  {pct:5.1f}%  {mb:.1f} / {total_mb:.1f} MB", end="", flush=True)


def download_movielens():
    if all((MOVIELENS_DIR / f).exists() for f in MOVIELENS_FILES):
        print("MovieLens 1M: files already exist — skipping.")
        return

    zip_path = MOVIELENS_DIR / "ml-1m.zip"
    print(f"Downloading MovieLens 1M ...\n  {MOVIELENS_URL}")
    urllib.request.urlretrieve(MOVIELENS_URL, zip_path, reporthook=_progress_hook)
    print()

    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            name = Path(member).name
            if name in MOVIELENS_FILES:
                (MOVIELENS_DIR / name).write_bytes(zf.read(member))
                print(f"  {name}")

    zip_path.unlink()
    print("MovieLens 1M ready.\n")


def download_tmdb():
    if all((TMDB_DIR / f).exists() for f in TMDB_FILES):
        print("TMDB 5000: files already exist — skipping.")
        return

    # ---- Try Kaggle API ----
    try:
        import kaggle  # noqa: F401

        print("Kaggle API detected. Downloading TMDB 5000 ...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET, path=str(TMDB_DIR), unzip=True
        )
        print("TMDB 5000 ready.\n")
        return
    except ImportError:
        pass
    except Exception as exc:
        print(f"Kaggle API error: {exc}\nFalling back to manual instructions.\n")

    # ---- Manual instructions ----
    sep = "=" * 62
    print(
        f"\n{sep}\n"
        "TMDB 5000 — Manual Download Required\n"
        f"{sep}\n"
        "1. Go to: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata\n"
        "2. Sign in (free account) and click Download.\n"
        f"3. Extract the two CSV files into:\n"
        f"     {TMDB_DIR}\n\n"
        "Expected files:\n"
        "  • tmdb_5000_movies.csv\n"
        "  • tmdb_5000_credits.csv\n\n"
        "─── OR set up the Kaggle API ────────────────────────────\n"
        "  pip install kaggle\n"
        "  # Download kaggle.json from Kaggle → Account → API\n"
        "  mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/\n"
        "  chmod 600 ~/.kaggle/kaggle.json\n"
        "  python data/download_data.py   # re-run this script\n"
        f"{sep}\n"
    )


if __name__ == "__main__":
    download_movielens()
    download_tmdb()
