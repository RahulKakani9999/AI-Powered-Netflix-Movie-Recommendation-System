import ast
import re
import sys
import pandas as pd
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MOVIELENS_DIR, TMDB_DIR, PROCESSED_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_eval(val) -> list:
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []


def _extract_year(title: str) -> Optional[int]:
    m = re.search(r"\((\d{4})\)\s*$", title)
    return int(m.group(1)) if m else None


def _clean_title(title: str) -> str:
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_movielens_ratings() -> pd.DataFrame:
    df = pd.read_csv(
        MOVIELENS_DIR / "ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    print(f"  Ratings : {len(df):,} rows")
    return df


def load_movielens_movies() -> pd.DataFrame:
    df = pd.read_csv(
        MOVIELENS_DIR / "movies.dat",
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres_raw"],
        encoding="latin-1",
    )
    df["year"] = df["title"].apply(_extract_year)
    df["clean_title"] = df["title"].apply(_clean_title)
    df["genres_ml"] = df["genres_raw"].str.split("|")
    df["title_norm"] = df["clean_title"].str.lower().str.strip()
    print(f"  Movies  : {len(df):,} rows")
    return df


def load_tmdb() -> Optional[pd.DataFrame]:
    movies_path = TMDB_DIR / "tmdb_5000_movies.csv"
    credits_path = TMDB_DIR / "tmdb_5000_credits.csv"

    if not movies_path.exists() or not credits_path.exists():
        print("  TMDB files not found — run download_data.py first.")
        print("  Continuing with MovieLens-only data (no TMDB enrichment).")
        return None

    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    # tmdb_5000_credits has movie_id; merge on movies.id
    credits = credits.rename(columns={"title": "credits_title"})
    df = movies.merge(credits, left_on="id", right_on="movie_id", how="left")
    df = df.drop(columns=["movie_id"], errors="ignore")

    # Parse JSON-encoded string columns
    for col in ["genres", "keywords", "cast", "crew"]:
        df[col] = df[col].apply(_safe_eval)

    # Director: first crew member whose job is "Director"
    df["director"] = df["crew"].apply(
        lambda crew: next(
            (
                m["name"]
                for m in crew
                if isinstance(m, dict) and m.get("job") == "Director"
            ),
            "",
        )
    )

    # Top-5 cast names
    df["cast"] = df["cast"].apply(
        lambda c: [m["name"] for m in c[:5] if isinstance(m, dict)]
    )

    # Flatten genres / keywords to plain name lists
    for col in ["genres", "keywords"]:
        df[col] = df[col].apply(
            lambda lst: [m["name"] for m in lst if isinstance(m, dict)]
        )

    # Year derived from release_date (used for matching with MovieLens)
    df["tmdb_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

    # Normalised title for matching — kept separate from ml title_norm
    df["tmdb_title_norm"] = df["title"].str.lower().str.strip()

    # Rename columns that would clash with MovieLens columns after merge
    df = df.rename(
        columns={
            "title": "tmdb_title",
            "genres": "tmdb_genres",
            "keywords": "tmdb_keywords",
            "cast": "tmdb_cast",
            "id": "tmdb_id",
        }
    )

    print(f"  TMDB    : {len(df):,} rows")
    return df


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_datasets(ml_movies: pd.DataFrame, tmdb: pd.DataFrame) -> pd.DataFrame:
    merged = ml_movies.merge(
        tmdb,
        left_on=["title_norm", "year"],
        right_on=["tmdb_title_norm", "tmdb_year"],
        how="left",
    )

    matched = merged["tmdb_id"].notna().sum()
    total = len(merged)
    pct = matched / total * 100 if total else 0
    print(f"  Matched : {matched:,} / {total:,} MovieLens movies ({pct:.1f}%)")
    return merged


# ---------------------------------------------------------------------------
# combined_text
# ---------------------------------------------------------------------------

def _build_combined_text(row) -> str:
    genres = row.get("tmdb_genres")
    if not isinstance(genres, list):
        genres = row.get("genres_ml") or []

    keywords = row.get("tmdb_keywords") or []
    cast = row.get("tmdb_cast") or []

    parts = [
        str(row.get("clean_title") or ""),
        str(row.get("overview") or ""),
        " ".join(genres),
        " ".join(keywords),
        " ".join(cast),
        str(row.get("director") or ""),
    ]
    return " ".join(p for p in parts if p).strip()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def preprocess():
    print("--- Loading MovieLens ---")
    ratings = load_movielens_ratings()
    ml_movies = load_movielens_movies()

    print("\n--- Loading TMDB ---")
    tmdb = load_tmdb()

    print("\n--- Merging ---")
    if tmdb is not None:
        movies = merge_datasets(ml_movies, tmdb)
    else:
        movies = ml_movies.copy()

    movies["combined_text"] = movies.apply(_build_combined_text, axis=1)

    # ── ratings.csv ──────────────────────────────────────────
    out = PROCESSED_DIR / "ratings.csv"
    ratings[["user_id", "movie_id", "rating", "timestamp"]].to_csv(out, index=False)
    print(f"\nSaved → {out}")

    # ── movies.csv (full merge) ───────────────────────────────
    out = PROCESSED_DIR / "movies.csv"
    movies.to_csv(out, index=False)
    print(f"Saved → {out}")

    # ── movies_metadata.csv (slim) ────────────────────────────
    meta = movies.copy()

    # Unified genre / keyword / cast columns (prefer TMDB, fall back to ML)
    meta["genres"] = meta.apply(
        lambda r: r["tmdb_genres"]
        if isinstance(r.get("tmdb_genres"), list)
        else (r.get("genres_ml") or []),
        axis=1,
    )
    meta["keywords"] = meta.apply(
        lambda r: r["tmdb_keywords"] if isinstance(r.get("tmdb_keywords"), list) else [],
        axis=1,
    )
    meta["cast"] = meta.apply(
        lambda r: r["tmdb_cast"] if isinstance(r.get("tmdb_cast"), list) else [],
        axis=1,
    )

    slim_cols = [
        "movie_id", "title", "year", "overview",
        "genres", "keywords", "cast", "director", "combined_text",
    ]
    available = [c for c in slim_cols if c in meta.columns]
    out = PROCESSED_DIR / "movies_metadata.csv"
    meta[available].to_csv(out, index=False)
    print(f"Saved → {out}")

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    preprocess()
