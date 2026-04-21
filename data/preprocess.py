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


def _normalize_title(title: str) -> str:
    """Normalize a title for fuzzy matching across MovieLens and TMDB.

    Handles MovieLens article-at-end ("Matrix, The" → "matrix"),
    TMDB article-at-front ("The Matrix" → "matrix"), and strips punctuation
    that differs between the two datasets.
    """
    t = str(title).strip()
    # Move trailing article to front: "Matrix, The" → "The Matrix"
    t = re.sub(r"^(.*),\s*(The|A|An)$", r"\2 \1", t, flags=re.IGNORECASE)
    t = t.lower()
    # Strip leading article
    for prefix in ("the ", "a ", "an "):
        if t.startswith(prefix):
            t = t[len(prefix):]
            break
    # Remove punctuation that causes mismatches (colons, hyphens, etc.)
    t = re.sub(r"[,:\-'\.&!?]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


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
    df["title_norm2"] = df["clean_title"].apply(_normalize_title)
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

    # Normalised titles for matching — kept separate from ml columns
    df["tmdb_title_norm"] = df["title"].str.lower().str.strip()
    df["tmdb_title_norm2"] = df["title"].apply(_normalize_title)

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
    # Build lookup: (normalized_title, year) → tmdb DataFrame index
    tmdb_lookup: dict[tuple, int] = {}
    for idx, row in tmdb.iterrows():
        yr = row["tmdb_year"]
        if pd.isna(yr):
            continue
        key = (row["tmdb_title_norm2"], int(yr))
        if key not in tmdb_lookup:   # first entry wins (avoids duplicates)
            tmdb_lookup[key] = idx

    # Match each ML movie: exact year first, then year−1, then year+1
    mapping: dict[int, int] = {}   # ml movie_id → tmdb index
    for _, ml_row in ml_movies.iterrows():
        yr = ml_row["year"]
        if pd.isna(yr):
            continue
        yr, mid = int(yr), int(ml_row["movie_id"])
        title = ml_row["title_norm2"]
        for offset in (0, -1, 1):
            if (title, yr + offset) in tmdb_lookup:
                mapping[mid] = tmdb_lookup[(title, yr + offset)]
                break

    # Build a TMDB slice aligned to matched ML movie IDs and merge
    if mapping:
        ml_ids = sorted(mapping.keys())
        tmdb_slice = tmdb.loc[[mapping[mid] for mid in ml_ids]].copy()
        tmdb_slice = tmdb_slice.reset_index(drop=True)
        tmdb_slice["movie_id"] = ml_ids
        merged = ml_movies.merge(tmdb_slice, on="movie_id", how="left")
    else:
        merged = ml_movies.copy()

    matched = merged["tmdb_id"].notna().sum() if "tmdb_id" in merged.columns else 0
    total = len(merged)
    pct = matched / total * 100 if total else 0
    print(f"  Matched : {matched:,} / {total:,} MovieLens movies ({pct:.1f}%)")
    return merged


# ---------------------------------------------------------------------------
# combined_text
# ---------------------------------------------------------------------------

def _build_combined_text(row) -> str:
    def _to_str(val) -> str:
        if isinstance(val, list):
            return ", ".join(str(x) for x in val)
        elif isinstance(val, str):
            return val
        else:
            return ""   # covers NaN, None, and any other non-iterable

    genres_val = row.get("tmdb_genres")
    if not isinstance(genres_val, list):
        genres_val = row.get("genres_ml")   # fall back to MovieLens genres

    parts = [
        _to_str(row.get("clean_title")),
        _to_str(row.get("overview")),
        _to_str(genres_val),
        _to_str(row.get("tmdb_keywords")),
        _to_str(row.get("tmdb_cast")),
        _to_str(row.get("director")),
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
