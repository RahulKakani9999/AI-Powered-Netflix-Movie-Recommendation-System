import ast
import sys
import pandas as pd
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert movie recommendation assistant. "
    "Your role is to explain why specific movies are being recommended to a user "
    "based on their viewing history and personal taste.\n\n"
    "Guidelines:\n"
    "- Be specific about themes, genres, directors, and narrative elements that "
    "connect each recommendation to the user's preferences.\n"
    "- Reference the user's highly-rated movies by name when they strengthen the "
    "justification.\n"
    "- Keep each explanation concise: 2–3 sentences per movie.\n"
    "- Use an enthusiastic but informative tone — you love movies.\n"
    "- For general movie questions, answer helpfully using your broader knowledge."
)


# ---------------------------------------------------------------------------
# Helper (shared with vector_search.py pattern)
# ---------------------------------------------------------------------------

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
    return [s.strip() for s in str(val).split(",") if s.strip()]


def _is_blank(val) -> bool:
    """True when a DataFrame cell carries no useful value."""
    if val is None:
        return True
    if pd.isna(val) if not isinstance(val, (list, dict)) else False:
        return True
    return str(val).strip() in ("", "nan", "[]", "None")


# ---------------------------------------------------------------------------
# LLMExplainer
# ---------------------------------------------------------------------------

class LLMExplainer:
    def __init__(self):
        self.provider = LLM_PROVIDER.strip().lower()

        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model = OPENAI_MODEL
        elif self.provider == "groq":
            from groq import Groq
            self.client = Groq(api_key=GROQ_API_KEY)
            self.model = GROQ_MODEL
        else:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{self.provider}'. "
                "Set LLM_PROVIDER to 'openai' or 'groq' in .env"
            )

        print(f"LLMExplainer ready — provider={self.provider}, model={self.model}")

    # ── Core LLM call ─────────────────────────────────────────

    def _call_llm(self, messages: list[dict]) -> str:
        """
        Chat completions call.  Both OpenAI and Groq use the same interface,
        so this method is provider-agnostic.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            return f"[LLM error — {type(exc).__name__}: {exc}]"

    # ── Public API ────────────────────────────────────────────

    def explain_recommendations(
        self,
        recommendations_df: pd.DataFrame,
        user_history_df: Optional[pd.DataFrame] = None,
    ) -> str:
        """
        Ask the LLM to explain why each recommended movie suits this user.

        recommendations_df : output of SVDRecommender.get_recommendations()
        user_history_df    : output of SVDRecommender.get_user_history() (optional)
        """
        sections: list[str] = []

        # ── User taste context ────────────────────────────────
        if user_history_df is not None and not user_history_df.empty:
            liked = (
                user_history_df[user_history_df["rating"] >= 4]
                .sort_values("rating", ascending=False)
                .head(10)
            )
            if not liked.empty:
                titles = liked["title"].dropna().tolist()
                bullet_list = "\n".join(f"  - {t}" for t in titles)
                sections.append(
                    f"User's highly-rated movies (≥ 4 / 5 stars):\n{bullet_list}"
                )

        # ── Recommendations ───────────────────────────────────
        sections.append(f"Movies recommended for this user:\n{self._format_movies(recommendations_df)}")

        user_msg = (
            "\n\n".join(sections)
            + "\n\nPlease explain why each recommended movie would appeal to this user. "
            "Reference their taste where relevant. "
            "2–3 sentences per movie."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        return self._call_llm(messages)

    def chat(
        self,
        user_message: str,
        movie_context: str = "",
        conversation_history: Optional[list[dict]] = None,
    ) -> str:
        """
        General-purpose conversational interface with optional RAG context.

        movie_context        : structured text from VectorSearch.get_movie_context()
        conversation_history : list of {"role": "user"|"assistant", "content": str}
                               representing prior turns in the conversation
        """
        if movie_context.strip():
            full_user_message = (
                f"Relevant movie information:\n{movie_context}\n\n"
                f"User question: {user_message}"
            )
        else:
            full_user_message = user_message

        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": full_user_message})

        return self._call_llm(messages)

    # ── Formatting ────────────────────────────────────────────

    @staticmethod
    def _format_movies(df: pd.DataFrame) -> str:
        """
        Convert a recommendations DataFrame to a numbered plain-text block
        suitable for an LLM prompt.  Handles list columns stored as either
        Python lists or their string representations.
        """
        blocks: list[str] = []

        for i, (_, row) in enumerate(df.iterrows(), start=1):
            title = row.get("title") or "Unknown"
            year_val = row.get("year")
            year = f" ({int(float(year_val))})" if not _is_blank(year_val) else ""

            predicted = row.get("predicted_rating")
            rating_tag = (
                f"  [predicted: {float(predicted):.2f}/5]"
                if not _is_blank(predicted)
                else ""
            )

            lines = [f"{i}. {title}{year}{rating_tag}"]

            # Genres
            genres_val = row.get("genres")
            if not _is_blank(genres_val):
                names = _parse_list_field(genres_val)
                if names:
                    lines.append(f"   Genres   : {', '.join(names)}")

            # Overview (truncated to 200 chars)
            overview = row.get("overview")
            if not _is_blank(overview):
                snippet = str(overview)[:200].rstrip()
                if len(str(overview)) > 200:
                    snippet += "…"
                lines.append(f"   Overview : {snippet}")

            # Director
            director = row.get("director")
            if not _is_blank(director):
                lines.append(f"   Director : {director}")

            # Cast
            cast_val = row.get("cast")
            if not _is_blank(cast_val):
                names = _parse_list_field(cast_val)
                if names:
                    lines.append(f"   Cast     : {', '.join(names)}")

            blocks.append("\n".join(lines))

        return "\n\n".join(blocks)


# ── Smoke test ────────────────────────────────────────────────

if __name__ == "__main__":
    explainer = LLMExplainer()

    # Minimal test: single chat turn
    response = explainer.chat(
        "What makes a great sci-fi movie? Name a few classics."
    )
    print("Chat response:")
    print(response)
    print()

    # Multi-turn test
    history = [
        {"role": "user", "content": "I love Christopher Nolan films."},
        {"role": "assistant", "content": "Great taste! His work on Inception and Interstellar is phenomenal."},
    ]
    response = explainer.chat(
        "What should I watch next?",
        conversation_history=history,
    )
    print("Multi-turn response:")
    print(response)
