import ast
import sys
import pandas as pd
import streamlit as st
from pathlib import Path

# ── Page config (must be the first Streamlit call) ────────────
st.set_page_config(
    page_title="AI Movie Recommender",
    page_icon="🎬",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Movie card grid */
    .cards-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
        gap: 12px;
        margin: 14px 0 8px 0;
    }

    /* Individual card */
    .movie-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-left: 4px solid #e50914;
        border-radius: 8px;
        padding: 14px 16px;
        color: #e0e0e0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.4);
        transition: transform 0.15s ease;
    }
    .movie-card:hover { transform: translateY(-2px); }

    .movie-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 6px;
        line-height: 1.3;
    }
    .movie-genres {
        font-size: 0.75rem;
        color: #a0a0b0;
        margin-bottom: 6px;
        line-height: 1.4;
    }
    .movie-rating {
        font-size: 0.82rem;
        font-weight: 600;
        color: #f5c518;
    }

    /* Sidebar footer */
    .sidebar-footer {
        font-size: 0.75rem;
        color: #808080;
        text-align: center;
        padding-top: 12px;
        border-top: 1px solid #2e2e2e;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Pipeline loader (cached across reruns) ─────────────────────
@st.cache_resource(show_spinner="Loading recommendation engine …")
def load_pipeline():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils.pipeline import RecommendationPipeline
    return RecommendationPipeline()


# ── Session-state initialisation ───────────────────────────────
_WELCOME = (
    "👋 Welcome! I'm your AI movie assistant.\n\n"
    "Here's what I can do:\n"
    "- **Personalised recommendations** — pick a User ID in the sidebar and click "
    "**Get Recommendations** to see SVD-powered suggestions with LLM explanations.\n"
    "- **Conversational search** — ask me anything: *'dark psychological thrillers'*, "
    "*'who directed Parasite?'*, *'movies like Interstellar'*.\n\n"
    "How can I help you today?"
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "type": "text", "content": _WELCOME}
    ]


# ── Helpers ────────────────────────────────────────────────────

def _parse_list_field(val) -> list[str]:
    if isinstance(val, list):
        return [str(x) for x in val]
    try:
        result = ast.literal_eval(str(val))
        if isinstance(result, list):
            return [str(x) for x in result]
    except (ValueError, SyntaxError):
        pass
    return [s.strip() for s in str(val).split(",") if s.strip()]


def _render_movie_cards(cards: list[dict]) -> str:
    """Build an HTML card-grid string from a list of movie dicts."""
    html_cards = []
    for card in cards:
        title = card.get("title") or "Unknown"

        year_val = card.get("year")
        year = (
            f" <span style='font-weight:400;color:#aaa;'>({int(float(year_val))})</span>"
            if year_val and str(year_val) not in ("", "nan")
            else ""
        )

        genres_raw = card.get("genres")
        genre_names = (
            _parse_list_field(genres_raw)
            if genres_raw and str(genres_raw) not in ("", "[]", "nan")
            else []
        )
        genres_str = " &bull; ".join(genre_names[:3]) if genre_names else "&mdash;"

        rating = card.get("predicted_rating")
        rating_str = f"⭐ {float(rating):.2f} / 5" if rating and str(rating) != "nan" else ""

        html_cards.append(
            f"""
            <div class="movie-card">
                <div class="movie-title">{title}{year}</div>
                <div class="movie-genres">{genres_str}</div>
                <div class="movie-rating">{rating_str}</div>
            </div>
            """
        )

    return f'<div class="cards-grid">{"".join(html_cards)}</div>'


def _render_message(msg: dict) -> None:
    """Render a single message dict inside a st.chat_message block."""
    if msg.get("type") == "recommendations":
        if msg.get("cards"):
            st.markdown(_render_movie_cards(msg["cards"]), unsafe_allow_html=True)
        if msg.get("content"):
            st.markdown(msg["content"])
    else:
        st.markdown(msg.get("content", ""))


def _build_llm_history(messages: list[dict], last_n: int = 6) -> list[dict]:
    """Return the last N messages formatted for the LLM conversation API."""
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    return history[-last_n:]


# ── Attempt pipeline load ──────────────────────────────────────
pipeline = None
_pipeline_error: str = ""
try:
    pipeline = load_pipeline()
except Exception as exc:
    _pipeline_error = str(exc)


# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎬 AI Movie\nRecommender")
    st.divider()

    if pipeline is None:
        st.error(
            "**Pipeline not ready.**\n\n"
            "Run the training scripts first:\n"
            "```\n"
            "python data/download_data.py\n"
            "python data/preprocess.py\n"
            "python models/train_svd.py\n"
            "python models/build_faiss_index.py\n"
            "```"
        )
        if _pipeline_error:
            with st.expander("Error details"):
                st.code(_pipeline_error)
        selected_user = None
        get_recs_clicked = False
    else:
        valid_users = pipeline.get_valid_users(20)
        selected_user = st.selectbox(
            "Select User ID",
            options=valid_users,
            help="Users with ≥ 50 ratings for meaningful recommendations",
        )
        get_recs_clicked = st.button(
            "🎯 Get Recommendations",
            use_container_width=True,
            type="primary",
        )
        st.divider()
        st.markdown(
            '<div class="sidebar-footer">'
            "SVD · FAISS · LLM<br>"
            "MovieLens 1M + TMDB 5000"
            "</div>",
            unsafe_allow_html=True,
        )


# ── Main area ──────────────────────────────────────────────────
st.title("🎬 AI Movie Recommender")
st.caption(
    "Collaborative filtering · Semantic search · LLM explanations"
)
st.divider()

# ── Display existing chat history ──────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        _render_message(msg)

# ── Handle "Get Recommendations" button ────────────────────────
if get_recs_clicked and selected_user is not None and pipeline is not None:
    user_msg_text = f"Get recommendations for **User {selected_user}**"

    with st.chat_message("user"):
        st.markdown(user_msg_text)
    st.session_state.messages.append(
        {"role": "user", "type": "text", "content": user_msg_text}
    )

    with st.chat_message("assistant"):
        with st.spinner("Crunching ratings and generating explanations …"):
            try:
                result = pipeline.get_explained_recommendations(selected_user, n=10)
                recs_df: pd.DataFrame = result["recommendations"]
                explanation: str = result["explanation"]

                # Build serialisable card dicts (DataFrames can't live in session_state)
                cards = [
                    {
                        "title": row.get("title"),
                        "year": row.get("year"),
                        "genres": row.get("genres") or row.get("genres_ml"),
                        "predicted_rating": row.get("predicted_rating"),
                    }
                    for _, row in recs_df.iterrows()
                ]

                st.markdown(_render_movie_cards(cards), unsafe_allow_html=True)
                st.markdown(explanation)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "type": "recommendations",
                        "content": explanation,
                        "cards": cards,
                    }
                )
            except Exception as exc:
                err = f"Sorry, something went wrong: `{exc}`"
                st.error(err)
                st.session_state.messages.append(
                    {"role": "assistant", "type": "text", "content": err}
                )

# ── Chat input ─────────────────────────────────────────────────
if prompt := st.chat_input(
    "Ask about movies, genres, directors, moods …",
    disabled=(pipeline is None),
):
    if pipeline is None:
        st.error("Pipeline is not ready. See the sidebar for setup instructions.")
    else:
        # Display and save user turn
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append(
            {"role": "user", "type": "text", "content": prompt}
        )

        # Build history from everything before the current turn
        history = _build_llm_history(st.session_state.messages[:-1], last_n=6)

        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching movies and thinking …"):
                try:
                    response = pipeline.chat(
                        user_message=prompt,
                        user_id=selected_user,
                        conversation_history=history,
                    )
                except Exception as exc:
                    response = f"Sorry, I hit an error: `{exc}`"
            st.markdown(response)

        st.session_state.messages.append(
            {"role": "assistant", "type": "text", "content": response}
        )
