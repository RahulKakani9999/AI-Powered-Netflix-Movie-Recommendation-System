import ast
import sys
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="CineMatch AI",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    font-family: 'DM Sans', system-ui, sans-serif !important;
    background: #ffffff !important;
    color: #1a1a1a !important;
}

/* Remove default padding */
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
section[data-testid="stMain"] > div { padding: 0 !important; }

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
.stDeployButton { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

/* ── Nav bar ── */
.nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 40px;
    height: 60px;
    background: #fff;
}
.nav-gradient-line {
    height: 2px;
    background: linear-gradient(135deg, #E81123, #FF6B00, #FFF100);
}
.logo-wrap {
    display: flex;
    align-items: center;
    gap: 10px;
}
.logo-text {
    font-family: 'Outfit', sans-serif;
    font-size: 19px;
    font-weight: 800;
    letter-spacing: 2.5px;
    background: linear-gradient(135deg, #E81123, #FF6B00, #C8A800);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.nav-tagline {
    font-size: 12px;
    color: #bbb;
    letter-spacing: 0.5px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 400;
}

/* ── User selector row ── */
.user-row {
    padding: 12px 40px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.user-row-label {
    font-size: 13px;
    color: #999;
    white-space: nowrap;
    font-family: 'DM Sans', sans-serif;
}
/* Style the selectbox to look inline */
.user-selector [data-testid="stSelectbox"] > label { display: none !important; }
.user-selector [data-testid="stSelectbox"] > div > div {
    background: #f7f7f7 !important;
    border: 1px solid #e8e8e8 !important;
    border-radius: 24px !important;
    font-size: 13px !important;
    color: #444 !important;
    padding: 4px 14px !important;
    min-height: 34px !important;
}

/* ── Hero ── */
.hero {
    padding: 36px 40px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.hero-heading {
    font-family: 'Outfit', sans-serif;
    font-size: 36px;
    font-weight: 800;
    color: #1a1a1a;
    margin: 0 0 6px;
    line-height: 1.15;
}
.hero-sub {
    font-size: 14px;
    color: #aaa;
    margin: 0;
    font-family: 'DM Sans', sans-serif;
}

/* ── 3D CTA button ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(180deg, #FF3333 0%, #E81123 50%, #C20E1E 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 11px 26px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 1.8px !important;
    box-shadow: 0 2px 0 #8B0A15, 0 4px 10px rgba(232,17,35,0.28) !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.25) !important;
    transition: transform 0.1s, box-shadow 0.1s !important;
    height: auto !important;
    min-height: unset !important;
    white-space: nowrap !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 3px 0 #8B0A15, 0 6px 14px rgba(232,17,35,0.32) !important;
}
[data-testid="stButton"] > button[kind="primary"]:active {
    transform: translateY(1px) !important;
    box-shadow: 0 1px 0 #8B0A15, 0 2px 5px rgba(232,17,35,0.2) !important;
}

/* ── Movie cards ── */
.movie-card {
    background: #fafafa;
    border: 0.5px solid #ececec;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 4px;
}
.card-grad-bar {
    height: 3px;
    background: linear-gradient(135deg, #E81123, #FF6B00, #FFF100);
}
.card-body {
    padding: 14px 16px 12px;
}
.card-title {
    font-family: 'Outfit', sans-serif;
    font-size: 14px;
    font-weight: 700;
    color: #1a1a1a;
    margin: 0 0 2px;
    line-height: 1.3;
}
.card-year {
    font-size: 12px;
    color: #bbb;
    margin: 0 0 9px;
}
.pills-row {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-bottom: 10px;
}
.gpill {
    font-size: 10px;
    font-weight: 500;
    color: #d4200e;
    background: linear-gradient(135deg,
        rgba(232,17,35,0.07),
        rgba(255,107,0,0.07),
        rgba(255,241,0,0.07));
    border: 0.5px solid rgba(232,17,35,0.13);
    border-radius: 20px;
    padding: 2px 9px;
    white-space: nowrap;
    font-family: 'DM Sans', sans-serif;
}
.card-rating {
    display: flex;
    align-items: center;
    gap: 5px;
    margin-bottom: 8px;
}
.rating-val {
    font-size: 12px;
    font-weight: 600;
    color: #E81123;
    font-family: 'Outfit', sans-serif;
}
.card-director {
    font-size: 11px;
    color: #c0c0c0;
    margin: 0;
    font-style: italic;
}

/* Watchlist add button */
.add-btn-wrap [data-testid="stButton"] > button {
    width: 28px !important;
    height: 28px !important;
    min-height: 28px !important;
    padding: 0 !important;
    border-radius: 50% !important;
    border: 1.5px solid #e0e0e0 !important;
    background: #fff !important;
    color: #888 !important;
    font-size: 17px !important;
    line-height: 1 !important;
    font-weight: 400 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: none !important;
    margin-left: auto !important;
}
.add-btn-wrap [data-testid="stButton"] > button:hover {
    border-color: #E81123 !important;
    color: #E81123 !important;
    background: rgba(232,17,35,0.04) !important;
}

/* ── Director's Notes ── */
.dir-notes {
    margin: 20px 40px 28px;
    border-left: 3px solid transparent;
    border-image: linear-gradient(180deg, #E81123, #FF6B00, #FFF100) 1;
    background: linear-gradient(135deg,
        rgba(232,17,35,0.025),
        rgba(255,107,0,0.025),
        rgba(255,241,0,0.025));
    border-radius: 0 8px 8px 0;
    padding: 18px 22px;
}
.dir-notes-label {
    font-family: 'Outfit', sans-serif;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    background: linear-gradient(135deg, #E81123, #FF6B00, #C8A800);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 10px;
}
.dir-notes-body {
    font-size: 14px;
    color: #555;
    line-height: 1.75;
    margin: 0;
}

/* ── Chat bubbles ── */
.chat-wrap { padding: 0 40px 10px; }
.chat-heading {
    font-family: 'Outfit', sans-serif;
    font-size: 16px;
    font-weight: 700;
    color: #1a1a1a;
    padding: 20px 0 14px;
    border-top: 0.5px solid #ececec;
    margin: 0;
}
.bubble-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 10px;
}
.bubble-user .bbl {
    background: linear-gradient(135deg, #E81123, #FF6B00, #E8C800);
    color: #fff;
    padding: 9px 16px;
    border-radius: 18px 18px 4px 18px;
    max-width: 68%;
    font-size: 14px;
    font-weight: 500;
    box-shadow: 0 2px 8px rgba(232,17,35,0.18);
    line-height: 1.5;
}
.bubble-asst {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 10px;
}
.bubble-asst .bbl {
    background: #f8f8f8;
    border: 1px solid #ececec;
    color: #444;
    padding: 9px 16px;
    border-radius: 18px 18px 18px 4px;
    max-width: 68%;
    font-size: 14px;
    line-height: 1.65;
}

/* Chat input pill */
[data-testid="stChatInput"] {
    border: 1.5px solid #e8e8e8 !important;
    border-radius: 32px !important;
    background: #fff !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(232,17,35,0.4) !important;
    box-shadow: 0 2px 14px rgba(232,17,35,0.08) !important;
}
[data-testid="stChatInputSubmitButton"] > button {
    background: linear-gradient(180deg, #FF3333 0%, #E81123 50%, #C20E1E 100%) !important;
    border-radius: 50% !important;
    border: none !important;
    box-shadow: 0 1px 0 #8B0A15 !important;
}

/* Spinner text */
[data-testid="stSpinner"] p {
    font-style: italic !important;
    color: #aaa !important;
    font-size: 13px !important;
}

/* Cards section padding */
.cards-pad { padding: 0 40px 16px; }

/* Section pad */
.sec-pad { padding: 0 40px; }

/* Error card */
.err-card {
    background: #fff8f8;
    border: 1px solid rgba(232,17,35,0.15);
    border-left: 3px solid #E81123;
    border-radius: 8px;
    padding: 20px 24px;
    margin: 24px 40px;
}
.err-title {
    font-family: 'Outfit', sans-serif;
    font-size: 15px;
    font-weight: 700;
    color: #E81123;
    margin: 0 0 8px;
}
.err-body {
    font-size: 13px;
    color: #666;
    line-height: 1.7;
    margin: 0;
}
.err-body code {
    background: rgba(232,17,35,0.06);
    padding: 1px 6px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 12px;
    color: #c00;
}
</style>
""", unsafe_allow_html=True)

# ── Pipeline ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_pipeline():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils.pipeline import RecommendationPipeline
    return RecommendationPipeline()


_FILM_ICON = """
<svg width="22" height="22" viewBox="0 0 24 24" fill="none" style="flex-shrink:0">
  <defs>
    <linearGradient id="fi" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#E81123"/>
      <stop offset="55%" stop-color="#FF6B00"/>
      <stop offset="100%" stop-color="#FFF100"/>
    </linearGradient>
  </defs>
  <rect x="1" y="3" width="22" height="18" rx="3" fill="url(#fi)"/>
  <polygon points="9.5,8.5 16.5,12 9.5,15.5" fill="white"/>
  <rect x="1" y="6.5" width="2.5" height="1.8" rx="0.4" fill="rgba(255,255,255,0.55)"/>
  <rect x="1" y="11.1" width="2.5" height="1.8" rx="0.4" fill="rgba(255,255,255,0.55)"/>
  <rect x="1" y="15.7" width="2.5" height="1.8" rx="0.4" fill="rgba(255,255,255,0.55)"/>
  <rect x="20.5" y="6.5" width="2.5" height="1.8" rx="0.4" fill="rgba(255,255,255,0.55)"/>
  <rect x="20.5" y="11.1" width="2.5" height="1.8" rx="0.4" fill="rgba(255,255,255,0.55)"/>
  <rect x="20.5" y="15.7" width="2.5" height="1.8" rx="0.4" fill="rgba(255,255,255,0.55)"/>
</svg>"""

_STAR_SVG = """<svg width="12" height="12" viewBox="0 0 24 24" style="flex-shrink:0;vertical-align:middle;margin-bottom:1px">
  <defs>
    <linearGradient id="sg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#E81123"/>
      <stop offset="55%" stop-color="#FF6B00"/>
      <stop offset="100%" stop-color="#D4A000"/>
    </linearGradient>
  </defs>
  <polygon fill="url(#sg)" points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26"/>
</svg>"""


# ── Helpers ────────────────────────────────────────────────────

def _parse_list(val) -> list[str]:
    if isinstance(val, list):
        return [str(x) for x in val]
    try:
        r = ast.literal_eval(str(val))
        if isinstance(r, list):
            return [str(x) for x in r]
    except (ValueError, SyntaxError):
        pass
    return [s.strip() for s in str(val).split(",") if s.strip()]


def _pills(raw) -> str:
    if not raw or str(raw) in ("", "[]", "nan"):
        return ""
    names = _parse_list(raw)[:4]
    if not names:
        return ""
    pills = "".join(f'<span class="gpill">{g}</span>' for g in names)
    return f'<div class="pills-row">{pills}</div>'


def _card_html(card: dict) -> str:
    title = card.get("title") or "Unknown"
    yr = card.get("year")
    year_str = f"({int(float(yr))})" if yr and str(yr) not in ("", "nan") else ""
    genres = _pills(card.get("genres") or card.get("genres_ml") or "")
    rating = card.get("predicted_rating")
    rating_html = ""
    if rating and str(rating) not in ("", "nan"):
        rating_html = (
            f'<div class="card-rating">{_STAR_SVG}'
            f'<span class="rating-val">{float(rating):.2f} / 5</span></div>'
        )
    director = card.get("director") or ""
    dir_html = (
        f'<p class="card-director">Dir. {director}</p>'
        if director and str(director) not in ("", "nan", "[]")
        else ""
    )
    return (
        f'<div class="movie-card">'
        f'  <div class="card-grad-bar"></div>'
        f'  <div class="card-body">'
        f'    <p class="card-title">{title}</p>'
        f'    <p class="card-year">{year_str}</p>'
        f'    {genres}'
        f'    {rating_html}'
        f'    {dir_html}'
        f'  </div>'
        f'</div>'
    )


# ── Session state ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rec_cards" not in st.session_state:
    st.session_state.rec_cards = []
if "rec_explanation" not in st.session_state:
    st.session_state.rec_explanation = ""

# ── Load pipeline ──────────────────────────────────────────────
pipeline = None
_err = ""
valid_users: list = []
try:
    pipeline = _load_pipeline()
    valid_users = pipeline.get_valid_users(20)
except Exception as exc:
    _err = str(exc)

# ── Nav bar ────────────────────────────────────────────────────
st.markdown(
    f'<div class="nav-bar">'
    f'  <div class="logo-wrap">'
    f'    {_FILM_ICON}'
    f'    <span class="logo-text">CINEMATCH</span>'
    f'  </div>'
    f'  <span class="nav-tagline">SVD &nbsp;+&nbsp; RAG &nbsp;+&nbsp; LLM</span>'
    f'</div>'
    f'<div class="nav-gradient-line"></div>',
    unsafe_allow_html=True,
)

# ── User selector ──────────────────────────────────────────────
if valid_users:
    sel_col, _ = st.columns([3, 9])
    with sel_col:
        st.markdown(
            '<div class="user-selector" style="padding: 10px 40px 0;">',
            unsafe_allow_html=True,
        )
        selected_user = st.selectbox(
            "Viewing as",
            valid_users,
            key="selected_user",
            label_visibility="visible",
        )
        st.markdown("</div>", unsafe_allow_html=True)
else:
    selected_user = None

# ── Pipeline error state ───────────────────────────────────────
if pipeline is None:
    st.markdown(
        f'<div class="err-card">'
        f'  <p class="err-title">Pipeline not ready</p>'
        f'  <p class="err-body">'
        f'    Run the setup scripts first:<br>'
        f'    <code>python data/download_data.py</code><br>'
        f'    <code>python data/preprocess.py</code><br>'
        f'    <code>python models/train_svd.py</code><br>'
        f'    <code>python models/build_faiss_index.py</code>'
        + (f'<br><br>Error: <code>{_err}</code>' if _err else "")
        + f'  </p>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Hero ───────────────────────────────────────────────────────
h_left, h_right = st.columns([5, 1])
with h_left:
    st.markdown(
        '<div style="padding: 28px 40px 0;">'
        '<p class="hero-heading">Your screening room</p>'
        '<p class="hero-sub">Personalized picks based on your taste profile</p>'
        '</div>',
        unsafe_allow_html=True,
    )
with h_right:
    st.markdown('<div style="padding-top: 30px; padding-right: 40px;">', unsafe_allow_html=True)
    curate_clicked = st.button(
        "CURATE MY LIST",
        type="primary",
        key="curate_btn",
        disabled=(pipeline is None or selected_user is None),
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ── Curate action ──────────────────────────────────────────────
if curate_clicked and pipeline and selected_user:
    with st.spinner("Finding your next watch..."):
        try:
            result = pipeline.get_explained_recommendations(selected_user, n=6)
            recs_df: pd.DataFrame = result["recommendations"]
            explanation: str = result["explanation"]
            st.session_state.rec_cards = [
                {
                    "title": row.get("title"),
                    "year": row.get("year"),
                    "genres": row.get("genres") or row.get("genres_ml"),
                    "predicted_rating": row.get("predicted_rating"),
                    "director": row.get("director"),
                }
                for _, row in recs_df.iterrows()
            ]
            st.session_state.rec_explanation = explanation
        except Exception as exc:
            st.session_state.rec_cards = []
            st.session_state.rec_explanation = f"Could not generate recommendations: {exc}"

# ── Movie cards grid ───────────────────────────────────────────
if st.session_state.rec_cards:
    st.markdown('<div class="cards-pad" style="padding-top:20px;">', unsafe_allow_html=True)
    cols = st.columns(3, gap="medium")
    for i, card in enumerate(st.session_state.rec_cards):
        with cols[i % 3]:
            st.markdown(_card_html(card), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Director's Notes
    if st.session_state.rec_explanation:
        st.markdown(
            f'<div class="dir-notes">'
            f'  <p class="dir-notes-label">Director\'s Notes</p>'
            f'  <p class="dir-notes-body">{st.session_state.rec_explanation}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── Chat history ───────────────────────────────────────────────
if st.session_state.messages or pipeline:
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    st.markdown('<p class="chat-heading">Ask anything about movies</p>', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        cls = "bubble-user" if msg["role"] == "user" else "bubble-asst"
        st.markdown(
            f'<div class="{cls}"><div class="bbl">{msg["content"]}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ── Chat input ─────────────────────────────────────────────────
if prompt := st.chat_input(
    "Ask about movies, genres, directors, moods...",
    disabled=(pipeline is None),
):
    st.session_state.messages.append({"role": "user", "content": prompt})

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
        if m.get("content")
    ][-6:]

    try:
        reply = pipeline.chat(
            user_message=prompt,
            user_id=selected_user,
            conversation_history=history,
        )
    except Exception as exc:
        reply = f"Sorry, something went wrong: {exc}"

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()
