"""
CineMatch AI — Premium Movie Recommendation Interface
Full-width layout with Discover / Watchlist / History tabs.
Red-yellow gradient theme on white background.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="CineMatch AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none !important; }
    #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    header { visibility: hidden !important; }
    [data-testid="stToolbar"] { display: none !important; }
    .stApp { background: #ffffff; }
    .block-container { max-width: 1100px; padding-top: 1rem; }

    .nav-bar {
        display: flex; align-items: center; justify-content: space-between;
        padding: 14px 0; position: relative; margin-bottom: 0;
    }
    .nav-bar::after {
        content: ''; position: absolute; bottom: 0; left: 0; right: 0;
        height: 2px; background: linear-gradient(90deg, #E81123, #FF6B00, #FFF100);
    }
    .nav-logo { display: flex; align-items: center; gap: 10px; }
    .nav-logo-text {
        font-size: 17px; font-weight: 600; letter-spacing: 2px;
        background: linear-gradient(135deg, #E81123, #FF6B00, #FFF100);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .nav-tagline { color: #b0b0b0; font-size: 11px; letter-spacing: 1.2px; }

    .movie-card {
        background: #fafafa; border: 0.5px solid #ececec; border-radius: 8px;
        padding: 16px 18px; position: relative; margin-bottom: 12px;
        border-top: 3px solid transparent;
        border-image: linear-gradient(90deg, #E81123, #FFF100) 1;
        border-image-slice: 1 1 0 1;
        transition: background 0.2s;
    }
    .movie-card:hover { background: #f5f5f5; }
    .card-title { color: #1a1a1a; font-size: 15px; font-weight: 500; padding-right: 55px; }
    .card-year { color: #aaa; font-size: 12px; }
    .card-rating {
        position: absolute; top: 14px; right: 14px;
        background: linear-gradient(135deg, rgba(232,17,35,0.08), rgba(255,241,0,0.08));
        border-radius: 12px; padding: 3px 9px;
        display: flex; align-items: center; gap: 4px;
    }
    .card-rating span { color: #E81123; font-size: 12px; font-weight: 500; }
    .genre-pill {
        display: inline-block;
        background: linear-gradient(135deg, rgba(232,17,35,0.07), rgba(255,241,0,0.07));
        color: #d4200e; font-size: 11px; padding: 3px 10px; border-radius: 20px;
        border: 0.5px solid rgba(232,17,35,0.12); margin: 2px;
    }
    .card-meta {
        border-top: 0.5px solid #f0f0f0; padding-top: 8px; margin-top: 8px;
        color: #bbb; font-size: 11px; font-style: italic;
    }

    .director-notes {
        border-left: 3px solid; border-image: linear-gradient(180deg, #E81123, #FFF100) 1;
        background: linear-gradient(135deg, rgba(232,17,35,0.02), rgba(255,241,0,0.02));
        padding: 16px 20px; margin: 20px 0; border-radius: 0;
    }
    .dn-label {
        font-size: 11px; letter-spacing: 1.5px; margin-bottom: 10px;
        background: linear-gradient(90deg, #E81123, #FF6B00);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .director-notes p { color: #555; font-size: 13px; line-height: 1.7; margin: 0 0 6px; }

    .chat-user {
        background: linear-gradient(135deg, #E81123, #FF6B00, #FFF100);
        color: #fff; font-size: 13px; padding: 10px 16px;
        border-radius: 16px 16px 4px 16px; max-width: 60%;
        line-height: 1.5; margin-left: auto; margin-bottom: 10px;
        box-shadow: 0 1px 4px rgba(232,17,35,0.15);
        text-shadow: 0 1px 1px rgba(0,0,0,0.1);
    }
    .chat-bot {
        background: #f8f8f8; border: 0.5px solid #ececec; color: #555;
        font-size: 13px; padding: 14px 18px;
        border-radius: 16px 16px 16px 4px; max-width: 70%;
        line-height: 1.7; margin-bottom: 10px;
    }
    .chat-bot strong { color: #1a1a1a; }

    .empty-box {
        text-align: center; padding: 60px 20px; color: #ccc; font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)


# ── Pipeline ───────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    from utils.pipeline import RecommendationPipeline
    return RecommendationPipeline()

try:
    pipeline = load_pipeline()
    pipeline_ready = True
except Exception as e:
    pipeline_ready = False
    pipeline_error = str(e)


# ── Session State ──────────────────────────────────────────────────
for key, default in [
    ("active_tab", "discover"),
    ("messages", []),
    ("recs", None),
    ("explanation", None),
    ("watchlist", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

STAR = (
    '<svg width="12" height="12" viewBox="0 0 12 12" style="vertical-align:middle;">'
    '<defs><linearGradient id="gs" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0%" stop-color="#E81123"/><stop offset="100%" stop-color="#FFF100"/>'
    '</linearGradient></defs>'
    '<polygon points="6,0.5 7.5,4.2 11.5,4.5 8.5,7 9.4,11 6,9 2.6,11 3.5,7 0.5,4.5 4.5,4.2" fill="url(#gs)"/></svg>'
)


def card_html(title, year, genres, rating, director=""):
    yr = ""
    if pd.notna(year):
        yr = str(int(year)) if isinstance(year, float) else str(year)
    rt = f"{float(rating):.2f}" if pd.notna(rating) else ""
    pills = ""
    if genres and str(genres) != "nan":
        for g in str(genres).split(","):
            g = g.strip()
            if g:
                pills += f'<span class="genre-pill">{g}</span>'
    meta = f'<div class="card-meta">{director}</div>' if director and str(director) != "nan" else ""
    return (
        f'<div class="movie-card">'
        f'<div class="card-rating">{STAR}<span>{rt}</span></div>'
        f'<div class="card-title">{title}</div>'
        f'<div class="card-year">{yr}</div>'
        f'<div style="margin:8px 0">{pills}</div>'
        f'{meta}</div>'
    )


# ── Nav Bar ────────────────────────────────────────────────────────
st.markdown(
    '<div class="nav-bar"><div class="nav-logo">'
    '<svg width="26" height="26" viewBox="0 0 26 26" fill="none">'
    '<defs><linearGradient id="nlg" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0%" stop-color="#E81123"/><stop offset="100%" stop-color="#FFF100"/>'
    '</linearGradient></defs>'
    '<rect x="2" y="3" width="22" height="17" rx="2.5" stroke="url(#nlg)" stroke-width="1.4" fill="none"/>'
    '<polygon points="10.5,10.5 10.5,16 16,13.25" fill="url(#nlg)"/></svg>'
    '<span class="nav-logo-text">CINEMATCH</span></div>'
    '<span class="nav-tagline">SVD &middot; RAG &middot; LLM</span></div>',
    unsafe_allow_html=True,
)


# ── Tab Buttons ────────────────────────────────────────────────────
tc1, tc2, tc3, _ = st.columns([1, 1.2, 1, 5])
with tc1:
    if st.button("Discover", use_container_width=True):
        st.session_state.active_tab = "discover"
        st.rerun()
with tc2:
    wl_count = len(st.session_state.watchlist)
    if st.button(f"Watchlist ({wl_count})", use_container_width=True):
        st.session_state.active_tab = "watchlist"
        st.rerun()
with tc3:
    if st.button("History", use_container_width=True):
        st.session_state.active_tab = "history"
        st.rerun()

# Visual tab indicator
tab = st.session_state.active_tab
disc_cls = "background:linear-gradient(135deg,#E81123,#FF6B00,#FFF100);color:#fff;box-shadow:0 2px 0 #8B0A15,0 3px 8px rgba(232,17,35,0.2);text-shadow:0 1px 1px rgba(0,0,0,0.15);"
off_cls = "background:#f5f5f5;color:#999;border:0.5px solid #e8e8e8;"
st.markdown(
    f'<div style="display:flex;gap:8px;margin:0 0 20px;">'
    f'<span style="padding:8px 20px;border-radius:20px;font-size:13px;font-weight:500;{disc_cls if tab=="discover" else off_cls}">Discover</span>'
    f'<span style="padding:8px 20px;border-radius:20px;font-size:13px;font-weight:500;{disc_cls if tab=="watchlist" else off_cls}">Watchlist ({wl_count})</span>'
    f'<span style="padding:8px 20px;border-radius:20px;font-size:13px;font-weight:500;{disc_cls if tab=="history" else off_cls}">History</span>'
    f'</div>',
    unsafe_allow_html=True,
)


# ── User Selector ──────────────────────────────────────────────────
if pipeline_ready:
    valid_users = pipeline.get_valid_users(sample=20)
    uc1, uc2, _ = st.columns([1, 1.5, 6])
    with uc1:
        st.markdown('<p style="color:#aaa;font-size:12px;margin:8px 0 0;">Viewer profile</p>', unsafe_allow_html=True)
    with uc2:
        user_id = st.selectbox("u", options=valid_users, label_visibility="collapsed")
else:
    user_id = 1
    st.markdown(
        f'<div style="background:#fdf6f6;border:0.5px solid #f0c0c0;border-radius:8px;'
        f'padding:14px 18px;color:#c44;font-size:13px;">Pipeline not ready. '
        f'Please run setup steps first.</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
#  DISCOVER
# ══════════════════════════════════════════════════════════════════
if tab == "discover" and pipeline_ready:

    h1, h2 = st.columns([4, 1.5])
    with h1:
        st.markdown(
            '<h2 style="color:#1a1a1a;font-size:22px;font-weight:500;margin:0;">Your screening room</h2>'
            '<p style="color:#aaa;font-size:13px;margin:4px 0 0;">Personalized picks based on your taste profile</p>',
            unsafe_allow_html=True,
        )
    with h2:
        curate = st.button("CURATE MY LIST", key="curate", use_container_width=True)

    if curate:
        with st.spinner("Finding your next watch..."):
            result = pipeline.get_explained_recommendations(user_id, n=6)
            st.session_state.recs = result["recommendations"]
            st.session_state.explanation = result["explanation"]

    if st.session_state.recs is not None:
        recs_df = st.session_state.recs
        cols = st.columns(3)
        for i, (_, r) in enumerate(recs_df.head(6).iterrows()):
            with cols[i % 3]:
                t = r.get("title", "Unknown")
                st.markdown(
                    card_html(
                        t,
                        r.get("year"),
                        r.get("genres_list", r.get("genres", "")),
                        r.get("predicted_rating", 0),
                        r.get("director", ""),
                    ),
                    unsafe_allow_html=True,
                )
                movie_info = {
                    "title": t,
                    "year": r.get("year"),
                    "genres": r.get("genres_list", r.get("genres", "")),
                    "rating": r.get("predicted_rating", 0),
                    "director": r.get("director", ""),
                }
                already = any(m["title"] == t for m in st.session_state.watchlist)
                label = "Added" if already else "+ Watchlist"
                if st.button(label, key=f"a_{i}", disabled=already):
                    st.session_state.watchlist.append(movie_info)
                    st.rerun()

        if st.session_state.explanation:
            exp = st.session_state.explanation.replace("\n", "<br>")
            st.markdown(
                f'<div class="director-notes">'
                f'<div class="dn-label">DIRECTOR\'S NOTES</div>'
                f'<p>{exp}</p></div>',
                unsafe_allow_html=True,
            )

    # Chat
    st.markdown('<hr style="border:none;border-top:0.5px solid #f0f0f0;margin:24px 0 16px;">', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        cls = "chat-user" if msg["role"] == "user" else "chat-bot"
        st.markdown(f'<div class="{cls}">{msg["content"]}</div>', unsafe_allow_html=True)

    if prompt := st.chat_input("Describe what you're in the mood for..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Finding your next watch..."):
            hist = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-6:-1]]
            resp = pipeline.chat(user_message=prompt, user_id=user_id, conversation_history=hist)
        st.session_state.messages.append({"role": "assistant", "content": resp})
        st.rerun()


# ══════════════════════════════════════════════════════════════════
#  WATCHLIST
# ══════════════════════════════════════════════════════════════════
elif tab == "watchlist":

    st.markdown(
        f'<h2 style="color:#1a1a1a;font-size:22px;font-weight:500;margin:0 0 4px;">Your watchlist</h2>'
        f'<p style="color:#aaa;font-size:13px;margin:0 0 20px;">{len(st.session_state.watchlist)} films saved</p>',
        unsafe_allow_html=True,
    )

    if not st.session_state.watchlist:
        st.markdown(
            '<div class="empty-box">'
            '<div style="font-size:36px;color:#e0e0e0;margin-bottom:12px;">&#9734;</div>'
            'Your watchlist is empty.<br>Discover movies and tap <strong>+ Watchlist</strong> to save them here.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        cols = st.columns(3)
        for i, m in enumerate(st.session_state.watchlist):
            with cols[i % 3]:
                st.markdown(
                    card_html(m["title"], m.get("year"), m.get("genres", ""), m.get("rating", 0), m.get("director", "")),
                    unsafe_allow_html=True,
                )
                if st.button("Remove", key=f"rm_{i}"):
                    st.session_state.watchlist = [x for x in st.session_state.watchlist if x["title"] != m["title"]]
                    st.rerun()


# ══════════════════════════════════════════════════════════════════
#  HISTORY
# ══════════════════════════════════════════════════════════════════
elif tab == "history" and pipeline_ready:

    st.markdown(
        '<h2 style="color:#1a1a1a;font-size:22px;font-weight:500;margin:0 0 4px;">Your viewing history</h2>'
        '<p style="color:#aaa;font-size:13px;margin:0 0 20px;">Movies you have rated, sorted by your rating</p>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading your history..."):
        hist_df = pipeline.recommender.get_user_history(user_id)

    if hist_df.empty:
        st.markdown(
            '<div class="empty-box">'
            '<div style="font-size:36px;color:#e0e0e0;margin-bottom:12px;">&#9201;</div>'
            'No viewing history found for this user.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        top = hist_df.head(30)
        cols = st.columns(3)
        for i, (_, r) in enumerate(top.iterrows()):
            with cols[i % 3]:
                t = r.get("title", "Unknown")
                st.markdown(
                    card_html(
                        t,
                        r.get("year"),
                        r.get("genres_list", r.get("genres", "")),
                        r.get("rating"),
                        r.get("director", ""),
                    ),
                    unsafe_allow_html=True,
                )
                already = any(m["title"] == t for m in st.session_state.watchlist)
                label = "Added" if already else "+ Watchlist"
                if st.button(label, key=f"h_{i}", disabled=already):
                    st.session_state.watchlist.append({
                        "title": t,
                        "year": r.get("year"),
                        "genres": r.get("genres_list", r.get("genres", "")),
                        "rating": r.get("rating"),
                        "director": r.get("director", ""),
                    })
                    st.rerun()

elif not pipeline_ready:
    st.markdown(
        '<div class="empty-box">'
        '<div style="font-size:36px;color:#e0e0e0;margin-bottom:12px;">&#9888;</div>'
        'Pipeline not ready. Run all setup steps first. See README.md.'
        '</div>',
        unsafe_allow_html=True,
    )
