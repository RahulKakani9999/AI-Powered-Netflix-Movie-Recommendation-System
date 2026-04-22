"""
CineMatch AI — Premium Movie Recommendation Interface
"""

import streamlit as st
import pandas as pd
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="CineMatch AI", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none !important; }
    #MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden !important; }
    .stApp { background: #ffffff; }
    .block-container { max-width: 1100px; padding-top: 1rem; }

    .nav-logo { display: flex; align-items: center; gap: 10px; }
    .nav-logo-text {
        font-size: 17px; font-weight: 600; letter-spacing: 2px;
        background: linear-gradient(135deg, #E81123, #FF6B00, #FFF100);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }

    .movie-card {
        background: #fafafa;
        border: 1px solid #e8e8e8;
        border-radius: 10px;
        padding: 16px 18px;
        position: relative;
        margin-bottom: 8px;
        min-height: 170px;
        overflow: hidden;
        transition: background 0.2s;
    }
    .movie-card::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #E81123, #FF6B00, #FFF100);
        border-radius: 10px 10px 0 0;
    }
    .movie-card:hover { background: #f2f2f2; }
    .card-title { color: #1a1a1a; font-size: 15px; font-weight: 500; padding-right: 55px; line-height: 1.3; }
    .card-year { color: #aaa; font-size: 12px; margin-top: 2px; }
    .card-rating {
        position: absolute; top: 14px; right: 14px;
        background: linear-gradient(135deg, rgba(232,17,35,0.08), rgba(255,241,0,0.08));
        border-radius: 12px; padding: 3px 9px; display: flex; align-items: center; gap: 4px;
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
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }

    .director-notes {
        border-left: 3px solid #E81123;
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
    }
    .chat-bot {
        background: #f8f8f8; border: 0.5px solid #ececec; color: #555;
        font-size: 13px; padding: 14px 18px;
        border-radius: 16px 16px 16px 4px; max-width: 70%;
        line-height: 1.7; margin-bottom: 10px;
    }
    .chat-bot strong { color: #1a1a1a; }
    .empty-box { text-align: center; padding: 60px 20px; color: #ccc; font-size: 14px; }

    .stButton > button {
        font-weight: 500 !important; font-size: 13px !important;
        border-radius: 20px !important; padding: 6px 18px !important;
        background: #f5f5f5 !important; color: #666 !important;
        border: 0.5px solid #e0e0e0 !important; transition: all 0.15s !important;
    }
    .stButton > button:hover { background: #eee !important; color: #333 !important; }
    .stButton > button:active { transform: scale(0.98) !important; }
    .stButton > button:disabled { background: #f0f0f0 !important; color: #ccc !important; }
</style>
""", unsafe_allow_html=True)


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


def get_recommender(pipe):
    for attr in ["recommender", "svd_recommender", "svd", "rec"]:
        if hasattr(pipe, attr):
            return getattr(pipe, attr)
    return None


for key, default in [("active_tab", "discover"), ("messages", []),
                      ("recs", None), ("explanation", None), ("watchlist", [])]:
    if key not in st.session_state:
        st.session_state[key] = default

STAR = (
    '<svg width="12" height="12" viewBox="0 0 12 12" style="vertical-align:middle;">'
    '<defs><linearGradient id="gs" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0%" stop-color="#E81123"/><stop offset="100%" stop-color="#FFF100"/>'
    '</linearGradient></defs>'
    '<polygon points="6,0.5 7.5,4.2 11.5,4.5 8.5,7 9.4,11 6,9 2.6,11 3.5,7 0.5,4.5 4.5,4.2" fill="url(#gs)"/></svg>'
)


def clean_title(title):
    if not title or str(title) == "nan":
        return str(title)
    return re.sub(r'\s*\(\d{4}\)\s*$', '', str(title)).strip()


def parse_genres(row):
    for col in ["genres_list", "genres"]:
        val = row.get(col, "")
        if val and str(val) != "nan":
            val = str(val)
            if "|" in val:
                return [g.strip() for g in val.split("|") if g.strip()]
            return [g.strip() for g in val.split(",") if g.strip()]
    return []


def build_genre_html(genres_list):
    if not genres_list:
        return ''
    html = ""
    for g in genres_list[:4]:
        html += '<span class="genre-pill">' + g + '</span>'
    return html


def card_html(title, year, genres_list, rating, director="", cast=""):
    ct = clean_title(title)
    yr = ""
    if pd.notna(year):
        if isinstance(year, float):
            yr = str(int(year))
        else:
            yr = str(year)
    rt = ""
    if pd.notna(rating):
        rt = "{:.2f}".format(float(rating))

    genre_html = build_genre_html(genres_list)

    meta_parts = []
    if director and str(director) != "nan":
        meta_parts.append(str(director))
    if cast and str(cast) != "nan":
        cast_short = ", ".join(str(cast).split(",")[:3])
        meta_parts.append(cast_short)
    meta_text = " · ".join(meta_parts)

    meta_html = ""
    if meta_text:
        meta_html = '<div class="card-meta">' + meta_text + '</div>'

    result = '<div class="movie-card">'
    result += '<div class="card-rating">' + STAR + '<span>' + rt + '</span></div>'
    result += '<div class="card-title">' + ct + '</div>'
    result += '<div class="card-year">' + yr + '</div>'
    if genre_html:
        result += '<div style="margin:8px 0">' + genre_html + '</div>'
    result += meta_html
    result += '</div>'
    return result


if pipeline_ready:
    valid_users = pipeline.get_valid_users(sample=20)
else:
    valid_users = [1]

nav_c1, nav_c2, nav_c3 = st.columns([3, 1.5, 1])
with nav_c1:
    st.markdown(
        '<div class="nav-logo">'
        '<svg width="26" height="26" viewBox="0 0 26 26" fill="none">'
        '<defs><linearGradient id="nlg" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0%" stop-color="#E81123"/><stop offset="100%" stop-color="#FFF100"/>'
        '</linearGradient></defs>'
        '<rect x="2" y="3" width="22" height="17" rx="2.5" stroke="url(#nlg)" stroke-width="1.4" fill="none"/>'
        '<polygon points="10.5,10.5 10.5,16 16,13.25" fill="url(#nlg)"/></svg>'
        '<span class="nav-logo-text">CINEMATCH</span></div>',
        unsafe_allow_html=True,
    )
with nav_c2:
    st.markdown(
        '<p style="color:#b0b0b0;font-size:11px;letter-spacing:1.2px;text-align:right;margin-top:8px;">'
        'SVD &middot; RAG &middot; LLM</p>',
        unsafe_allow_html=True,
    )
with nav_c3:
    user_id = st.selectbox("Viewer", options=valid_users, label_visibility="collapsed")

st.markdown(
    '<div style="height:2px;background:linear-gradient(90deg,#E81123,#FF6B00,#FFF100);margin:-8px 0 16px;"></div>',
    unsafe_allow_html=True,
)

if not pipeline_ready:
    st.markdown(
        '<div style="background:#fdf6f6;border:0.5px solid #f0c0c0;border-radius:8px;'
        'padding:14px 18px;color:#c44;font-size:13px;">Pipeline not ready. Run setup steps first.</div>',
        unsafe_allow_html=True,
    )

wl_count = len(st.session_state.watchlist)
tc1, tc2, tc3, _ = st.columns([1, 1.2, 1, 5])
with tc1:
    if st.button("Discover", key="t_d", use_container_width=True):
        st.session_state.active_tab = "discover"
        st.rerun()
with tc2:
    if st.button("Watchlist (" + str(wl_count) + ")", key="t_w", use_container_width=True):
        st.session_state.active_tab = "watchlist"
        st.rerun()
with tc3:
    if st.button("History", key="t_h", use_container_width=True):
        st.session_state.active_tab = "history"
        st.rerun()

tab = st.session_state.active_tab
tab_idx = {"discover": 1, "watchlist": 2, "history": 3}[tab]
st.markdown(
    "<style>"
    "div.stColumns:nth-of-type(2) > div:nth-child(" + str(tab_idx) + ") .stButton > button {"
    "    background: linear-gradient(135deg, #E81123, #FF6B00, #FFF100) !important;"
    "    color: #fff !important; border: none !important;"
    "    box-shadow: 0 2px 0 #8B0A15, 0 3px 8px rgba(232,17,35,0.2) !important;"
    "}"
    "</style>",
    unsafe_allow_html=True,
)


if tab == "discover" and pipeline_ready:
    h1, h2 = st.columns([4, 1.5])
    with h1:
        st.markdown(
            '<h2 style="color:#1a1a1a;font-size:22px;font-weight:500;margin:0;">Your screening room</h2>'
            '<p style="color:#aaa;font-size:13px;margin:4px 0 16px;">Personalized picks based on your taste profile</p>',
            unsafe_allow_html=True,
        )
    with h2:
        curate = st.button("CURATE MY LIST", key="curate", use_container_width=True)

    st.markdown(
        "<style>"
        "div.stColumns:nth-of-type(3) > div:nth-child(2) .stButton > button {"
        "    background: linear-gradient(180deg, #FF3333 0%, #E81123 50%, #C20E1E 100%) !important;"
        "    color: #fff !important; border: none !important; border-radius: 8px !important;"
        "    box-shadow: 0 2px 0 #8B0A15, 0 4px 8px rgba(232,17,35,0.25) !important;"
        "    font-weight: 600 !important; letter-spacing: 0.8px !important; padding: 10px 24px !important;"
        "}"
        "</style>",
        unsafe_allow_html=True,
    )

    if curate:
        with st.spinner("Finding your next watch..."):
            result = pipeline.get_explained_recommendations(user_id, n=6)
            st.session_state.recs = result["recommendations"]
            st.session_state.explanation = result["explanation"]

    if st.session_state.recs is not None:
        recs_df = st.session_state.recs
        recs_list = list(recs_df.head(6).iterrows())
        for row_start in range(0, len(recs_list), 3):
            cols = st.columns(3)
            for j in range(3):
                idx = row_start + j
                if idx >= len(recs_list):
                    break
                _, r = recs_list[idx]
                with cols[j]:
                    t = r.get("title", "Unknown")
                    genres = parse_genres(r)
                    html = card_html(
                        t, r.get("year"), genres,
                        r.get("predicted_rating", 0),
                        r.get("director", ""),
                        r.get("cast_list", ""),
                    )
                    st.markdown(html, unsafe_allow_html=True)
                    movie_info = {
                        "title": t,
                        "year": r.get("year"),
                        "genres": ", ".join(genres),
                        "rating": r.get("predicted_rating", 0),
                        "director": r.get("director", ""),
                    }
                    already = any(m["title"] == t for m in st.session_state.watchlist)
                    btn_label = "Added" if already else "+ Watchlist"
                    if st.button(btn_label, key="a_" + str(idx), disabled=already):
                        st.session_state.watchlist.append(movie_info)
                        st.rerun()

        if st.session_state.explanation:
            exp = st.session_state.explanation.replace("\n", "<br>")
            st.markdown(
                '<div class="director-notes">'
                '<div class="dn-label">DIRECTOR\'S NOTES</div>'
                '<p>' + exp + '</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<hr style="border:none;border-top:0.5px solid #f0f0f0;margin:24px 0 16px;">',
        unsafe_allow_html=True,
    )
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown('<div class="chat-user">' + msg["content"] + '</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="chat-bot">' + msg["content"] + '</div>', unsafe_allow_html=True)

    if prompt := st.chat_input("Describe what you're in the mood for..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Finding your next watch..."):
            hist = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[-6:-1]
            ]
            resp = pipeline.chat(
                user_message=prompt, user_id=user_id, conversation_history=hist,
            )
        st.session_state.messages.append({"role": "assistant", "content": resp})
        st.rerun()


elif tab == "watchlist":
    st.markdown(
        '<h2 style="color:#1a1a1a;font-size:22px;font-weight:500;margin:0 0 4px;">Your watchlist</h2>'
        '<p style="color:#aaa;font-size:13px;margin:0 0 20px;">'
        + str(len(st.session_state.watchlist)) + ' films saved</p>',
        unsafe_allow_html=True,
    )

    if not st.session_state.watchlist:
        st.markdown(
            '<div class="empty-box">'
            '<div style="font-size:36px;color:#e0e0e0;margin-bottom:12px;">&#9734;</div>'
            'Your watchlist is empty.<br>Tap <strong>+ Watchlist</strong> on any movie to save it here.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        wl = st.session_state.watchlist
        for row_start in range(0, len(wl), 3):
            cols = st.columns(3)
            for j in range(3):
                idx = row_start + j
                if idx >= len(wl):
                    break
                m = wl[idx]
                with cols[j]:
                    g_str = m.get("genres", "")
                    if g_str and str(g_str) != "nan":
                        g_list = [x.strip() for x in str(g_str).split(",") if x.strip()]
                    else:
                        g_list = []
                    html = card_html(
                        m["title"], m.get("year"), g_list,
                        m.get("rating", 0), m.get("director", ""),
                    )
                    st.markdown(html, unsafe_allow_html=True)
                    if st.button("Remove", key="rm_" + str(idx)):
                        st.session_state.watchlist = [
                            x for x in st.session_state.watchlist
                            if x["title"] != m["title"]
                        ]
                        st.rerun()


elif tab == "history" and pipeline_ready:
    st.markdown(
        '<h2 style="color:#1a1a1a;font-size:22px;font-weight:500;margin:0 0 4px;">Your viewing history</h2>'
        '<p style="color:#aaa;font-size:13px;margin:0 0 20px;">Movies you rated, sorted by rating</p>',
        unsafe_allow_html=True,
    )

    rec = get_recommender(pipeline)
    hist_df = pd.DataFrame()

    if rec is not None and hasattr(rec, "get_user_history"):
        with st.spinner("Loading history..."):
            hist_df = rec.get_user_history(user_id)
    else:
        try:
            with st.spinner("Loading history..."):
                hist_df = pipeline.get_user_history(user_id)
        except Exception:
            try:
                from config import PROCESSED_DIR
                ratings = pd.read_csv(PROCESSED_DIR / "ratings.csv")
                movies = pd.read_csv(PROCESSED_DIR / "movies.csv")
                user_ratings = ratings[ratings["user_id"] == user_id]
                hist_df = user_ratings.merge(movies, on="movie_id", how="left")
                hist_df = hist_df.sort_values("rating", ascending=False)
            except Exception as ex:
                st.error("Could not load history: " + str(ex))

    if hist_df.empty:
        st.markdown(
            '<div class="empty-box">'
            '<div style="font-size:36px;color:#e0e0e0;margin-bottom:12px;">&#9201;</div>'
            'No history found for this user.</div>',
            unsafe_allow_html=True,
        )
    else:
        top = hist_df.head(30)
        top_list = list(top.iterrows())
        for row_start in range(0, len(top_list), 3):
            cols = st.columns(3)
            for j in range(3):
                idx = row_start + j
                if idx >= len(top_list):
                    break
                _, r = top_list[idx]
                with cols[j]:
                    t = r.get("title", "Unknown")
                    genres = parse_genres(r)
                    html = card_html(
                        t, r.get("year"), genres,
                        r.get("rating"),
                        r.get("director", ""),
                        r.get("cast_list", ""),
                    )
                    st.markdown(html, unsafe_allow_html=True)
                    already = any(m["title"] == t for m in st.session_state.watchlist)
                    btn_label = "Added" if already else "+ Watchlist"
                    if st.button(btn_label, key="h_" + str(idx), disabled=already):
                        st.session_state.watchlist.append({
                            "title": t,
                            "year": r.get("year"),
                            "genres": ", ".join(genres),
                            "rating": r.get("rating"),
                            "director": r.get("director", ""),
                        })
                        st.rerun()

elif not pipeline_ready:
    st.markdown(
        '<div class="empty-box">'
        '<div style="font-size:36px;color:#e0e0e0;margin-bottom:12px;">&#9888;</div>'
        'Pipeline not ready. Run setup steps first.</div>',
        unsafe_allow_html=True,
    )
