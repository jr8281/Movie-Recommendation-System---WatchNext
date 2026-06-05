import streamlit as st
import pandas as pd
import requests
from recommendation import (
    create_spark_session,
    load_data,
    tune_als_model,
    precompute_all_recommendations,
    precompute_popular,
    get_user_recommendations,
    get_popular_movies,
    get_tuning_results,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WatchNext",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark cinema theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] {
    background: #0a0a0f;
    color: #e8e8f0;
}
[data-testid="stSidebar"] {
    background: #12121a;
    border-right: 1px solid #2a2a3a;
}

/* ── Typography ── */
h1, h2, h3, h4 { color: #f0f0ff; letter-spacing: -0.02em; }

/* ── Movie card grid ── */
.movie-card {
    background: #16161f;
    border: 1px solid #2a2a3a;
    border-radius: 10px;
    padding: 0.75rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
    height: 100%;
}
.movie-card:hover {
    transform: translateY(-4px);
    border-color: #e50914;
}
.movie-card img {
    width: 100%;
    border-radius: 6px;
    margin-bottom: 0.5rem;
    aspect-ratio: 2/3;
    object-fit: cover;
}
.movie-card .movie-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: #f0f0ff;
    margin-bottom: 0.25rem;
    line-height: 1.3;
}
.movie-card .movie-genre {
    font-size: 0.7rem;
    color: #888;
    margin-bottom: 0.4rem;
}
.movie-card .movie-rating {
    font-size: 0.85rem;
    font-weight: 700;
    color: #e50914;
}

/* ── Stat cards ── */
.stat-card {
    background: #16161f;
    border: 1px solid #2a2a3a;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.stat-card .stat-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #e50914;
}
.stat-card .stat-label {
    font-size: 0.75rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Badge ── */
.badge {
    display: inline-block;
    background: #e50914;
    color: white;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 0.15rem 0.5rem;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: #e50914;
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: background 0.2s;
}
[data-testid="stButton"] > button:hover { background: #b0070f; }

/* ── Table ── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ── Divider ── */
hr { border-color: #2a2a3a; }

/* ── Disclaimer ── */
.disclaimer {
    background: #1a1a10;
    border-left: 3px solid #f0a500;
    border-radius: 4px;
    padding: 0.6rem 1rem;
    font-size: 0.8rem;
    color: #b0a060;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TMDB poster helper
# ─────────────────────────────────────────────────────────────────────────────
PLACEHOLDER_POSTER = "https://via.placeholder.com/200x300/16161f/555?text=No+Poster"

def get_poster_url(title: str) -> str:
    """Fetch movie poster from TMDB. Returns placeholder on any failure."""
    try:
        api_key = st.secrets.get("TMDB_API_KEY", "")
        if not api_key:
            return PLACEHOLDER_POSTER
        # Strip year from title e.g. "Toy Story (1995)" → "Toy Story"
        clean_title = title.split("(")[0].strip()
        r = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": api_key, "query": clean_title},
            timeout=3,
        )
        results = r.json().get("results", [])
        if results and results[0].get("poster_path"):
            return f"https://image.tmdb.org/t/p/w200{results[0]['poster_path']}"
    except Exception:
        pass
    return PLACEHOLDER_POSTER


# ─────────────────────────────────────────────────────────────────────────────
# Model loading — cached so it runs only once per session
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🎬 Training model and precomputing recommendations…")
def load_everything():
    spark = create_spark_session()
    ratings, movies = load_data(spark, "data/ratings.csv", "data/movies.csv")

    # Hyperparameter tuning
    model, best_rmse, best_params, tuning_table = tune_als_model(ratings)

    # Precompute caches
    precompute_all_recommendations(model, movies, n=20)
    precompute_popular(ratings, movies)

    return spark, ratings, movies, model, best_rmse, best_params, tuning_table


try:
    spark, ratings, movies, model, best_rmse, best_params, tuning_table = load_everything()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 WatchNext")
    st.caption("Personalised movie recommendations")
    st.markdown("---")

    st.markdown("### Model Performance")
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{best_rmse}</div>
        <div class="stat-label">Best RMSE</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    cols = st.columns(2)
    with cols[0]:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{best_params['rank']}</div>
            <div class="stat-label">Rank</div>
        </div>""", unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{best_params['regParam']}</div>
            <div class="stat-label">RegParam</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Dataset")
    st.markdown("""
    | | |
    |---|---|
    | 👥 Users | 610 |
    | 🎬 Movies | 9,742 |
    | ⭐ Ratings | 100,836 |
    | 🤖 Algorithm | ALS (PySpark) |
    """)

    st.markdown("---")
    st.markdown("### What is RMSE?")
    st.info("Root Mean Square Error — how far predicted ratings deviate from actual ratings. Lower = better.")

    st.markdown("---")
    st.caption("Developed by **Jaswanth Babu Reddi**")
    st.caption("SRM Institute of Science and Technology")


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="font-size:2.2rem; margin-bottom:0;">🎬 WatchNext</h1>
<p style="color:#888; margin-top:0.2rem; font-size:1rem;">
    AI-powered movie recommendations using PySpark ALS collaborative filtering
</p>
""", unsafe_allow_html=True)

st.markdown('<div class="disclaimer">📽️ Recommendations are based on collaborative filtering from the MovieLens dataset. Results reflect historical rating patterns, not personal taste.</div>', unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Get Recommendations", "📊 Model Insights", "ℹ️ About"])


# ════════════════════════════════════════════════════════════
# TAB 1 — Recommendations
# ════════════════════════════════════════════════════════════
with tab1:
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        user_id = st.number_input(
            "User ID (1 – 610)",
            min_value=1, max_value=610, value=1,
            help="Enter any User ID from the MovieLens dataset.",
        )
    with col2:
        n_recs = st.slider("Number of Recommendations", 5, 20, 10)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        show_posters = st.checkbox("Show Posters", value=True,
                                   help="Requires TMDB_API_KEY in Streamlit secrets")

    run = st.button("🎬 Get Recommendations", type="primary")

    if run:
        recs = get_user_recommendations(int(user_id), n_recs)
        is_cold_start = int(user_id) not in [uid for uid in range(1, 611)]

        if is_cold_start or recs.empty:
            st.warning(
                f"⚠️ User {user_id} not found in training data. "
                "Showing globally popular movies instead."
            )
            recs = get_popular_movies(n_recs)
            st.markdown('<span class="badge">Popular Picks</span>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<span class="badge">Personalised for User {user_id}</span>',
                unsafe_allow_html=True,
            )

        st.markdown(f"#### Top {len(recs)} Movies")

        if not recs.empty:
            # ── Top 3 metrics ────────────────────────────────────────────────
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("🥇 Top Pick", recs.iloc[0]["title"][:30] + ("…" if len(recs.iloc[0]["title"]) > 30 else ""))
            with m2:
                st.metric("⭐ Predicted Rating", f"{recs.iloc[0]['predicted_rating']:.2f} / 5.0")
            with m3:
                top_genre = recs.iloc[0]["genres"].split("|")[0]
                st.metric("🎭 Top Genre", top_genre)

            st.markdown("---")

            # ── Movie poster grid ────────────────────────────────────────────
            if show_posters:
                st.markdown("##### Recommendations")
                num_cols = 5
                rows = [recs.iloc[i:i+num_cols] for i in range(0, len(recs), num_cols)]

                for row_df in rows:
                    cols = st.columns(num_cols)
                    for col, (_, movie) in zip(cols, row_df.iterrows()):
                        poster = get_poster_url(movie["title"])
                        genre  = movie["genres"].split("|")[0]
                        rating = movie["predicted_rating"]
                        title  = movie["title"]
                        with col:
                            st.markdown(f"""
                            <div class="movie-card">
                                <img src="{poster}" alt="{title}" 
                                     onerror="this.src='{PLACEHOLDER_POSTER}'"/>
                                <div class="movie-title">{title[:35]}{"…" if len(title)>35 else ""}</div>
                                <div class="movie-genre">{genre}</div>
                                <div class="movie-rating">⭐ {rating}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
            else:
                # ── Plain table fallback ─────────────────────────────────────
                display_df = recs.copy()
                display_df.index = range(1, len(display_df) + 1)
                display_df.columns = ["Title", "Genres", "Predicted Rating"]
                display_df["Genres"] = display_df["Genres"].str.replace("|", ", ", regex=False)
                st.dataframe(display_df, use_container_width=True)

            # ── Genre breakdown ──────────────────────────────────────────────
            with st.expander("📊 Genre Breakdown"):
                all_genres = []
                for g in recs["genres"]:
                    all_genres.extend(g.split("|"))
                genre_counts = pd.Series(all_genres).value_counts().reset_index()
                genre_counts.columns = ["Genre", "Count"]
                st.bar_chart(genre_counts.set_index("Genre"))

    else:
        st.markdown("""
        <div style="text-align:center; padding:4rem; color:#444;">
            <div style="font-size:5rem">🎬</div>
            <div style="font-size:1.3rem; margin-top:1rem; color:#666;">
                Enter a User ID and click <b style="color:#e50914">Get Recommendations</b>
            </div>
            <div style="font-size:0.9rem; margin-top:0.5rem; color:#444;">
                Powered by ALS collaborative filtering · PySpark MLlib · MovieLens
            </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 2 — Model Insights
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Hyperparameter Tuning Results")
    st.markdown(
        "The model was trained with four different parameter combinations. "
        "The configuration with the **lowest RMSE** was selected automatically."
    )

    if tuning_table:
        tuning_df = pd.DataFrame(tuning_table)
        tuning_df.columns = ["Rank", "RegParam", "MaxIter", "RMSE"]
        # Highlight best row
        best_idx = tuning_df["RMSE"].idxmin()

        def highlight_best(row):
            return ["background-color: #1a2a1a; color: #4caf50; font-weight:bold"
                    if row.name == best_idx else "" for _ in row]

        st.dataframe(
            tuning_df.style.apply(highlight_best, axis=1),
            use_container_width=True,
        )
        st.success(
            f"✅ Best configuration: Rank={best_params['rank']}, "
            f"RegParam={best_params['regParam']}, "
            f"MaxIter={best_params['maxIter']} → RMSE = {best_rmse}"
        )

    st.markdown("---")
    st.markdown("### What the Metrics Mean")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **RMSE (Root Mean Square Error)**
        Measures the average prediction error in rating units (1–5 scale).
        An RMSE of ~0.87 means predictions are off by less than 1 star on average.

        **Rank**
        Number of latent factors. Higher rank = more expressive model but slower training
        and higher risk of overfitting on small datasets.
        """)
    with c2:
        st.markdown("""
        **RegParam (Regularisation)**
        Controls overfitting. Higher values penalise complexity more aggressively.
        Too high → underfitting. Too low → overfitting.

        **MaxIter**
        Number of ALS iterations. More iterations = better convergence,
        but diminishing returns beyond ~15 on this dataset size.
        """)

    st.markdown("---")
    st.markdown("### Known Limitations")
    st.markdown("""
    | Limitation | Status |
    |---|---|
    | Cold start for new users | ✅ Handled — falls back to popularity ranking |
    | Fixed hyperparameters | ✅ Resolved — grid search selects best params |
    | No cross-validation | ⚠️ Single 80/20 split — CV would improve confidence |
    | Small dataset (100K ratings) | ⚠️ Production systems use billions of interactions |
    | No user history / age / preferences | ❌ Not implemented |
    | Predicted ratings are relative scores | ⚠️ Not absolute quality measures |
    """)


# ════════════════════════════════════════════════════════════
# TAB 3 — About
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### About WatchNext")
    st.markdown("""
    **WatchNext** is a personalised movie recommendation system built using
    **PySpark ALS (Alternating Least Squares)** collaborative filtering,
    deployed as an interactive web application with Streamlit.

    The system learns latent preference patterns from 100,836 user ratings
    across 9,742 movies and generates personalised recommendations for any
    of the 610 users in the MovieLens dataset.
    """)

    st.markdown("### Tech Stack")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        **Machine Learning**
        - PySpark MLlib — ALS model
        - Hyperparameter grid search
        - RMSE evaluation
        """)
    with c2:
        st.markdown("""
        **Data & Backend**
        - Apache Spark 3.5
        - Pandas
        - MovieLens dataset
        """)
    with c3:
        st.markdown("""
        **Frontend & Deployment**
        - Streamlit
        - TMDB API (posters)
        - Streamlit Cloud
        """)

    st.markdown("### What I Learned Building This")
    st.info("""
    The biggest challenge was deploying PySpark to Streamlit Cloud — PySpark requires Java,
    which is not pre-installed on Streamlit Cloud's containers. I solved this by adding a
    `packages.txt` file specifying `default-jdk` as a system dependency, which Streamlit Cloud
    installs before the Python environment. This was a non-obvious deployment problem that took
    real debugging to solve, and it taught me the difference between Python dependencies and
    system-level dependencies in cloud deployments.

    I also learned that precomputing all recommendations at startup and caching them in a
    dictionary gives O(1) per-user lookup — far more efficient than computing recommendations
    on every request.
    """)

    st.markdown("### Developed By")
    st.markdown("""
    **Jaswanth Babu Reddi**

    B.Tech CSE — Big Data Analytics
    
    SRM Institute of Science and Technology, Kattankulathur
    """)