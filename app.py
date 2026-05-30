import streamlit as st
import pandas as pd
from recommendation import (
    create_spark_session,
    load_data,
    train_als_model,
    get_user_recommendations,
    precompute_all_recommendations,
)

st.set_page_config(page_title="WatchNext", page_icon="🎬", layout="wide")

st.markdown("""
    <style>
    .disclaimer-box {
        background: #fff8e1;
        border-left: 4px solid #ff9800;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #5d4037;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h2 style="color:#1a73e8;">🎬 WatchNext — Movie Recommendation System</h2>', unsafe_allow_html=True)
st.markdown('<p style="color:#666;margin-top:-0.5rem;">Developed by <strong>Jaswanth Babu Reddi</strong></p>', unsafe_allow_html=True)
st.markdown(
    '<div class="disclaimer-box">📽️ Recommendations are based on <strong>collaborative filtering</strong> '
    'from the MovieLens dataset. Results reflect historical rating patterns, not personal taste.</div>',
    unsafe_allow_html=True,
)
st.markdown("---")


@st.cache_resource(show_spinner="Training ALS model and precomputing recommendations... ⚙️")
def load_model():
    spark = create_spark_session()
    ratings, movies = load_data(spark, "data/ratings.csv", "data/movies.csv")
    model, rmse = train_als_model(ratings)
    precompute_all_recommendations(model, movies, n=20)  # ← compute once, cache for all users
    return spark, model, ratings, movies, rmse


try:
    spark, model, ratings, movies, rmse = load_model()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

with st.sidebar:
    st.markdown("## 🎬 WatchNext")
    st.markdown("---")
    st.markdown("### About")
    st.info("Enter a User ID to get personalised movie recommendations powered by the ALS collaborative filtering algorithm.")
    st.markdown("### Model Info")
    st.success(
        f"⚡ Algorithm: **ALS (PySpark)**\n\n"
        f"📊 RMSE: **{rmse}**\n\n"
        f"🎬 Dataset: **MovieLens**\n\n"
        f"👥 Users: **610**"
    )
    st.markdown("---")
    st.markdown("### What is RMSE?")
    st.info("Root Mean Square Error — measures how far predicted ratings are from actual ratings. Lower is better.")

st.markdown("### 🔍 Get Recommendations")

col1, col2 = st.columns(2)
with col1:
    user_id = st.number_input("Enter User ID", min_value=1, max_value=610, value=1,
                               help="Enter any User ID between 1 and 610 from the MovieLens dataset.")
with col2:
    n_recs = st.slider("Number of Recommendations", min_value=5, max_value=20, value=10)

if st.button("🎬 Get Recommendations", type="primary"):
    with st.spinner(f"Finding top {n_recs} movies for User {user_id}..."):
        recs = get_user_recommendations(spark, model, movies, int(user_id), n_recs)

    if recs.empty:
        st.warning(f"⚠️ No recommendations found for User {user_id}. This user may not have enough ratings history in the dataset.")
    else:
        st.markdown(f"#### 🎯 Top {n_recs} Recommendations for User {user_id}")

        m1, m2, m3 = st.columns(3)
        with m1:
            title = recs.iloc[0]["title"]
            st.metric("🎬 Top Pick", title)
        with m2:
            st.metric("⭐ Predicted Rating", f"{recs.iloc[0]['predicted_rating']:.2f} / 5.0")
        with m3:
            genres = recs.iloc[0]["genres"].split("|")[0]
            st.metric("🎭 Top Genre", genres)

        st.markdown("---")

        display_df = recs.copy()
        display_df.index = range(1, len(display_df) + 1)
        display_df.columns = ["Title", "Genres", "Predicted Rating"]
        display_df["Genres"] = display_df["Genres"].str.replace("|", ", ", regex=False)
        st.dataframe(display_df, use_container_width=True)

        with st.expander("📊 Genre Breakdown"):
            all_genres = []
            for g in recs["genres"]:
                all_genres.extend(g.split("|"))
            genre_counts = pd.Series(all_genres).value_counts().reset_index()
            genre_counts.columns = ["Genre", "Count"]
            st.bar_chart(genre_counts.set_index("Genre"))

else:
    st.markdown("""
        <div style="text-align:center;padding:3rem;color:#aaa;">
            <div style="font-size:4rem">🎬</div>
            <div style="font-size:1.2rem;margin-top:1rem">
                Enter a User ID above and click <b>Get Recommendations</b>
            </div>
            <div style="font-size:0.9rem;margin-top:0.5rem">
                Powered by ALS collaborative filtering on the MovieLens dataset
            </div>
        </div>
    """, unsafe_allow_html=True)