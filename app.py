import streamlit as st
import pandas as pd
from recommendation import (
    create_spark_session,
    load_data,
    train_als_model,
    get_user_recommendations
)

st.set_page_config(
    page_title="WatchNext",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System - WatchNext")
st.caption("developed by Jaswanth Babu Reddi")

@st.cache_resource
def load_model():
    spark = create_spark_session()
    ratings, movies = load_data(spark, "data/ratings.csv", "data/movies.csv")
    model, rmse = train_als_model(ratings)
    return model, ratings, movies, rmse

with st.spinner("Loading model... This may take a minute!"):
    model, ratings, movies, rmse = load_model()

st.success("Model is ready!")

st.subheader("Find Movies For a User")

col1, col2 = st.columns(2)

with col1:
    user_id = st.number_input(
        "Enter User ID",
        min_value=1,
        max_value=610,
        value=1
    )

with col2:
    n_recs = st.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10
    )

if st.button("Get Recommendations"):
    with st.spinner("Finding movies..."):
        recs = get_user_recommendations(model, movies, int(user_id), n_recs)
    
    if recs.empty:
        st.warning("No recommendations found for this user.")
    else:
        st.subheader(f"Top {n_recs} Movies for User {user_id}")
        st.dataframe(recs, use_container_width=True)

