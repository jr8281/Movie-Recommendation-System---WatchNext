🎬 WatchNext

A Movie Recommendation System powered by PySpark ALS

Developed by Jaswanth Babu Reddi

Project Description

WatchNext is a movie recommendation system built using the ALS (Alternating Least Squares) collaborative filtering algorithm from PySpark MLlib. It analyzes user rating patterns from the MovieLens dataset and suggests personalized movie recommendations for any given user.

The app is deployed as an interactive web application using Streamlit, allowing users to simply enter a User ID and instantly get their top movie picks.

Features

🔍 Personalized Recommendations — Enter any User ID (1–610) and get top N movie recommendations tailored specifically for that user

⚡ Powered by PySpark — All data processing and model training is done using PySpark DataFrames — no Pandas used in the pipeline

🎯 ALS Collaborative Filtering — Uses matrix factorization to learn hidden patterns from user rating history

📊 Model Accuracy Display — Shows the RMSE score of the trained model so you can evaluate its performance

🎛️ Adjustable Results — Slider to control how many recommendations to display (5 to 20)

🌐 Live Web App — Fully deployed and accessible from any browser via Streamlit Cloud


PySpark - Data processing and ALS model training

Streamlit - Web application UI

Pandas - Final result display

Python 3.11 - Runtime environment

MovieLens - Dataset
