# 🎬 WatchNext — Movie Recommendation System

A personalised movie recommendation system powered by **PySpark ALS** collaborative filtering, deployed as an interactive web app with Streamlit.

## 🔗 Live Demo
👉 **[Try it on Streamlit Cloud](https://your-app-link.streamlit.app)** 

---

## What it does

WatchNext takes a User ID from the MovieLens dataset and returns personalised movie recommendations using the ALS (Alternating Least Squares) matrix factorisation algorithm. The app displays the top N recommended movies with predicted ratings and a genre breakdown.

---

## How it works

1. **Data** — MovieLens dataset (610 users, 9,742 movies, 100,836 ratings)
2. **Algorithm** — ALS collaborative filtering via PySpark MLlib. Learns latent factors from user-movie rating patterns without requiring any movie metadata
3. **Efficiency** — Uses `recommendForUserSubset()` to compute recommendations only for the requested user, not all 610 users
4. **Evaluation** — Model performance measured by RMSE on a held-out 20% test split

---

## Model performance

| Metric | Value |
|---|---|
| Algorithm | ALS (PySpark MLlib) |
| Training split | 80% |
| Test split | 20% |
| Evaluation metric | RMSE |
| Dataset | MovieLens (ml-latest-small) |

---

## Project structure

```
WatchNext/
├── app.py                  # Streamlit UI
├── recommendation.py       # Spark session, data loading, ALS training, inference
├── requirements.txt        # Python dependencies
├── packages.txt            # System dependencies (Java for PySpark)
├── runtime.txt             # Python version
├── .gitignore
├── README.md
└── data/
    ├── movies.csv          # movieId, title, genres
    └── ratings.csv         # userId, movieId, rating, timestamp
```

---

## Run locally

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Movie-Recommendation-System---WatchNext.git
cd Movie-Recommendation-System---WatchNext

# 2. Install Java (required for PySpark)
# Download from https://adoptium.net and set JAVA_HOME

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## Deploy to Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select this repo → set main file to `app.py`
4. Click **Deploy**

Streamlit Cloud reads `packages.txt` to install Java automatically, so PySpark works without any manual setup.

---

## Known limitations

- Only works for existing users (IDs 1–610) — no cold start support for new users
- ALS hyperparameters (rank, regParam, maxIter) are fixed — no tuning performed
- Dataset is small (100K ratings) — production systems use billions of interactions
- Predicted ratings are relative scores, not absolute quality measures

---

## Tech stack

- **PySpark MLlib** — ALS model training and inference
- **Streamlit** — web application UI
- **Pandas** — final result formatting and display
- **MovieLens** — open dataset by GroupLens Research

---

## Developed by

Jaswanth Babu Reddi