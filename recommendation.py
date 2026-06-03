import sys
import os

# Dynamically use the current Python environment — no hardcoded paths
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd

# ─────────────────────────────────────────────
# In-memory cache: populated once at startup
# ─────────────────────────────────────────────
_recs_cache = {}
_popular_cache = None          # fallback for cold-start / unknown users
_tuning_results  = []          # stores param-grid comparison for README / UI


def create_spark_session():
    """Create and return a configured SparkSession."""
    return (
        SparkSession.builder
        .appName("WatchNext")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "50")   # reduced: dataset is small
        .config("spark.default.parallelism", "50")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def load_data(spark, ratings_path: str, movies_path: str):
    """Load and clean ratings + movies CSVs."""
    ratings = (
        spark.read.csv(ratings_path, header=True, inferSchema=True)
        .dropna()
        .drop("timestamp")
    )
    movies = spark.read.csv(movies_path, header=True, inferSchema=True).dropna()
    return ratings, movies


def tune_als_model(ratings):
    """
    Run a small parameter grid search over ALS hyperparameters.
    Returns the best model, its RMSE, the best params dict,
    and the full comparison table as a list of dicts.
    """
    global _tuning_results

    train, test = ratings.randomSplit([0.8, 0.2], seed=42)

    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction",
    )

    # ── Parameter grid ──────────────────────────────────────────────────────
    param_grid = [
        {"rank": 10,  "regParam": 0.1,  "maxIter": 10},
        {"rank": 20,  "regParam": 0.1,  "maxIter": 10},
        {"rank": 20,  "regParam": 0.01, "maxIter": 15},
        {"rank": 50,  "regParam": 0.1,  "maxIter": 10},
    ]

    best_rmse   = float("inf")
    best_params = None
    best_model  = None
    results     = []

    for params in param_grid:
        als = ALS(
            rank=params["rank"],
            regParam=params["regParam"],
            maxIter=params["maxIter"],
            userCol="userId",
            itemCol="movieId",
            ratingCol="rating",
            coldStartStrategy="drop",
            nonnegative=True,
        )
        model       = als.fit(train)
        predictions = model.transform(test)
        rmse        = round(evaluator.evaluate(predictions), 4)

        results.append({**params, "rmse": rmse})
        print(f"  rank={params['rank']:>2}  regParam={params['regParam']}  "
              f"maxIter={params['maxIter']:>2}  →  RMSE={rmse}")

        if rmse < best_rmse:
            best_rmse   = rmse
            best_params = params
            best_model  = model

    _tuning_results = results
    print(f"\n✅ Best params: {best_params}  |  RMSE: {best_rmse}")
    return best_model, best_rmse, best_params, results


def precompute_all_recommendations(model, movies, n: int = 20):
    """
    Compute recommendations for ALL users once at startup.
    Results are cached in _recs_cache as a dict {userId: DataFrame}.
    O(1) lookup for all subsequent per-user requests.
    """
    global _recs_cache

    all_recs = model.recommendForAllUsers(n)
    all_recs = (
        all_recs
        .select(
            F.col("userId"),
            F.explode(F.col("recommendations")).alias("rec"),
        )
        .select(
            F.col("userId"),
            F.col("rec.movieId").alias("movieId"),
            F.col("rec.rating").alias("predicted_rating"),
        )
    )

    result = (
        all_recs
        .join(movies, on="movieId", how="left")
        .select("userId", "title", "genres", "predicted_rating")
        .orderBy(F.col("userId"), F.col("predicted_rating").desc())
        .toPandas()
    )

    result["predicted_rating"] = (
        result["predicted_rating"].round(2).clip(lower=0.5, upper=5.0)
    )

    for uid, group in result.groupby("userId"):
        _recs_cache[uid] = group[["title", "genres", "predicted_rating"]].reset_index(drop=True)


def precompute_popular(ratings, movies, min_ratings: int = 50, n: int = 20):
    """
    Build a global popularity fallback for cold-start / unknown users.
    Criteria: Bayesian-weighted average (avoids bias toward rarely-rated movies).
    """
    global _popular_cache

    stats = (
        ratings
        .groupBy("movieId")
        .agg(
            F.count("rating").alias("num_ratings"),
            F.avg("rating").alias("avg_rating"),
        )
        .filter(F.col("num_ratings") >= min_ratings)
    )

    # Bayesian average: (v/(v+m)) * R + (m/(v+m)) * C
    # v = num_ratings, m = min_ratings threshold, R = avg_rating, C = global mean
    global_mean = stats.agg(F.avg("avg_rating")).collect()[0][0]

    stats = stats.withColumn(
        "score",
        (F.col("num_ratings") / (F.col("num_ratings") + min_ratings)) * F.col("avg_rating")
        + (min_ratings / (F.col("num_ratings") + min_ratings)) * global_mean,
    )

    _popular_cache = (
        stats
        .orderBy(F.col("score").desc())
        .limit(n)
        .join(movies, on="movieId", how="left")
        .select("title", "genres", F.col("score").alias("predicted_rating"),
                "num_ratings")
        .toPandas()
    )
    _popular_cache["predicted_rating"] = _popular_cache["predicted_rating"].round(2)


def get_user_recommendations(user_id: int, n: int = 10) -> pd.DataFrame:
    """
    Return top-n recommendations for a known user.
    Falls back to popular movies for unknown / cold-start users.
    """
    if user_id in _recs_cache:
        return _recs_cache[user_id].head(n)

    # Cold-start fallback
    if _popular_cache is not None:
        return _popular_cache[["title", "genres", "predicted_rating"]].head(n)

    return pd.DataFrame(columns=["title", "genres", "predicted_rating"])


def get_popular_movies(n: int = 10) -> pd.DataFrame:
    """Direct access to the popular-movies fallback (used by UI)."""
    if _popular_cache is not None:
        return _popular_cache[["title", "genres", "predicted_rating"]].head(n)
    return pd.DataFrame(columns=["title", "genres", "predicted_rating"])


def get_tuning_results() -> list:
    """Return the hyperparameter comparison table."""
    return _tuning_results