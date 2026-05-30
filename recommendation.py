import os
os.environ["PYSPARK_PYTHON"] = r"C:\Users\JASWANTH REDDI\AppData\Local\Programs\Python\Python311\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\JASWANTH REDDI\AppData\Local\Programs\Python\Python311\python.exe"

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd

# Cache all recommendations in memory after first compute
_recs_cache = {}


def create_spark_session():
    return SparkSession.builder \
        .appName("Movie Recommendation") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.default.parallelism", "200") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()


def load_data(spark, ratings_path, movies_path):
    ratings = spark.read.csv(ratings_path, header=True, inferSchema=True).dropna().drop("timestamp")
    movies = spark.read.csv(movies_path, header=True, inferSchema=True).dropna()
    return ratings, movies


def train_als_model(ratings):
    train, test = ratings.randomSplit([0.8, 0.2], seed=42)
    als = ALS(
        maxIter=10,
        regParam=0.1,
        rank=10,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True
    )
    model = als.fit(train)
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    return model, round(rmse, 3)


def precompute_all_recommendations(model, movies, n=20):
    """
    Compute recommendations for ALL users once and cache as a pandas DataFrame.
    Called once at startup — fast for all subsequent per-user lookups.
    """
    global _recs_cache

    all_recs = model.recommendForAllUsers(n)

    all_recs = all_recs.select(
        F.col("userId"),
        F.explode(F.col("recommendations")).alias("rec")
    ).select(
        F.col("userId"),
        F.col("rec.movieId").alias("movieId"),
        F.col("rec.rating").alias("predicted_rating"),
    )

    result = (
        all_recs
        .join(movies, on="movieId", how="left")
        .select("userId", "title", "genres", "predicted_rating")
        .orderBy(F.col("userId"), F.col("predicted_rating").desc())
        .toPandas()
    )

    result["predicted_rating"] = result["predicted_rating"].round(2)
    result["predicted_rating"] = result["predicted_rating"].clip(lower=0.5, upper=5.0)

    # Cache per userId for O(1) lookup
    for uid, group in result.groupby("userId"):
        _recs_cache[uid] = group[["title", "genres", "predicted_rating"]].reset_index(drop=True)


def get_user_recommendations(spark, model, movies, user_id, n=10):
    global _recs_cache

    # Precompute once if cache is empty
    if not _recs_cache:
        precompute_all_recommendations(model, movies, n=20)

    user_recs = _recs_cache.get(user_id, pd.DataFrame(columns=["title", "genres", "predicted_rating"]))
    return user_recs.head(n)