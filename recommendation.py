from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

def create_spark_session():
  return SparkSession.builder \
    .appName("Movie Recommendation") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

def load_data(spark, ratings_path, movies_path):
  ratings = spark.read.csv(ratings_path, header=True, inferSchema=True).dropna().drop("timestamp")
  movies = spark.read.csv(movies_path, header=True, inferSchema=True).dropna()
  return ratings, movies

def train_als_model(ratings):
  train, test = ratings.randomSplit([0.8, 0.2], seed = 42)
  als = ALS(
      maxIter=10,
      regParam=0.1,
      rank = 10,
      userCol="userId",
      itemCol="movieId",
      ratingCol="rating",
      coldStartStrategy="drop",
      nonnegative = True
  )
  model = als.fit(train)
  predictions = model.transform(test)
  evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
  rmse = evaluator.evaluate(predictions)
  return model, round(rmse, 3)

def get_user_recommendations(model, movies, user_id, n = 10):
  user_recs = model.recommendForAllUsers(n)
  user_recs = user_recs.filter(F.col("userId") == user_id)
  user_recs = user_recs.select(
      F.col("userId"),
      F.explode(F.col("recommendations")).alias("recommendations")
  ).select(
      F.col("userId"),
      F.col("recommendations.movieId").alias("movieId"),
      F.col("recommendations.rating").alias("rating")
  )
  result = user_recs.join(movies, on="movieId", how="left").select("title", "genres", "rating").orderBy(F.col("rating").desc())
  result = result.toPandas()
  return result