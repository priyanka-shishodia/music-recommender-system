'''
Script evaluates our best model, with tuned rank and reg parameters, on the full test set

Usage:
    $ spark-submit --driver-memory=4g --executor-memory=24g --executor-cores=50 test_eval.py
'''

import sys
import time
import getpass
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql import dataframe

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

def data_prep(spark, data):
	'''
	Function to prepare the training and validation data

	Parameters:
	----------
	spark: Spark Session object
	data: Unprocessed DataFrame
	'''
	indexer_1 = StringIndexer(inputCol = 'user_id', outputCol = 'user_id_index')
	indexer_2 = StringIndexer(inputCol = 'track_id', outputCol = 'track_id_index')
	indexed = indexer_1.fit(data).transform(data)
	transformed = indexer_2.fit(indexed).transform(indexed)

	df = transformed.select(transformed['user_id_index'].astype('int'), transformed['track_id_index'].astype('int'), transformed['count'])
	return df

def main(spark):
	# Load training parquet file into a dataframe
	train_data = spark.read.parquet('hdfs:/user/bm106/pub/MSD/cf_train_new.parquet')
	train_data_sample = train_data.sample(0.1, seed=125)
	
	# Pre-processing dataframe according to ALS requirements
	train_df = data_prep(spark, train_data_sample)

	# Load test parquet file into a dataframe
	test_data = spark.read.parquet('hdfs:/user/bm106/pub/MSD/cf_test.parquet')

	# Pre-processing dataframe according to ALS requirements
	test_df = data_prep(spark, test_data)

	#prepare test_df for later evaluation
	test_df = test_df.sort(["user_id_index", "count"], ascending=[1,0]) #ascending user_id_index, descending counts per user
	#creates DF where each row is a user; first col is a user_id and second col is an an array of track_ids in order of descending play counts
	test_df_agg = test_df.groupby("user_id_index").agg(F.collect_list("track_id_index").alias("sorted_ground_truth_track_Ids"))
	
	users = test_df_agg.select(test_df_agg.user_id_index).distinct()
	
	#initialize parameters
	maxIter= 5
	k = 500
	rank=8
	regParam=0.01

	# get ALS model
	als = ALS(userCol='user_id_index', itemCol='track_id_index', ratingCol='count', maxIter=maxIter, regParam=regParam, rank=rank, implicitPrefs=True, coldStartStrategy='drop')
	# train ALS model
	print("Fitting ALS model with rank={} and regularization parameter={}".format(rank, regParam))
	model = als.fit(train_df)

# 	# evaluate the model by computing the RMSE on the TEST data
# 	predictions = model.transform(test_df)
# 	evaluator = RegressionEvaluator(metricName="rmse", labelCol="count", predictionCol="prediction")
# 	rmse = evaluator.evaluate(predictions)
# 	print("RMSE on full test set with {} latent factors and regularization = {} is {}".format(rank, regParam, rmse))

	#get predictions
	recs = model.recommendForUserSubset(users, 500) #returns DF with first column user ID, second column array of Row(track_id_index, "rating")
	recs = recs.withColumn("ordered_predicted_track_Ids", recs.recommendations.track_id_index)
	recs = recs.select("user_id_index", "ordered_predicted_track_Ids")
            
        #prepare RDD for RankingMetrics
	predictionAndLabels = recs.join(test_df_agg, "user_id_index")
	predictionAndLabels = predictionAndLabels.select("ordered_predicted_track_Ids", "sorted_ground_truth_track_Ids")
	predictionAndLabels_rdd = spark.sparkContext.parallelize(predictionAndLabels)

	#evaluate
	metrics = RankingMetrics(predictionAndLabels_rdd)

	map = metrics.meanAveragePrecision
	print("Mean Average Precision of full test set with {} latent factors and regularization = {} is {}").format(rank, regParam, map)
	prec_at_k = metrics.precisionAt(k)
	print("Precision at {} of full test set with {} latent factors and regularization = {} is {}").format(k, rank, regParam, prec_at_k)
	ndcg_at_k = metrics.ndcgAt(k)
	print("NDCG at k = {} of full test set with {} latent factors and regularization = {} is {}").format(k, rank, regParam, ndcg_at_k)

	
if __name__ == "__main__":

	# Creating the Spark session object
	# Disabled the blacklist in the spark session config
	spark = SparkSession.builder.appName('hyperparam_tuning').config('spark.blacklist.enabled', False).getOrCreate()

	main(spark)
