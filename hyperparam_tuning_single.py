'''
Usage:
    $ spark-submit --driver-memory=4g --executor-memory=6g --executor-cores=50 hyperparam_tuning_single.py <rank> <reg>
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

def main(spark, rank, reg):
	# Load training parquet file into a dataframe
	train_data = spark.read.parquet('hdfs:/user/bm106/pub/MSD/cf_train_new.parquet')

	# Sampling percentage of the train-data with a seed value to regenerate the same sample set
	train_data_sample =  train_data.sample(0.1, seed = 125)

    	# Pre-processing dataframe according to ALS requirements
	train_df = data_prep(spark, train_data_sample)

    	# Load validation parquet file into a dataframe
	val_data = spark.read.parquet('hdfs:/user/bm106/pub/MSD/cf_validation.parquet')
	#val_data_sample = val_data.sample(0.01, seed = 125)
	
    	# Pre-processing dataframe according to ALS requirements
	val_df = data_prep(spark, val_data)
	
	#prepare val_df for later evaluation
	val_df = val_df.sort(["user_id_index", "count"], ascending=[1,0]) #ascending user_id_index, descending counts per user
	#creates DF where each row is a user; first col is a user_id and second col is an an array of track_ids in order of descending play counts
	val_df_agg = val_df.groupby("user_id_index").agg(F.collect_list("track_id_index").alias("sorted_ground_truth_track_Ids"))
	
	users = val_df_agg.select(val_df_agg.user_id_index).distinct()
	
    # initializing hyperparameters over which to tune
    # ranks = [10, 50, 100, 150]
    # regParams = [0.01, 0.1, 1.0, 1.5]
	maxIter=5
	k = 500

    # start hyperparameter tuning
    # for rank in ranks:
    #     for reg in regParams:
    	# get ALS model
	als = ALS(userCol='user_id_index', itemCol='track_id_index', ratingCol='count', maxIter=maxIter, regParam=float(reg), rank=int(rank), implicitPrefs=True, coldStartStrategy='drop')
	# train ALS model
	print("Fitting ALS model with rank={} and regularization parameter={}".format(rank, reg))
	model = als.fit(train_df)

# 	# evaluate the model by computing the RMSE on the validation data
# 	predictions = model.transform(val_df)
# 	print("printing predictions schema")
# 	predictions.printSchema()
# # 	evaluator = RegressionEvaluator(metricName="rmse", labelCol="count", predictionCol="prediction")
# # 	rmse = evaluator.evaluate(predictions)
# # 	#rmse_storage.append(rmse)
# # 	print("RMSE on validation set with {} latent factors and regularization = {} is {}".format(rank, reg, rmse))

	#get predictions
	recs = model.recommendForUserSubset(users, 500) #returns DF with first column user ID, second column array of Row(track_id_index, "rating")
	recs = recs.withColumn("ordered_predicted_track_Ids", recs.recommendations.track_id_index)
	recs = recs.select("user_id_index", "ordered_predicted_track_Ids")
            
        #prepare RDD for RankingMetrics
	predictionAndLabels = recs.join(val_df_agg, "user_id_index")
	predictionAndLabels = predictionAndLabels.select("ordered_predicted_track_Ids", "sorted_ground_truth_track_Ids")
	predictionAndLabels_rdd = spark.sparkContext.parallelize(predictionAndLabels.rdd)	
	#evaluate
	metrics = RankingMetrics(predictionAndLabels_rdd)
		#each row of predictionAndLabels represents a single user
		#tuple of 2 arrays
		#first array =. ordered predicted track_IDs
		#second array = array of ordered ground track_IDs
		#arrays can be different lengths
	
	map = metrics.meanAveragePrecision
	print("Mean average precision of validation set with {} latent factors and regularization = {} is {}".format(rank, reg, map))
	prec_at_k = metrics.precisionAt(k)
	print("Precision at {} of validation set with {} latent factors and regularization = {} is {}".format(k, rank, reg, prec_at_k))
	ndcg_at_k = metrics.ndcgAt(k)
	print("NDCG at {} of validation set with {} latent factors and regularization = {} is {}".format(k, rank, reg, ndcg_at_k))

if __name__ == "__main__":

	# Creating the Spark session object
	# Disabled the blacklist in the spark session config
	spark = SparkSession.builder.appName('hyperparam_tuning').config('spark.blacklist.enabled', False).getOrCreate()

	rank=sys.argv[1]
	reg=sys.argv[2]

	main(spark, rank, reg)
