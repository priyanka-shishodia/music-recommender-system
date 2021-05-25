'''
Usage:
    $ spark-submit --driver-memory=4g --executor-memory=6g --executor-cores=50 RMSE.py <rank> <reg>
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
	print("First 5 rows of train_df")
	train_df.show(5)

    	# Load validation parquet file into a dataframe
	val_data = spark.read.parquet('hdfs:/user/bm106/pub/MSD/cf_validation.parquet')
	#val_data_sample = val_data.sample(0.1, seed = 125)

    	# Pre-processing dataframe according to ALS requirements
	val_df = data_prep(spark, val_data)
	print("First 5 rows of val_df")
	val_df.show(5)

	maxIter= 5
	k = 500

    	# get ALS model
	als = ALS(userCol='user_id_index', itemCol='track_id_index', ratingCol='count', maxIter=maxIter, regParam=float(reg), rank=int(rank), implicitPrefs=True, coldStartStrategy='drop')
	# train ALS model
	print("Fitting ALS model with rank={} and regularization parameter={}".format(rank, reg))
	model = als.fit(train_df)

	# evaluate the model by computing the RMSE on the validation data
	predictions = model.transform(val_df)
	evaluator = RegressionEvaluator(metricName="rmse", labelCol="count", predictionCol="prediction")
	rmse = evaluator.evaluate(predictions)
	#rmse_storage.append(rmse)
	print("RMSE on validation set with {} latent factors and regularization = {} is {}".format(rank, reg, rmse))


if __name__ == "__main__":

	# Creating the Spark session object
	# Disabled the blacklist in the spark session config
	spark = SparkSession.builder.appName('hyperparam_tuning').config('spark.blacklist.enabled', False).getOrCreate()

	rank=sys.argv[1]
	reg=sys.argv[2]

	main(spark, rank, reg)
