'''
Usage:
    $ spark-submit --driver-memory=4g --executor-memory=4g baseline_model.py
'''

from pyspark.ml.feature import StringIndexer, IndexToString
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms.bias import Bias
from lenskit.algorithms.basic import Popular
from lenskit.metrics.predict import rmse
from lenskit.batch import predict

import pandas as pd
import matplotlib.pyplot as plt

def data_prep_convert(spark, data):
	'''
	Function to convert string IDs (user_id & track_id) to index and convert back to original

	Parameters:
	----------
	spark: Spark Session object
	data: Unprocessed DataFrame
	'''

	indexer_1 = StringIndexer(inputCol = 'user_id', outputCol = 'user_id_index')
  	indexer_2 = StringIndexer(inputCol = 'track_id', outputCol = 'track_id_index')
  	indexed = indexer_1.fit(data).transform(data)
  	transformed = indexer_2.fit(indexed).transform(indexed)

  	# Converting back to original string IDs
  	converter = IndexToString(inputCol = 'track_id_index', outputCol='track_id_original')
  	converted = converter.transform(transformed)
  
  	converted_df = converted.select(converted['track_id_index'], converted['track_id_original'])

  	df = transformed.select(transformed['user_id_index'].astype('int'), transformed['track_id_index'].astype('int'), transformed['count'])
  	return df, converted_df


def biasModel(spark, train_df, valid_df):
	'''
	Function to implement a user-item bias rating prediction algorithm using LensKit

	Parameters:
	----------
	spark: Spark Session object
	train_df, valid_df: Pre-processed DataFrames
	'''

	# Defining the bias model: s(u, i) = global_mean + item_bias + user_bias
	bias_model = Bias(items = True, users = True)
	bias_model.fit(train_df)

	# Prediction scores on Validation data
	predictions = predict(bias_model, valid_df)
	print('Prediction scores on validation dataframe')
	predictions.head(5)

	# Evaluating the predictions using rmse
	rmse_score = rmse(predictions['prediction'], predictions['rating'])
	print('RMSE score on the training sample using the Bias model is: ', rmse_score)

	# Calculating RMSE per user in the training sample taken
	rmse_user = predictions.groupby('user').apply(lambda df: rmse(df.prediction, df.rating))

	# Calculating a mean over all the per-user RMSE calculated above
	print('Mean of per-user RMSE scores:', rmse_user.mean())


def popularityModel(spark, train_df, valid_df):
	'''
	Function to implement a most-popular-item recommendation algorithm

	Parameters:
	----------
	spark: Spark Session object
	train_df, valid_df: Pre-processed DataFrames
	'''

	# Defining the model and training on the preprocessed dataframe
	popular_model = Popular()
	popular_model.fit(train_df_pd)

	# Top 10 most popular recommendations
	recommendations = popular_model.recommend(valid_df_pd.user[1], n=10)
	popular_tracks = list(recommendations.item)

	# Extracting the top recommended tracks using the track_id_index converted back to original strings
	for track in popular_tracks:
  		query = valid_reversed_df.filter(valid_reversed_df.track_id_index.isin(track)).select(valid_reversed_df.track_id_original)
  		query.show(1)


def main(spark):

	# Load training and validation parquet files into a dataframe
	train_data = spark.read.parquet('hdfs:/user/bm106/pub/MSD/cf_train_new.parquet')
	val_data = spark.read.parquet('hdfs:/user/bm106/pub/MSD/cf_validation.parquet')

	train_df, _ = data_prep_convert(spark, train_data)
	valid_df, valid_reversed_df = data_prep_convert(spark, valid_data)

	train_df_pd = train_df.toPandas()
	valid_df_pd = valid_df.toPandas()

	# Rename the columns as required by LensKit's algorithms. A ratings matrix of user, item and product rating
	mapping = ["user", "item", "rating"]
	train_df_pd.columns = mapping
	valid_df_pd.columns = mapping

	biasModel(spark, train_df_pd, valid_df_pd)
	popularityModel(spark, train_df_pd, valid_df_pd)


 if __name__ == "__main__":
 	# Creating the Spark session object. Disabled the blacklist in the spark session config
 	spark = SparkSession.builder.appName('hyperparam_tuning').config('spark.blacklist.enabled', False).getOrCreate()

 	main(spark)
