"""
This script contains utility functions for alumni hall related processing
"""

"""
This script contains all the data processing activities that are needed
before it can be provided to any other object
"""
import os
from typing import Union
import numpy as np
import pandas as pd


# sources of data for which processing can be done
DATASOURCE = ['BdX']
DATATYPE = ['.csv', '.xlsx', '.pkl']

class dataframescaler():
	"""
	An attempt at creating a class that can scale or inverse scale any variable
	individually or collectively of a dataframe once, the original raw and cleaned values
	are given
	"""
	def __init__(self, df: Union[str, pd.DataFrame, dict]):
		
		if isinstance(df, str):
			self.stats = pd.read_pickle(df)
			self.columns = self.stats.columns
		elif isinstance(df, pd.DataFrame):
			self.df = df  # get the dataframe
			self.stats = df.describe()  # collect the statistics of a dataframe
			self.columns = df.columns  # save column names
		elif isinstance(df, dict):
			df = pd.DataFrame(df)
			self.stats = df  # get the statistics from dictionary
			self.columns = df.columns  # save column names

	def minmax_scale(self,
					 input_data : Union[pd.DataFrame, np.ndarray], 
					 input_columns: list, df_2scale_columns: list):

		x_min = self.stats.loc['min',input_columns]
		x_max = self.stats.loc['max',input_columns]

		if isinstance(input_data,pd.DataFrame):
			return (input_data[df_2scale_columns].to_numpy()-x_min.to_numpy())/(x_max.to_numpy()-x_min.to_numpy())
		else:
			return (input_data-x_min.to_numpy())/(x_max.to_numpy()-x_min.to_numpy())

	def minmax_inverse_scale(self,
					 input_data : Union[pd.DataFrame, np.ndarray], 
					 input_columns: list, df_2scale_columns: list):

		x_min = self.stats.loc['min',input_columns]
		x_max = self.stats.loc['max',input_columns]

		if isinstance(input_data,pd.DataFrame):
			return input_data[df_2scale_columns].to_numpy()*(x_max.to_numpy()-x_min.to_numpy()) + x_min.to_numpy()
		else:
			return input_data*(x_max.to_numpy()-x_min.to_numpy()) + x_min.to_numpy()
		
	def standard_scale(self, 
						input_data : Union[pd.DataFrame, np.ndarray], 
						input_columns: list, df_2scale_columns: list):
		
		x_mean= self.stats.loc['mean',input_columns]
		x_std = self.stats.loc['std',input_columns]

		if isinstance(input_data,pd.DataFrame):
			return (input_data[df_2scale_columns].to_numpy()-x_mean.to_numpy())/x_std.to_numpy()
		else:
			return (input_data-x_mean.to_numpy())/x_std.to_numpy()

	def standard_inverse_scale(self, 
						input_data : Union[pd.DataFrame, np.ndarray], 
						input_columns: list, df_2scale_columns: list):
		
		x_mean= self.stats.loc['mean',input_columns]
		x_std = self.stats.loc['std',input_columns]

		if isinstance(input_data,pd.DataFrame):
			return input_data[df_2scale_columns].to_numpy()*x_std.to_numpy() + x_mean.to_numpy()
		else:
			return input_data*x_std.to_numpy() + x_mean.to_numpy()

	def save_stats(self, path: str):

		if not path.endswith('/'):
			path += '/'

		self.stats.to_pickle(path + 'datastats.pkl')


# save test performance of the rl agent
def rl_perf_save(test_perf_log_list: list, log_dir: str, save_as: str = 'csv', header = True):

	# assert that perf metric has data from at least one episode
	assert all([len(i.metriclist) != 0 for i in test_perf_log_list]), 'Need metric data for at least one episode'

	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	# iterate throguh each environment in a single Trial
	for idx, test_perf_log in enumerate(test_perf_log_list):
	
		# performance metriclist is a list where each element has 
		# performance data for each episode in a dict
		perf_metric_list = test_perf_log.metriclist

		# iterating through the list to save the data
		for episode_dict in perf_metric_list:

			if save_as == 'csv':
				df = pd.DataFrame(data=episode_dict)
				df.to_csv(log_dir+'/results.csv', index=False, mode='a+', header=header)
				header = False
			else:
				for key, value in episode_dict.items():
					f = open(log_dir + '/EnvId{}-'.format(idx) + key + '.txt', 'a+')
					f.writelines("%s\n" % j for j in value)
					f.close()
				header = False
