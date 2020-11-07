"""
This script contains utility functions for testbed related processing
"""

"""
This script contains all the data processing activities that are needed
before it can be provided to any other object
"""
import os
from typing import Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

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



# get zone occupancy pattern based on time
def simulate_zone_occupancy_default(step_time):
	# Should return two lists
	"""
	Considers the current time point and generates whether the next step of the simulation should have occupancy True or False for the following zones in order
	Cor, Nor, Sou, Eas, Wes
	here days of week are numbered starting from Monday=0 to Sunday=6
	here hours of day are numbered starting from 12:00:00AM=0 to 23:59:59=23
	"""
	ctime = secs2datetime(step_time)
	hour, _ = ctime.hour, ctime.weekday()

	# Set the default and extraneous rules
	if 6<hour<19:
		occ_list = [True,True,True,True,True]
		tNexOccAll = [0.0]
	else:
		occ_list = [False,False,False,False,False]
		tNexOccAll = [86400.0]

	return occ_list, tNexOccAll

def simulate_zone_occupancy(step_time):
	# Should return two lists
	"""
	Considers the current time point and generates whether the next step of the simulation should have occupancy True or False for the following zones in order
	Cor, Nor, Sou, Eas, Wes
	here days of week are numbered starting from Monday=0 to Sunday=6
	here hours of day are numbered starting from 12:00:00AM=0 to 23:59:59=23
	"""
	ctime = secs2datetime(step_time)
	hour, _ = ctime.hour, ctime.weekday()

	# Set the default and extraneous rules
	if 6<hour<19:
		occ_list = [True,True,True,True,True]
		tNexOccAll = [0.0]
	else:
		occ_list = [False,random.random() >= 0.7,False,random.random() >= 0.7,False]
		tNexOccAll = [86400.0]

	return occ_list, tNexOccAll


# convert seconds from beginning of a base date to a datetime.datetime type
def secs2datetime(x):
	"""
	converts time in seconds to datetime.datetime with a certain base_date
	"""
	base_date = datetime(2020, 1, 1)
	return base_date + timedelta(seconds=x)



# get zone internal load pattern based on time of day
def simulate_internal_load_default(step_time, zone_occupancy_status):
	"""
	It will accept the current time of the day and return the zone based internal
	gain/load. This method implements the default rules described here
	(https://obc.lbl.gov/specification/example.html#internal-loads)
	and coded(https://simulationresearch.lbl.gov/modelica/releases/latest/help/
	Buildings_Examples_VAVReheat_ThermalZones.html#Buildings.Examples.VAVReheat.ThermalZones.Floor)
	The end user would have to change this or implement their own method depending on how they
	want to simulate the internal gain load.
	"""
	ctime = secs2datetime(step_time)
	hour, _ = ctime.hour, ctime.weekday()

	# create the default case from the lbnl website
	xp = [   0,    8,   9,  12,  12,  13, 13, 17,  19,   24]
	fp = [0.05, 0.05, 0.9, 0.9, 0.8, 0.8,  1,  1, 0.1, 0.05]
	intGaiFra = [np.interp(hour, xp=xp, fp=fp)]*5

	return intGaiFra



def simulate_internal_load(step_time, zone_occupancy_status):
	"""
	It will accept the current time of the day and return the zone based internal
	gain/load. This method implements the default rules described here
	(https://obc.lbl.gov/specification/example.html#internal-loads)
	and coded(https://simulationresearch.lbl.gov/modelica/releases/latest/help/
	Buildings_Examples_VAVReheat_ThermalZones.html#Buildings.Examples.VAVReheat.ThermalZones.Floor)
	The end user would have to change this or implement their own method depending on how they
	want to simulate the internal gain load.
	"""
	ctime = secs2datetime(step_time)
	hour, _ = ctime.hour, ctime.weekday()

	# create the default case from the lbnl website
	xp = [   0,    8,   9,  12,  12,  13, 13, 17,  19,   24]
	fp = [0.05, 0.05, 0.9, 0.9, 0.8, 0.8,  1,  1, 0.1, 0.05]
	intGaiFra = [np.interp(hour, xp=xp, fp=fp)]*5
	intGaiFra_rand = []
	for i,j in zip(intGaiFra,zone_occupancy_status):
		if (i <= 0.1) & (j == True) :
			intGaiFra_rand.append(i+0.6*np.random.random())  # pylint: disable=no-member
		else:
			intGaiFra_rand.append(i)

	return intGaiFra_rand
	