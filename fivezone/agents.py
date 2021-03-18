import numpy as np
from datetime import datetime, timedelta

class BaseExtAgent():

	def __init__(self, lb: np.array, ub: np.array):
		self.flat_shape = ub.shape[0]
		self.lb = lb
		self.ub = ub

	def predict(self, _):
		if self.flat_shape==0:
			return np.array([])
		else:
			return np.random.uniform(self.lb, self.ub, self.flat_shape)

	def train(self,*args,**kwargs):
		raise NotImplementedError



class BaseInternalAgent():
	"""
	Use this agent as the base class for selecting actions that are not chosen by the user of the
	testbed. For example if the testbed needs actions for temperature and valve. but user only decides to choose temeprature actions, this class should be inherited to build an internal agent for the testbed that should take care the valve actions according to some default rule
	based on domain knowledge
	"""

	def __init__(self, *args,**kwargs):
		raise NotImplementedError

	def default_rules(self, *args,**kwargs):
		raise NotImplementedError

	def predict(self, *args,**kwargs):
		raise NotImplementedError



class PerformanceMetrics():
	"""
	Store the history of performance metrics. Useful for evaluating the
	agent's performance:
	"""

	def __init__(self):
		self.metriclist = []  # store multiple performance metrics for multiple episodes
		self.metric = {}  # store performance metric for each episode

	def on_episode_begin(self):
		self.metric = {}  # flush existing metric data from previous episode

	def on_episode_end(self):
		self.metriclist.append(self.metric)

	def on_step_end(self, info = {}):
		for key, value in info.items():
			if key in self.metric:
				self.metric[key].append(value)
			else:
				self.metric[key] = [value]



class InternalAgent_1(BaseInternalAgent):
	"""
	An internal agent for the testbed_v2 to take care of actions not specified by the user. 
	Based on **occupancy condition**, it will choose appropriate temperature values for the rooms
	 and AHUs based on some default guidelines from the lbnl page. Any other rule can be implemented
	 in the default_rules section
	"""
	def __init__(self, action_idx_by_user, num_axns):

		self._num_actions = num_axns
		# calcualte complementary ids
		self.internal_agent_action_idx = [idx for idx in range(self._num_actions) \
			if idx not in action_idx_by_user] # is already sorted
		self.default_rules()

	def default_rules(self,):
		"""
		Implement the default or any other rules for the internal agent here
		"""
		# Here default values are written in order of 
		# 'TSupSetHea','CorTRooSetCoo','CorTRooSetHea','NorTRooSetCoo','NorTRooSetHea',
		# 'SouTRooSetCoo','SouTRooSetHea','EasTRooSetCoo','EasTRooSetHea','WesTRooSetCoo',
		# 'WesTRooSetHea' following the order in the config file
		self.default_action_vals_occupied = np.array([285.0, 297.15, 293.15,297.15, 293.15,
			297.15, 293.15, 297.15, 293.15,297.15, 293.15])
		self.default_action_vals_unoccupied = np.array([285.0, 303.15, 285.15,303.15, 285.15,
			303.15, 285.15, 303.15, 285.15, 303.15, 285.15])
		self.non_default_action_vals_occupied = np.array([285.0, 292.15, 291.15,300.15, 298.15,
			297.15, 295.15, 288.15, 286.15, 297.15, 296.15]) # ~ 18C,25C,23C,15C,24C
			# TODO: in Celsius summer -> all zones b/w 18C and 25C; add time/oat to make the setpoint
			# TODO: in Celsius winter->  all zones b/w 22C and 27C
		self.non_default_action_vals_unoccupied = np.array([285.0, 303.15, 285.15,303.15, 285.15,
			303.15, 285.15, 303.15, 285.15, 303.15, 285.15]) # ~ Normal Range
			# TODO: in Celsius summer -> all zones b/w 18C and 25C -- go up and down by 5C or don't penalize that much
			# TODO: in Celsius winter->  all zones b/w 22C and 27C -- go up and down by 5C or don't penalize that much
		self.get_suitable_init_vals()

	def get_suitable_init_vals(self,):
		self.unocc_mid_vals = [296.15,301.15,298.15,293.15,300.15]
		

	def get_internal_agent_action_idx(self,):
		return self.internal_agent_action_idx
		
	def predict(self, zone_occupancy_status, time):

		# convert seconds from beginning of a base date to a datetime.datetime type
		date = datetime(2020, 1, 1) + timedelta(seconds=time)
		_, weekofyear, _ = date.isocalendar()

		zone_occ = np.array([[i, j] for i, j in zip(zone_occupancy_status, 
													zone_occupancy_status)]).ravel()
		zone_occ = np.concatenate((np.array([True]),zone_occ))  # add true for ahu setpoint
		
		if (weekofyear < 15) | (weekofyear > 35) :
			self.action_vals = self.default_action_vals_occupied*zone_occ \
									+ self.default_action_vals_unoccupied*~zone_occ
		else:
			self.action_vals = self.non_default_action_vals_occupied*zone_occ \
									+ self.non_default_action_vals_unoccupied*~zone_occ

		return self.action_vals[self.internal_agent_action_idx]