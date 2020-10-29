import numpy as np

class RandomAgent():

	def __init__(self, lb: np.array, ub: np.array):
		self.flat_shape = ub.shape[0]
		self.lb = lb
		self.ub = ub

	def predict(self, _):
		return np.random.uniform(self.lb, self.ub, self.flat_shape)

	def train(self,*args,**kwargs):
		raise NotImplementedError

class InternalAgent():
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

class InternalAgent_testbed_v1(InternalAgent):
	"""
	An internal agent for the testbed_v1 to take care of actions not specified by the user. 
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
		self.default_action_vals_occupied = np.array([285.0, 297.15, 293.15,297.15, 293.15,
			297.15, 293.15, 297.15, 293.15,297.15, 293.15])
		self.default_action_vals_unoccupied = np.array([285.0, 303.15, 285.15,303.15, 285.15,
			303.15, 285.15, 303.15, 285.15, 303.15, 285.15])

	def get_internal_agent_action_idx(self,):
		return self.internal_agent_action_idx
		
	def predict(self, occupancy_status):
		if occupancy_status:
			return self.default_action_vals_occupied[self.internal_agent_action_idx]
		else:
			return self.default_action_vals_unoccupied[self.internal_agent_action_idx]

class InternalAgent_testbed_v2(InternalAgent):
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
		
		

	def get_internal_agent_action_idx(self,):
		return self.internal_agent_action_idx
		
	def predict(self, zone_occupancy_status):

		zone_occ = np.array([[i, j] for i, j in zip(zone_occupancy_status, 
													zone_occupancy_status)]).ravel()
		zone_occ = np.concatenate((np.array([True]),zone_occ))
		self.default_action_vals = self.default_action_vals_occupied*zone_occ \
									+ self.default_action_vals_unoccupied*~zone_occ

		return self.default_action_vals[self.internal_agent_action_idx]
		

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