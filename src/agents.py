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