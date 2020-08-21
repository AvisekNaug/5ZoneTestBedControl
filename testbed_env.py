"""
This script will act as an interface between custom 'Buildings.Examples VAVReheat.*' FMU for the 5
Zone System and a reinforcement learning agent. It will implement the standard methods needed for
the gym.env class.
"""

# imports
import gym
from pyfmi import load_fmu
from pyfmi.fmi import FMUModelCS2, FMUModelCS1  # pylint: disable=no-name-in-module

class testbed(gym.Env):
	"""
	The class which will interface between custom 'Buildings.Examples VAVReheat.*' FMU for the 5
	 Zone System and a reinforcement learning agent
	"""
	def __init__(self, *args, **kwargs):

		# path to fmu
		fmu_path : str = kwargs['fmu']
		self._load_fmu(**{'fmu':fmu_path, 'kind':'CS'})

	def reset(self, *args, **kwargs):
		
		raise NotImplementedError

	def step(self, action):
		
		# process the action if needed
		action = self.action_processor(action)
		
	# Extra processes needed for this environment
	def action_processor(self, a):

		raise NotImplementedError

	def load_fmu(self, *args, **kwargs):

		assert 'fmu' in kwargs.keys(), 'Pass fmu path using "fmu" argument'
		if 'kind' not in kwargs.keys():
			kwargs.update({'kind':'CS'})
		else:
			assert kwargs['kind']=='CS', 'Models can only be loaded in Co-simulation model, use kind="CS"'
		self._load_fmu(**kwargs)

	def _load_fmu(self,*args,**kwargs):

		# load the fmu
		self.fmu_model : FMUModelCS2 = load_fmu(**kwargs)