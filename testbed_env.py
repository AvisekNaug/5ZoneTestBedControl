"""
This script will act as an interface between custom 'Buildings.Examples VAVReheat.*' FMU for the 5 Zone System and a reinforcement learning agent. It will implement the standard methods needed for the gym.env class.
"""

# imports
import gym
from pyfmi import load_fmu
from pyfmi.fmi import FMUModelCS2, FMUModelCS1  # pylint: disable=no-name-in-module

class testbed(gym.Env):
	"""
	The class which will interface between custom 'Buildings.Examples VAVReheat.*' FMU for the 5 Zone System and a reinforcement learning agent
	"""
	def __init__(self, *args, **kwargs):

		# path to fmu
		fmu_path : str = kwargs['fmu_path']
		# load the fmu
		self.fmu_model : FMUModelCS2 = load_fmu(fmu_path, kind='CS')

	def reset(self, *args, **kwargs):
		
		raise NotImplementedError

	def step(self, action):
		
		# process the action if needed
		action = self.action_processor(action)
		
	def action_processor(self, a):

		raise NotImplementedError