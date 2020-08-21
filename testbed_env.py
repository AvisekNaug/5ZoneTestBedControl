"""
This script will act as an interface between custom 'Buildings.Examples VAVReheat.*' FMU for the 5
Zone System and a reinforcement learning agent. It will implement the standard methods needed for
the gym.env class.
"""

# imports
import numpy as np

import gym
from gym import spaces
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
		# observation variables
		obs_vars : list[str] = kwargs['observed_variables']
		# action variables
		act_vars : list[str] = kwargs['action_variables']
		# observation space bounds
		obs_space_bounds : list[list] = kwargs['observation_space_bounds']
		# observation space bounds
		obs_space_bounds : list[list] = kwargs['observation_space_bounds']
		# action space bounds
		action_space_bounds : list[list] = kwargs['action_space_bounds']


		'''methods to perform to set up the environment'''

		# load the fmu
		self._load_fmu(**{'fmu':fmu_path, 'kind':'CS'})

		# obsservation space bounds
		assert len(obs_space_bounds)==2, "Exactly two lists are needed: low and high bounds"
		self.observation_space = spaces.Box(low = np.array(obs_space_bounds[0]),
											high = np.array(obs_space_bounds[1]),
											dtype = np.float32)

		# the action space bounds
		assert len(action_space_bounds)==2, "Exactly two lists are needed: low and high bounds"
		assert all([actlow == -1*acthigh for actlow,
		 acthigh in zip(action_space_bounds[0], action_space_bounds[1])]), "Action Space bounds have to be symmetric"
		self.action_space = spaces.Box(low = np.array(action_space_bounds[0]),
										high = np.array(action_space_bounds[1]),
										dtype = np.float32)

		# determine whether the first reset has been called once or not
		self.first_reset = True

	def reset(self, *args, **kwargs):
		
		if  self.first_reset:
			obs = self.fmu_model.get_fmu_state()  # TODO: how to parse necessart obs from this
		'''Standard requirements for interfacing with gym environments'''
		self.steps_beyond_done = None

	def step(self, action):
		
		# process the action if needed
		action = self.action_processor(action)

		# perform state transition
		obs_next = self.state_transition()
		
	# Extra processes needed for this environment
	def action_processor(self, a):

		raise NotImplementedError

	def state_transition(self, *args, **kwargs):
		
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