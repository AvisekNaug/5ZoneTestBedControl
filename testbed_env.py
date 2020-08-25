"""
This script will act as an interface between custom 'Buildings.Examples VAVReheat.*' FMU for the 5
Zone System and a reinforcement learning agent. It will implement the standard methods needed for
the gym.env class.
"""

# imports
import numpy as np
from json import load as jsonload

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
		# path to fmu model variables
		fmu_vars_path : str = kwargs['fmu_variables_path']
		# observation variables
		self.obs_vars : list[str] = kwargs['observed_variables']
		# action variables
		self.act_vars : list[str] = kwargs['action_variables']
		# observation space bounds
		obs_space_bounds : list[list] = kwargs['observation_space_bounds']
		# observation space bounds
		obs_space_bounds : list[list] = kwargs['observation_space_bounds']
		# action space bounds
		action_space_bounds : list[list] = kwargs['action_space_bounds']


		'''methods to perform to set up the environment'''
		
		# load model variables
		self.load_fmu_vars(fmu_vars_path)
		# check membership of observation and action space
		all_vars_present, absent_vars = self.variables_in_fmu(self.obs_vars+self.act_vars)
		assert all_vars_present, "There are missing variables in the FMU:{}".format(absent_vars)

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
		# current fmu_time
		# self.fmu_time = self.fmu_model.time

	def reset(self, *args, **kwargs):
		
		# reset the fmu
		self.fmu.reset()

		if  self.first_reset:
			# initialize the fmu to t=0
			self.fmu.initialize()
			obs = self.fmu.get_fmu_state()  # TODO: how to parse necessart obs from this
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
		self.fmu : FMUModelCS2 = load_fmu(**kwargs)

	def variables_in_fmu(self, vars: list):
		"""
		Assert whether the variable names in var are part of the fmu model variables
		"""
		membership = [var in self.fmu_var_names for var in vars]
		absent_vars = [var for var , member in zip(vars, membership) if not member]
		return not all(membership), absent_vars


	def load_fmu_vars(self, fmu_vars_path):
		with open(fmu_vars_path, 'r') as f: 
			self.fmu_var_names = jsonload(f).keys()
		f.close()

