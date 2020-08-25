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
		# fmu simulation start time
		self.fmu_start_time = kwargs['fmu_start_time']
		# time delta to advance simulation
		self.simulation_time_delta_s = kwargs['simulation_time_delta_s']

		# simulation time elapsed
		self.simulation_time_elapsed = 0.0


		'''methods to perform to set up the environment'''
		# load model variables into a json
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
		self.global_fmu_reset = True

	def reset(self, *args, **kwargs):
		
		# new fmu_start_time
		self.fmu_start_time += self.simulation_time_elapsed
		# simulation time elapsed
		self.simulation_time_elapsed = 0.0

		# reset the fmu
		self.fmu.reset()
		# initialize the fmu to t = fmu_start_time
		self.fmu.initialize(start_time = self.fmu_start_time, stop_time_defined = False)
		# do not reinitialize simulation from now on
		self.opts = self.fmu.simulate_options()
		self.opts['initialize'] = False
		# get current value of observation variables
		self.obs : list = self.fmu.get(self.obs_vars)
		# Standard requirements for interfacing with gym environments
		self.steps_beyond_done = None

		return np.array(self.obs)

	def step(self, action):
		
		# process the action from the agent before sending it to the fmu
		action = self.action_processor(action)

		# perform state transition and get next state observation
		self.obs_next = self.state_transition(self.obs, action)

		# calculate reward and other info
		reward, info_dict = self.calculate_reward_and_info(self.obs, action, self.obs_next)

		# log the info_dict for our purpose
		self.log_info(info_dict)

		# advance simulation and check for episode completion
		self.simulation_time_elapsed = self.fmu_start_time+self.simulation_time_delta_s
		done = self.check_done(self.simulation_time_elapsed)

		# process the observation before sending it to the agent
		obs_next_processed : np.array = self.obs_processor(self.obs_next)

		return obs_next_processed, reward, done, {}

	# Calculate the observations for the next state of the system
	def state_transition(self, obs, action):

		raise NotImplementedError

	# Process the observation
	def obs_processor(self, obs):

		raise NotImplementedError
		
	# Process the action
	def action_processor(self, a):

		raise NotImplementedError

	# check if episode has completed
	def check_done(self, time_elapsed):

		raise NotImplementedError

	# calculate reward and other info
	def calculate_reward_and_info(self, obs, action, obs_next):

		raise NotImplementedError

	# public access method to load fmu
	def load_fmu(self, *args, **kwargs):

		assert 'fmu' in kwargs.keys(), 'Pass fmu path using "fmu" argument'
		if 'kind' not in kwargs.keys():
			kwargs.update({'kind':'CS'})
		else:
			assert kwargs['kind']=='CS', 'Models can only be loaded in Co-simulation model, use kind="CS"'
		self._load_fmu(**kwargs)

	# internal method to load fmu
	def _load_fmu(self,*args,**kwargs):

		# load the fmu
		self.fmu : FMUModelCS2 = load_fmu(**kwargs)

	# check whether all the variables are in an fmu model variable dicitonary keys
	def variables_in_fmu(self, vars: list):
		"""
		Assert whether the variable names in var are part of the fmu model variables
		"""
		membership = [var in self.fmu_var_names for var in vars]
		absent_vars = [var for var,member in zip(vars, membership) if not member]
		return not all(membership), absent_vars


	# load the fmu model variable dicitonary keys
	def load_fmu_vars(self, fmu_vars_path):
		with open(fmu_vars_path, 'r') as f: 
			self.fmu_var_names = jsonload(f).keys()
		f.close()

	# log info_dict
	def log_info(self, info_dict):

		raise NotImplementedError

