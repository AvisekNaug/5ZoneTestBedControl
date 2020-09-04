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

class testbed_base(gym.Env):
	"""
	The base class which will interface between custom 'Buildings.Examples VAVReheat.*' FMU for the 5
	 Zone System and a reinforcement learning agent
	"""
	def __init__(self, *args, **kwargs):

		# observation variables
		self.obs_vars : list[str] = kwargs['observed_variables']
		# action variables
		self.act_vars : list[str] = kwargs['action_variables']
		# observation space bounds
		obs_space_bounds : list[list] = kwargs['observation_space_bounds']
		# action space bounds
		action_space_bounds : list[list] = kwargs['action_space_bounds']
		# time delta in seconds to advance simulation for every step
		self.step_size = kwargs['step_size']
		# path to fmu
		fmu_path : str = kwargs['fmu']
		# path to fmu model variables
		fmu_vars_path : str = kwargs['fmu_variables_path']

		# initialize start time and global end time
		self.re_init(**{'fmu_start_time':kwargs['fmu_start_time'], 'global_fmu_end_time' : kwargs['global_fmu_end_time']})

		# simulation time elapsed
		self.simulation_time_elapsed = 0.0

		'''methods to set up the environment'''
		# load model variable names into a json
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


	def re_init(self, *args, **kwargs):
		"""
		This method exists to change the start time for the fmu during relearning"""
		# fmu simulation start time
		self.fmu_start_time = kwargs['fmu_start_time']
		# global_fmu_end_time : time used to end the simulation for fmu: need not be actual end of the weather file
		self.global_fmu_end_time = kwargs['global_fmu_end_time']
		# determine whether the first reset needs to be called once or not
		self.global_fmu_reset = True


	def reset(self, *args, **kwargs):
		
		if self.global_fmu_reset: #don't advance simulaiton on fmu any further on actual time
			# Reset start_time to fmu_start_time
			self.start_time = self.fmu_start_time
			# reset the fmu
			self.fmu.reset()
			# initialize the fmu to t = start_time
			self.fmu.initialize(start_time = self.start_time, stop_time_defined = False)
			# set self.global_fmu_reset to False
			self.global_fmu_reset = False
		
		# simulation time elapsed
		self.simulation_time_elapsed = 0.0
		# get current value of observation variables
		self.obs : np.array = np.array([i[0] for i in self.fmu.get(variable_name=self.obs_vars)])

		# Standard requirements for interfacing with gym environments
		self.steps_beyond_done = None

		return self.obs


	def step(self, action):
		
		# process the action from the agent before sending it to the fmu
		action : np.array = self.action_processor(action)

		# perform state transition and get next state observation
		self.obs_next : np.array = self.state_transition(self.obs, action)

		# calculate reward and other info
		reward, info_dict = self.calculate_reward_and_info(self.obs, action, self.obs_next)

		# log the info_dict for our purpose
		self.log_info(info_dict)

		# advance simulation and check for episode completion
		self.start_time += self.step_size
		self.simulation_time_elapsed += self.step_size
		done = self.check_done(self.start_time, self.simulation_time_elapsed)

		# process the observation before sending it to the agent
		obs_next_processed : np.array = self.obs_processor(self.obs_next)

		return obs_next_processed, reward, done, {}


	# Process the action
	def action_processor(self, a):
		"""
		TODO: Inverse scale the action from the agent
		"""
		raise NotImplementedError


	# Calculate the observations for the next state of the system
	def state_transition(self, obs, action):

		# check input type
		if isinstance(action,np.array):
			action = list(action)
		elif isinstance(action, list):
			pass
		else:
			raise TypeError

		# set input to the action variables
		self.fmu.set(variable_name=self.act_vars, value=action)
		# simulate the system
		self.simulation_status = self.fmu.do_step(current_t = self.start_time,
								step_size = self.step_size, new_step=True)
		if self.simulation_status!=0: 
			print("Something wrong with the Simulation: {}".format(self.simulation_status))
		# get the observation variables
		obs_next : list = [i[0] for i in self.fmu.get(variable_name=self.obs_vars)]

		return np.array(obs_next)


	# calculate reward and other info
	def calculate_reward_and_info(self, obs, action, obs_next):

		raise NotImplementedError


	# log info_dict
	def log_info(self, info_dict):

		raise NotImplementedError


	# check if episode has completed
	def check_done(self, start_time, time_elapsed):

		"""
		If the number of seconds is greater than a week, terminate the episode. Also do a fmu reset
		if the next stepsize cannot be accomodated in the simulation
		"""
		if start_time>self.global_fmu_end_time-self.step_size: # won't be able to do a full step
			self.global_fmu_reset = True
		return (time_elapsed>=float(3600*24*7)) | self.global_fmu_reset


	# Process the observation
	def obs_processor(self, obs):

		raise NotImplementedError


	# public access method to load fmu
	def load_fmu(self, *args, **kwargs):

		assert 'fmu' in kwargs.keys(), 'Pass fmu path using "fmu" argument'
		if 'kind' not in kwargs.keys():
			kwargs.update({'kind':'CS'})
		else:
			assert kwargs['kind']=='CS', 'Models can only be loaded in Co-simulation model, use kind : "CS"'
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

# Example of one testbed where we control only the AHU heating coil.
class testbed_v0(testbed_base):
	"""
	Inherits the base testbed class. This version has the following characteristics

	1. Receives the delta changes in AHU set point for AHU heating coil and creates 
	the resultant heating coil set point using action_processor.

	2. 0-1 Scales the set of "observed_variables" before sending it back to the agent
	using the obs_processor method. Must include a method which knows the upper and 
	lower bounds for each of the observed variables

	3. In this version we will try to incentivize lower energy consumption and better 
	comfort.

	"""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.params = kwargs['testbed_v0_params']
		self.ahu_stpt = np.array(self.params['initial_ahu_stpt'])
		self.ahu_heat_stpt_ub = np.array(self.params['ahu_heat_stpt_ub'])
		self.ahu_heat_stpt_lb = np.array(self.params['ahu_heat_stpt_lb'])
		self.obs_vars_max : np.array = np.array(self.params['obs_vars_max'])
		self.obs_vars_min : np.array = np.array(self.params['obs_vars_min'])

	# Process the action
	def action_processor(self, a):
		"""
		Receives the positive / negative change in the heating set point. Will be used to 
		decide the actual heating set point.
		"""
		self.ahu_stpt += a
		self.ahu_stpt = np.clip(self.ahu_stpt, self.ahu_heat_stpt_lb, self.ahu_heat_stpt_ub)
		return self.ahu_stpt

	# Process the observation
	def obs_processor(self, obs : np.array):
		"""
		Scale the observations to 0-1 range 
		"""
		return (obs-self.obs_vars_min)/(self.obs_vars_max-self.obs_vars_min)


	# calculate reward and other info
	def calculate_reward_and_info(self, obs, action, obs_next):
		pass

		