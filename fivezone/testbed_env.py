"""
This script will act as an interface between custom 'Buildings.Examples VAVReheat.*' FMU for the 5
Zone System and a reinforcement learning agent. It will implement the standard methods needed for
the gym.env class.
"""

# imports
import numpy as np
from json import load as jsonload
from typing import List

import gym
from gym import spaces
from pyfmi import load_fmu
from pyfmi.fmi import FMUModelCS2, FMUModelCS1  # pylint: disable=no-name-in-module

from agents import InternalAgent_testbed_v1
from testbed_utils import dataframescaler

# base testbed class
class testbed_base(gym.Env):
	"""
	The base class which will interface between custom 'Buildings.Examples VAVReheat.*' FMU for the 5
	 Zone System and a reinforcement learning agent
	"""
	def __init__(self, *args, **kwargs):

		# user observation variables
		self.user_observations : List[str] = kwargs['user_observations']
		# user action variables
		self.user_actions : List[str] = kwargs['user_actions']
		# observation variables
		self.obs_vars : List[str] = kwargs['observed_variables']
		# action variables
		self.act_vars : List[str] = kwargs['action_variables']
		# observation space bounds
		self.obs_space_bounds : List[List] = kwargs['observation_space_bounds']
		# action space bounds
		self.action_space_bounds : List[List] = kwargs['action_space_bounds']

		self.create_gym_obs_axn_space(self.user_observations, self.user_actions,
									self.obs_vars, self.act_vars,
									self.obs_space_bounds, self.action_space_bounds)


		# time delta in seconds to advance simulation for every step
		self.step_size = kwargs['step_size']
		# number of days in one episode
		self.episode_days = kwargs['episode_days']
		# path to fmu
		fmu_path : str = kwargs['fmu']
		# path to fmu model variables
		fmu_vars_path : str = kwargs['fmu_vars_path']

		# initialize start time and global end time
		self.re_init(**{'fmu_start_time_step':kwargs['fmu_start_time_step'],
						 'global_fmu_end_time_step' : kwargs['global_fmu_end_time_step']})

		# simulation time elapsed
		self.simulation_time_elapsed = 0.0

		'''methods to set up the environment'''
		# load model variable names into a json
		self.load_fmu_vars(fmu_vars_path)
		# check membership of observation and action space
		# all_vars_present, absent_vars = self.variables_in_fmu(self.obs_vars+self.act_vars)
		# assert all_vars_present, "There are missing variables in the FMU:{}".format(absent_vars)
		# load the fmu
		self._load_fmu(**{'fmu':fmu_path, 'kind':'CS'})
		
	def create_gym_obs_axn_space(self, usr_obs, usr_axn, obs, axn, obs_bounds, axn_bounds):

		self.usr_axn_idx_unsrtd = [axn.index(name) for name in usr_axn]
		self.usr_axn_idx = sorted(self.usr_axn_idx_unsrtd)
		self.usr_obs_idx_unsrtd = [obs.index(name) for name in usr_obs]
		self.usr_obs_idx = sorted(self.usr_obs_idx_unsrtd)

		#  observation space bounds
		if len(self.usr_obs_idx)!=0:  # not empty
			self.usr_obs_space_bounds = [[obs_bounds[0][idx] for idx in self.usr_obs_idx],
									 	 [obs_bounds[1][idx] for idx in self.usr_obs_idx]]
			assert len(self.usr_obs_space_bounds)==2, "Exactly two lists are needed: low and high bounds"
			self.observation_space = spaces.Box(low = np.array(self.usr_obs_space_bounds[0]),
												high = np.array(self.usr_obs_space_bounds[1]),
												dtype = np.float32)
			self.no_usr_obs = False
		else:
			self.no_usr_obs = True

		# action space bounds
		if len(self.usr_axn_idx)!=0:  # not empty
			self.usr_axn_space_bounds = [[axn_bounds[0][idx] for idx in self.usr_axn_idx],
									     [axn_bounds[1][idx] for idx in self.usr_axn_idx]]

			assert len(self.usr_axn_space_bounds)==2, "Exactly two lists are needed: low and high bounds"
			assert all([actlow == -1*acthigh for actlow,
					acthigh in zip(self.usr_axn_space_bounds[0],
					 self.usr_axn_space_bounds[1])]), "Action Space bounds have to be symmetric"
			self.action_space = spaces.Box(low = np.array(self.usr_axn_space_bounds[0]),
										high = np.array(self.usr_axn_space_bounds[1]),
										dtype = np.float32)
			self.no_usr_action = False
		else:
			self.no_usr_action = True

	def re_init(self, *args, **kwargs):
		"""
		This method exists to change the start time for the fmu during relearning"""
		# fmu simulation start time
		self.fmu_start_time = 3600.0*kwargs['fmu_start_time_step']
		# global_fmu_end_time : time used to end the simulation for fmu: need not be actual end of the weather file
		self.global_fmu_end_time = 3600.0*kwargs['global_fmu_end_time_step']
		# determine whether the first reset needs to be called once or not
		self.global_fmu_reset = True


	def reset(self,):
		
		if self.global_fmu_reset: #don't advance simulaiton on fmu any further on actual time
			# Reset start_time to fmu_start_time
			self.start_time = self.fmu_start_time
			# reset the fmu
			self.fmu.reset()
			# initialize the fmu to t = start_time
			self.fmu.initialize(start_time = self.start_time, stop_time = self.global_fmu_end_time)
			# set self.global_fmu_reset to False
			self.global_fmu_reset = False
		
		# simulation time elapsed
		self.simulation_time_elapsed = 0.0

		# get current value of observation variables
		self.obs : np.array = np.array([i[0] for i in self.fmu.get(self.obs_vars)])
		# Standard requirements for interfacing with gym environments
		self.steps_beyond_done = None

		if self.no_usr_obs:
			return None
		else:
			return self._obs_processor(np.copy(self.obs))


	def step(self, action=None):
		
		# process the action from the agent before sending it to the fmu
		action : np.array = self._action_processor(action)

		# perform state transition and get next state observation
		self.obs_next : np.array = self.state_transition(self.obs, action)

		# calculate reward and other info
		reward, info_dict = self.calculate_reward_and_info(self.obs, action, self.obs_next)

		# log the info_dict for our purpose
		# self.log_info(info_dict)

		# process the observation before sending it to the agent
		if self.no_usr_obs:
			obs_next_processed = None
		else:
			obs_next_processed : np.array = self._obs_processor(np.copy(self.obs_next))

		# assign obs_next to obs
		self.obs = self.obs_next

		# advance simulation time and check for episode completion
		self.start_time += self.step_size
		self.simulation_time_elapsed += self.step_size
		done = self.check_done(self.start_time, self.simulation_time_elapsed)

		return obs_next_processed, reward, done, info_dict


	def _action_processor(self,a):

		if not self.no_usr_action:
			usr_actions = a  # don't reorder anything
		else: 
			usr_actions = a[[self.usr_axn_idx_unsrtd.index(i) for i in self.usr_axn_idx]]  # sort
		return self.action_processor(usr_actions)
	
	# Process the action
	def action_processor(self, a):
		"""
		TODO: Inverse scale the action from the agent
		"""
		raise NotImplementedError


	# Calculate the observations for the next state of the system
	# this function does not need obs since it stores obs internally in fmu
	def state_transition(self, _, action):

		# check input type
		if isinstance(action,np.ndarray):
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
		obs_next : list = [i[0] for i in self.fmu.get(self.obs_vars)]

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
		return (time_elapsed>=float(3600*24*self.episode_days)) | self.global_fmu_reset


	def _obs_processor(self, obs):
		usr_obs = obs[self.usr_obs_idx_unsrtd]  # chose the obs user wants in their prefererd order
		return self.obs_processor(usr_obs)
	
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

		return all(membership), absent_vars


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
		# reset the fmu to appropriate time point to get correct values; no need to set global_fmu_reset
		_ = self.reset()
		self.stpt_vals = np.array(kwargs['initial_stpt_vals'])
		self.stpt_vals_ub = np.array(kwargs['stpt_vals_ub'])
		self.stpt_vals_lb = np.array(kwargs['stpt_vals_lb'])
		self.scaler : dataframescaler = kwargs['scaler']
		self.r_energy_wt, self.r_comfort_wt = kwargs['r_energy_wt'], kwargs['r_comfort_wt']
		# energy variables used to calculate reward
		self.power_variables = ['res.PFan', 'res.PHea', 'res.PCooSen', 'res.PCooLat']
		# * Here 'res.PCooSen','res.PCooLat' will be negative so we have to negate the values *
		self.power_sign = [1.0, 1.0, -1.0, -1.0]
		# zone temperature var names
		self.zone_temp_vars : List[str] = ['TSupCor.T','TSupEas.T','TSupWes.T','TSupNor.T','TSupSou.T']
		# zone temperature cooling and heating bounds
		self.zone_temp_cool = ['conVAVCor.TRooCooSet','conVAVEas.TRooCooSet','conVAVWes.TRooCooSet',
								'conVAVNor.TRooCooSet','conVAVSou.TRooCooSet']
		self.zone_temp_heat = ['conVAVCor.TRooHeaSet','conVAVEas.TRooHeaSet','conVAVWes.TRooHeaSet',
								'conVAVNor.TRooHeaSet','conVAVSou.TRooHeaSet']
		# temp ub
		self.temp_ub : np.array = np.array([i[0] for i in self.fmu.get(self.zone_temp_cool)])
		# temp lb
		self.temp_lb : np.array = np.array([i[0] for i in self.fmu.get(self.zone_temp_heat)])

	# Process the action
	def action_processor(self, a):
		"""
		Receives the positive / negative change in the heating set point. Will be used to 
		decide the actual heating set point.
		"""
		self.stpt_vals += a
		self.stpt_vals = np.clip(self.stpt_vals, self.stpt_vals_lb, self.stpt_vals_ub)
		return self.stpt_vals

	# Process the observation
	def obs_processor(self, obs : np.array):
		"""
		Scale the observations to 0-1 range 
		"""
		return self.scaler.minmax_scale(obs, input_columns= self.user_observations,
											 df_2scale_columns= self.user_observations)


	# calculate reward and other info
	def calculate_reward_and_info(self, obs, action, obs_next):
		"""
		Here we will get the power consumsed over the last 1 hour- initially use only this
		Also use the deviation between the room temperature and set point during day time
		"""
		
		'''energy component of the reward '''
		power_values = np.array([i[0]*j for i,j in zip(self.fmu.get(self.power_variables),
														self.power_sign)])
		# scale the power values
		power_values = self.scaler.minmax_scale(power_values, input_columns= self.power_variables,
											 df_2scale_columns= self.power_variables)
		# reward incentivizes lower energy; reward lower energy
		r_energy = -1.0*np.sum(power_values)

		'''comfort component of the reward: we only penalize deviation when there is occupancy'''
		# see if zone is occupied
		occupancy_status : int = int(self.fmu.get('occSch.occupied')[0])
		# zone temperatures
		zone_temp : np.array = np.array([i[0] for i in self.fmu.get(self.zone_temp_vars)])
		# check whether zone_temp is within the range per zone
		temp_within_range : np.array = (zone_temp>self.temp_lb) & (zone_temp<self.temp_ub)
		r_comfort = occupancy_status*np.sum(temp_within_range)

		reward = self.r_energy_wt*r_energy +  self.r_comfort_wt*r_comfort

		info = {}
		info['reward_energy'] = r_energy
		info['reward_comfort'] = r_comfort
		info['action'] = action[0]
		for name,val in zip(self.power_variables, power_values):
			info[name] = val
		for name,val in zip(self.zone_temp_vars, zone_temp):
			info[name] = val
		for name, val in zip(self.user_observations, obs):
			info[name] = val
		
		return reward, info



# Example testbed where we control AHU heating coil and terminal heating coil set points.
class testbed_v1(testbed_base):
	"""
	Inherits the base testbed base class. This version has the following characteristics

	1. Receives the delta changes in temperature set points for AHU heating coils as well as the  temperature set points for cooling and heating in the terminal zones. It then creates the resultant temperature set points for the corresponding cases using action_processor. The terminal VAVs have dead band control

	2. 0-1 Scales the set of "observed_variables" before sending it back to the agent
	using the obs_processor method. Must include an object which knows the upper and 
	lower bounds for each of the observed variables

	3. In this version we will try to incentivize lower energy consumption.
	"""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self._num_actions = len(kwargs['initial_stpt_vals'])
		self.stpt_vals = np.array(kwargs['initial_stpt_vals'])
		self.stpt_vals_ub = np.array(kwargs['stpt_vals_ub'])
		self.stpt_vals_lb = np.array(kwargs['stpt_vals_lb'])
		# energy variables used to calculate reward
		self.power_variables = ['res.PFan', 'res.PHea', 'res.PCooSen', 'res.PCooLat']
		self.energy_lb = kwargs['energy_lb']
		self.energy_ub = kwargs['energy_ub']
		# * Here 'res.PCooSen','res.PCooLat' will be negative so we have to negate the values *
		self.power_sign = [1.0, 1.0, -1.0, -1.0]

		# create the internal agent to handle unused actions for the testbed
		self.internal_agent_created = False
		if len(self.usr_axn_idx)<self._num_actions:  # internal agent needed
			self.internal_agent = InternalAgent_testbed_v1(self.usr_axn_idx)
			self.internal_agent_action_idx = self.internal_agent.get_internal_agent_action_idx()
			self.internal_agent_created = True
		elif len(self.usr_axn_idx)==self._num_actions:  # internal agent not needed
			pass
		else:
			raise IndexError

		# create the scaler dictionary from above information
		scaler_dict = {}
		var_names = kwargs['observed_variables'] + kwargs['action_variables'] + self.power_variables
		var_lb = kwargs['observation_space_bounds'][0] + kwargs['stpt_vals_lb'] + self.energy_lb
		var_ub = kwargs['observation_space_bounds'][1] + kwargs['stpt_vals_ub'] + self.energy_ub
		for key,var_lb,var_ub in zip(var_names, var_lb, var_ub):
			scaler_dict[key] = {'min':var_lb,'max':var_ub}
		self.scaler : dataframescaler = dataframescaler(scaler_dict)

		self.r_energy_wt, self.r_comfort_wt = kwargs['r_energy_wt'], kwargs['r_comfort_wt']
		# zone temperature var names
		self.zone_temp_vars : List[str] = ['TSupCor.T','TSupEas.T','TSupWes.T','TSupNor.T','TSupSou.T']
		# zone temperature cooling and heating bounds
		self.zone_temp_cool = ['conVAVCor.TRooCooSet','conVAVEas.TRooCooSet','conVAVWes.TRooCooSet',
								'conVAVNor.TRooCooSet','conVAVSou.TRooCooSet']
		self.zone_temp_heat = ['conVAVCor.TRooHeaSet','conVAVEas.TRooHeaSet','conVAVWes.TRooHeaSet',
								'conVAVNor.TRooHeaSet','conVAVSou.TRooHeaSet']

		# reset the fmu to appropriate time point to get correct values; no need to set global_fmu_reset
		_ = self.reset()								
	
	# Process the action
	def action_processor(self, a):
		"""
		Receives the positive / negative change in the actions supplied by the user agent and
		calculates the current values of those action idx. Rest axn idx vals are provided by
		the internal testbed agent.
		"""
		# get final action values  for user agent
		if not self.no_usr_action:
			self.stpt_vals[self.usr_axn_idx] += a
			self.stpt_vals[self.usr_axn_idx] = \
							np.clip(self.stpt_vals[self.usr_axn_idx], 
									self.stpt_vals_lb[self.usr_axn_idx], 
									self.stpt_vals_ub[self.usr_axn_idx])

		# get final action values for internal agent
		if self.internal_agent_created:
			internal_axns = self.internal_agent.predict(self.fmu.get('occSch.occupied')[0])
			self.stpt_vals[self.internal_agent_action_idx] = internal_axns

		return self.stpt_vals

	# Process the observation
	def obs_processor(self, obs : np.array):
		"""
		Scale the observations to 0-1 range 
		"""
		return self.scaler.minmax_scale(obs, input_columns= self.user_observations,
											 df_2scale_columns= self.user_observations)


	# calculate reward and other info
	def calculate_reward_and_info(self, obs, action, obs_next):
		"""
		Here we will get the power consumsed over the last 1 hour- initially use only this
		Also use the deviation between the room temperature and set point during day time
		"""
		
		'''energy component of the reward '''
		power_values = np.array([i[0]*j for i,j in zip(self.fmu.get(self.power_variables),
														self.power_sign)])
		# scale the power values
		power_values = self.scaler.minmax_scale(power_values, input_columns= self.power_variables,
											 df_2scale_columns= self.power_variables)
		# reward incentivizes lower energy; reward lower energy
		r_energy = -1.0*np.sum(power_values)

		'''comfort component of the reward: we only penalize deviation when there is occupancy'''
		# see if zone is occupied
		occupancy_status : int = int(self.fmu.get('occSch.occupied')[0])
		# zone temperatures
		zone_temp : np.array = np.array([i[0] for i in self.fmu.get(self.zone_temp_vars)])
		# check whether zone_temp is within the range per zone

		# temp ub
		self.temp_ub : np.array = np.array([i[0] for i in self.fmu.get(self.zone_temp_cool)])
		# temp lb
		self.temp_lb : np.array = np.array([i[0] for i in self.fmu.get(self.zone_temp_heat)])

		temp_within_range : np.array = (zone_temp>self.temp_lb) & (zone_temp<self.temp_ub)
		r_comfort = occupancy_status*np.sum(temp_within_range)

		reward = self.r_energy_wt*r_energy +  self.r_comfort_wt*r_comfort

		info = {}
		# add time
		info['time'] = self.fmu.time
		info['reward_energy'] = r_energy
		info['reward_comfort'] = r_comfort
		for name,val in zip(self.act_vars, action):
			info[name] = val
		for name,val in zip(self.power_variables, power_values):
			info[name] = val
		for name,val in zip(self.zone_temp_vars, zone_temp):
			info[name] = val
		for name, val in zip(self.user_observations, obs):
			info[name] = val
		
		return reward, info