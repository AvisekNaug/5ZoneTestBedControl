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

import fivezone.testbed_utils as tu
import fivezone.agents as ag

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

		# time delta in seconds to advance simulation for every step
		self.step_size = kwargs['step_size']
		# number of days in one episode
		self.episode_days = kwargs['episode_days']
		# path to fmu
		fmu_path : str = kwargs['fmu']

		# initialize start time and global end time
		self.re_init(**{'fmu_start_time_step':kwargs['fmu_start_time_step'],
						 'global_fmu_end_time_step' : kwargs['global_fmu_end_time_step']})

		# simulation time elapsed
		self.simulation_time_elapsed = 0.0

		'''methods to set up the environment'''
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
			# set paramters to be set before initialization
			# Note that we cannot initialize any Real dependent parameter.
			self.initialize_vals()
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


	def initialize_vals(self,*args,**kwargs):
		pass


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
			usr_actions = a[[self.usr_axn_idx_unsrtd.index(i) for i in self.usr_axn_idx]]  # sort
		else: 
			usr_actions = a  # don't reorder anything
		return self.action_processor(usr_actions)
	

	# Process the action
	def action_processor(self, a):
		"""
		TODO: Inverse scale the action from the agent
		"""
		raise NotImplementedError


	# Calculate the observations for the next state of the system this function does not need obs since it stores obs internally in fmu 
	def state_transition(self, _, action):

		raise NotImplementedError


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
		usr_obs = obs[self.usr_obs_idx_unsrtd]  # chose the obs, user wants, in their prefererd order
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



class testbed_5zone(testbed_base):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.create_gym_obs_axn_space(self.user_observations, self.user_actions,
									  self.obs_vars, self.act_vars,
									  self.obs_space_bounds, self.action_space_bounds)

		self._num_actions = len(kwargs['initial_stpt_vals'])
		self.stpt_vals = np.array(kwargs['initial_stpt_vals'])
		self.stpt_vals_ub = np.array(kwargs['stpt_vals_ub'])
		self.stpt_vals_lb = np.array(kwargs['stpt_vals_lb'])

		# zone occupancy variables
		self.zone_occupancy_vars = ['occSchCor', 'occSchNor','occSchSou','occSchEas','occSchWes']
		# zone internal heat gain variables
		self.zone_internal_heat_gain_vars = ['pCorintGaiFra','pNorintGaiFra','pSouintGaiFra',
											'pEasintGaiFra','pWesintGaiFra']

		# occupancy simulation function
		self.occupancy_function = eval('tu.'+kwargs['occupancy_function'])
		# internal load simulation function
		self.load_function = eval('tu.'+kwargs['load_function'])
		# internal agent to be used					
		int_agent_class = eval('ag.'+kwargs['internal_agent'])
		# whether to use usual delta or preferential ub lb delta
		self.delta_fn = eval('self.'+kwargs['delta_fn'])

		# energy variables used to calculate reward
		self.power_variables = ['res.PHea', 'res.PCooSen', 'res.PCooLat']
		self.energy_lb = kwargs['energy_lb']
		self.energy_ub = kwargs['energy_ub']
		# * Here 'res.PCooSen','res.PCooLat' will be negative so we have to negate the values *
		self.power_sign = [1.0, -1.0, -1.0]

		# create the internal agent to handle unused actions for the testbed
		self.internal_agent_created = False
		if len(self.usr_axn_idx)<self._num_actions:  # internal agent needed
			self.internal_agent = int_agent_class(self.usr_axn_idx, self._num_actions)
			self.internal_agent_action_idx = self.internal_agent.get_internal_agent_action_idx()
			self.internal_agent_created = True
		elif len(self.usr_axn_idx)==self._num_actions:  # internal agent not needed
			pass
		else:
			raise IndexError

		# create training phase info_dict
		self.train_info_list = []

		# create the scaler dictionary from above information
		scaler_dict = {}
		var_names = kwargs['observed_variables'] + kwargs['action_variables'] + self.power_variables
		var_lb = kwargs['observation_space_bounds'][0] + kwargs['stpt_vals_lb'] + self.energy_lb
		var_ub = kwargs['observation_space_bounds'][1] + kwargs['stpt_vals_ub'] + self.energy_ub
		for key,var_lb,var_ub in zip(var_names, var_lb, var_ub):
			scaler_dict[key] = {'min':var_lb,'max':var_ub}
		self.scaler = tu.dataframescaler(scaler_dict)

		self.r_energy_wt, self.r_comfort_wt, self.r_delta_wt = kwargs['r_energy_wt'], kwargs['r_comfort_wt'], kwargs['r_delta_wt']
		# zone temperature var names ;should we switch to room temps? Done!
		self.zone_temp_vars : List[str] = ['TRooAir.y5[1]','TRooAir.y3[1]','TRooAir.y1[1]',
											'TRooAir.y2[1]','TRooAir.y4[1]']
		# zone temperature cooling and heating bounds
		self.zone_temp_cool = ['conVAVCor.TRooCooSet','conVAVNor.TRooCooSet','conVAVSou.TRooCooSet',
								'conVAVEas.TRooCooSet','conVAVWes.TRooCooSet']
		self.zone_temp_heat = ['conVAVCor.TRooHeaSet','conVAVNor.TRooHeaSet','conVAVSou.TRooHeaSet',
								'conVAVEas.TRooHeaSet','conVAVWes.TRooHeaSet']

		# reset the fmu to appropriate time point to get correct values; no need to set global_fmu_reset
		_ = self.reset()

		# list the variables to be included for debugging1 vav damper position,
		# "Signal for VAV damper", Signal for heating coil valve, 
		self.debug_vars = ['cor.y_actual','nor.y_actual','sou.y_actual','eas.y_actual','wes.y_actual',
						'conVAVCor.yDam','conVAVNor.yDam','conVAVSou.yDam','conVAVEas.yDam','conVAVWes.yDam',
						'conVAVCor.yVal','conVAVNor.yVal','conVAVSou.yVal','conVAVEas.yVal','conVAVWes.yVal']



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

		# get True False status of each zone based on current fmu time and  tnextOcc
		self.zone_occupancy_status, self.tNexOccAll = self.occupancy_function(self.start_time)
		# get internal heat gain values for each zone
		self.zone_internal_heat_gain = self.load_function(self.start_time,self.zone_occupancy_status)
		# get final action values for internal agent if needed
		if self.internal_agent_created:
			internal_axns = self.internal_agent.predict(np.array(self.zone_occupancy_status), self.start_time)
			self.stpt_vals[self.internal_agent_action_idx] = internal_axns

		return self.stpt_vals



	# Process the observation
	def obs_processor(self, obs : np.array):
		"""
		Scale the observations to 0-1 range 
		"""
		return self.scaler.minmax_scale(obs, input_columns= self.user_observations,
											 df_2scale_columns= self.user_observations)



	def state_transition(self, _, action):

		# check input type
		if isinstance(action,np.ndarray):
			action = list(action)
		elif isinstance(action, list):
			pass
		else:
			raise TypeError

		# set the inputs and actions
		self.set_inputs(action)

		# try to simulate the system until it produces no error
		self.simulation_status = 100  # any value != 0

		while self.simulation_status!=0:
			self.simulation_status = self.fmu.do_step(current_t = self.start_time,
									step_size = self.step_size, new_step=True)
			if self.simulation_status!=0:
				print("Something wrong with the Simulation: {}".format(self.simulation_status))
				# reset simulation to the next time step and see if error persists
				# Again set the inputs to the simulation

				# reset simulation to the next time step
				self.start_time += self.step_size  # 1 advance time to a hopefully correct simulation step
				# self.simulation_time_elapsed += self.step_size  # 2 advance simulation time elapsed
				# ^^  not needed as it will be reset to 0
				# 3 now do the reinit first to change the simulation global start time to current value
				# of self.start_time and already existing value of global end time
				self.re_init(**{'fmu_start_time_step':self.start_time//self.step_size,
						 		'global_fmu_end_time_step' : self.global_fmu_end_time//self.step_size})
				# now reset the environment
				_ = self.reset()
				
				# set inputs and  the action variables
				self.set_inputs(action)

		# get the observation variables
		obs_next : list = [i[0] for i in self.fmu.get(self.obs_vars)]

		return np.array(obs_next)



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
		power_values_scaled = self.scaler.minmax_scale(power_values, input_columns= self.power_variables,
											 df_2scale_columns= self.power_variables)
		# reward incentivizes lower energy; reward lower energy
		r_energy = -1.0*np.sum(power_values_scaled)

		'''comfort component of the reward: we only penalize deviation when there is occupancy'''
		# see if zone is occupied; not needed anymore since we want to calculate reward at everytime
		# occupancy_status : int = all(self.zone_occupancy_status)
		# zone temperatures
		zone_temp : np.array = np.array([i[0] for i in self.fmu.get(self.zone_temp_vars)])
		# check whether zone_temp is within the range per zone
		# temp ub
		self.temp_ub : np.array = np.array([i[0] for i in self.fmu.get(self.zone_temp_cool)])
		# temp lb
		self.temp_lb : np.array = np.array([i[0] for i in self.fmu.get(self.zone_temp_heat)])
		temp_within_range : np.array = (zone_temp>self.temp_lb) & (zone_temp<self.temp_ub)
		r_comfort = np.sum(temp_within_range)
		# scale to 0-1 using reasonable bounds(0,5: since there are 5 zones at most)
		r_comfort = r_comfort/5

		'''decentralized energy use because of difference b/w ahu temp and zone setpoints'''
		ahu_temp : np.array = np.array([i[0] for i in self.fmu.get(['TSupSetHea'])]*5) # get ahu temp for all zones
		r_delta = self.delta_fn(obs[0], ahu_temp, self.temp_lb, self.temp_ub)

		reward = self.r_energy_wt*r_energy +  self.r_comfort_wt*r_comfort + self.r_delta_wt*r_delta

		info = {}
		# add time
		info['time'] = self.fmu.time
		info['reward_energy'] = r_energy
		info['reward_comfort'] = r_comfort
		info['reward_delta'] = r_delta
		for name,val in zip(self.power_variables, power_values):
			info[name] = val
		for name,val in zip(self.zone_temp_vars, zone_temp):
			info[name] = val
		for name, val in zip(self.user_observations, obs):
			info[name] = val
		for name,val in zip(self.act_vars, action):
			info[name] = val
		for name, val in zip(self.zone_occupancy_vars, self.zone_occupancy_status):
			info[name] = val
		info['tNexOccAll'] = self.tNexOccAll
		for name, val in zip(self.zone_internal_heat_gain_vars, self.zone_internal_heat_gain):
			info[name] = val
		# add other variables for debugging
		self.debug_vals = [i[0] for i in self.fmu.get(self.debug_vars)]
		for name, val in zip(self.debug_vars, self.debug_vals):
			info[name] = val

		self.train_info_list.append(info)
		
		return reward, info


	def delta_ublb(self, oat, ahu_temp, temp_lb, temp_ub):
		if oat>290.15:
			ahu_temp_penalty : np.array = np.abs((ahu_temp>temp_ub)*(ahu_temp-temp_ub))
		else:
			ahu_temp_penalty : np.array = np.abs((ahu_temp<temp_lb)*(temp_lb-ahu_temp))
		
		zone_delta_avg = np.sum((ahu_temp_penalty))/5.0  # flattened average
		r_delta = -1.0*zone_delta_avg/5.0  # scale by maximum deviation seen in historical data
		return r_delta

	
	def delta_usual(self, oat, ahu_temp, temp_lb, temp_ub):
		ahu_temp_below_lb_penalty : np.array = np.abs((ahu_temp<temp_lb)*(temp_lb-ahu_temp))
		ahu_temp_above_ub_penalty : np.array = np.abs((ahu_temp>temp_ub)*(ahu_temp-temp_ub))
		zone_delta_avg = np.sum((ahu_temp_above_ub_penalty,ahu_temp_below_lb_penalty))/5.0  # flattened average
		r_delta = -1.0*zone_delta_avg/5.0  # scale by maximum deviation seen in historical data
		return r_delta


	def set_inputs(self,action):
		# set input to the action variables
		self.fmu.set(variable_name=self.act_vars, value=action)
		# set the input to zone occupancy variables
		self.fmu.set(variable_name=self.zone_occupancy_vars, value=self.zone_occupancy_status)
		# set the input to tnextOcc variable
		self.fmu.set(variable_name=['tNexOccAll'], value=self.tNexOccAll)
		# set the input to internal heat gain values
		self.fmu.set(variable_name=self.zone_internal_heat_gain_vars, value=self.zone_internal_heat_gain)


	def initialize_vals(self,*args,**kwargs):
		"""Will initialize certain vars of our interest"""
		init_var_names = ['flo.cor.T_start','flo.nor.T_start',
							'flo.sou.T_start','flo.eas.T_start','flo.wes.T_start',]
		init_vals = self.internal_agent.unocc_mid_vals
		self.fmu.set(init_var_names,init_vals)


	

