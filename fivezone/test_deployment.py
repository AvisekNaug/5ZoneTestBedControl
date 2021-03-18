import os
os.environ['KMP_WARNINGS'] = '0'
import sys
import json

import ast
import configparser
from argparse import ArgumentParser
import logging
import time

import numpy as np

from notify_run import Notify
notify = Notify()

from testbed_env import testbed_5zone
from agents import PerformanceMetrics
import agents as ag
from testbed_utils import rl_perf_save, dataframescaler

NOTIFY_ME = False
WORKING_DIR = os.path.abspath(os.getcwd())
DEFAULT_CONFIG_FILE = os.path.join(WORKING_DIR, 'config.cfg')
DEFAULT_TESTBED = 'testbed_5zone'
DEFAULT_CONIFG_SECTION = "TESTBED_5ZONE"
DEFAULT_TIMESTEPS = 100 # actual time elapsed depends on chosen testbed
OUTPUT_DIR = WORKING_DIR+'/tmp'
NUM_DAYS_IN_ONE_EPISODE = 7
DEFAULT_OCC = 'simulate_zone_occupancy_default'
DEFAULT_LOAD = 'simulate_internal_load_default'
INTERNAL_AGENT = 'InternalAgent_1'
DELTA_FN = 'delta_usual'
EXTERNAL_AGENT = 'BaseExtAgent'


# argument parser
parser = ArgumentParser(description='Deploy a random Reinforcement Learning Control \
										agent on 5 zone FMU Testbed')
parser.add_argument('-c', '--config_path', type=str, required=False, default=DEFAULT_CONFIG_FILE,
						help='Path to the configuration file.')
parser.add_argument('-s', '--config_section', type=str, required=False, default=DEFAULT_CONIFG_SECTION,
						help='Configuration section to use.')
parser.add_argument('-t', '--time_steps', type=int, required=False, default=DEFAULT_TIMESTEPS,
						help='Time Steps to run the deployment.')
parser.add_argument('-d', '--output_dir', type=str, required=False, default=OUTPUT_DIR,
						help='Output directory path to store results.')
parser.add_argument('-b', '--testbed', type=str, required=False, default=DEFAULT_TESTBED,
						help='The testbed to use.')
parser.add_argument('-e', '--episode_days', type=int, required=False, default=NUM_DAYS_IN_ONE_EPISODE,
						help='Number of days in one episode.')
parser.add_argument('-o', '--occupancy_function', type=str, required=False, default=DEFAULT_OCC,
						help='Function to simulate occupancy.')	
parser.add_argument('-i', '--load_function', type=str, required=False, default=DEFAULT_LOAD,
						help='Function to simulate internal load.')
parser.add_argument('-g', '--internal_agent', type=str, required=False, default=INTERNAL_AGENT,
						help='Class to simulate internal agent for non-user-controlled inputs.')
parser.add_argument('-x', '--ext_agent', type=str, required=False, default=EXTERNAL_AGENT,
						help='Class to simulate internal agent for non-user-controlled inputs.')
parser.add_argument('-j', '--delta_fn', type=str, required=False, default=DELTA_FN,
						help='Function to use for delta component of the reward')



parser.add_argument('--notify', dest='notify', action='store_true',help='Notify user of different checkpoints')
parser.set_defaults(notify=NOTIFY_ME)				



# set up logger
def create_logger(settings):
	logging.captureWarnings(True)
	logger = logging.getLogger(__name__)
	formatter = logging.Formatter('%(asctime)s, %(levelname)s, %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
	handler_file = logging.FileHandler(filename=settings['logs'], mode='w')
	handler_file.setFormatter(formatter)
	logger.addHandler(handler_file)
	handler_stream = logging.StreamHandler(stream=sys.stderr)
	handler_stream.setFormatter(formatter)
	logger.addHandler(handler_stream)
	logger.setLevel(logging.DEBUG)
	return logger



def get_settings(config_path, config_section):

	cfg = configparser.RawConfigParser()
	cfg.read(config_path)
	params = dict()
	for key,_ in cfg[config_section].items():
		params[key]=ast.literal_eval(cfg.get(config_section, key))
	return params


def deploy_testbed(args):
	try:

		if args.notify==True:
			notify.send('Script Started')	

		settings = get_settings(args.config_path, args.config_section)

		# create logger
		log = create_logger(settings)

		# There has to be an agent that can generate actions(possibly empty arrays if useer doesn't want to provide any action.)
		# example: Create a Random agent(can issue no actions too)
		settings['action_idx_by_user'] = [settings['action_variables'].index(name) for name in settings['user_actions']]
		
		agent_class = eval('ag.'+args.ext_agent) 
		agent = agent_class(lb=np.array(settings['action_space_bounds'][0])[settings['action_idx_by_user']], \
							ub=np.array(settings['action_space_bounds'][1])[settings['action_idx_by_user']])

		log.info('Agent Created')

		# set up the environment
		settings['episode_days'] = args.episode_days  # set days in an episode
		settings['occupancy_function'] = args.occupancy_function  # Function to simulate occupancy
		settings['load_function'] = args.load_function  # Function to simulate internal load
		settings['internal_agent'] = args.internal_agent  # get internal agent class
		settings['delta_fn'] = args.delta_fn  # fn to use for delta reward
		env = testbed_5zone(**settings)
		log.info('Environment Created')

		# get initial state of the system
		obs = env.reset()
		log.info('Agent Resets environment')
		# logger for deployment performance
		performance_logger = PerformanceMetrics()
		# create new empty metric dictionary
		performance_logger.on_episode_begin()

		# run the prediction for t timesteps
		s_time = time.time()
		for iter in range(args.time_steps):
			log.info('Iteration {} of {}'.format(iter+1,args.time_steps))
			# get action
			action = agent.predict(obs)
			log.info('Action Taken = {}'.format(action))
			# send action to environment
			time_start = time.time()
			obs, _, done, info = env.step(action)
			time_end = time.time()
			# log.info('Observation received = {}'.format(obs))
			log.info('Took {:.2f} s to complete the simulation iteration {} of {}'.format(time_end-time_start, iter+1, args.time_steps))
			performance_logger.on_step_end(info=info)
			if done:
				# add info to metric list
				performance_logger.on_episode_end()
				# create new empty metric dictionary
				performance_logger.on_episode_begin()
				obs = env.reset()
				log.info('Agent Resets environment')
		e_time = time.time()
		log.info('Took {:.2f} s to complete the simulation for {} iterations'.format(e_time-s_time, args.time_steps))
		# add info to metric list in case time steps lower than episode length
		performance_logger.on_episode_end()
		# save the performance logs
		rl_perf_save(test_perf_log_list=[performance_logger], log_dir=args.output_dir,
										save_as= 'csv', header=True)
		if args.notify:
			notify.send('Script Ended Without Error')										
	except Exception as e:
		log.critical('Script stopped due to:\n{}'.format(e))
		log.debug(e, exc_info=True)
		if args.notify:
			err_pt = settings['fmu_start_time_step'] + iter
			notify.send('Script stopped at at iteration {} due to \n {}'.format(err_pt,e))
		exit(-1)


if __name__ == '__main__':
	args = parser.parse_args()
	deploy_testbed(args)