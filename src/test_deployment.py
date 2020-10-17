import os
import sys
import json

import ast
import configparser
from argparse import ArgumentParser
import logging
import time

import numpy as np

from testbed_env import testbed_v0, testbed_v1
from simple_agents import RandomAgent, PerformanceMetrics
from testbed_utils import rl_perf_save, dataframescaler

WORKING_DIR = os.path.abspath(os.getcwd())
DEFAULT_CONFIG_FILE = os.path.join(WORKING_DIR, 'config.cfg')
DEFAULT_CONIFG_SECTION = "TESTBED_V0"
DEFAULT_TIMESTEPS = 100
OUTPUT_DIR = WORKING_DIR+'/tmp/'
META_DATA_FILE = os.path.join(WORKING_DIR, 'resource/meta_data.json')
DEFAULT_TESTBED = 'testbed_v0'
DEFAULT_AGENT = 'random'

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
parser.add_argument('-a', '--agent', type=str, required=False, default=DEFAULT_AGENT,
						help='The testbed to use.')

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

def testbed_v0_random_agent(args):
	settings = get_settings(args.config_path, args.config_section)
	# get agg type and data stats from meta_data.json
	with open(META_DATA_FILE, 'r') as fp:
			meta_data_ = json.load(fp)
	scaler = dataframescaler(meta_data_)
	# update settings with scaler
	settings['scaler'] = scaler
	# create logger
	log = create_logger(settings)
	# create agent
	agent = RandomAgent(lb=np.array(settings['action_space_bounds'][0]), \
						ub=np.array(settings['action_space_bounds'][1]))
	log.info('Agent Created')
	# set up the environment
	env = testbed_v0(**settings)
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


def testbed_v1_random_agent(args):
	settings = get_settings(args.config_path, args.config_section)
	# create logger
	log = create_logger(settings)
	# USER CAN CREATE AGENT ANYWAY THEY WANT
	# example: create agent
	settings['action_idx_by_user'] = [idx for idx,name in enumerate(settings['action_variables'])
													 if name in settings['user_actions']]
	
	agent = RandomAgent(lb=np.array(settings['action_space_bounds'][0])[settings['action_idx_by_user']], \
						ub=np.array(settings['action_space_bounds'][1])[settings['action_idx_by_user']])

	log.info('Agent Created')
	# set up the environment
	env = testbed_v1(**settings)
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

if __name__ == '__main__':
	args = parser.parse_args()
	if args.agent == 'random':
		if args.testbed=='testbed_v0':
			testbed_v0_random_agent(args)
		elif args.testbed=='testbed_v1':
			testbed_v1_random_agent(args)