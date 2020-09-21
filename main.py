# imports
import numpy as np
import os
import configparser
import ast
from argparse import ArgumentParser

from src import on_policy_learn

# ast.literal_eval(cfg.get("TESTBED_V0_DEFAULT", "fmu"))

# SRC_DIR = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR = os.path.abspath(os.getcwd())
CONFIG_FILE = os.path.join(WORKING_DIR, 'config.cfg')
LOG_PATH = os.path.join(WORKING_DIR, 'log.txt')


