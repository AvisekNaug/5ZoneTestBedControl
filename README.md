# 5 Zone Test Bed Control
This repository can be used to combine any supervisory control strategy (like standard reinforcement learning agents which respect the OpenAI Gym protocol) with a [5 Thermal Zones Test Bed](https://github.com/AvisekNaug/buildings_library_dev). This 5 Zone model is built based on the [corresponding open loop control implementation]((https://simulationresearch.lbl.gov/modelica/releases/latest/help/Buildings_Examples_VAVReheat_BaseClasses.html#Buildings.Examples.VAVReheat.BaseClasses.PartialOpenLoop)) of the 5 Zone testbed in the [Modelica Buildings Library](https://github.com/lbl-srg/modelica-buildings). The proposed extension of the existing open loop testbed will allow the end user to implement any type of **supervisory control only**(ie setpoint control) **through a Python Programming interface**for different plant models located in the 5 Zone test bed. The ability to implement these controls will vary across different versions of the proposed extension. Below we detail one such version. With development newer versions will be added with more functionality.

# Quickstart Example
If you want to quickly get running with this repo follow the two steps below
## Installation
The stable installation can be done through conda only
```bash
git clone https://github.com/AvisekNaug/5ZoneTestBedControl.git
cd 5ZoneTestBedControl
```
Install necessary requirements inside a virtual environment using anaconda
```bash
conda env create -f environment.yml
```
Activate the virtual environment
```bash
conda activate 5Zone
```

## Example run

We created a simple testbed `testbed_v1` where we use `ambient temperature, humidity, solar radiation, ahu supply air temperature, zone temperatures` as `observation` variables. `AHU heating temperatue setpoint, South zone heating temperatue setpoint and Noth zone rcooling temperatue setpoint` as the `action` variables. The other actions are adjusted internally by a [default agent with default rules](#Selecting-actions). The `reward` incentivizes less energy consumption, better zone comfort for each zone.

The example output for testbed_v1 with the controller using a `RandomAgent` and the rest of the variables controlled using a `InternalAgent_testbed_v1` class is shown below.
```bash
(5Zone) $ python fivezone/test_deployment.py -t 25 -d tb1 -b testbed_v1 -s TESTBED_V1
```

# Documentation

# Versions of the testbed(Modelica File)
We plan to develop multiple versions of the testbed each with more functionalities to allow more control of the testbed.

## 1.0 Testbed_v1
This is the testbed version [testbed_v1](https://github.com/AvisekNaug/buildings_library_dev/blob/master/Buildings/Examples/VAVReheat/testbed_v1.mo). 

### Action Space for the testbed:

It allows the **supervisory control only** of the following components of the 5 Zone Testbed.

* <u>Heating Coil Temperature Setpoint of the Air Handling Unit</u>. It heats the air whenever the mixed air temperature falls below the setpoint and the building is occupied or in warmup mode. The low level controller is PI based. This means that in the current set up the coils won't turn on when the building is unoccupied (i.e. at night/after office hours. This is an existing feature of the default testbed which we plan to override in a future version.)
* <u>Heating and Cooling Temperature Setpoints of the individual Terminal Reheat Units/Rooms</u>. There are 5 such units. The low level controller implemented in those units are deadband controllers. They remain turned off as long as the temperature of the room is within those ranges of the set point.

All these variables can be adjusted by the user of the testbed using the supervisory controller of their choice. The user can also choose to control only a certain subste of the controller set points and the rest would be controlled by the Python interface for the testbed using default rules.

### Observation Space for the testbed:

Though every possible variable can be queried as a part of the observation space, for the purposes of demnstration we name a few of the observation variables of interest to our own reinforcement learning control. These are:

* 'weaBus.TDryBul', # dry bulb temperature
* 'weaBus.relHum', # relative humidity
* 'weaBus.HGloHor', # global horz irradiation
* 'TSup.T',# ahu final air temperature
* 'TRooAir.y5[1]',# corridor zone temperature
* 'TRooAir.y3[1]',# nor zone temperature
* 'TRooAir.y1[1]',# sou zone temperature
* 'TRooAir.y2[1]',# eas zone temperature
* 'TRooAir.y4[1]']# wes zone temperature

The generic method to query the value of an observation at any time point using the PyFMI library is `fmu.get([variable _name'])`

A complete description of all possible variables that can be treated as part of the observation space is provided [here](resource/testbed_v1_variable_explanation.json).

### State Variables for the testbed:

The fmu also maintains internally a set of state variables used to keep tab of the current state of the system. Since it is a large list we chose not to show the entire list here. The PyFMI library allows to query all the state variables in the FMU using the `fmu.get_state_list()` command.

<!-- prettier-ignore 

# Compiling the 5 Zone TestBed into an FMU
This testbed is compiled into an FMU using any modelica compiler. We used the **pymodelica** package which uses the JModelica compiler backend to compile the testbed. 

A simple example compilation preocedure is demonstrated below. The requirements for this compilation procedure is to have the **pymodelica** package which uses the JModelica compiler backend installed in the local machine. Detailed steps to do this installation is discussed as a part of installing the JModelica compiler with python support inside a Docker [here](https://github.com/AvisekNaug/JModelica_docker).

First clone the [5 Thermal Zones Test Bed](https://github.com/AvisekNaug/buildings_library_dev) and this library. Then provide the path to the library to the `MODELICAPATH` environment variable.

```bash
git clone https://github.com/AvisekNaug/buildings_library_dev
git clone https://github.com/AvisekNaug/5ZoneTestBedControl
export MODELICAPATH=<path to the 5 Thermal Zones Test Bed library on the local machine>:$MODELICAPATH
```

for example on Linux this would be

```bash
export MODELICAPATH=$HOME/buildings_library_dev:$MODELICAPATH
```

Navigate to fmu_models folder to store the fmu
```bash
cd 5ZoneTestBedControl/fmu_models
```
Compile the FMU by providing apprpriate path to the .mo file
```python
# assuming the proper packages are installed with their appropriate 
# backend modelica compiler and MODELICAPATH is set
from pymodelica import compile_fmu
import pymodelica
pymodelica.environ['JVM_ARGS'] = '-Xmx4096m'  # Increase memory in case compilation fails
model_name = 'Buildings.Examples.VAVReheat.testbed_v1'
fmu_path = compile_fmu(model_name, target='cs') # fmu is now compiled
```

# Load an FMU 

To use a compiled FMU from above we do the following steps

```bash
cd $HOME/5ZoneTestBedControl
```
Now inside the python do the following
```python
from pyfmi import load_fmu
import numpy as np
import time

fmu_path = 'fmu_models/Buildings.Examples.VAVReheat.testbed_v1.fmu'
fmu = load_fmu(fmu_path)
```

Now that the FMU is created a standard Python interface is provided to interact with the testbed. This is provided by the `src/testbed_env.py` script discussed below.

-->

# Interaction with the testbed using a Python, OpenAIGym, PyFMI library and the testbed FMU

## The base testbed
The base testbed class called `testbed_base`(located in src/testbed_env.py) implements some of the lower level functionalities following the OpenAI Gym method of creating an environment, stepping through the testbed simulation etc. These are the following

`__init__` : used to pass the `names` and `gym space` type of the variables used as observations and actions for the process. IT also needs the path to the complied `fmu` model and the path to the json file containing all the `fmu variables`.

`re_init` : To enable resetting the fmu model to its initial state without creating a new instance.

`reset` : To allow the reinforcement learning agent to reset the environment to start a new learning or testing episode.

`step` : Used to perform a step of the simulation based on actions sent by the agent to this method. It performs the following methods:

* `action_processor` : Abstract method used to process the action sent by the rl controller.
* `obs_processor` :  Abstract method used to process the observation to be sent to the rl controller.
* `state_transition` : Perform a step of the simulation based on the processed action and return the next state.
* `calculate_reward_and_info` : Calculate step reward based on current state, action, next_state.
* `check_done` : Return True of condition for episode end is met.
`load_fmu` : Load the compiled fmu from a given path.

All these methods can be overridden by any class inheriting from this based environment.

## Selecting actions

The testbed provides the user to control a set of actions. The user has to speify which actions they wish to control using their own controller in the `config.cfg` file. The rest of the actions will be handled internally by an internal agent in `src/simple_agents.py`. For the current versions of the testbed the default rules are implemented. If the user wishes to create separate default rules, they can subclass the `InternalAgent` class in `simple_agents.py` to provide their own default rules.