# 5ZoneTestBedControl
This repository will be used to develop and contain all the code needed for the control of the 
5 Zone Test Bed.

## Notes on the system
* time is not a variable that can be get or set. It can only be adjusted during simulate option.
* have to verify --> can changing start time after issuing 
* values like time and weather cannot be changed using set method from outside
* states are a part of the system model_variables

## TODO

* decide how to set ncp
* reset to 0 if weater file exhausted ie find default end time of the simulation