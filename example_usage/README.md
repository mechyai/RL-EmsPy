**This folder contains example scripts of **emspy** usage along with the building models (.IDF and .OSM) and weather files (.EPW) 
necessary to run the EnergyPlus building simulation. Below will be a brief explanation of each example file.*

##simple_emspy_control.py
This is a simple example to show how to set up and simulation and utilize some of **emspy**'s features.
This implements simple rule-based thermostat control based on the time of day, for a single zone of a 5-zone office
building. Other data is tracked and reported just for example.

The same functionality implemented in this script could be done much more simply, but I wanted to provide some exposure
to the more complex features that are really useful when working with more complicated RL control tasks; Such as the use
of the `MdpManager` to handle all of the simulation data and EMS variables.

<img src="" width = "750">
<img src="" width = "750">