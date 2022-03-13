 # RL - EmsPy (work In Progress...)
### The EmsPy Python package was made to facilitate Reinforcement Learning (RL) algorithm research for developing and testing Building Control Agents (BCAs) for intelligent building and HVAC automation using EnergyPlus's (E+) building energy simulator and wrapping their Energy Management System (EMS) Python API. 

*This repo was constructed by someone with little experience with EnergyPlus and software/programming, but wanted to 
assist in creating a simplistic but flexible building 'environment' interface for RL building control research. Any feedback or improvements to the repo is welcomed.* 

### Introduction

The RL-centered wrapper, **EmsPy**, is meant to simplify and somewhat constrain the EnergyPlus (E+) Energy Management System API (EMS). The popular/intended use of the
EMS API is to interface with a running E+ building simulation and/or inject custom code, which is not so easily done otherwise. 
EMS exposes E+ real-time simulation data such as variables, internal variables, meters, actuators, and weather. 

Recently, a Python API was created for EMS so users aren't constrained to using the E+ Runtime Language (ERL) and can more naturally interact with 
a running building simulation to gather state information and implement custom control at runtime (subhourly timesteps).
EMS can be used to create Python plugins or call E+ as a library and run simulations from Python - **EmsPy** utilizes the latter. 
Please see the documentation hyperlinks below to learn more about EnergyPlus EMS and its Python API. 

Although this repo is meant to simplify EMS and interfacing with E+ - making this research space more accessible to AI and controls 
people - a good understanding of E+ and building modeling may still be necessary, especially if you intend to create, link, and
 control your own building models. 

*Eventually*, some standard building models and template scripts will be created so that 
user's can simply experiment with them through Python for control purposes with no E+ experience needed. A natural formfactor would be
to resemble OpenAI's Gym Environment API. This standardization building models and interaction may also help institute performance benchmarks for the research community. 
 
 Regardless of your use case, you will 
 need to have the proper versioned (**9.5.0**) E+ simulation engine downloaded onto your system https://energyplus.net/downloads. 

### Further Documentation:
- [EnergyPlus](https://energyplus.net/)
- [EnergyPlus Documentation](https://energyplus.net/documentation) *(including EMS Application Guide!)*
- [EnergyPlus EMS Python API 0.2 Documentation](https://energyplus.readthedocs.io/en/stable/api.html)
- [EnerrgyPlus EMS API Homepage](https://nrel.github.io/EnergyPlus/api/)
- [OpenStudio SDK Documentation](http://nrel.github.io/OpenStudio-user-documentation/) (for building model creation and simulation GUI)
- [OpenStudio Coalition](https://openstudiocoalition.org/)
- [Unmet Hours Help Forum](https://unmethours.com/questions/) (community forum for EnergyPlus related help)

### Major Dependencies:
- EnergyPlus 9.5 (building energy simulation engine)
- EnergyPlus EMS Python API 0.2 (included in E+ 9.5 download)
- Python >= 3.8 
- pyenergyplus Python package (included in E+ download)
- [openstudio Python package](https://pypi.org/project/openstudio/) (not currently used, but plan to add functionality)

### Usage Explanation:

The diagram below depicts the RL-interaction-loop within a timestep at simulation runtime. Because of the unchangeable technicalities of the 
interaction between EMS and the E+ simulator - through the use of callback function(s) and the many calling points available
 per timestep - the underlying RL interface and algorithm must be implemented in a very specific manner, which will be explained in 
detail below. 

<img src="https://user-images.githubusercontent.com/65429130/119517258-764bbc00-bd45-11eb-97bf-1af9ab0444cb.png" width = "750"> 

<br/>There are likely 4 main use-cases for this repo, if you are hoping to implement RL algorithms at runtime. <ins>In order of increasing complexity</ins>:

- You want to use an existing EMS interface template and linked building model to only implement RL control
- You have an existing E+ building model (with *no* model or .idf modification needed) that you want to link and 
implement RL control on
- You have an existing E+ building model (with *some amount* of model or .idf modification needed) that you want to 
link and implement RL control on
- You want to create a new E+ building model to integrate and implement RL control on (another project in itself)

**EmsPy**'s usage for these use-cases is all the same - the difference is what must be done beforehand. Creating building models, 
understanding their file makeup, configuring HVAC systems, modifying .idf files, and adding/linking EMS variables and actuators brings extra challenges.
This guide will focus on utilizing **EmsPy** (EMS-RL wrapper). The former components, before utilizing **EmsPy**, will
be discussed briefly at the end, with basic guidance to get you started in the right direction. 

At the very least, even if solely using **EmsPy** for a given model, it is important to understand the EMS metrics of a given
model: <ins>*variables, internal variables, meters, actuators,* and *weather*</ins>. These represent the types of simulation data exposed
through EMS that can be used to build the state and action space of your control framework. 
See the *9.5 EMS Application Guide* and *9.5 Input Output Reference* documents for detailed documentation on these topics at either 
[EnergyPlus Documentation](https://energyplus.net/documentation) or [Big Ladder Software](https://bigladdersoftware.com/epx/docs/9-5/index.html).
  
### How to use EmsPy with an E+ Model:
 
This guide follows the design of the template Python scripts provided. The integration of the control (RL) algorithm and 
the flow of the calling points and callback functions at runtime is depicted in the image above. The image below loosely
represents the logic of EmsPy and its usage.

<img src="https://user-images.githubusercontent.com/65429130/121730967-598de300-cabe-11eb-9364-051bb993e8d1.png" width = "750">

**1.** First, you will create an **BcaEnv object** (Building Control Agent + Environment) from proper inputs.
`BcaEnv` is a simplified UI that wraps `EmsPy` that should provide all necessary functionallity. Using `EmsPy`, this object encapsulates your building simulation environment and helps manage all EMS data produced and recorded during runtime. 
The inputs include paths to the E+ directory and the building model .idf file to be simulated, information about all types of desired EMS metrics, and the simulation timestep.
Specifying the callback functions (organized by *Observation* and *Actuation* functions) with their linked calling points will come later.

```python
sim_environment = BcaEnv(ep_path: str, ep_idf_to_run: str, timesteps: int, tc_var: dict, tc_intvar: dict, tc_meter: dict, tc_actuator: dict, tc_weather: dict)
```
- `ep_path` sets the path to your EnergyPlus 9.5 installation directory
- `ep_idf_to_run` sets the path to your EnergyPlus building model, likely .idf file
- `timesteps` the number of timesteps per hour of the simulation. This *must* match the timestep detailed in your model .idf
- Define all EMS metrics you want to call or interact with in your model:
  - Build the **Table of Contents (ToC) dictionaries** for EMS variables, internal variables, meters, actuators, and weather 
  - ***Note:*** *this requires an understanding of EnergyPlus model input and output files, especially for actuators*
  - Each EMS category's ToC should be a dictionary with each EMS metric's user-defined name (key) and its required arguments (value) for
    fetching the 'handle' or data from the model. See [Data Transfer API documentation](https://energyplus.readthedocs.io/en/stable/datatransfer.html) for more info on this process.     
    - Variables: `'user_var_name': ['variable_name', 'variable_key']` elements of `tc_vars` dict
    - Internal Variables: `'user_intvar_name': ['variable_type', 'variable_key']` elements of `tc_intvars` dict 
    - Meters: `'user_meter_name': 'meter_name'` element of `tc_meter` dict
    - Actuators: `'user_actuator_name': ['component_type', 'control_type', 'actuator_key']` elements of `tc_actuator` dict
    - Weathers: `'user_weather_name': 'weather_name'` elements of `tc_weather` dict
 
Once this has been completed, your ***BcaEnv*** instance has all it needs to manage your runtime EMS needs - implementing 
various data collection/organization and dataframes attributes, as well as finding the EMS handles from the ToCs, etc. 

***Note:*** *At this point, the <ins>simulation can be ran</ins> but nothing useful will happen (in terms of control or data collection) as no calling points, callback functions, or actuation functions have been defined and linked.* 

*It may be helpful to run the simulation with only this 'environment' object initialization and then review its contents to see all that the class has created. 

**2.** Next, you must define the "Calling Point & Actuation Function dictionary" to define and enable callback functionality at runtime. This dictionary links a calling point(s) with a callback function(s) and the  arguments related to data/actuation update frequencies.
 A given <ins>calling point</ins> defines when a *linked* <ins>callback function</ins> (and optionally an embedded <ins>actuation function</ins>) will be ran during the simulation timestep calculations.
The diagram above represents the simulation flow and RL integration with calling points and callback functions. 
 
 The Calling Point & Actuation Function dictionary should be built one key-value at a time using the method:

# TODO update new method arguments
 ```python
 BcaEnv.set_calling_point_and_callback_function(
    calling_point: str, actuation_fxn, update_state: bool, update_state_freq: int = 1, update_act_freq: int = 1)
 ```
- `calling_point` a single calling point from the available list `EmsPy.available_calling_points`
- `actuation_fxn` the control algorithm function (one of potentially many throughout a timestep), which <ins>must take no argument and return a dictionary</ins> (or `None` if no custom actuation) of actuator name(s) *(key)* and floating point setpoint value(s) *(value)* to be implemented at the linked calling point. 
    Be sure to pass the function, not its result.
    - **Note:** due to the scope and passing of the callback function, please use a custom class and instantiate a global object in your script to encapsulate any custom data for the control algorithm (RL agent parameters) and then utilize the global object in your actuation function. The callback functions can reference object/class data at runtime.
    - ***Warning:*** *actual actuator **setpoint values** can be floating point, integer, and boolean values (or `None` to relinquish control back to E+) and have a variety of input domain spans. Since the API input <ins>must be floating point</ins>, the setpoint values will be automatically cast to nearest integer (1/2 rounds up) and all but ~1.0 casts to False, respective to the specific actuator's needs.
    **Internal variables** may be able to be used to understand an actuators input domain. You must have an understanding of the actuator(s) to function as intended.*  
- `update_state` T/F to whether or not the entire EMS ToCs should be updated for that calling point, this acts as a complete state update (use `BcaEnv.update_ems_data` for more selective udpates at specific calling points, if needed)
- `update_state_freq` the number of simulation timesteps in between each state update, default is every timestep
- `update_act_freq` set to the number of simulation timesteps in between each actuation function call, default is every timestep
   
***Note:*** *there are multiple calling points per timestep, each signifying the start/end of an event in the process. The majority of calling points occur consistently throughout the simulation, but several occur *once* before during simulation setup.* 

The user-defined `actuation_function` should encapsulate any sort of control algorithm (more than one can be created and linked to unique calling points, but it's likely that only 1 will be used as the entire RL algorithm). Using the 'agent/environment' object attributes, or better, the methods `BcaEnv.get_ems_data` and `BcaEnv.get_weather_forecast`, to collect state information, a control algorithm/function can be created and passed. Using a decorator function, this actuation function will automatically be attached to a base callback function and linked to the defined calling point. At that calling point during runtime, the actuation function will be ran and the returned actuator dict will be passed to the simulation to update actuator setpoint values. 
The rest of the arguments are also automatically passed to the base-callback function to dictate the update frequency of state data and actuation. This means that data collection or actuation updates do not need to happen every timestep. 

```python
BcaEnv.get_ems_data(ems_metric_list: list, time_rev_index: list=[0]) -> list
```
- This method will return an ordered nested list of ordered data points for then given EMS metrics and timing index(s), or entire EMS type ToC
- Its intended use is to return updated state information at each timestep, it can be called as many times as need in a actuation function. 
- This method must be used during runtime from an actuation function 
- `ems_metric_list` pass one or more EMS metrics (of any type) OR ONLY a single EMS type (var, intvar, meter, actuator, weather) in a list. Passing an EMS type will utilize and return data for that entire EMS ToC
- `time_rev_index` indicates the time index of the data you want to return, indexing backwards from the most recent timestep at 0. Leaving this list empty [ ] will return the entire data list collected thus far in the simulation for each given EMS metric. *Note that data will only be returned once the number of simulation timesteps has surpassed the maximum prior-time index given* 

```python
BcaEnv.get_weather_forecast(when: str, weather_metrics: list, hour: int, zone_ts: int) -> list
```
- This method is used to fetch and return an ordered list of future weather data, resembling weather forecasts. Weather events that have already occurred in simulation can be gathered using `BcaEnv.get_ems_data`
- This method must be used during runtime from an actuation function 
- `weather_metrics` is the list of user-defined weather variable names, defined in the weather ToC, you want to fetch data for 
- `when` either 'today' or 'tomorrow' dictates which day is in question, relative to current simulation time
- `hour` the hour of the day to collect the weather forecast data
- `zone_ts` the timestep within the given hour you want to collect weather forecast data for

 ***Note*** *: If you wish to use callback functions just for <ins>defualt data collection</ins> pass `None` as the actuation function. If you wish to use the callback functions for custom data collection and/or other actions other than any <ins>actuation/control</ins> at a specific calling point, implement an actuation function that returns `None` as the actuation dict.*

Also, if there is a need to <ins>update</ins> specific EMS metrics at a certain calling point separately from the rest (all EMS ToCs), you can use the method below within an actuation function to update specific EMS metrics. However, this <ins>does not also exclude them</ins> from the `state_update` that updates ALL EMS metrics.

```python
BcaEnv.update_ems_data(ems_metric_list: list, return_data: bool) -> list
```
- This method will update the given EMS metrics, or entire EMS type ToC and optionally return an ordered list of the updated data 
- Its intended use is if you want to update specific EMS data (or types) at a unique calling point, separate from the default state update of all EMS ToCs at another calling point.
- This method must be used during runtime from an actuation function 
- `ems_metric_list` pass one or more EMS metrics (of any type) OR ONLY a single EMS type (var, intvar, meter, actuator, weather) in a list. Passing an EMS type will utilize that entire EMS ToC
- `return_data` if True, this will automatically return the ordered list of data from `Bca.Env.get_ems_data`   

 ***Warning*** *: EMS data (and actuation) can be 'updated' by the user (but not necessarily internally by the simulation) <ins>for each calling point</ins> (and actuation function) assigned within a single timestep. You likely want to avoid this and manually only implement one state update `state_update=True` per timestep. Otherwise, you will screw up zone timestep increments (with current software design) and may accidentally be collecting data and actuating multiple times per timestep.
Just because you want to update data/actuation does not necessary mean it will be implmented at all or how you intended.
An understanding of calling points and when to collect data or actuate is ***crucial*** - Please see the [EMS Application Guide](https://energyplus.net/documentation) for more information on calling points.*
  
           
**TIPS**:
- *(in progress)*

**CAUTION**:
- Make sure your hourly timestep matches that of your EnergyPlus .idf model
- EMS data (and actuation) can be 'updated' by the user (but not necessarily internally by the simulation) <ins>for each calling point</ins> (and actuation function) assigned within a single timestep. You likely want to avoid this and manually only implement one state update `state_update=True` per timestep. Otherwise, you will screw up zone timestep increments (with current software design) and may accidentally be collecting data and actuating multiple times per timestep.
Just because you want to update data/actuation does not necessary mean it will be implmented at all or how you intended.
An understanding of calling points and when to collect data or actuate is ***crucial*** - Please see the [EMS Application Guide](https://energyplus.net/documentation) for more information on calling points.

### Future Planned Functionality & Repo Improvements:
- EmsPy improvements
  - more automatic user oversight to verify that user's have not violated logical errors in calling points, callback functions, and/or EMS updates
  - verify that given model timestep matches the .idf file, OR have it overwrite the model if not (say using openstudio somehow)
  - assist users in understanding actuator input ranges
  - further detailed documentation
- Data Dashboard class to automatically compile E+ performance and RL learning data into subplots via Matplotlib
- Openstudio wrapper class to assist in simple modifications of the .idf/.osm that impact simulation experiments (timesteps, start-end dates, etc.)
- Provide tips to documentation on how to construct and/or modify building models to be linked with EmsPy 
- A handful of various building models already set up with EmsPy so that user's can just focus on control algorithms given readily available state and action space, and pre-linked calling points. 


### Creating an E+ Building Energy Model:
- TOOD

### Setting up a E+ Model for EMS API Usage:
- TODO

### Linking EMS Metrics to Your EmsPy Script:
- TODO

### References:
- *(in progress)*
