 # RL - EmsPy (work in progress...)
### The EmsPy Python package was made to facilitate Reinforcement Learning (RL) algorithm research for developing and testing Building Control Agents (BCAs) for intelligent building and HVAC automation using EnergyPlus's (E+) building energy simulator and wrapping their Energy Management System (EMS) Python API. 

*This repo was constructed by someone with little experience with EnergyPlus and software/programming, but wanted to 
assist in creating a simplistic but flexible building 'environment' interface for RL building control research. Any feedback or improvements to the repo is welcomed.* 

### Introduction

The RL-centered wrapper, **EmsPy**, is meant to simplify and somewhat constrain the EnergyPlus (E+) Energy Management System API (EMS). The popular/intended use of the
EMS API is to interface with a running E+ building simulation and/or inject custom code, which is not so easily done otherwise. 
EMS exposes E+ real-time simulation data such as variables, internal variables, meters, actuators, and weather. 

Recently, a Python API was created for EMS so users aren't limited by the E+ Runtime Language (ERL) and can more naturally interact with 
a running building simulation to gather state information and implement custom control and simulation modifications at runtime (subhourly timesteps).
EMS can be used to create Python plugins or call E+ as a library and run simulations directly from Python - **EmsPy** utilizes the latter. 
Please see the documentation hyperlinks to learn more about [EnergyPlus EMS](https://bigladdersoftware.com/epx/docs/9-5/ems-application-guide/index.html), [Using EnergyPlus as a library](https://bigladdersoftware.com/epx/docs/9-5/input-output-reference/api-usage.html#sec:api-usage), and its [Python API](https://nrel.github.io/EnergyPlus/api/python/index.html). 

**Note:**
Although this repo is meant to wrap EMS and simplify interfacing with E+ for RL purposes - making this research space more readily accessible to AI and controls 
researchers and hobbyiest - a good understanding of E+ and building modeling may still be necessary, especially if you intend to create, link, and
control your own building models, or, need more advanced EMS features. EMS offers many entry points into the simulation during runtime through calling points, each with their own utility, and this API does not limit that functionality. With this API, observations and actuation functions can be enacted on none, one, to all calling points - whatever fits your specific control and BEM needs. This flexibility naturally leads to more complexity.
 
*Eventually*, some standard building models and template scripts will be created so that 
user's can simply experiment with them through Python with no E+ experience needed. A natural formfactor would be
to resemble OpenAI's Gym Environment API. This standardization building models and interaction may also help institute performance benchmarks for the research community. 
 
 Regardless of your use case, you will 
 need to have the proper versioned (**9.5.0**) [E+ simulation engine](https://github.com/NREL/EnergyPlus/releases/tag/v9.5.0) downloaded onto your system.

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

### Other Helpful E+ Software Tools:
- [OpenStudio SDK](https://openstudio-sdk-documentation.s3.amazonaws.com/index.html)

### Overview

The diagram below depicts the RL-interaction-loop within a timestep at simulation runtime. Because of the unchangeable technicalities of the 
interaction between EMS and the E+ simulator - through the use of callback function(s) and the many calling points available
 per timestep - the underlying RL interface and algorithm must be implemented in a very specific manner. This was done in a way as to provide maximal flexibility and not constrain usage, but at the inherent cost of some extra complexity and greater learning curve. However, once understood, it is simple to use and fit to your custom needs. This be explained in detail below and in the Wiki pages. 

<img src="https://user-images.githubusercontent.com/65429130/158069215-6a4c654d-5658-4765-8828-0728fcc34c88.PNG" width = "750"> 

<br/>There are likely 4 main use-cases for this repo, if you are hoping to implement RL algorithms on E+ building simulationss at runtime. 

<ins>In order of increasing complexity</ins>:

- You want to use an existing EMS interface template and linked building model to only implement RL control
- You have an existing E+ building model (with *no* model or .idf modification needed) that you want to link and 
implement RL control on
- You have an existing E+ building model (with *some amount* of model or .idf modification needed) that you want to 
link and implement RL control on
- You want to create a new E+ building model to integrate and implement RL control on (another project in itself)

**EmsPy**'s usage for these use-cases is all the same - the difference is what must be done beforehand. Creating building models, 
understanding their file makeup, configuring HVAC systems, modifying .idf files, and adding/linking EMS variables and actuators brings extra challenges.
This guide will focus on utilizing **EmsPy** (EMS-RL wrapper). The former components, before utilizing **EmsPy**, will
be discussed elsewhere, with basic guidance to get you started in the right direction if you are new to EnergyPlus and/or EMS 

At the very least, even if solely using **EmsPy** for a given model, it is important to understand the types EMS metrics of a given
model: <ins>*variables, internal variables, meters, actuators,* and *weather*</ins>. These represent specific types of simulation data exposed
through EMS that can be used to build the state and action space of your control framework. For each type, there are many specific entities within the building model whose data can be looked up throughout the simulation. For instance, at each timestep for a specific calling point, I may use a *meter* to track all HVAC energy use, *variables* to track zone temperatures and occupancy schedules, and thermostat *actuator* to control the heating and cooling setpoints of a zone. The calling point I choose, say `callback_after_predictor_before_hvac_managers` determines exactly when in the flow of the simulation-solver that my callback function will be called.

See the *9.5 EMS Application Guide* and *9.5 Input Output Reference* documents for detailed documentation on these topics at either 
[EnergyPlus Documentation](https://energyplus.net/documentation) or [Big Ladder Software](https://bigladdersoftware.com/epx/docs/9-5/index.html).
  
### How to use EmsPy with an E+ Model
 
This guide provides a *very brief* overview of how to use EmsPy. Please see the Wiki, code documentation, and example scripts for more detailed information. The integration of the control (RL) algorithm and the flow of the calling points and callback functions at runtime is depicted in the image above. The image below loosely
represents the logic of the **EmsPy** API.

<img src="https://user-images.githubusercontent.com/65429130/158071407-116125be-18e4-4e94-bbfe-a8e1860aa86e.PNG" width = "750">

**1.** First, you will create an **BcaEnv object** (Building Control Agent + Environment) from proper inputs.
`BcaEnv` is a simplified UI that wraps `EmsPy` that should provide all necessary functionallity. Using `EmsPy`, this object encapsulates your building simulation environment and helps manage all your specificed EMS data produced and recorded during runtime. 
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
    - **Variables**: `'user_var_name': ['variable_name', 'variable_key']` elements of `tc_vars` dict
    - **Internal Variables**: `'user_intvar_name': ['variable_type', 'variable_key']` elements of `tc_intvars` dict 
    - **Meters**: `'user_meter_name': ['meter_name']` element of `tc_meter` dict
    - **Weather**: `'user_weather_name': ['weather_name']` elements of `tc_weather` dict
    - **Actuators**: `'user_actuator_name': ['component_type', 'control_type', 'actuator_key']` elements of `tc_actuator` dict
 
Once this has been completed, your ***BcaEnv*** object has all it needs to manage your runtime EMS needs - implementing 
various data collection/organization and dataframes attributes, as well as finding the EMS handles from the ToCs, etc. 

***Note:*** *At this point, the <ins>simulation can be ran</ins> but nothing useful will happen (in terms of control or data collection) as no calling points, callback functions, or actuation functions have been defined and linked. It may be helpful to run the simulation with only this 'environment' object initialization and then review its contents to see all that the class has created.*

**2.** Next, you must define the "Calling Point & Callback Function dictionary" with `BcaEnv.set_calling_point_and_callback_function()` to define and enable your callback functionality at runtime. This dictionary links a calling point(s) to a callback function(s) with optionally 1) **Obvservation** function, 2) **Actuation** function, 3) and the arguments dictating at what frequncy (with respect to the simulation timestep) these observation and actuations occur.
 A given <ins>calling point</ins> defines when a *linked* <ins>callback function</ins> (and optionally an embedded <ins>actuation function</ins>) will be ran during the simulation timestep calculations.
The diagram above represents the simulation flow and RL integration with calling points and callback functions. 

*A brief word on **Observation** and **Actuation** functions*: 

- Each callback function (linked with a specific calling point) permits two custom functions to be attached. One is termed the **Observation** function and the other the **Actuation** function, and they're meant for capturing the state and taking actions, respectively. Your actual usage and implementation of these functions - if at all since they are optional, and only 1 is necessary for custom control and data tracking - is up to you. The two main differences is that the **Observation** function is called *before* the **Actuation** function in the callback and what each should/can return when called. The **Obvservation** function can return 'reward(s)' to be automatically tracked. And the **Actuation** function must return an actuation dictionary, linking an actuator to its new setpoint value. Technically, for control purposes, you could do everything in just the **Actuation** function; but the **Observation** function grants extra flexibility to accessing the state and helpful automatic reward tracking. Also, since *each calling point* can have its own callback function, many seperate **Observation** and **Actuation** functions could be used across a single timestep, however, these usage is more advanced and may only be needed is special circumstances.
 
 The Calling Point & Actuation Function dictionary should be built one key-value at a time using the method for each desired calling point callback:

 ```python
 BcaEnv.set_calling_point_and_callback_function(
    calling_point: str, observation_function, actuation_function, update_state: bool, update_observation_frequency: int = 1, update_actuation_frequency: int = 1)
 ```
- `calling_point` a single calling point from the available list `EmsPy.available_calling_points`
- `actuation_function` the control algorithm function, which <ins>must take no arguments and must return a dictionary</ins> (or `None` if no custom actuation) of actuator name(s) *(key)* and floating point setpoint value(s) *(value)* to be implemented at the linked calling point. 
    Be sure to pass the function itself, don't call it.
    - **Note:** due to the scope and passing of the callback function, please use a custom class and instantiate a global object in your script to encapsulate any custom data for the control algorithm (RL agent parameters) and then utilize the global object in your actuation function. The callback functions can reference object/class data at runtime.
    - ***Warning:*** *actual actuator **setpoint values** can be floating point, integer, and boolean values (or `None` to relinquish control back to E+) and have a variety of input domain spans. Since the API input <ins>must be floating point</ins>, the setpoint values will be automatically cast to nearest integer (1/2 rounds up) and all but ~1.0 casts to False, respective to the specific actuator's needs. These details are defined in the E+ EMS API Documentation
    **Internal variables** may be able to be used to understand an actuators input domain. You must have an understanding of the actuator(s) to control them as intended.*  
- `update_state` T/F to whether or not the entire EMS ToCs' data should be updated from simulation for that calling point, this acts as a complete state update (use `BcaEnv.update_ems_data` for more selective udpates at specific calling points, if needed)
- `update_observation_frequency` the number of simulation timesteps between each time the associated **Observation** function is called, default is every timestep
- `update_actuation_frequnecy` the number of simulation timesteps between each time the associated **Actuation** function called, default is every timestep
   
***Note:*** *there are multiple calling points per timestep, each signifying the start/end of an event in the process. The majority of calling points occur consistently throughout the simulation, but several occur *once* before during simulation setup.* 

The user-defined `actuation_function` should encapsulate any sort of control algorithm (more than one can be created and linked to unique calling points, but it's likely that only 1 will be used as the entire RL algorithm). Using the methods `BcaEnv.get_ems_data` and `BcaEnv.get_weather_forecast`, to collect state information, a control algorithm/function can be created and its actions returned. In `emspy` using a decorator function, this **Actuation** function will automatically be attached to the standard callback function and linked to the defined calling point. At that calling point during runtime, the actuation function will be ran and the returned actuator dict will be passed to the simulation to update actuator setpoint values. 
The rest of the arguments are also automatically passed to the base-callback function to dictate the update frequency of observation and actuation. This means that data collection or actuation updates do not need to happen every timestep or in tandem with each other. 

### Please refer to the Wiki or `EmsPy` and `BcaEnv` code documentation on how to utilize this API.

Below, is a sample sub-script of EmsPy usage: controlling the thermostat setpoints of a single zone of a 5-Zone Office Building based on the time of day. 
```python
"""
This is a simple example to show how to set up and simulation and utilize some of emspy's features.
This implements simple rule-based thermostat control based on the time of day, for a single zone of a 5-zone office
building. Other data is tracked and reported just for example.

This is a simplified/cleaned version (no MdpManager, less comments, etc.) of the 'simple_emspy_control.py' example,
meant for the README.md.
"""
import datetime
import matplotlib.pyplot as plt

from emspy import EmsPy, BcaEnv


# -- FILE PATHS --
# * E+ Download Path *
ep_path = 'A:/Programs/EnergyPlusV9-5-0/'  # path to E+ on system
# IDF File / Modification Paths
idf_file_name = r'BEM_simple/simple_office_5zone_April.idf'  # building energy model (BEM) IDF file
# Weather Path
ep_weather_path = r'BEM_simple/5B_USA_CO_BOULDER_TMY2.epw'  # EPW weather file

# Output .csv Path (optional)
cvs_output_path = r'dataframes_output_test.csv'

# STATE SPACE (& Auxiliary Simulation Data)

zn0 = 'Core_ZN ZN'

tc_intvars = {}  # empty, don't need any

tc_vars = {
    # Building
    'hvac_operation_sched': ('Schedule Value', 'OfficeSmall HVACOperationSchd'),  # is building 'open'/'close'?
    # -- Zone 0 (Core_Zn) --
    'zn0_temp': ('Zone Air Temperature', zn0),  # deg C
    'zn0_RH': ('Zone Air Relative Humidity', zn0),  # %RH
}

tc_meters = {
    # Building-wide
    'electricity_facility': ('Electricity:Facility'),  # J
    'electricity_HVAC': ('Electricity:HVAC'),  # J
    'electricity_heating': ('Heating:Electricity'),  # J
    'electricity_cooling': ('Cooling:Electricity'),  # J
    'gas_heating': ('NaturalGas:HVAC')  # J
}

tc_weather = {
    'oa_rh': ('outdoor_relative_humidity'),  # %RH
    'oa_db': ('outdoor_dry_bulb'),  # deg C
    'oa_pa': ('outdoor_barometric_pressure'),  # Pa
    'sun_up': ('sun_is_up'),  # T/F
    'rain': ('is_raining'),  # T/F
    'snow': ('is_snowing'),  # T/F
    'wind_dir': ('wind_direction'),  # deg
    'wind_speed': ('wind_speed')  # m/s
}

# ACTION SPACE
tc_actuators = {
    # HVAC Control Setpoints
    'zn0_cooling_sp': ('Zone Temperature Control', 'Cooling Setpoint', zn0),  # deg C
    'zn0_heating_sp': ('Zone Temperature Control', 'Heating Setpoint', zn0),  # deg C
}

# -- Simulation Params --
calling_point_for_callback_fxn = EmsPy.available_calling_points[6]  # 6-16 valid for timestep loop during simulation
sim_timesteps = 6  # every 60 / sim_timestep minutes (e.g 10 minutes per timestep)

# -- Create Building Energy Simulation Instance --
sim = BcaEnv(
    ep_path=ep_path,
    ep_idf_to_run=idf_file_name,
    timesteps=sim_timesteps,
    tc_vars=tc_vars,
    tc_intvars=tc_intvars,
    tc_meters=tc_meters,
    tc_actuator=tc_actuators,
    tc_weather=tc_weather
)


class Agent:
    """
    Create agent instance, which is used to create actuation() and observation() functions (both optional) and maintain
    scope throughout the simulation.
    Since EnergyPlus' Python EMS using callback functions at calling points, it is helpful to use a object instance
    (Agent) and use its methods for the callbacks. * That way data from the simulation can be stored with the Agent
    instance.
    """
    def __init__(self, bca: BcaEnv):
        self.bca = bca

        # simulation data state
        self.zn0_temp = None  # deg C
        self.time = None

    def observation_function(self):
        # -- FETCH/UPDATE SIMULATION DATA --
        self.time = self.bca.get_ems_data(['t_datetimes'])

        # Get data from simulation at current timestep (and calling point) using ToC names
        var_data = self.bca.get_ems_data(list(self.bca.tc_var.keys()))
        meter_data = self.bca.get_ems_data(list(self.bca.tc_meter.keys()), return_dict=True)
        weather_data = self.bca.get_ems_data(list(self.bca.tc_weather.keys()), return_dict=True)

        # get specific values from MdpManager based on name
        self.zn0_temp = var_data[1]  # index 1st element to get zone temps, based on EMS Variable ToC
        # OR if using "return_dict=True"
        outdoor_temp = weather_data['oa_db']  # outdoor air dry bulb temp

        # print reporting
        if self.time.hour % 2 == 0 and self.time.minute == 0:  # report every 2 hours
            print(f'\n\nTime: {str(self.time)}')
            print('\n\t* Observation Function:')
            print(f'\t\tVars: {var_data}'  # outputs ordered list
                  f'\n\t\tMeters: {meter_data}'  # outputs dictionary
                  f'\n\t\tWeather:{weather_data}')  # outputs dictionary
            print(f'\t\tZone0 Temp: {round(self.zn0_temp,2)} C')
            print(f'\t\tOutdoor Temp: {round(outdoor_temp, 2)} C')

    def actuation_function(self):
        work_hours_heating_setpoint = 18  # deg C
        work_hours_cooling_setpoint = 22  # deg C

        off_hours_heating_setpoint = 15  # deg C
        off_hours_coolng_setpoint = 30  # deg C

        work_day_start = datetime.time(6, 0)  # day starts 6 am
        work_day_end = datetime.time(20, 0)  # day ends at 8 pm

        # Change thermostat setpoints based on time of day
        if work_day_start < self.time.time() < work_day_end:  #
            # during workday
            heating_setpoint = work_hours_heating_setpoint
            cooling_setpoint = work_hours_cooling_setpoint
            thermostat_settings = 'Work-Hours Thermostat'
        else:
            # off work
            heating_setpoint = off_hours_heating_setpoint
            cooling_setpoint = off_hours_coolng_setpoint
            thermostat_settings = 'Off-Hours Thermostat'

        # print reporting
        if self.time.hour % 2 == 0 and self.time.minute == 0:  # report every 2 hours
            print(f'\n\t* Actuation Function:'
                  f'\n\t\t*{thermostat_settings}*'
                  f'\n\t\tHeating Setpoint: {heating_setpoint}'
                  f'\n\t\tCooling Setpoint: {cooling_setpoint}\n'
                  )

        # return actuation dictionary, referring to actuator EMS variables set
        return {
            'zn0_heating_sp': heating_setpoint,
            'zn0_cooling_sp': cooling_setpoint
        }


#  --- Create agent instance ---
my_agent = Agent(sim)

# --- Set your callback function (observation and/or actuation) function for a given calling point ---
sim.set_calling_point_and_callback_function(
    calling_point=calling_point_for_callback_fxn,
    observation_function=my_agent.observation_function,  # optional function
    actuation_function=my_agent.actuation_function,  # optional function
    update_state=True,  # use this callback to update the EMS state
    update_observation_frequency=1,  # linked to observation update
    update_actuation_frequency=1  # linked to actuation update
)

# -- RUN BUILDING SIMULATION --
sim.run_env(ep_weather_path)
sim.reset_state()  # reset when done

# -- Sample Output Data --
output_dfs = sim.get_df(to_csv_file=cvs_output_path)  # LOOK at all the data collected here, custom DFs can be made too

# -- Plot Results --
fig, ax = plt.subplots()
output_dfs['var'].plot(y='zn0_temp', use_index=True, ax=ax)
output_dfs['weather'].plot(y='oa_db', use_index=True, ax=ax)
output_dfs['meter'].plot(y='electricity_HVAC', use_index=True, ax=ax, secondary_y=True)
output_dfs['actuator'].plot(y='zn0_heating_sp', use_index=True, ax=ax)
output_dfs['actuator'].plot(y='zn0_cooling_sp', use_index=True, ax=ax)
plt.title('Zn0 Temps and Thermostat Setpoint for Year')

# Analyze results in "out" folder, DView, or directly from your Python variables and Pandas Dataframes
```
<ins>5 Zone Office Building Model</ins>

<img src="https://user-images.githubusercontent.com/65429130/158045813-914259d1-a0ba-45a5-b81d-35520e685b23.PNG" width = "750">

<ins>Sample Results for the Month of April</ins>

<img src="https://user-images.githubusercontent.com/65429130/158045876-da914d81-f705-43c6-9815-5c5c5cff9778.PNG" width = "750">


### References:
- *(in progress)*
