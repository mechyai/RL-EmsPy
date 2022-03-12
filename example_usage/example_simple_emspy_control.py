"""
This is a simple example to show how to set up and simulation and utilize some of emspy's features.
This implements simple rule-based thermostat control on a single zone of a 5-zone office building. Other data is
tracked and reported just for example.

The same functionality implemented in this script could be done much more simply, but I wanted to provide some exposure
to the more complex features that are really useful when working with more complicated RL control tasks; Such as the use
of the MdpManager to handle all of the simulation data and EMS variables.
"""

import os

from emspy import EmsPy, BcaEnv, MdpManager, idf_editor

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.

# -- FILE PATHS --
# * E+ Download Path *
ep_path = 'A:/Programs/EnergyPlusV9-5-0/'  # path to E+ on system

# IDF File / Modification Paths
idf_file_name = r'/BEM_5z_V1.idf'  # building model IDF file

# Weather Path
ep_weather_path = r'/DallasTexas_2019CST.epw'  # EPW weather file

# Output .csv Path (optional)
cvs_output_path = r'dataframes_output_test.csv'


# --- create EMS Table of Contents (TC) for sensors/actuators ---
# int_vars_tc = {"attr_handle_name": [("variable_type", "variable_key")],...}
# vars_tc = {"attr_handle_name": [("variable_type", "variable_key")],...}
# meters_tc = {"attr_handle_name": [("meter_name")],...}
# actuators_tc = {"attr_handle_name": [("component_type", "control_type", "actuator_key")],...}
# weather_tc = {"attr_name": [("weather_metric"),...}

# STATE SPACE (& Auxiliary Simulation Data)
zn0 = 'Core_ZN ZN'

"""
See mdpmanager.MdpManager.generate_mdp_from_tc() to understand this structure for initializing and managing variables, 
and creating optional encoding functions/args for automatic encoding/normalization on variables values.

Note, however, MdpManager is not required. You can choose to manage variables on your own.
"""


def temp_c_to_f(temp_c: float, arbitrary_arg=None):
    """Convert temp from C to F. Test function with arbitrary argument, for example."""
    x = arbitrary_arg

    return 1.8 * temp_c + 32


tc_intvars = {}

tc_vars = {
    # Building
    'hvac_operation_sched': [('Schedule Value', 'OfficeSmall HVACOperationSchd')],  # is building 'open'/'close'?
    # -- Zone 0 (Core_Zn) --
    'zn0_temp': [('Zone Air Temperature', zn0), temp_c_to_f, 2],  # deg C
    'zn0_RH': [('Zone Air Relative Humidity', zn0)],  # %RH
}

tc_meters = {
    # Building-wide
    'electricity_facility': ['Electricity:Facility'],  # J
    'electricity_HVAC': ['Electricity:HVAC'],  # J
    'electricity_heating': ['Heating:Electricity'],  # J
    'electricity_cooling': ['Cooling:Electricity'],  # J
    'gas_heating': ['NaturalGas:HVAC']  # J
}

tc_weather = {
    'oa_rh': ['outdoor_relative_humidity'],  # %RH
    'oa_db': ['outdoor_dry_bulb', temp_c_to_f],  # deg C
    'oa_pa': ['outdoor_barometric_pressure'],  # Pa
    'sun_up': ['sun_is_up'],  # T/F
    'rain': ['is_raining'],  # T/F
    'snow': ['is_snowing'],  # T/F
    'wind_dir': ['wind_direction'],  # deg
    'wind_speed': ['wind_speed']  # m/s
}

# ACTION SPACE
tc_actuators = {
    # HVAC Control Setpoints
    'zn0_cooling_sp': [('Zone Temperature Control', 'Cooling Setpoint', zn0)],  # deg C
    'zn0_heating_sp': [('Zone Temperature Control', 'Heating Setpoint', zn0)],  # deg C
}


# -- INSTANTIATE 'MDP' --
my_mdp = MdpManager.generate_mdp_from_tc(tc_intvars, tc_vars, tc_meters, tc_weather, tc_actuators)

# -- Simulation Params --
calling_point_for_callback_fxns = EmsPy.available_calling_points[6]  # 5-15 valid for timestep loop during simulation
sim_timesteps = 6  # every 60 / sim_timestep minutes (e.g 10 minutes per timestep)

# -- Create Building Energy Simulation Instance --
sim = BcaEnv(
    ep_path=ep_path,
    ep_idf_to_run=idf_file_name,
    timesteps=sim_timesteps,
    tc_vars=my_mdp.tc_var,
    tc_intvars=my_mdp.tc_intvar,
    tc_meters = my_mdp.tc_meter,
    tc_actuator=my_mdp.tc_actuator,
    tc_weather=my_mdp.tc_weather
)


class Agent:
    """
    Create agent instance, which is used to create actuation() and observation() functions (both optional) and maintain
    scope throughout the simulation.
    Since EnergyPlus' Python EMS using callback functions at calling points, it is helpful to use a object instance
    (Agent) and use its methods for the callbacks. * That way data from the simulation can be stored with the Agent
    instance.

    NOTE: The observation and actuation functions are callback functions that, depending on your configuration (calling
    points,

    Depending on your implementation, an observation function may not be needed. The observation function just
    allows for extra flexibility in gathering state data from the simulation, as it happens before the actuation
    function is calle, and it can be used to return and store "reward" throughout the simulation, if desired.

    The actuation function is used to take actions. And must return an "actuation dictionary" that has the specific
    EMS actuator variable with its corresponding control value.
    """
    def __init__(self, bca: BcaEnv, mdp: MdpManager):
        self.bca = bca
        self.mdp = mdp

        # simplify naming of all MDP elements/types
        self.vars = mdp.tc_var
        self.meters = mdp.tc_meter
        self.weather = mdp.tc_weather
        self.actuators = mdp.tc_actuator

        # get just the names of EMS variables to use with other functions
        self.var_names = mdp.get_ems_names(self.vars)
        self.meter_names = mdp.get_ems_names(self.meters)
        self.weather_names = mdp.get_ems_names(self.weather)
        self.actuator_names = mdp.get_ems_names(self.actuators)

        # simulation data state
        self.zn0_temp = None  # deg C
        self.time = None

    def observation_function(self):
        # -- FETCH/UPDATE SIMULATION DATA --
        self.time = self.bca.get_ems_data(['t_datetimes'])

        # Get data from simulation at current timestep (and calling point)
        var_data = self.bca.get_ems_data(self.var_names)
        meter_data = self.bca.get_ems_data(self.meter_names)
        weather_data = self.bca.get_ems_data(self.weather_names)

        # update our MdpManager and all MdpElements, automatically calling any encoding/normalization functions
        vars = self.mdp.update_ems_value(self.vars, var_data)
        meters = self.mdp.update_ems_value(self.meters, meter_data)
        weather = self.mdp.update_ems_value(self.weather, weather_data)

        print(f'\n\n{str(self.time)}')
        print(f'Vars: {vars}\nMeters: {meters}, Weather:{weather}')

        # get specific values from MdpManager based on name
        self.zn0_temp = self.mdp.get_mdp_element_from_name('zn0_temp').value

    def actuation_function(self):
        work_hours_heating_setpoint = 18  # deg C
        work_hours_cooling_setpoint = 22  # deg C

        off_hours_heating_setpoint = 15  # deg C
        off_hours_cooilng_setpoint = 30  # deg C


        if work_day_start < current_time < workd_day_end:
            # during workday
            heating_setpoint = work_hours_heating_setpoint
            cooling_setpoint = work_hours_cooling_setpoint
        else:
            # off work
            heating_setpoint = off_hours_heating_setpoint
            cooling_setpoint = off_hours_cooilng_setpoint

        print(f'Action:\n\tHeating Setpoint: {heating_setpoint}\n\tCooling Setpoint: {cooling_setpoint}')

        # return actuation dictionary, refering to actuator EMS variables set
        return {
            'zn0_heating_sp': heating_setpoint,
            'zn0_cooling_sp': cooling_setpoint
        }


# create agent instance
my_agent = Agent(sim, my_mdp)


# set your callback function (observation and/or actuation) function for a given calling point
sim.set_calling_point_and_callback_function(
    calling_point=calling_point_for_callback_fxns,
    observation_fxn=my_agent.observation_function,  # optional
    actuation_fxn=my_agent.actuation_function,  # optional
    update_state=True,
    update_observation_frequency=1,
    update_actuation_frequency=1
)

# -- RUN BUILDING SIMULATION --
sim.run_env(ep_weather_path)
# reset when done
sim.reset_state()

# Analyze results in "out" folder, DView, or directly from your Python variables and Pandas Dataframes





