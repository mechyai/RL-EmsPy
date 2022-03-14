"""
This is a simple example to show how to set up and simulation and utilize some of emspy's features.
This implements simple rule-based thermostat control based on the time of day, for a single zone of a 5-zone office
building. Other data is tracked and reported just for example.

The same functionality implemented in this script could be done much more simply, but I wanted to provide some exposure
to the more complex features that are really useful when working with more complicated RL control tasks; Such as the use
of the MdpManager to handle all of the simulation data and EMS variables.
"""
import datetime
import os

import matplotlib.pyplot as plt

from emspy import EmsPy, BcaEnv, MdpManager


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # a workaround to an error I encountered when running sim
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.


# -- FILE PATHS --
# * E+ Download Path *
ep_path = 'A:/Programs/EnergyPlusV9-5-0/'  # path to E+ on system

# IDF File / Modification Paths
idf_file_name = r'BEM_simple/simple_office_5zone_April.idf'  # building energy model (BEM) IDF file

# Weather Path
ep_weather_path = r'BEM_simple/5B_USA_CO_BOULDER_TMY2.epw'  # EPW weather file

# Output .csv Path (optional)
cvs_output_path = r'dataframes_output_test.csv'


def temp_c_to_f(temp_c: float, arbitrary_arg=None):
    """Convert temp from C to F. Test function with arbitrary argument, for example."""
    x = arbitrary_arg

    return 1.8 * temp_c + 32


# STATE SPACE (& Auxiliary Simulation Data)
"""
See mdpmanager.MdpManager.generate_mdp_from_tc() to understand this structure for initializing and managing variables, 
and creating optional encoding functions/args for automatic encoding/normalization on variables values.

Note, however, MdpManager is not required. You can choose to manage variables on your own.
"""

# --- create EMS Table of Contents (TC) for sensors/actuators ---
# int_vars_tc = {"attr_handle_name": [("variable_type", "variable_key")],...}
# vars_tc = {"attr_handle_name": [("variable_type", "variable_key")],...}
# meters_tc = {"attr_handle_name": [("meter_name")],...}
# actuators_tc = {"attr_handle_name": [("component_type", "control_type", "actuator_key")],...}
# weather_tc = {"attr_name": [("weather_metric"),...}

zn0 = 'Core_ZN ZN'

tc_intvars = {}

tc_vars = {
    # Building
    'hvac_operation_sched': [('Schedule Value', 'OfficeSmall HVACOperationSchd')],  # is building 'open'/'close'?
    # 'people_occupant_count': [('People Occupant Count', zn0)],  # number of people per Zn0
    # -- Zone 0 (Core_Zn) --
    'zn0_temp': [('Zone Air Temperature', zn0), temp_c_to_f, 2],  # deg C
    'zn0_RH': [('Zone Air Relative Humidity', zn0)],  # %RH
}

"""
NOTE: meters currently do not accumulate their values within there sampling interval during runtime, this happens at the
end of the simulation as a post-processing step. These will behave a lot like just a collection of EMS variables. See
UnmetHours for more info.
"""
tc_meters = {
    # Building-wide
    'electricity_facility': ['Electricity:Facility'],  # J
    'electricity_HVAC': [('Electricity:HVAC')],  # J
    'electricity_heating': [('Heating:Electricity')],  # J
    'electricity_cooling': [('Cooling:Electricity')],  # J
    'gas_heating': [('NaturalGas:HVAC')]  # J
}

tc_weather = {
    'oa_rh': [('outdoor_relative_humidity')],  # %RH
    'oa_db': [('outdoor_dry_bulb'), temp_c_to_f],  # deg C
    'oa_pa': [('outdoor_barometric_pressure')],  # Pa
    'sun_up': [('sun_is_up')],  # T/F
    'rain': [('is_raining')],  # T/F
    'snow': [('is_snowing')],  # T/F
    'wind_dir': [('wind_direction')],  # deg
    'wind_speed': [('wind_speed')]  # m/s
}

# ACTION SPACE
"""
NOTE: only zn0 (CoreZn) has been setup in the model to allow 24/7 HVAC setpoint control. Other zones have default
HVAC operational schedules and night cycle managers that prevent EMS Actuator control 24/7. Essentially, at times the 
HVAV is "off" and can't be operated. If all zones are to be controlled 24/7, they must be implemented as CoreZn.
See the "HVAC Systems" tab in OpenStudio to zone configurations.
"""
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
    tc_meters=my_mdp.tc_meter,
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
        self.vars = mdp.ems_type_dict['var']  # all MdpElements of EMS type var
        self.meters = mdp.ems_type_dict['meter']  # all MdpElements of EMS type meter
        self.weather = mdp.ems_type_dict['weather']  # all MdpElements of EMS type weather
        self.actuators = mdp.ems_type_dict['actuator']  # all MdpElements of EMS type actuator

        # get just the names of EMS variables to use with other functions
        self.var_names = mdp.get_ems_names(self.vars)
        self.meter_names = mdp.get_ems_names(self.meters)
        self.weather_names = mdp.get_ems_names(self.weather)
        self.actuator_names = mdp.get_ems_names(self.actuators)

        # simulation data state
        self.zn0_temp = None  # deg C
        self.time = None

        # print reporting
        self.print_every_x_hours = 2

    def observation_function(self):
        # -- FETCH/UPDATE SIMULATION DATA --
        # Get data from simulation at current timestep (and calling point)
        self.time = self.bca.get_ems_data(['t_datetimes'])

        var_data = self.bca.get_ems_data(self.var_names)
        meter_data = self.bca.get_ems_data(self.meter_names, return_dict=True)
        weather_data = self.bca.get_ems_data(self.weather_names, return_dict=True)  # just for example, other usage

        # Update our MdpManager and all MdpElements, returns same values
        # Automatically runs any encoding functions to update encoded values
        vars = self.mdp.update_ems_value(self.vars, var_data)  # outputs dict based on ordered list of names & values
        meters = self.mdp.update_ems_value_from_dict(meter_data)  # other usage, outputs same dict w/ dif input
        weather = self.mdp.update_ems_value_from_dict(weather_data)   # other usage, outputs same dict w/ dif input

        """
        Below, we show various redundant ways of looking at EMS values and encoded values. A variety of approaches are 
        provided for a variety of use-cases. Please inspect the usage and code to see what best suites your needs. 
        Note: not all usage examples are presented below.
        """
        # Get specific values from MdpManager based on name
        self.zn0_temp = self.mdp.get_mdp_element('zn0_temp').value
        # OR get directly from BcaEnv
        self.zn0_temp = self.bca.get_ems_data(['zn0_temp'])
        # OR directly from output
        self.zn0_temp = var_data[1]  # from BcaEnv list output
        self.zn0_temp = vars['zn0_temp']  # from MdpManager list output
        # outdoor air dry bulb temp
        outdoor_temp = weather_data['oa_db']  # from BcaEnv dict output
        outdoor_temp = weather['oa_db']  # from MdpManager dict output

        # use encoding function values to see temperature in Fahrenheit
        zn0_temp_f = self.mdp.ems_master_list['zn0_temp'].encoded_value  # access the Master list dictionary directly
        outdoor_temp_f = self.mdp.get_mdp_element('oa_db').encoded_value  # using helper function
        # OR call encoding function on multiple elements, even though encoded values are automatically up to date
        encoded_values_dict = self.mdp.get_ems_encoded_values(['oa_db', 'zn0_temp'])
        zn0_temp_f = encoded_values_dict['zn0_temp']
        outdoor_temp_f = encoded_values_dict['oa_db']

        # print reporting
        if self.time.hour % 2 == 0 and self.time.minute == 0:  # report every 2 hours
            print(f'\n\nTime: {str(self.time)}')
            print('\n\t* Observation Function:')
            print(f'\t\tVars: {var_data}\n\t\tMeters: {meter_data}\n\t\tWeather:{weather_data}')
            print(f'\t\tZone0 Temp: {round(self.zn0_temp,2)} C, {round(zn0_temp_f,2)} F')
            print(f'\t\tOutdoor Temp: {round(outdoor_temp, 2)} C, {round(outdoor_temp_f,2)} F')

    def actuation_function(self):
        work_hours_heating_setpoint = 18  # deg C
        work_hours_cooling_setpoint = 22  # deg C

        off_hours_heating_setpoint = 15  # deg C
        off_hours_cooilng_setpoint = 30  # deg C

        work_day_start = datetime.time(6, 0)  # day starts 6 am
        work_day_end = datetime.time(20, 0)  # day ends at 8 pm

        if work_day_start < self.time.time() < work_day_end:  #
            # during workday
            heating_setpoint = work_hours_heating_setpoint
            cooling_setpoint = work_hours_cooling_setpoint
            thermostat_settings = 'Work-Hours Thermostat'
        else:
            # off work
            heating_setpoint = off_hours_heating_setpoint
            cooling_setpoint = off_hours_cooilng_setpoint
            thermostat_settings = 'Off-Hours Thermostat'

        # print reporting
        if self.time.hour % self.print_every_x_hours == 0 and self.time.minute == 0:
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


# Create agent instance
my_agent = Agent(sim, my_mdp)

# Set your callback function (observation and/or actuation) function for a given calling point
sim.set_calling_point_and_callback_function(
    calling_point=calling_point_for_callback_fxns,
    observation_function=my_agent.observation_function,  # optional
    actuation_function=my_agent.actuation_function,  # optional
    update_state=True,
    update_observation_frequency=1,
    update_actuation_frequency=1
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





