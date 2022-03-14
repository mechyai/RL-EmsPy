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
calling_point_for_callback_fxn = EmsPy.available_calling_points[6]  # 5-15 valid for timestep loop during simulation
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
output_dfs = sim.get_df()  # LOOK at all the data collected here, custom DFs can be made too

# -- Plot Results --
fig, ax = plt.subplots()
output_dfs['var'].plot(y='zn0_temp', use_index=True, ax=ax)
output_dfs['weather'].plot(y='oa_db', use_index=True, ax=ax)
output_dfs['meter'].plot(y='electricity_HVAC', use_index=True, ax=ax, secondary_y=True)
output_dfs['actuator'].plot(y='zn0_heating_sp', use_index=True, ax=ax)
output_dfs['actuator'].plot(y='zn0_cooling_sp', use_index=True, ax=ax)
plt.title('Zn0 Temps and Thermostat Setpoint for Year')

# Analyze results in "out" folder, DView, or directly from your Python variables and Pandas Dataframes
