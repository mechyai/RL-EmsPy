import os
import matplotlib.pyplot as plt
import pandas as pd

import openstudio  # ver 3.2.0 !pip list

import emspy

# work arouund # TODO find reference to
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.

# insert the repo build tree or install path into the search Path, then import the EnergyPlus API
ep_path = 'A:/Programs/EnergyPlusV9-5-0/'
project_name = '/EmsPy/'
project_path = 'A:/Files/PycharmProjects/RL-BCA' + project_name

# ep_file_path = ''  # path to .idf file for simulation
ep_idf_to_run = project_path + 'test_CJE_act.idf'
ep_weather_path = ep_path + '/WeatherData/USA_CO_Golden-NREL.724666_TMY3.epw'

# TODO update to dict usage
# define EMS sensors and actuators to be used via 'Table of Contents'
# vars_tc = {"attr_handle_name": ["variable_type", "variable_key"],...}
# int_vars_tc = {"attr_handle_name": "variable_type", "variable_key"],...}
# meters_tc = {"attr_handle_name": "meter_name",...}
# actuators_tc = {"attr_handle_name": ["component_type", "control_type", "actuator_key"],...}
# weather_tc = {"attr_name": "weather_metric",...}

# create EMS Table of Contents (TC)
zone = 'Thermal Zone 1'
vars_tc = {'oa_temp': ['site outdoor air drybulb temperature', 'environment'],
           'zone_temp': ['zone mean air temperature', zone]}

int_vars_tc = None
meters_tc = None
# still not working
actuators_tc = {'act_odb_temp': ['weather data', 'outdoor dry bulb', 'environment']}
weather_tc = {'sun': 'sun_is_up', 'rain': 'is_raining', 'wind_dir': 'wind_direction',
              'out_rh': 'outdoor_relative_humidity', 'out_db_temp': 'outdoor_dry_bulb'}

# create calling point with actuation function and required callback fxn arguments
cp1 = 'callback_begin_zone_timestep_after_init_heat_balance'
cp2 = 'callback_begin_zone_timestep_before_set_current_weather'
cp3 = 'callback_begin_system_timestep_before_predictor'


ts = 12

agent = emspy.BcaEnv(ep_path, ep_idf_to_run, ts, vars_tc, int_vars_tc, meters_tc, actuators_tc, weather_tc)


class test_class:
    def __init__(self):
        self.param_data = 0
        self.weather_db_temp_val = 20


agent_params = test_class()


def actuation_fxn1():
    agent_params.param_data += 1
    print(agent_params.param_data)
    data = agent.get_ems_data(['wind_dir'], [0, 1, 2])
    print(f'Data: {data}')
    return None


def actuation_test():
    act_dbt = agent_params.weather_db_temp_val
    if act_dbt > 50:
        update_dbt = 10
    else:
        update_dbt = act_dbt + 1
    agent_params.weather_db_temp_val = update_dbt

    dbt, dbt_actuator, timestep = agent.get_ems_data(['oa_temp', 'act_odb_temp', 'timesteps'])

    print(f'A: Timestep:{timestep}, DBT Act: {update_dbt}, DBT Current Act: {dbt_actuator}, DBT Current: {dbt}')

    return {'act_odb_temp': update_dbt}


def read_data():
    dbt, dbt_actuator, timestep = agent.get_ems_data(['oa_temp', 'act_odb_temp', 'timesteps'])
    print(f'B: Timestep:{timestep}, DBT Act: "prev", DBT Current Act: {dbt_actuator}, DBT Current: {dbt}\n')

# agent.set_calling_point_and_actuation_function(cp1, actuation_fxn1, False, 1, 1)
agent.set_calling_point_and_callback_function(cp2, actuation_test, True)
# agent.set_calling_point_and_actuation_function(cp3, read_data, True)

# create custom dict
# agent.init_custom_dataframe_dict('df1', cp1, 4, ['act_odb_temp', 'sun'])
# agent.init_custom_dataframe_dict('df2', cp1, 2, ['rain', 'zone_temp'])

agent.run_env(ep_weather_path)
# agent.reset_state()

