import pandas as pd

from emspy import EmsPy


class BcaEnv(EmsPy):
    """
    Building Control Agent & ENVironemnt (BcaEnv)

    This class represents the interface to the Building Control Agent (bca) and Environment for RL. It exposes a higher
    level of abstraction to its parent class, emspy, encapsulating more complex features.
    Users should be able to perform all necessary interactions for control algorithm experimentation mostly or
    completely through this class.
    """

    # Building Control Agent (bca) & Environment
    def __init__(self, ep_path: str,
                 ep_idf_to_run: str,
                 timesteps: int,
                 tc_vars: dict,
                 tc_intvars: dict,
                 tc_meters: dict,
                 tc_actuator: dict,
                 tc_weather: dict):

        """See emspy.__init__() documentation."""

        # follow same init procedure as parent class emspy
        super().__init__(ep_path, ep_idf_to_run, timesteps, tc_vars, tc_intvars, tc_meters, tc_actuator, tc_weather)
        self.ems_list_update_checked = False  # TODO get rid off, doesnt work with multiple method instances

    def set_calling_point_and_callback_function(self, calling_point: str,
                                                observation_function,
                                                actuation_function,
                                                update_state: bool,
                                                update_observation_frequency: int = 1,
                                                update_actuation_frequency: int = 1):
        """
        Sets connection for runtime calling points and custom callback function specification with defined arguments.

        This will be used to created user-defined callback functions, including an (optional) observation function,
        (optional) actuation function, state update condition, and state update and action update frequencies.
        This allows the user to create the conventional RL agent interaction -> get state -> take action, each with
        desired timestep frequencies of implementation.

        :param calling_point: the calling point at which the callback function will be called during simulation runtime
        :param observation_function: the user defined observation function to be called at runtime calling point and desired
        timestep frequency, to be used to gather state data for agent before taking actions.
        :param actuation_function: the user defined actuation function to be called at runtime calling point and desired
        timestep frequency, function must return dict of actuator names (key) and associated setpoint (value)
        :param update_state: whether EMS and time/timestep data should be updated from simulation for this calling point.
        :param update_observation_frequency: the number of zone timesteps per running the observation function
        :param update_actuation_frequency: the number of zone timesteps per updating the actuators from the actuation
        function
        """

        if update_actuation_frequency > update_observation_frequency:
            print(f'\n*WARNING: It is unusual to have your action update more frequent than your state update\n')
        if calling_point in self.calling_point_callback_dict:  # overwrite error
            raise Exception(
                f'ERROR: You have overwritten the calling point \'{calling_point}\'. Keep calling points unique.')
        else:
            self.calling_point_callback_dict[calling_point] = [observation_function, actuation_function, update_state,
                                                               update_observation_frequency, update_actuation_frequency]

    def _check_ems_metric_input(self, ems_metric):
        """Verifies user-input of EMS metric/type list is valid."""

        # catch EMS type names (var, intvar, actuator, etc.)
        if ems_metric in self.ems_num_dict:
            raise Exception(f'ERROR: EMS categories can only be called by themselves, please only call one at a '
                            f'time.')
        # catch invalid EMS metric names
        elif ems_metric not in self.ems_names_master_list and ems_metric not in self.available_timing_metrics:
            raise Exception(f'ERROR: The EMS/timing metric [{ems_metric}] is not valid. Please see your EMS ToCs'
                            f' or emspy.ems_master_list & emspy.times_master_list for available EMS & '
                            f'timing metrics')

    def get_ems_data(self, ems_metric_list: list, time_rev_index=0) -> list:
        """
        This takes desired EMS metric(s) (or type) & returns the entire current data set(s) OR at specific time indices.

        This function should be used to collect user-defined state space OR time information at each timestep. 1 to all
        EMS metrics and timing can be called, or just EMS category (var, intvar, meter, actuator, weather), and 1 data
        point to the entire current data set can be returned.
        It is likely that the user will want the most recent data point only.

        If calling any default timing data, see emspy.times_master_list for available timing data metrics.

        :param ems_metric_list: list of strings (or single element) of any available EMS/timing metric(s) to be called,
        or ONLY ONE entire EMS category ('var', 'intvar', 'meter', 'actuator', 'weather', 'time')
        :param time_rev_index: list (or single value) of timestep indexes, applied to all EMS/timing metrics starting
        from index 0 as most recent available data point. An empty list [] will return the entire current data list for
        each metric.
        :return return_data_list: nested list of data for each EMS metric at each time index specified, or entire list
        """

        # handle single val inputs -> convert to list for rest of function
        single_val = False
        single_metric = False
        # metrics
        if type(ems_metric_list) is not list:  # assuming single metric
            ems_metric_list = [ems_metric_list]
        if len(ems_metric_list) == 1:
            single_metric = True
        # time indexes
        if type(time_rev_index) is not list and type(time_rev_index) is not range:  # assuming single time
            time_rev_index = [time_rev_index]
        if len(time_rev_index) == 1:
            single_val = True

        return_data_list = []

        # check if only EMS category called
        ems_type = ems_metric_list[0]
        if ems_type in self.ems_num_dict and single_metric:
            # reassign entire EMS type list
            if ems_type == 'time':
                ems_metric_list = self.available_timing_metrics
            else:
                ems_metric_list = list(getattr(self, 'tc_' + ems_metric_list[0]).keys())
            if len(ems_metric_list) > 1:
                single_metric = False

        for ems_metric in ems_metric_list:
            # verify valid input #TODO do once
            self._check_ems_metric_input(ems_metric)
            ems_type = self._get_ems_type(ems_metric)  # for attribute variable name
            # no time index specified, return ALL current data
            if not time_rev_index:
                return_data_list.append(getattr(self, 'data_' + ems_type + '_' + ems_metric))
            else:
                return_data_indexed = []
                # iterate through previous time indexes
                for time in time_rev_index:
                    if ems_type != 'time':
                        ems_name = 'data_' + ems_type + '_' + ems_metric
                    else:
                        ems_name = ems_metric
                    try:
                        data_indexed = getattr(self, ems_name)[-1 - time]
                        # so that a single-element nested list is not returned
                        if single_val:
                            return_data_indexed = data_indexed
                        else:
                            return_data_indexed.append(data_indexed)
                    except IndexError:
                        print('\n*NOTE: Not enough simulation time elapsed to collect data at specified index.\n')
                # no unnecessarily nested lists
                if single_metric:
                    return return_data_indexed
                else:
                    return_data_list.append(return_data_indexed)
        return return_data_list

    def get_weather_forecast(self, weather_metrics: list, when: str, hour: int, zone_ts: int):
        """
        Fetches given weather metric from today/tomorrow for a given hour of the day and timestep within that hour.

        :param weather_metrics: list of desired weather metric(s) (1 to all) from weather ToC dict
        :param when: 'today' or 'tomorrow' relative to current timestep
        :param hour: hour of day
        :param zone_ts: timestep of hour
        """

        return self._get_weather(weather_metrics, when, hour, zone_ts)

    def update_ems_data(self, ems_metric_list: list, return_data: bool):
        """
        This takes desired EMS metric(s) (or type) to update from the sim (and opt return val) at calliing point.

        This OPTIONAL function can be used to update/collect specific EMS at each timestep for a given calling point.
        One to all EMS metrics can be called, or just EMS category (var, intvar, meter, actuator, weather) and have the
        data point updated & returned. This is ONLY NEEDED if the user wants to update specific EMS metrics at a unique
        calling point separate from ALL EMS metrics if using default state update.

        This also works for default timing data.  # TODO does not work currently with _update_ems_and_weather_val

        :param ems_metric_list: list of any available EMS metric(s) to be called, or ONLY ONE entire EMS category
        (var, intvar, meter, actuator, weather) in a list
        :param return_data: Whether or not to return an order list of the data points fetched
        :return return_data_list: list of fetched data for each EMS metric at each time index specified or None
        """

        # if only EMS category called
        if ems_metric_list[0] in self.ems_num_dict and len(ems_metric_list) == 1:
            ems_metric_list = list(getattr(self, 'tc_' + ems_metric_list[0]).keys())
        else:  # TODO get rid off, doesnt work with multiple method instances
            for ems_metric in ems_metric_list:
                if not self.ems_list_update_checked:
                    self._check_ems_metric_input(ems_metric)
                    self.ems_list_update_checked = True

        self._update_ems_and_weather_vals(ems_metric_list)
        if return_data:
            return self.get_ems_data(ems_metric_list)
        else:
            return None

    def init_custom_dataframe_dict(self, df_name: str, calling_point: str, update_freq: int, ems_metrics: list):
        """
        Initialize custom EMS metric dataframes attributes at specific calling points & frequencies to be tracked.

        Desired setpoint data from actuation actions can be acquired and compared to updated system setpoints - Use
        'setpoint' + your_actuator_name as the EMS metric name. Rewards can also be fetched. These are NOT collected
        by default dataframes.

        :param df_name: user-defined df variable name
        :param calling_point: the calling point at which the df should be updated
        :param update_freq: how often data will be posted, it will be posted every X timesteps
        :param ems_metrics: list of EMS metric names, 'setpoint+...', or 'rewards', to store their data points in df
        """
        self.df_count += 1
        self.df_custom_dict[df_name] = [ems_metrics, calling_point, update_freq]

    def track_standard_dfs(self, track: bool = False):
        """Only necessary if you don't want to track standard DFs, otherwise they will be tracked automatically."""
        if not track:
            self.default_dfs_tracked = False

    def get_df(self, df_names: list = None, to_csv_file: str = None):
        """
        Returns selected EMS-type default dataframe based on user's entered ToC(s) or custom DF, or ALL df's by default.

        :param df_names: default EMS metric type (var, intvar, meter, actuator, weather) OR custom df name. Leave
        argument empty if you want to return ALL dataframes together (all default, then all custom)
        :param to_csv_file: path/file name you want the dataframe to be written to
        :return: (concatenation of) pandas dataframes in order of entry or [vars, intvars, meters, weather, actuator] by
        default.
        """
        if not self.calling_point_callback_dict:
            raise Exception('ERROR: There is no dataframe data to collect and return, please specific calling point(s)'
                            ' first.')
        if self.simulation_success != 0:
            raise Exception('ERROR: Simulation must be run successfully first to fetch data. See EnergyPlus error file,'
                            ' eplusout.err')

        if df_names is None:
            df_names = []
        if to_csv_file is None:
            to_csv_file = ''

        all_df = pd.DataFrame()  # merge all into 1 df
        return_df = {}

        # handle DEFAULT dfs
        df_default_names = list(self.ems_num_dict.keys()) + ['reward'] if self.rewards else self.ems_num_dict.keys()

        for df_name in df_default_names:  # iterate thru available EMS types
            if df_name in df_names or not df_names:  # specific or ALL dfs
                df = (getattr(self, 'df_' + df_name))  # create df for EMS type
                return_df[df_name] = df

                # create complete DF of all default vars with only 1 set of time/index columns
                if all_df.empty:
                    all_df = df.copy(deep=True)
                else:
                    if df_name == 'reward' and len(self.rewards) != self.t_datetimes:
                        # include reward to ALL DFs only if its the same size
                        print('*NOTE: Rewards DF will not be included on ALL DF as it is not the same size.')
                    else:
                        all_df = pd.merge(all_df, df, on=['Datetime', 'Timestep', 'Calling Point'])

                # remove from list since accounted for
                if df_name in df_names:
                    df_names.remove(df_name)

        # handle CUSTOM dfs
        for df_name in self.df_custom_dict:
            if df_name in df_names or not df_names:
                df = (getattr(self, df_name))
                return_df[df_name] = df
                if all_df.empty:
                    all_df = df.copy(deep=True)
                else:  # TODO verify robustness of merging of custom df with default, can it be compressed for same time indexes
                    all_df = pd.concat([all_df, df], axis=1)
                    # TODO determine why custom dfs do not add to all_df well, num of indexes is wrong
                if df_name in df_names:
                    df_names.remove(df_name)

        # leftover dfs not fetched and returned
        if df_names:
            raise ValueError(f'ERROR: Either dataframe custom name or default type: {df_names} is not valid or was not'
                             ' collected during simulation.')
        else:
            if to_csv_file:
                # write DFs to file
                all_df.to_csv(to_csv_file, index=False)
            return_df['all'] = all_df

            return return_df

    def run_env(self, weather_file_path: str):
        """Runs E+ simulation for given .IDF building model and EPW Weather File"""

        self.run_simulation(weather_file_path)
