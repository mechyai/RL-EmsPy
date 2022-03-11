import time
import numpy as np

import torch

from BCA import bdq
from BCA import mdpmanager

from EmsPy import emspy


class Agent:
    def __init__(self,
                 sim: emspy.BcaEnv,
                 mdp: mdpmanager.MdpManager,
                 dqn_model: bdq.BranchingDQN,
                 policy: bdq.EpsilonGreedyStrategy,
                 replay_memory: bdq.ReplayMemory,
                 interaction_frequency: int,
                 learning_loop: int = 1
                 ):

        # -- SIMULATION STATES --
        self.sim = sim
        self.mdp = mdp
        # Get list of EMS objects
        self.vars = mdp.ems_type_dict['var']
        self.meters = mdp.ems_type_dict['meter']
        self.weather = mdp.ems_type_dict['weather']

        # -- STATE SPACE --
        self.state_var_names = {}
        self.termination = 0
        self.state_normalized = None
        self.next_state_normalized = None

        # -- ACTION SPACE --
        self.action = None
        self.actuation_dict = {}
        self.epsilon = policy.start
        self.fixed_epsilon = None  # optional fixed exploration rate
        self.greedy_epsilon = policy
        # misc
        self.temp_deadband = 5  # distance between heating and cooling setpoints
        self.temp_buffer = 3  # new setpoint distance from current temps

        # -- REWARD --
        self.reward_dict = None
        self.reward = 0
        self.reward_sum = 0

        # -- CONTROL GOALS --
        self.indoor_temp_ideal_range = np.array([21.1, 23.89])  # occupied hours, based on OS model
        self.indoor_temp_unoccupied_range = np.array(
            [15.6 - 0.5, 29.4 + 0.5])  # mimic night cycle manager, + 1/2 temp tolerance
        self.indoor_temp_limits = np.array([15, 30])  # ??? needed?

        # -- TIMING --
        self.interaction_frequency = interaction_frequency
        self.n_ts = 0
        self.current_step = 0

        # -- INTERACTION FREQUENCIES --
        self.observation_ts = 15  # how often agent will observe state & keep fixed action - off-policy
        self.action_ts = 15  # how often agent will observe state & act - on-policy
        self.action_delay = 15  # how many ts will agent be fixed at beginning of simulation

        # -- REPLAY MEMORY --
        self.memory = replay_memory

        # -- BDQ --
        self.bdq = dqn_model

        # -- PERFORMANCE RESULTS --
        self.comfort_dissatisfaction = 0
        self.hvac_rtp_costs = 0
        self.comfort_dissatisfaction_total = 0
        self.hvac_rtp_costs_total = 0

        # -- Misc. --
        self.learning = True
        self.learning_loop = learning_loop
        self.once = True

    def observe(self):
        # SKIP FIRST TIMESTEP ???
        if self.once:
            self.once = False
            return 0

        # -- FETCH/UPDATE SIMULATION DATA --
        time = self.sim.get_ems_data(['t_datetimes'])
        vars = self.mdp.update_ems_value(self.vars, self.sim.get_ems_data(self.mdp.get_ems_names(self.vars)))
        meters = self.mdp.update_ems_value(self.meters, self.sim.get_ems_data(self.mdp.get_ems_names(self.meters)))
        weather = self.mdp.update_ems_value(self.weather, self.sim.get_ems_data(self.mdp.get_ems_names(self.weather)))

        print(f'\n\n{str(time)}')  # \n\n\tVars: {vars}\n\tMeters: {meters}\n\tWeather: {weather}')

        # -- ENCODING --
        self.next_state_normalized = np.array(list(vars.values()) + list(weather.values()), dtype=float)

        # -- ENCODED STATE --
        self.termination = self._is_terminal()

        # -- REWARD --
        self.reward_dict = self._reward()
        self.reward = self._get_total_reward('mean')  # aggregate 'mean' or 'sum'

        # -- STORE INTERACTIONS --
        if self.action is not None:  # after first action, enough data available
            self.memory.push(self.state_normalized, self.action, self.next_state_normalized,
                             self.reward, self.termination)  # <S, A, S', R, t> - push experience to Replay Memory

        # -- LEARN BATCH --
        if self.learning:
            if self.memory.can_provide_sample():  # must have enough interactions stored
                for i in range(self.learning_loop):
                    batch = self.memory.sample()
                    self.bdq.update_policy(batch)  # batch learning

        # -- UPDATE DATA --
        self.state_normalized = self.next_state_normalized
        self.reward_sum += self.reward
        self.current_step += 1

        # -- PERFORMANCE RESULTS --
        self.comfort_dissatisfaction = self._get_comfort_results()
        self.hvac_rtp_costs = self._get_rtp_hvac_cost_results()
        self.comfort_dissatisfaction += self.comfort_dissatisfaction
        self.hvac_rtp_costs_total += self.hvac_rtp_costs

        # -- DO ONCE --
        # if self.once:
        #     self.state_var_names = list(vars.keys()) + list(weather.keys())
        #     self.once = False

        # -- REPORTING --
        # self._report_time()  # time
        print(f'\n\tReward: {round(self.reward, 2)}, Cumulative: {round(self.reward_sum, 2)}')

        # -- TRACK REWARD --
        return self.reward  # return reward for EmsPy pd.df tracking

    def act_heat_cool_off(self):
        """
        Action callback function:
        Takes action from network or exploration, then encodes into HVAC commands and passed into running simulation.

        :return: actuation dictionary - EMS variable name (key): actuation value (value)
        """

        if False:  # mdp.ems_master_list['hvac_operation_sched'].value == 0:
            self.action = [None] * self.bdq.action_branches  # return actuation to E+
            action_type = 'Availability OFF'

        # -- EXPLOITATION vs EXPLORATION --
        else:
            self.epsilon = self.greedy_epsilon.get_exploration_rate(self.current_step, self.fixed_epsilon)
            if np.random.random() < self.epsilon:
                # Explore
                self.action = np.random.randint(0, 3, self.bdq.action_branches)
                action_type = 'Explore'
            else:
                # Exploit
                self.action = self.bdq.get_greedy_action(torch.Tensor(self.state_normalized).unsqueeze(1))
                action_type = 'Exploit'

        print(f'\n\tAction: {self.action} ({action_type}, eps = {self.epsilon})')

        # -- ENCODE ACTIONS TO HVAC COMMAND --
        action_cmd_print = {0: 'OFF', 1: 'HEAT', 2: 'COOL', None: 'Availability OFF'}
        for zone, action in enumerate(self.action):
            zone_temp = self.mdp.ems_master_list[f'zn{zone}_temp'].value

            if all((self.indoor_temp_limits - zone_temp) < 0) or all((self.indoor_temp_ideal_range - zone_temp) > 0):
                # outside safe comfortable bounds
                # print('unsafe temps')
                pass

            # adjust thermostat setpoints accordingly
            if action == 0:
                # OFF
                heating_sp = zone_temp - self.temp_deadband / 2
                cooling_sp = zone_temp + self.temp_deadband / 2
            elif action == 1:
                # HEAT
                heating_sp = zone_temp + self.temp_buffer
                cooling_sp = zone_temp + self.temp_buffer + self.temp_deadband
            elif action == 2:
                # COOL
                heating_sp = zone_temp - self.temp_buffer - self.temp_deadband
                cooling_sp = zone_temp - self.temp_buffer
            else:
                # HVAC Availability OFF
                heating_sp = action  # None
                cooling_sp = action  # None

            self.actuation_dict[f'zn{zone}_heating_sp'] = heating_sp
            self.actuation_dict[f'zn{zone}_cooling_sp'] = cooling_sp

            print(f'\t\tZone{zone} ({action_cmd_print[action]}): Temp = {round(zone_temp, 2)},'
                  f' Heating Sp = {round(heating_sp, 2)},'
                  f' Cooling Sp = {round(cooling_sp, 2)}')

        aux_actuation = {
            # Data Tracking
            'reward': self.reward,
            'reward_cumulative': self.reward_sum,
        }
        # combine
        self.actuation_dict.update(aux_actuation)

        return self.actuation_dict

    def act_strict_setpoints(self):
        # -- EXPLOITATION vs EXPLORATION --
        self.epsilon = self.greedy_epsilon.get_exploration_rate(self.current_step, self.fixed_epsilon)
        if np.random.random() < self.epsilon:
            # Explore
            self.action = np.random.randint(0, self.bdq.action_dim, self.bdq.action_branches)
            action_type = 'Explore'
        else:
            # Exploit
            self.action = self.bdq.get_greedy_action(torch.Tensor(self.state_normalized).unsqueeze(1))
            action_type = 'Exploit'

        print(f'\n\tAction: {self.action} ({action_type}, eps = {self.epsilon})')

        # -- ENCODE ACTIONS TO THERMOSTAT SETPOINTS --
        action_cmd_print = {0: 'LOWEST', 1: 'LOWER', 2: 'IDEAL', 3: 'HIGHER', 4: 'HIGHEST'}

        for zone_i, action in enumerate(self.action):
            zone_temp = self.mdp.ems_master_list[f'zn{zone_i}_temp'].value

            actuation_cmd_dict = {
                0: [15.1, 18.1],  # LOWEST
                1: [18.1, 21.1],  # LOWER
                2: [21.1, 23.9],  # IDEAL*
                3: [23.9, 26.9],  # HIGHER
                4: [26.9, 29.9]  # HIGHEST
            }

            heating_sp = actuation_cmd_dict[action][0]
            cooling_sp = actuation_cmd_dict[action][1]

            self.actuation_dict[f'zn{zone_i}_heating_sp'] = heating_sp
            self.actuation_dict[f'zn{zone_i}_cooling_sp'] = cooling_sp

            print(f'\t\tZone{zone_i} ({action_cmd_print[action]}): Temp = {round(zone_temp, 2)},'
                  f' Heating Sp = {round(heating_sp, 2)},'
                  f' Cooling Sp = {round(cooling_sp, 2)}')

        aux_actuation = {
            # Data Tracking
            'reward': self.reward,
            'reward_cumulative': self.reward_sum,
        }

        # combine actuations
        self.actuation_dict.update(aux_actuation)

        return self.actuation_dict

    def _reward(self):
        """Reward function - per component, per zone."""
        # TODO add some sort of normalization and lambda

        n_zones = self.bdq.action_branches
        reward_components_per_zone_dict = {f'zn{zone_i}': None for zone_i in range(n_zones)}

        # -- GET DATA SINCE LAST INTERACTION --
        # interaction_frequency = min(self.interaction_frequency, self.current_step)
        interaction_span = range(self.interaction_frequency)

        # COMFORT
        temp_schedule = self.sim.get_ems_data(['hvac_operation_sched'], interaction_span)
        # if not isinstance(temp_schedule, list):
        #     temp_schedule = [temp_schedule]  # handle edge case at beginning of sim
        temp_bounds = []
        for schedule_value in temp_schedule:
            if schedule_value == 1:  # during operating hours
                temp_bounds.append(self.indoor_temp_ideal_range)
            else:
                temp_bounds.append(self.indoor_temp_unoccupied_range)
        temp_bounds = np.asarray(temp_bounds)

        # $RTP
        rtp_since_last_interaction = np.asarray(self.sim.get_ems_data(['rtp'], interaction_span))

        # Per Controlled Zone
        for zone_i, reward_list in reward_components_per_zone_dict.items():
            reward_per_component = np.array([])

            # -- COMFORTABLE TEMPS --
            """
            For each zone, and array of minute interactions, each temperature is compared with the comfortable
            temperature bounds for the given timestep. If temperatures are out of bounds, the (-) MSE of that
            temperature from the nearest comfortable bound will be accounted for the reward.
            """
            zone_temps_since_last_interaction = np.asarray(
                self.sim.get_ems_data([f'{zone_i}_temp'], interaction_span))

            too_cold_temps = np.multiply(zone_temps_since_last_interaction < temp_bounds[:, 0],
                                         zone_temps_since_last_interaction)
            temp_bounds_cold = temp_bounds[too_cold_temps != 0]
            too_cold_temps = too_cold_temps[too_cold_temps != 0]  # only cold temps left


            too_warm_temps = np.multiply(zone_temps_since_last_interaction > temp_bounds[:, 1],
                                         zone_temps_since_last_interaction)
            temp_bounds_warm = temp_bounds[too_warm_temps != 0]
            too_warm_temps = too_warm_temps[too_warm_temps != 0]  # only warm temps left


            # MSE penalty for temps above and below comfortable bounds
            reward = - ((too_cold_temps - temp_bounds_cold[:, 0]) ** 2).sum() \
                     - ((too_warm_temps - temp_bounds_warm[:, 1]) ** 2).sum()

            reward_per_component = np.append(reward_per_component, reward)

            # -- DR, RTP $ --
            """
            For each zone, and array of minute interactions, heating and cooling energy will be compared to see if HVAC
            heating, cooling, or Off actions occurred per each timestep. If heating or cooling occurs, the -$RTP for the
            timestep will be accounted for. Note, that this is not multiplied by the energy used, such that this reward
            is agnostic to the zone size and incident load. 
            """
            heating_electricity_since_last_interaction = np.asarray(
                self.sim.get_ems_data([f'{zone_i}_heating_electricity'], interaction_span)
            )
            heating_gas_since_last_interaction = np.asarray(
                self.sim.get_ems_data([f'{zone_i}_heating_gas'], interaction_span)
            )
            heating_energy = heating_gas_since_last_interaction + heating_electricity_since_last_interaction

            cooling_energy = np.asarray(
                self.sim.get_ems_data([f'{zone_i}_cooling_electricity'], interaction_span)
            )

            # timestep-wise RTP cost, not accounting for energy-usage, only that energy was used
            cooling_factor = 1
            heating_factor = 1
            cooling_timesteps_cost = - cooling_factor * np.multiply(cooling_energy > heating_energy,
                                                                    rtp_since_last_interaction)
            heating_timesteps_cost = - heating_factor * np.multiply(heating_energy > cooling_energy,
                                                                    rtp_since_last_interaction)
            reward = (cooling_timesteps_cost + heating_timesteps_cost).sum()

            reward_per_component = np.append(reward_per_component, reward)

            # -- RENEWABLE ENERGY USAGE --
            """
            TODO
            """

            # Reward Components per Zone
            reward_components_per_zone_dict[zone_i] = reward_per_component

        return reward_components_per_zone_dict

    def _get_total_reward(self, aggregate_type: str):
        """Aggregates value from reward dict organized by zones and reward components"""
        if aggregate_type == 'sum':
            return np.array(list(self.reward_dict.values())).sum()
        elif aggregate_type == 'mean':
            return np.array(list(self.reward_dict.values())).mean()

    def _is_terminal(self):
        """Determines whether the current state is a terminal state or not. Dictates TD update values."""
        return 0

    def _get_comfort_results(self):
        """
        For each timestep, and each zone, we calculate the weighted sum of temps outside the comfortable bounds.
        This represents a comfort compliance metric.
        :return: comfort compliance metric. A value of dissatisfaction. 0 is optimal.
        """

        n_zones = self.bdq.action_branches
        interaction_span = range(self.interaction_frequency)
        controlled_zone_names = [f'zn{zone_i}' for zone_i in range(n_zones)]

        # Temp Bounds
        temp_schedule = np.asarray(
            self.sim.get_ems_data(['hvac_operation_sched'], interaction_span)
        )
        temp_bounds = []
        for schedule_value in temp_schedule:
            if schedule_value == 1:  # during operating hours
                temp_bounds.append(self.indoor_temp_ideal_range)
            else:
                temp_bounds.append(self.indoor_temp_unoccupied_range)
        temp_bounds = np.asarray(temp_bounds)

        # Per Controlled Zone
        uncomfortable_metric = 0
        for zone_i in controlled_zone_names:
            # -- COMFORTABLE TEMPS --
            """
            For each zone, and array of minute interactions, each temperature is compared with the comfortable
            temperature bounds for the given timestep. If temperatures are out of bounds, the (-) MSE of that
            temperature from the nearest comfortable bound will be accounted for the reward.
            """
            zone_temps_since_last_interaction = np.asarray(
                self.sim.get_ems_data([f'{zone_i}_temp'], interaction_span))

            too_cold_temps = np.multiply(zone_temps_since_last_interaction < temp_bounds[:, 0],
                                         zone_temps_since_last_interaction)
            temp_bounds_cold = temp_bounds[too_cold_temps != 0]
            too_cold_temps = too_cold_temps[too_cold_temps != 0]  # only cold temps left

            too_warm_temps = np.multiply(zone_temps_since_last_interaction > temp_bounds[:, 1],
                                         zone_temps_since_last_interaction)
            temp_bounds_warm = temp_bounds[too_warm_temps != 0]
            too_warm_temps = too_warm_temps[too_warm_temps != 0]  # only warm temps left

            # MSE penalty for temps above and below comfortable bounds
            uncomfortable_metric += ((too_cold_temps - temp_bounds_cold[:, 0]) ** 2).sum() + \
                         ((too_warm_temps - temp_bounds_warm[:, 1]) ** 2).sum()

        return uncomfortable_metric

    def _get_rtp_hvac_cost_results(self):
        """
        For each timestep, and each zone, we calculate the cost of HVAC electricity use based on RTP.
        This represents a DR compliance and monetary cost metric.
        :return: monetary cost of HVAC per interaction span metric. $0 is optimal.
        """

        n_zones = self.bdq.action_branches
        interaction_span = range(self.interaction_frequency)
        controlled_zone_names = [f'zn{zone_i}' for zone_i in range(n_zones)]

        # RTP of last X timesteps
        rtp_since_last_interaction = np.asarray(self.sim.get_ems_data(['rtp'], interaction_span))

        # Per Controlled Zone
        rtp_hvac_costs = 0
        for zone_i in controlled_zone_names:
            # -- DR, RTP $ --
            """
            For each zone, and array of minute interactions, heating and cooling energy will be compared to see if HVAC
            heating, cooling, or Off actions occurred per each timestep. If heating or cooling occurs, the -$RTP for the
            timestep will be accounted for proportional to the energy used.
            """

            heating_electricity = np.asarray(
                self.sim.get_ems_data([f'{zone_i}_heating_electricity'], interaction_span)
            )

            cooling_electricity = np.asarray(
                self.sim.get_ems_data([f'{zone_i}_cooling_electricity'], interaction_span)
            )

            joules_to_MWh = 2.77778e-10
            total_hvac_electricity = (heating_electricity + cooling_electricity) * joules_to_MWh

            # timestep-wise RTP cost, accounting for HVAC electricity usage
            hvac_electricity_costs = np.multiply(total_hvac_electricity, rtp_since_last_interaction)
            rtp_hvac_costs += hvac_electricity_costs.sum()

        return rtp_hvac_costs

    def _report_daily(self):
        self.time = self.sim.get_ems_data('t_datetimes')
        if self.time.day != self.prev_day and self.time.hour == 1:
            self.day_update = True
            print(f'{self.time.strftime("%m/%d/%Y")} - Trial: {self.trial} - Reward Daily Sum: '
                  f'{self.reward_sum - self.prior_reward_sum:0.4f}')
            print(f'Elapsed Time: {(time.time() - self.tictoc) / 60:0.2f} mins')
            # updates
            self.prior_reward_sum = self.reward_sum
            # update current/prev day
            self.prev_day = self.time.day
