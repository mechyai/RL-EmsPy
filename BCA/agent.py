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
                 learning_loop: int = 1):

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
        self.indoor_temp_unoccupied_range = np.array([15.6 + 0.5, 29.4 + 0.5])  # mimic night cycle manager, + 1/2 temp tolerance
        self.indoor_temp_limits = np.array([15, 30])  # ??? needed?

        # -- TIMING --
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

        # -- misc --
        self.learning = True
        self.learning_loop = learning_loop
        self.once = True

    def observe(self):
        # -- FETCH/UPDATE SIMULATION DATA --
        time = self.sim.get_ems_data(['t_datetimes'])
        vars = self.mdp.update_ems_value(self.vars, self.sim.get_ems_data(self.mdp.get_ems_names(self.vars)))
        # meters = mdp.update_ems_value(self.meters, self.sim.get_ems_data(mdp.get_ems_names(self.meters)))
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

        # -- REPORT --
        # self._report_time()  # time

        # -- DO ONCE --
        if self.once:
            self.state_var_names = list(vars.keys()) + list(weather.keys())
            self.once = False

        # -- REPORTING --
        print(f'\n\tReward: {round(self.reward, 2)}, Cumulative: {round(self.reward_sum, 2)}')

        # -- TRACK REWARD --
        return self.reward  # return reward for EmsPy pd.df tracking

    def act(self):
        """
        Action callback function:
        Takes action from network or exploration, then encodes into HVAC commands and passed into running simulation.

        :return: actuation dictionary - EMS variable name (key): actuation value (value)
        """

        if False: #mdp.ems_master_list['hvac_operation_sched'].value == 0:
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
                self.action = self.bdq.get_action(torch.Tensor(self.state_normalized).unsqueeze(1))
                action_type = 'Exploit'

        print(f'\n\tAction: {self.action} ({action_type}, eps = {self.epsilon})')

        # -- ENCODE ACTIONS TO HVAC COMMAND --
        action_cmd = {0: 'OFF', 1: 'HEAT', 2: 'COOL', None: 'Availability OFF'}
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

            print(f'\t\tZone{zone} ({action_cmd[action]}): Temp = {round(zone_temp,2)},'
                  f' Heating Sp = {round(heating_sp,2)},'
                  f' Cooling Sp = {round(cooling_sp,2)}')

        aux_actuation = {
            # Data Tracking
            'reward': self.reward,
            'reward_cumulative': self.reward_sum,
        }
        # combine
        self.actuation_dict.update(aux_actuation)

        return self.actuation_dict
        # return aux_actuation

    def _reward(self):
        """Reward function - per component, per zone."""

        # TODO add some sort of normalization and lambda

        n_zones = self.bdq.action_branches
        n_reward_components = 2
        reward_components_per_zone_dict = {f'zn{zone_i}': None for zone_i in range(n_zones)}

        if self.mdp.ems_master_list['hvac_operation_sched'].value == 0:
            # UNOCCUPIED TIMES
            temp_bounds = self.indoor_temp_unoccupied_range
        else:
            # OCCUPIED TIMES
            temp_bounds = self.indoor_temp_ideal_range

        zone_i = 0
        for zone, reward_list in reward_components_per_zone_dict.items():  # for all controlled zones

            reward_per_component = np.array([])

            # -- COMFORTABLE TEMPS --
            if True:
                zone_temp = self.mdp.ems_master_list[f'{zone}_temp'].value
                if all((temp_bounds - zone_temp) < 0) or all((temp_bounds - zone_temp) > 0):
                    # outside range - penalty
                    reward = -min(self.indoor_temp_ideal_range - zone_temp) ** 2
                else:
                    # inside range - no penalty
                    reward = 0
                reward_per_component = np.append(reward_per_component, reward)

            # -- DR, RTP $ --
            if self.action is not None:
                # TODO
                rtp = self.mdp.ems_master_list['rtp'].value
                # hvac_energy_use = self.mdp.ems_master_list[f'{zone}_heating_electricity'].value
                # hvac_energy_use += self.mdp.ems_master_list[f'{zone}_cooling_electricity'].value

                if int(self.action[zone_i]) != 0:
                    reward = -rtp
                else:
                    reward = 0
                reward_per_component = np.append(reward_per_component, reward)
                zone_i += 1

            # -- RENEWABLE ENERGY USAGE --
            # TODO
            if False:
                pass

            reward_components_per_zone_dict[zone] = reward_per_component


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

    def _report_daily(self):
        self.time = self.sim.get_ems_data('t_datetimes')
        if self.time.day != self.prev_day and self.time.hour == 1:
            self.day_update = True
            print(f'{self.time.strftime("%m/%d/%Y")} - Trial: {self.trial} - Reward Daily Sum: '
                  f'{self.reward_sum - self.prior_reward_sum:0.4f}')
            print(f'Elapsed Time: {(time.time() - self.tictoc)/60:0.2f} mins')
            # updates
            self.prior_reward_sum = self.reward_sum
            # update current/prev day
            self.prev_day = self.time.day