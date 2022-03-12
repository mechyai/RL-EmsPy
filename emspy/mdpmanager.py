"""
Classes aimed at helping managing all the EMS variables used during simulations to track data and control systems.
MDP = Markov Decision Process, and reference to Reinforcement Learning
"""

from collections.abc import Sequence
from typing import Callable


class MdpElement:

    def __init__(self,
                 ems_type: str,
                 ems_element_name: str,
                 ems_handle_identifiers,
                 encoding_fxn: Callable = None,
                 *args):
        """
        Creates EMS element given its EMS type, variable name, handle identifiers, and optionally an encoding
        function and its optional arguments.

        :param ems_type: EMS type: ['var','intvar','meter','actuator','weather']
        :param ems_element_name: Variable name of EMS element
        :param ems_handle_identifiers: EMS element's handle identifiers used to fetch handle ID at start of simulation
        :param encoding_fxn: Function to run when encoding the true values
        :param args: Args of the encoding_fxn. DON'T included the EMS element value, that is included explicitly
                     elsewhere and MUST be first argument of encoding_fxn when written. IF any arg is to be determined
                     elsewhere manually (say, at runtime) then use NONE for that argument.
        """
        # meta data
        self.ems_type = ems_type
        self.name = ems_element_name
        self.handle_identifiers = ems_handle_identifiers
        # data
        self.value = None
        self.encoded_value = None
        self.encoding_fxn = encoding_fxn
        self.encoding_fxn_args = [*args]

    def set_encoded_value(self, element_name: str, encoded_value):
        """Simply sets encoded value attribute for given EMS element. Done because of constant reuse. Returns value."""
        setattr(self, element_name + '_encoded_value', encoded_value)
        return encoded_value

    def set_value(self, element_name: str, value):
        """Returns encoded value for given EMS element."""
        setattr(self, element_name + '_value', value)
        return value

    def set_encoding_fxn_args(self, element_name: str, *args):
        """Sets encoding function arguments for given EMS element."""
        setattr(self, element_name + '_encoding_fxn_args', *args)

    def set_encoding_fxn(self, element_name: str, encoding_fxn):
        """Sets encoding function for given EMS element."""
        setattr(self, element_name + '_encoding_fxn', encoding_fxn)

    def get_encoded_value(self, element_name: str):
        """Returns encoded value for given EMS element."""
        return getattr(self, element_name + '_encoded_value')

    def get_value(self, element_name: str):
        """Returns value for given EMS element."""
        return getattr(self, element_name + '_value')

    def get_encoding_fxn_args(self, element_name: str):
        """Returns encoding function arguments for given EMS element."""
        return getattr(self, element_name + '_encoding_fxn_args')

    def get_encoding_fxn(self, element_name: str):
        """Returns encoding function for given EMS element."""
        return getattr(self, element_name + '_encoding_fxn')


class MdpManager:
    EMS_types = ('var', 'intvar', 'meter', 'weather', 'actuator')

    def __init__(self):
        """
        A manager class for all the created EMS element objects. Each MDP element added creates a new MdpElement
        instance and attribute referencing that MdpElement obj.
        """
        # store EMS element variable names and handle IDs
        self.tc_var = {}
        self.tc_intvar = {}
        self.tc_meter = {}
        self.tc_weather = {}
        self.tc_actuator = {}

        self.ems_type_dict = {'var': [], 'intvar': [], 'meter': [], 'weather': [], 'actuator': []}
        self.ems_master_list = {}  # stores list of ALL MdpElemnts used

    def add_ems_element(self,
                        ems_type: str,
                        ems_element_name: str,
                        ems_handle_identifiers,
                        encoding_fxn: Callable = None,
                        *args):
        """
        Creates EMS element given its EMS type, variable name, handle identifiers, and optionally an encoding
        function and its optional arguments. The adds this element to the proper MdpManager lists

        :param ems_type: EMS type: ['var','intvar','meter','actuator','weather']
        :param ems_element_name: Variable name of EMS element
        :param ems_handle_identifiers: EMS element's handle identifiers used to fetch handle ID at start of simulation
        :param encoding_fxn: Function to run when encoding the true values
        :param args: Args of the encoding_fxn. DON'T included the EMS element value, that is included explicitly
                     elsewhere and MUST be first argument of encoding_fxn when written. IF any arg is to be determined
                     elsewhere manually (say, at runtime) then use NONE for that argument.
        """
        # input checking
        if ems_type not in self.EMS_types:
            raise ValueError(f'EMS Type must be in {self.EMS_types}')

        ems_obj_name = ems_type + '_' + ems_element_name
        setattr(self, ems_obj_name, MdpElement(ems_type,
                                               ems_element_name,
                                               ems_handle_identifiers,
                                               encoding_fxn,
                                               *args))

        # add to existing attributes
        getattr(self, 'tc_' + ems_type)[ems_element_name] = ems_handle_identifiers  # add to handle dict
        ems_obj = getattr(self, ems_obj_name)
        self.ems_type_dict[ems_type].append(ems_obj)
        self.ems_master_list[ems_element_name] = ems_obj

    def get_ems_names(self, ems_objects: [MdpElement] = None) -> list:
        """From list of MdpElement objects, their names are returned in order."""

        if None:
            # All MdpElements
            ems_objects = self.ems_master_list.values()

        names_list = []
        for ems_obj in ems_objects:
            names_list.append(ems_obj.name)

        return names_list

    def update_ems_value(self, ems_objects: [MdpElement], ems_values: Sequence) -> dict:
        """
        Takes list of EMS objects and their values, stores all data, and returns encoded values in order of name list.

        :param ems_objects: List of EMS name(s) that are tracked
        :param ems_values: List of same EMS value(s) gathered from simulation runtime, in same order as names
        :return: List of encoded values, in same order as names. If NO encoding function available, returns just value.
        """
        if len(ems_objects) != len(ems_values):
            raise (f'EMS element names {ems_objects} and their values {ems_values} are out of sync. They '
                   f'do not consist of the same number of values. They must align, element-wise.')

        encoded_values = {}
        for i, ems_obj in enumerate(ems_objects):
            # update current value
            val = ems_values[i]
            ems_obj.value = val
            if ems_obj.encoding_fxn is not None:
                # if encoding function exists
                val = self.run_encoding_fxn(ems_obj, val)
            # return encoded val, or if not encoding return normal val
            encoded_values[ems_obj.name] = val  # using ordered dict

        return encoded_values

    @staticmethod
    def run_encoding_fxn(ems_object: MdpElement, value: float = None):
        """This carefully runs the encoding function for a given EMS obj and value, returning the encoded value."""

        # update encoded value
        args = ems_object.encoding_fxn_args
        if None in args:  # this handles encodings that must be handled manually in runtime
            return None
        if value is None:
            # use stored value if not given from update
            value = ems_object.value

        # run encoding fxn
        if args:  # if args exist
            encoded_value = ems_object.encoding_fxn(value, *args)
        else:  # if no args exist, empty
            encoded_value = ems_object.encoding_fxn(value)
        # obj update
        ems_object.encoded_value = encoded_value
        return encoded_value

    def get_mdp_element_from_name(self, ems_name: str) -> MdpElement:
        """Looks up and returns EMS object instance from its name."""
        return self.ems_master_list[ems_name]

    def read_ems_values(self, ems_objects: [MdpElement]):
        """Iterates through list of EMS objects (or names) and returns fetched values dictionary."""

        values_dict = {}
        for ems_obj in ems_objects:
            if type(ems_obj) == str:
                ems_obj = self.get_mdp_element_from_name(ems_obj)
            values_dict[ems_obj.name] = ems_obj.value

        return values_dict

    def read_ems_encoded_values(self, ems_objects: [MdpElement]):
        """
        Iterates through list of EMS objects and returns fetched encoded values dictionary.

        IF the value is None, the encoding fxn will be reran before being returned. This is done to check if any new
        information has been provided to the encoding arguments. Other elements without None value are NOT expected
        to change between the usage of this function.
        """

        values_dict = {}
        for ems_obj in ems_objects:
            encoded_value = ems_obj.encoded_value
            if encoded_value is None and ems_obj.encoding_fxn is not None:
                # rerun in case of encoding fxn argument changes
                encoded_value = self.run_encoding_fxn(ems_obj, ems_obj.value)
            values_dict[ems_obj.name] = encoded_value

        return values_dict

    @staticmethod
    def generate_mdp_from_tc(tc_intvars: dict = None,
                             tc_vars: dict = None,
                             tc_meters: dict = None,
                             tc_weather: dict = None,
                             tc_actuators: dict = None):
        """
        Given Table of Contents (TC) for each EMS-type, we create a MdpManager instance to more easily handle and store
        the simulation variables generated at runtime.
        This form-factor also helps implement automatic on normalizations on values for input to function approximators.

        The dictionary for each EMS variable type [intvars: internal variables, vars: variables, etc.] must follow a
        specific form:

        = {
            'ems_var1_name': [(...EMS handle ID parameters...)],  # no optional fxn and arguments
            'ems_var2_name': [(...EMS handle ID parameters...), optional_fxn, fxn_arg_2,..., fxn_arg_n],
            ...
        }

        Where each dictionary key is the user-defined name of that EMS element, and the value is a list of required
        parameters (its 'handle' ID values) to find the EMS element in the E+ simulation & (optional) attached
        normalization functions and its arguments. (Everytime the value of an EMS element is updated from the
        simulation, so will its encoded value. This is used to simplify the normalization process of input variables
        when working with Neural Networks, etc.)

        Ex:
        tc_vars = {
            'hvac_operation_sched': [('Schedule Value', 'OfficeSmall HVACOperationSchd')],
            'zn0_temp': [('Zone Air Temperature', zn0), normalize_min_max_saturate, 60, 80]
        }

        Looking at 'hvac_operation_sched': (a Schedule Value - EMS Variable)

            The 1st element of the list value, the tuple, contains all the parameters specific to that EMS elements
            'handle' ID. This is specific to your building model .IDF and requires EnergyPlus understanding to setup.
            There is no encoding/normalization function included.

        Looking at 'zn0_tmp': (a Zone temperature value - EMS Variable)

            We see this variable includes an encoding/normalization function. Next to it are the 2nd and 3rd agrument
            values to be called everytime this EMS element is updated, so that its encoded/normalized value will also
            be updated.

            These are the 2nd and 3rd arguments because for inorder for this user-written function
            'normalize_min_max_saturate' to work, the first argument must be the EMS values itself.

            EX:
            def normalize_min_max_saturate(value: float, min: float, max: float):
                lower = -1
                upper = 1
                if value > max:
                    return upper
                elif value < min:
                    return lower
                else:
                    return ((value - min) / (max - min))*(upper - lower) + lower
        """
        mdp_instance = MdpManager()

        # compile all ToCs and add to MDP class
        master_toc = {'intvar': tc_intvars, 'var': tc_vars, 'meter': tc_meters, 'weather': tc_weather,
                      'actuator': tc_actuators}
        for ems_type, ems_tc in master_toc.items():
            for ems_name, tc_values in ems_tc.items():
                mdp_instance.add_ems_element(ems_type, ems_name, *tc_values[0:])

        return mdp_instance