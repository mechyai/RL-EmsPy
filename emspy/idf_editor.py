"""
This program is to help automate simple repetitive tasks with EnergyPlus building model file (.IDF) modifications.
"""

import os
import tempfile


def append_idf(idf_file, idf_append, output_file_name=""):
    """ This takes two files (or their paths) as input, concatenates them, and writes it to a third file.

    :param idf_file: Path or file object of base file to be appended to.
    :param idf_append: Path or file object to be appended to base file.
    :param output_file_name: File name / path of output file. Leave blank to overwrite base "idf_file" in place.
    """

    # check for file/path input
    if isinstance(idf_file, str):
        # if path passed
        if not output_file_name:  # overwrite base file
            output_file_name = idf_file
        with open(idf_file) as idf_data:
            idf_file = idf_data.read()
    if isinstance(idf_append, str):
        with open(idf_append) as append_data:
            idf_append = append_data.read()

    # merge files segments
    idf_file += '\n' + idf_append

    with open(output_file_name, 'w') as output:
        output.write(idf_file)

    return output_file_name


def insert_custom_data_tracking(custom_name: str, idf_file_path: str, unit_type: str = 'Dimensionless'):
    """
    This inserts an workaround into the IDF to allow for custom data insertion and tracking through EnergyPlys.

    This feature is useful if you want to track your own data as an "Output:Variable", thus including it in the SQL
    output generation, and usable in DView.
    This is done by creating a "Schedule:Const" object, then the user must utilize a schedule actuator to insert data
    at each timestep. Thus, both a "Output:Variable" and "Schedule:Const" will be created.
    The ScheduleTypeLimits are also created accordingly to meet specifications for the custom Schedule:Const.

    Using this with the ToC Actuators is shown as such, where 'Custom Data Tracked' is your name for the IDF object and
    'my_data_var' is used to reference it via ToC in your script.
    Ex:
        'my_data_var': ['Schedule:Constant', 'Schedule Value', 'Custom Data Tracked'],

    :param custom_name: name (str) of data object to create and track (can have spaces).
    :param idf_file_path: Path to IDF model file to be modified, new objects will be appended to this file
    :param unit_type: name (str) of Unit Type for ScheduleTypeLimits object - default is dimensionless
    """

    allowable_unit_types = ['dimensionless', 'temperature', 'deltatemperature', 'precipitationRate', 'angle',
                            'convection coefficient', 'activity level', 'velocity', 'capacity', 'power', 'availability',
                            'percent', 'control', 'mode']

    if unit_type.lower() not in allowable_unit_types:
        raise ValueError(f'Specified unit type for ScheduleTypeLimits object must be in [{allowable_unit_types}]')

    schedule_type_limit_name = '\t' + 'SchedTypeLim ' + custom_name + ','
    custom_name = '\t' + custom_name + ','
    unit_type = '\t' + unit_type + ';'

    schedule_type_limit_obj = ['ScheduleTypeLimits,',
                               schedule_type_limit_name,
                               '\t,',  # no min limit
                               '\t,',  # no max limit
                               '\tContinuous,',
                               unit_type]

    schedule_const_obj = ['Schedule:Constant,',
                          custom_name,
                          schedule_type_limit_name,
                          '\t0;']

    output_var_obj = ['Output:Variable,',
                      custom_name,
                      '\tSchedule Value,',
                      '\tTimestep;',
                      '! ----------------------------------------------------------------------']

    # create, write, then delete temporary file, used for appending
    temp_file = '_temp_write'
    with open(temp_file, 'w') as idf:
        idf.writelines('\n')
        idf.writelines(f'!----------- Custom Schedule Tracking ({custom_name[:-1]}) -----------')
        idf.writelines('\n')
        idf.writelines('\n'.join(schedule_type_limit_obj))
        idf.writelines('\n\n')
        idf.writelines('\n'.join(schedule_const_obj))  # insert schedule obj for actutation
        idf.writelines('\n\n')
        idf.writelines('\n'.join(output_var_obj))  # insert output var for SQL
        # append temp IDF to base IDF

    append_idf(idf_file_path, temp_file)
    # delete temp IDF
    os.remove(temp_file)

    return 0


def create_schedule_file(csv_file_path: str, idf_output_file: str,
                         obj_name: str, col_num: int, rows_skip: int, min_per_item: int):
    """Create Schedule:File object IDF file given inputs."""

    sched_file_obj = ['Schedule:File,',
                      f'\t{obj_name},',
                      f'\tAny Number,',  # schedule type limits
                      f'\t{csv_file_path},',
                      f'\t{col_num},',
                      f'\t{rows_skip},',
                      f'\t,',  # #hrs of data
                      f'\t,',  # column delimiter, comma default
                      f'\tNo,',  # interpolation
                      f'\t{min_per_item};',
                      ]

    with open(idf_output_file, 'w') as idf:
        idf.writelines('\n')
        idf.writelines('!- Custom Schedule File')
        idf.writelines('\n')
        idf.writelines('\n'.join(sched_file_obj))

    return 0


def change_simulation_timestep(ts: int, idf_file_path: str):
    # TODO
    pass

