"""
This file is used to generate Schedule:File IDF objects and usable CSV files, for E+ simulation, from raw historic
data from ERCOT.
Usage: select year, file paths of EXCEL data from ERCOT and this will import the EXCEL data, transform them with
Pandas, and spit out the .CSV & .IDF to be used in E+ Schedule:Files
"""
import pandas as pd

import ErcotRTM_fromCSV as rtm
import ErcotDAM_fromCSV as dam
import ErcotFuelMix_fromCSV as fmix

from emspy import idf_editor

# -- OUTPUT PARAMS --
# metadata
year = 2019
# ERCOT location
settlement_point_name = 'HB_NORTH'
settlement_point_type = 'HU'
# output paths
save_schedule_file_csv = r'A:/Files/PycharmProjects/RL-BCA/Current_Prototype/BEM/ScheduleFiles/'
save_schedule_file_idf = r'A:/Files/PycharmProjects/RL-BCA/Current_Prototype/BEM/CustomIdfFiles/Automated/'


# -- RTP --
def rtm_schedule_file():
    """Real-time market/pricing processes to create Schedule:File dependencies."""

    min_per_rtp = 15
    file_rtm_path = 'raw_FifteenRTP' + str(year) + '.xlsx'
    file_out_rtm_name = f'ERCOT_RTM_{year}'

    # process raw data from source
    clean_rtp_df = rtm.clean_from_raw(rtm.raw_from_CSV(file_rtm_path), settlement_point_name, settlement_point_type)
    dt_rtm_df = rtm.datetime_from_clean(clean_rtp_df)

    # IDF Schedule:File creation
    idf_save_file = save_schedule_file_idf + file_out_rtm_name + '.idf'
    output_rtm_csv = save_schedule_file_csv + file_out_rtm_name + '.csv'
    rtm.to_csv_schedule_file(dt_rtm_df, output_rtm_csv)
    idf_editor.create_schedule_file(output_rtm_csv, idf_save_file, f'ERCOT RTM {year}', 2, 1, min_per_rtp)

    return dt_rtm_df


# -- DAP --
def dam_schedule_file():
    """Day-ahead market/pricing processes to create Schedule:File dependencies."""

    min_per_dmp = 60
    file_dam_path = 'raw_DAM' + str(year) + '.xlsx'
    file_out_dam_name = f'ERCOT_DAM_{year}'

    # process raw data from source
    clean_dam_df = dam.clean_from_raw(dam.raw_from_CSV(file_dam_path), settlement_point_name)
    dt_dam_df = dam.datetime_from_clean(clean_dam_df)

    # IDF Schedule:File creation
    idf_save_file = save_schedule_file_idf + file_out_dam_name + '.idf'
    output_rtm_csv = save_schedule_file_csv + file_out_dam_name + '.csv'
    dam.to_csv_schedule_file(dt_dam_df, output_rtm_csv)
    idf_editor.create_schedule_file(output_rtm_csv, idf_save_file, f'ERCOT DAM {year}', 2, 1, min_per_dmp)

    return dt_dam_df


def dam_forecast_schedule_file():
    """Day-ahead market/pricing processes. Creates 12-hr window lookahead for each timestep of available DAP data."""

    min_per_dmp = 60
    default_dap_eof = 18  # End of File default DAP
    file_dam_path = 'raw_DAM' + str(year) + '.xlsx'
    file_out_dam_name = f'ERCOT_DAM_12hr_forecast_{year}'

    # process raw data from source
    clean_dam_df = dam.clean_from_raw(dam.raw_from_CSV(file_dam_path), settlement_point_name)
    forecast_df = clean_dam_df.drop(columns=['Delivery Date', 'Hour Ending'])
    # create forecast window
    forecast_df.rename(columns={'Settlement Point Price': 'Hr0'}, inplace=True)
    column_forecasts = ['Hr1', 'Hr2', 'Hr3', 'Hr4', 'Hr5', 'Hr6', 'Hr7', 'Hr8', 'Hr9', 'Hr10', 'Hr11']
    forecast_df = pd.concat([forecast_df, pd.DataFrame(columns=column_forecasts, dtype=float)], axis=1)
    forecast_df.reset_index(drop=True, inplace=True)
    for r in range(len(forecast_df)):
        for i in range(11):
            # iterate thru 12 hr window
            try:
                future_dap = forecast_df.at[r + i + 1, 'Hr0']
                forecast_df.at[r, column_forecasts[i]] = future_dap
            except KeyError:
                # manage end of file missing data
                forecast_df.at[r, column_forecasts[i]] = default_dap_eof  # default filler

    # IDF Schedule:File creation
    output_dap_csv = save_schedule_file_csv + file_out_dam_name + '.csv'
    # save CSV
    dam.to_csv_schedule_file(forecast_df, output_dap_csv)
    for h in range(12):
        idf_save_file = save_schedule_file_idf + file_out_dam_name + f'_{h}hr_ahead.idf'
        idf_editor.create_schedule_file(output_dap_csv, idf_save_file, f'ERCOT DAM 12-Hr Forecast {year} - {h}hr Ahead',
                                        h + 2, 1, min_per_dmp)

    return forecast_df


# -- FuelMix --
def fuel_mix_file():
    """This organizes fuel mix data for 15-min increments."""

    min_per_fmix = 15
    file_dam_path = 'raw_IntGenbyFuel' + str(year) + '.xlsx'
    file_out_fmix_name = f'ERCOT_FMIX_{year}'

    # process raw data from source
    fmix_df = fmix.transpose_from_raw(fmix.raw_from_CSV(file_dam_path))

    # IDF Schedule:File creation

    output_fmix_csv = save_schedule_file_csv + file_out_fmix_name + '.csv'
    dam.to_csv_schedule_file(fmix_df, output_fmix_csv)
    fuel_types = fmix_df.columns.to_list()
    for fuel in fuel_types:
        fuel_index = fuel_types.index(fuel) + 2  # get index of specific fuel of interest, +2 time & Python 0 indexing
        idf_save_file = save_schedule_file_idf + file_out_fmix_name + f'_{fuel}.idf'
        idf_editor.create_schedule_file(output_fmix_csv, idf_save_file, f'ERCOT FMIX {year} - {fuel}', fuel_index, 1,
                                        min_per_fmix)

    return fmix_df


if __name__ == "__main__":
    rtm_df = rtm_schedule_file()
    # dam_df = dam_schedule_file()
    # dam_forecast_df = dam_forecast_schedule_file()
    # fmix_df = fuel_mix_file()
