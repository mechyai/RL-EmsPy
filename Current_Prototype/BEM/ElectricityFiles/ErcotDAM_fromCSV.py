"""
This file is used for quick cleaning and visual analysis of the ERCOT DAM data. Bootstraps functions from RTM module.
# TODO what is "Repeated Hourly Flag"
"""

import pandas as pd
import matplotlib.pyplot as plt

import ErcotRTM_fromCSV as rtm


def raw_from_CSV(file_path: 'str'):
    """Create single DF from ERCOT DAM Excel file consisting of 1 page per month."""
    return rtm.raw_from_CSV(file_path)


def clean_from_raw(raw_df: pd.DataFrame, settlement_point_name: str):
    """Process pricing data for one location, discarding the rest and unneeded columns."""
    # gather only 1 Location Name for annual period, using mask
    df = raw_df[(raw_df['Settlement Point'] == settlement_point_name)]
    # remove unneeded cols
    df = df.drop(columns=['Repeated Hour Flag', 'Settlement Point'])
    return df


def datetime_from_clean(clean_df: pd.DataFrame):
    """Compresses 2 timing columns to 1 DateTime column, considering hourly intervals."""

    clean_df.reset_index(drop=True, inplace=True)
    # must reduce 1-24 hrs to 0-23, conver hh:mm to h
    for r in range(len(clean_df)):
        hhmm = int(clean_df.at[r, 'Hour Ending'].split(':')[0]) - 1
        clean_df.at[r, 'Hour Ending'] = hhmm

    # correct temporal columns into single Datetime index
    datetime_col = pd.to_datetime(clean_df['Delivery Date']) + clean_df['Hour Ending'].astype('timedelta64[h]')  # hrs already in HH:MM format
    # replace date, remove unneeded time cols
    clean_df['Delivery Date'] = datetime_col
    clean_df = clean_df.drop(columns=['Hour Ending'])
    # make datetime index
    return clean_df.set_index(['Delivery Date'])


def to_csv_schedule_file(df: pd.DataFrame, output_file_name: str):
    """Saves DAM column to CSV to be used as Schedule:File in EnergyPlus"""

    df.to_csv(output_file_name)
    return 0


# visual data analysis, as needed
if __name__ == "__main__":
    year = 2019
    file_path = 'raw_DAM' + str(year) + '.xlsx'
    raw_df = raw_from_CSV(file_path)

    # gather only 1 Location Name for annual period
    settlement_point_name = 'HB_NORTH'

    cleaned_df = clean_from_raw(raw_df, settlement_point_name)
    daily_df = datetime_from_clean(cleaned_df)

    # weekly, downsample
    weekly_df = daily_df.resample('1W').mean()
    # monthly, downsample
    monthly_df = daily_df.resample('1M').mean()

    # PLOT
    # daily
    plt.figure()
    daily_df.plot(y='Settlement Point Price', title=f'DAM$ for {settlement_point_name}', label='Daily')
    plt.legend()
    # weekly
    plt.figure()
    weekly_df.iloc[:,0].plot(x='Weekly', y='DAP $ / MW', title=f'DAM$ for {settlement_point_name}'
                                                             f'- Mean', label='Weekly')
    # monthly
    monthly_df.iloc[:,0].plot(x='Monthly', y='DAP $ / MW', title=f'DAM$ for {settlement_point_name}'
                                                             f' - Mean', label='Monthly')
    plt.legend()

