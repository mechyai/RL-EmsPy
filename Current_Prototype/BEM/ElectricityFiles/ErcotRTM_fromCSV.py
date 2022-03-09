"""
This file is used for quick cleaning and visual analysis of the ERCOT RTP data.
# TODO what is "Repeated Hourly Flag"
"""

import pandas as pd
import matplotlib.pyplot as plt


def raw_from_CSV(file_path: 'str'):
    """Create single DF from ERCOT RTP Excel file consisting of 1 page per month."""

    raw_df = pd.DataFrame()
    for i, month in enumerate(('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')):
        # iterate through all Month sheets
        month_df = pd.read_excel(file_path, sheet_name=month)
        raw_df = raw_df.append(month_df, sort=False)
    # clean up unneeded columns
    return raw_df


def clean_from_raw(raw_df: pd.DataFrame, settlement_point_name: str, settlement_point_type: str):
    """Process pricing data for one location, discarding the rest and unneeded columns."""

    # gather only 1 Location Name for annual period, using mask
    df = raw_df[(raw_df['Settlement Point Name'] == settlement_point_name) &
                      (raw_df['Settlement Point Type'] == settlement_point_type)]
    # remove unneeded cols
    df = df.drop(columns=['Repeated Hour Flag', 'Settlement Point Name', 'Settlement Point Type'])
    return df


def datetime_from_clean(clean_df: pd.DataFrame):
    """Compresses 3 timing columns to 1 DateTime column, considering 15 minute intervals."""

    # correct temporal columns into single Datetime index
    datetime_col = pd.to_datetime(clean_df['Delivery Date']) + (clean_df['Delivery Hour']-1).astype('timedelta64[h]')  # hrs
    datetime_col += (clean_df['Delivery Interval'] * 15).astype('timedelta64[m]')  # 15 min intervals
    # replace date, remove unneeded time cols
    clean_df['Delivery Date'] = datetime_col
    clean_df = clean_df.drop(columns=['Delivery Hour', 'Delivery Interval'])
    # make datetime index
    return clean_df.set_index(['Delivery Date'])


def to_csv_schedule_file(df: pd.DataFrame, output_file_name: str):
    """Saves RTP column to CSV to be used as Schedule:File in EnergyPlus"""

    df.to_csv(output_file_name, columns=['Settlement Point Price'])
    return 0


# visual data analysis, as needed
if __name__ == "__main__":
    year = 2019
    file_path = 'raw_FifteenRTP' + str(year) + '.xlsx'
    raw_df = raw_from_CSV(file_path)

    # gather only 1 Location Name for annual period
    settlement_point_name = 'HB_NORTH'
    settlement_point_type = 'HU'

    cleaned_df = clean_from_raw(raw_df, settlement_point_name, settlement_point_type)
    daily_df = datetime_from_clean(cleaned_df)

    # weekly, downsample
    weekly_df = daily_df.resample('1W').mean()
    # monthly, downsample
    monthly_df = daily_df.resample('1M').mean()

    # PLOT
    # daily
    plt.figure()
    daily_df.iloc[:,0].plot(x='Daily', y='RTP $ / MW', title=f'RTP$ for {settlement_point_name}:{settlement_point_type} '
                                                             f'- Mean', label='Daily')
    plt.legend()
    # weekly
    plt.figure()
    weekly_df.iloc[:,0].plot(x='Weekly', y='RTP $ / MW', title=f'RTP$ for {settlement_point_name}:{settlement_point_type}'
                                                             f'- Mean', label='Weekly')
    # monthly
    monthly_df.iloc[:,0].plot(x='Monthly', y='RTP $ / MW', title=f'RTP$ for {settlement_point_name}:{settlement_point_type}'
                                                             f' - Mean', label='Monthly')
    plt.legend()
