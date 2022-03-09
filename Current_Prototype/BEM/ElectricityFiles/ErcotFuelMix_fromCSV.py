"""
This file is used for quick cleaning and visual analysis of the ERCOT fuel mix data, with the intentions to understand
renewable (wind) generation better. The main function here converts the uncomfortable raw format to a format organized
by time (row) and fuel types (columns)
"""

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt


def raw_from_CSV(file_path: 'str'):
    year_df = pd.DataFrame()
    for i, month in enumerate(('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')):
        # iterate through all Month sheets
        month_df = pd.read_excel(file_path, sheet_name=month)
        year_df = year_df.append(month_df, sort=False, ignore_index=True)
    # clean up unneededd & erroneous columns
    year_df = year_df.drop(columns=['Settlement Type', '01:15 (DST)', '01:30 (DST)', '01:45 (DST)', '02:00 (DST)'])
    # rename final span
    year_df = year_df.rename(columns={"0:00": "24:00"})  # last col of day
    return year_df


def transpose_from_raw(raw_df: pd.DataFrame):
    """Put into form ordered sequentially by datetime with fuels as columns."""
    # create 15 min intervals thru day
    minutes = ['00', '15', '30', '45']
    day_segments = []
    for hh in range(24):
        for mm in minutes:
            day_segments.append(dt.datetime.strptime(str(hh) + ':' + mm, '%H:%M').time())

    # init
    segment_sum = [0] * len(day_segments)
    date = raw_df.at[0, 'Date']  # timestamp
    start_year = date.year
    year_datetime = []
    segment_totals = []
    generation_history = {'Biomass': [], 'Coal': [], 'Gas': [], 'Gas-CC': [], 'Hydro': [], 'Nuclear': [], 'Wind': [],
                          'Other': []}
    fuel_columns = list(generation_history.keys())

    while date.year == start_year:
        for fuel_type in fuel_columns:
            # select row for given fuel and day
            row = raw_df[(raw_df['Date'] == date) & (raw_df['Fuel'] == fuel_type)]
            row = row.iloc[0, 3:].to_list()  # get only fuel generation data
            # add data
            generation_history[fuel_type] += row
            segment_sum = [sum(x) for x in zip(segment_sum, row)]  # element wise addition of day
            if fuel_type == fuel_columns[-1]:  # do once, at end of all fuel types
                # collect all dates
                for time_segment in day_segments:
                    year_datetime.append(dt.datetime.combine(date.date(), time_segment))
                # updates
                segment_totals += segment_sum
                segment_sum = [0] * len(day_segments)
                date += dt.timedelta(days=1)  # next day

    df = pd.DataFrame.from_dict(generation_history)
    df.insert(0, 'Total', segment_totals)
    df.insert(0, 'Datetime', year_datetime)
    df.set_index('Datetime', inplace=True)

    return df


if __name__ == "__main__":
    year = 2019
    year_df = raw_from_CSV('raw_IntGenbyFuel' + str(year) + '.xlsx')
    # gather only 1 fuel type for annual period
    df = transpose_from_raw(year_df)

    fuel_type = 'Wind'
    aggr = 'Mean'
    # daily
    wind_daily_df = year_df[year_df['Fuel'] == fuel_type].set_index('Date')  # using mask, set index
    # weekly, downsample
    wind_weekly_df = wind_daily_df.resample('1W').mean()
    # monthly, downsample
    wind_monthly_df = wind_daily_df.resample('1M').mean()

    # PLOTS
    # Annual Plots
    wind_daily_df.iloc[:,1].plot(x='Daily', y='kWh Generation', title=f'Annual {fuel_type} Generation of {year} - {aggr}',label='Daily')
    wind_weekly_df.iloc[:,0].plot(x='Weekly', y='kWh Generation', title=f'Annual {fuel_type} Generation of {year} - {aggr}',label='Weekly')
    wind_monthly_df.iloc[:,0].plot(x='Monthly', y='kWh Generation', title=f'Annual {fuel_type} Generation of {year} - {aggr}',label='Monthly')
    plt.legend()
