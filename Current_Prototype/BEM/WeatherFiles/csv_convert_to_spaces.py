"""
Convert .CSV file (comma delimited) to SPACE delimited file. Weather Converter is having issue understanding
DataDelimiter field, not reading comma from CSV.
"""

city = 'Snyder'
year = 2019

file_read = f"{city}Texas_{year}/{city}Texas_CC_{year}.csv"
file_write = f"{city}Texas_{year}/{city}Texas_CC_{year}"

with open(file_read) as infile, open(file_write, 'w') as outfile:
    outfile.write(infile.read().replace(",", " "))
