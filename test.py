import pandas as pd

# Set display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Read the CSV file
df = pd.read_csv('Combined\ppg\combined_ppg_data.csv')

# Display full description
print(df.describe())
df.describe().to_csv('Combined\ppg\combined_ppg_data_description.csv')

df = pd.read_csv('Combined\eda\combined_eda_data.csv')
print(df.describe())
df.describe().to_csv('Combined\eda\combined_eda_data_description.csv')

