# find datatype of each column
data_types = pd.DataFrame(df.dtypes, columns=['Data Type'])

# count number of missing observations by column
missing_data_counts = pd.DataFrame(df.isnull().sum(), columns=['Missing Values'])

# count number of present observations by column
present_data_counts = pd.DataFrame(df.count(), columns=['Present Values'])

# count number of unique observations by column
unique_value_counts = pd.DataFrame(columns=['Unique Values'])
for v in list(df.columns.values):
    unique_value_counts.loc[v] = [df[v].nunique()]
    
# find minimum value for each column
minimum_values = pd.DataFrame(columns=['Minimum Values'])
for v in list(df.columns.values):
    minimum_values.loc[v] = [df[v].min()]

# find maximum value for each column
maximum_values = pd.DataFrame(columns=['Maximum Values'])
for v in list(df.columns.values):
    maximum_values.loc[v] = [df[v].max()]

data_quality_report = pd.concat([present_data_counts, missing_data_counts, unique_value_counts, minimum_values, maximum_values], axis=1)
print(data_quality_report)