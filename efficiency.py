import pandas as pd

# Load the datasets
co2_file_path = "Data/CO2LevelsInAtmosphere.csv"
temp_file_path = "Data/GlobalTemperatures.csv"

co2_df = pd.read_csv(co2_file_path)
temp_df = pd.read_csv(temp_file_path)
temp_df["Year"] = pd.to_datetime(temp_df["dt"]).dt.year

# Check the initial number of rows in each dataset
initial_co2_rows = co2_df.shape[0]
initial_temp_rows = temp_df.shape[0]

# Filter datasets to include only data from 1958 to 2015
co2_df_filtered = co2_df[(co2_df['Year'] >= 1958) & (co2_df['Year'] <= 2015)]
temp_df_filtered = temp_df[(temp_df['Year'] >= 1958) & (temp_df['Year'] <= 2015)]

# Drop rows with missing values
co2_df_cleaned = co2_df_filtered.dropna()
temp_df_cleaned = temp_df_filtered.dropna()

# Check the number of rows after cleaning
cleaned_co2_rows = co2_df_cleaned.shape[0]
cleaned_temp_rows = temp_df_cleaned.shape[0]

# Calculate the percentage improvement in data readiness
co2_improvement = ((initial_co2_rows - cleaned_co2_rows) / initial_co2_rows) * 100
temp_improvement = ((initial_temp_rows - cleaned_temp_rows) / initial_temp_rows) * 100

# Average improvement across both datasets
average_improvement = (co2_improvement + temp_improvement) / 2
print(average_improvement)
