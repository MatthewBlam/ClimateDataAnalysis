COLUMNS = [
    "Carbon Dioxide in Atmosphere (ppm)",
    "Seasonally Adjusted Carbon Dioxide in Atmosphere (ppm)",
    "Land Temperature (celsius)",
    "Land and Ocean Temperature (celsius)"
]
COLUMNS_AVG = [
    "avg. Carbon Dioxide in Atmosphere (ppm)",
    "avg. Seasonally Adjusted Carbon Dioxide in Atmosphere (ppm)",
    "avg. Land Temperature (celsius)",
    "avg. Land and Ocean Temperature (celsius)"
]

COLUMNS_ABR = [
    "avg. CO2 Levels",
    "avg. Adjusted CO2 Levels",
    "avg. Land Temp",
    "avg. Land & Ocean Temp"
]

def log(dataframe):
    print(dataframe.to_string())