# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adarsh Kalikadien                     #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #

"""Use data from 2010 to 2017 to predict biomass availability for 2018 and 2019"""

import os
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA

input_folder = os.path.join('dataset', '1.initial_datasets')
df = pd.read_csv(os.path.join(input_folder, 'Biomass_History.csv'))
# Correct the column names for longitude and latitude
df.rename(columns={'Longitude': 'longitude', 'Latitude': 'latitude'}, inplace=True)

# # # 1. Simplest approach, just a rolling mean of the last 3 years
# # Reset the index to include it as a separate column in the DataFrame
# df.reset_index(inplace=True)
#
# # Create a new DataFrame to store the predictions for 2018 and 2019
# predictions_df = df[['index', 'longitude', 'latitude']].copy()
#
# # Calculate the rolling mean for biomass availability over the last 3 years (you can adjust the window size)
# predictions_df['2018'] = df.iloc[:, -4:-1].mean(axis=1)
# predictions_df['2019'] = df.iloc[:, -3:].mean(axis=1)
#
# # Melt the DataFrame to have the year as a separate column
# predictions_df = pd.melt(predictions_df, id_vars=['index', 'longitude', 'latitude'], var_name='year', value_name='biomass')
#
# # Convert the 'year' column to numeric type
# predictions_df['year'] = predictions_df['year'].astype(int)
#
# # Display the DataFrame with predicted biomass availability for 2018 and 2019
# print(predictions_df)

df.reset_index(inplace=True)
# Create a new DataFrame to store the predictions for 2018 and 2019
predictions_df = df[['index', 'longitude', 'latitude']].copy()


# Function to train ARIMA and make predictions for a specific location
def predict_biomass_ARIMA(location_data):
    model = ARIMA(location_data, order=(1, 0, 1))  # ARIMA(1, 0, 1) model, you can adjust the order
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=2)  # Predict 2018 and 2019
    return predictions


# Predict biomass availability for 2018 and 2019 for each unique location
predictions_list = []

unique_locations = df[['longitude', 'latitude']].drop_duplicates()

for _, location in unique_locations.iterrows():
    longitude = location['longitude']
    latitude = location['latitude']

    # Filter data for the specific location
    location_data = df[(df['longitude'] == longitude) & (df['latitude'] == latitude)].iloc[:, 4:]
    location_data = location_data.squeeze()  # Convert the DataFrame to a Series

    # Predict biomass availability for 2018 and 2019 using ARIMA
    predictions = predict_biomass_ARIMA(location_data)
    predictions_list.append(predictions)

# Create a DataFrame with the predictions for each unique location
predictions_df[['2018', '2019']] = pd.DataFrame(predictions_list, index=unique_locations.index)

# write the predictions to a csv file
output_folder = os.path.join('dataset', '3.predictions')
predictions_df.to_csv(os.path.join(output_folder, 'biomass_predictions.csv'), index=False)