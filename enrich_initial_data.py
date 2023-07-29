# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adarsh Kalikadien                     #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #

"""Add features for each year and each location to the dataset"""

# rainfall data
# import xarray as xr
# import os
#
# input_filename = 'merged_merged_data_06_2018.nc'
# file_path = os.path.join('dataset', '2.additional_data', 'rainfall', input_filename)
#
# # Open the NetCDF file using xarray
# data = xr.open_dataset(file_path)
#
# # Print the details and attributes of the dataset
# # print(data)
#
# # Convert the xarray dataset to a pandas DataFrame
# df = data.to_dataframe()
#
# # You can now work with the DataFrame as you would with any other pandas DataFrame
# df.to_csv(file_path.replace('.nc', '.csv'))
#
# # If you want to see the summary statistics of the data variables, you can use
# # print(data.describe())
#
# # You can also access specific variables in the dataset
# # For example, if your dataset has a variable named 'rainfall', you can access it as follows:
# # rainfall_data = data['rainfall']
# # print(rainfall_data)
#
# # To access specific slices of the data, you can use indexing on the variables
# # For example, to get the first record of rainfall data, you can use:
# # first_record = rainfall_data[0]
# # print(first_record)

# max T data is binary thus needs a different way of processing
# import struct
# import pandas as pd
# import numpy as np
#
# # Define the file paths
# input_file_path = "dataset/2.additional_data/max_temp/Maxtemp_MaxT_2018.GRD"
# output_file_path = "Maxtemp_MaxT_2018.csv"
#
# # Assuming the data in the binary file is stored as a 31x31 grid of 4-byte floating-point numbers (float in C)
# grid_size = 31
# element_size = 4  # Size of each element in bytes (float)
#
# # Calculate the total number of elements in one daily record (31x31 grid)
# record_elements = grid_size * grid_size
#
# # Function to read the binary data for one day (31x31 grid) and return it as a 2D array
# def read_binary_data(file):
#     # Read the entire day's data as a flat array
#     data = np.fromfile(file, dtype=np.float32, count=record_elements)
#     # Check if the data was successfully read
#     if data.size == record_elements:
#         # Reshape the flat array to a 2D grid (31x31)
#         data = data.reshape(grid_size, grid_size)
#         return data
#     else:
#         return None
#
# # Open the binary file for reading
# with open(input_file_path, "rb") as fin:
#     # Create a list to hold all the daily data
#     all_data = []
#
#     for k in range(366):
#         # Read the 31x31 grid data for one day
#         daily_data = read_binary_data(fin)
#         if daily_data is not None:
#             all_data.append(daily_data)
#
# # Check if any data was successfully read
# if len(all_data) == 0:
#     print("No data was read from the binary file.")
# else:
#     # Stack the daily grids to create a 3D NumPy array (366 days, 31x31 grid for each day)
#     stacked_data = np.stack(all_data)
#
#     # Reshape the stacked data to a 2D array (366115 data points, 1 column)
#     flattened_data = stacked_data.reshape(-1, 1)
#
#     # # read rainfall data and add max_t column to it with flattened_data
#     # DIMENSIONS DO NOT MATCH
#     # rainfall_df = pd.read_csv('dataset/2.additional_data/rainfall/RFone_imd_rf_1x1_2018.csv')
#     # rainfall_df['max_t'] = flattened_data
#
#
#     # Convert the 2D NumPy array to a pandas DataFrame
#     df = pd.DataFrame(flattened_data)
#
#     # Save the DataFrame to a CSV file
#     df.to_csv(output_file_path, index=False)
#
#     # Display the DataFrame (optional)
#     # print(df)