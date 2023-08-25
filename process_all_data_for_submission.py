# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adarsh Kalikadien                     #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #

"""process everything into a single solution.csv file for submission, see dataset/1.initial_datasets/sample_submission.csv
We need the following data: biomass_forecast for 2018 and 2019, depot_location and refinery_location for 2018/2019
(stays the same for both years), pellet_demand_supply and biomass_demand_supply for 2018 and 2019"""
import pandas as pd
import numpy as np


# function to process the biomass forecast df into a df with the required format
def process_biomass_forecast(df):
    # ToDo: this function is now specific for 230809_RF_biomass_prediction.csv, change if needed e.g. when index is already in df as column
    # unpivot 2018 and 2019 column such that we have the following columns:
    # year	data_type	source_index	destination_index	value
    # 2018  biomass_forecast  index_from_df  NaN  biomass_forecast_value

    # drop longitude and latitude columns
    df = df.drop(columns=['Longitude', 'Latitude'])
    # unpivot 2018 and 2019 columns
    df = pd.melt(df.reset_index(), id_vars=['index'], var_name='year', value_name='value')
    # drop index column
    df = df.drop(columns=['index'])
    # convert year column to int
    df['year'] = df['year'].astype(int)
    # add data_type column
    df['data_type'] = 'biomass_forecast'
    # add source_index and destination_index columns
    df['source_index'] = df.index
    # set destination_index to NaN
    df['destination_index'] = np.nan

    return df


def process_flow_matrix(df, year, biomass_or_pellet):
    # in the flow matrix the source_index is the index of the df and the destination_index is the column
    # the value is then the flow from source_index to destination_index in the matrix
    # year	data_type	source_index	destination_index	value
    # 2018  {biomass/pellet}_demand_supply  index_from_df  column_from_df  {biomass/pellet}_demand_supply_value



    pass


if __name__ == "__main__":
    # biomass forecast processing
    # read and process biomass forecast, write to csv file for example submission
    biomass_forecast_df = pd.read_csv('dataset/3.predictions/230809_RF_biomass_prediction.csv')
    biomass_forecast_df = process_biomass_forecast(biomass_forecast_df)
    # biomass_forecast_df.to_csv('dataset/3.predictions/submission.csv', index=False)
    # read example submission and concat biomass_forecast_df to bottom of df
    example_submission_df = pd.read_csv('dataset/1.initial_datasets/sample_submission.csv')
    example_submission_df = example_submission_df[example_submission_df['data_type'] != 'biomass_forecast']
    example_submission_df = pd.concat([example_submission_df, biomass_forecast_df])
    # write to csv file
    example_submission_df.to_csv('dataset/3.predictions/submission.csv', index=False)




