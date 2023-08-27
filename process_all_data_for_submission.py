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
    df = df.drop(columns=['Longitude', 'Latitude', 'Index'])
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
    # Create an empty DataFrame to store the transformed data
    transformed_df = pd.DataFrame(columns=['year', 'data_type', 'source_index', 'destination_index', 'value'])
    # if df is a numpy array, convert to pandas df
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df)

    transformed_data = []

    for source_index in df.index:
        for destination_index in df.columns:
            if source_index != destination_index:
                value = df.loc[source_index, destination_index]
                if value > 0:
                    data_type = f"{biomass_or_pellet}_demand_supply"
                    transformed_data.append({'year': year, 'data_type': data_type, 'source_index': source_index,
                                             'destination_index': destination_index, 'value': value})

    transformed_df = pd.DataFrame(transformed_data)

    # convert year column to int
    transformed_df['year'] = transformed_df['year'].astype(int)
    return transformed_df


def process_depot_biorefinery_location(list_of_location_indices, year, depot_or_biorefinery):
    # year  data_type  source_index  destination_index  value
    # 2018  {depot/biorefinery}_location  index_from_list  NaN  NaN
    # Create an empty DataFrame to store the transformed data
    transformed_df = pd.DataFrame(columns=['year', 'data_type', 'source_index', 'destination_index', 'value'])
    # iterate over list and add to transformed_df
    for index in list_of_location_indices:
        transformed_df = pd.concat([transformed_df, pd.DataFrame({'year': year, 'data_type': f"{depot_or_biorefinery}_location",
                                                                  'source_index': index, 'destination_index': np.nan,
                                                                  'value': np.nan}, index=[0])], ignore_index=True)
    # convert year column to int
    transformed_df['year'] = transformed_df['year'].astype(int)
    return transformed_df


if __name__ == "__main__":
    # create empty df to store all data
    all_data_df = pd.DataFrame(columns=['year', 'data_type', 'source_index', 'destination_index', 'value'])
    # biomass forecast processing
    # read and process biomass forecast, write to csv file for example submission
    biomass_forecast_df = pd.read_csv('dataset/3.predictions/20230826Biomass_Predictions.csv')
    biomass_forecast_df = process_biomass_forecast(biomass_forecast_df)
    # read biomass flows for 2018 and 2019
    # for vivek's file
    # biomass_flow_2018_df = pd.read_csv('dataset/3.predictions/2018_flow.csv')
    # biomass_flow_2019_df = pd.read_csv('dataset/3.predictions/2019_flow.csv')
    biomass_flow_2018_df = pd.read_csv('dataset/3.predictions/biomass_flow_2018.csv')
    biomass_flow_2019_df = pd.read_csv('dataset/3.predictions/biomass_flow_2019.csv')
    # read pellet flows for 2018 and 2019
    pellet_flow_2018_df = pd.read_csv('dataset/3.predictions/pellet_flow_2018.csv')
    pellet_flow_2019_df = pd.read_csv('dataset/3.predictions/pellet_flow_2019.csv')
    # read depot and biorefinery locations (unique source and destination indices from pellet_flow_2018_df)
    # for vivek's file we get this from the correct biomass flow matrix instead of pellet flow matrix
    # depot_location_indices = biomass_flow_2018_df['destination_index'].unique()
    depot_location_indices = pellet_flow_2018_df['source_index'].unique()
    biorefinery_location_indices = pellet_flow_2018_df['destination_index'].unique()
    # process locations
    depot_location_df = process_depot_biorefinery_location(depot_location_indices, 20182019, 'depot')
    biorefinery_location_df = process_depot_biorefinery_location(biorefinery_location_indices, 20182019, 'refinery')

    # concat all dataframes in all_data_df
    all_data_df = pd.concat([all_data_df, biomass_forecast_df, biomass_flow_2018_df, biomass_flow_2019_df,
                                pellet_flow_2018_df, pellet_flow_2019_df, depot_location_df, biorefinery_location_df])

    # write to csv file
    all_data_df.to_csv('dataset/3.predictions/submission.csv', index=False)



    # biomass_forecast_df.to_csv('dataset/3.predictions/submission.csv', index=False)
    # read example submission and concat biomass_forecast_df to bottom of df
    # example_submission_df = pd.read_csv('dataset/1.initial_datasets/sample_submission.csv')
    # example_submission_df = example_submission_df[example_submission_df['data_type'] != 'biomass_forecast']
    # example_submission_df = pd.concat([example_submission_df, biomass_forecast_df])
    # # write to csv file
    # example_submission_df.to_csv('dataset/3.predictions/submission.csv', index=False)




