# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adarsh Kalikadien                     #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #

import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids

from process_all_data_for_submission import process_flow_matrix


class DecisionBasedApproach:
    def __init__(self, distance_matrix_file, predictions_file, num_depots, num_biorefineries, max_depot_capacity,
                 max_refinery_capacity, year, year_1):
        self.distance_matrix = pd.read_csv(distance_matrix_file, index_col=0)
        self.predictions_df = pd.read_csv(predictions_file)
        self.num_depots = num_depots
        self.num_biorefineries = num_biorefineries
        self.max_depot_capacity = max_depot_capacity
        self.max_refinery_capacity = max_refinery_capacity

        # locations of depots and refineries do not change over the years, so use both to cluster
        self.year = year
        self.year_1 = year_1
        self.depot_indices = None
        self.refinery_indices = None
        self.flow_sites_to_depots = None
        self.flow_depots_to_refineries = None

    def fit(self):
        # Cluster sites into depots based on biomass availability for both years
        kmedoids_depots = KMedoids(n_clusters=self.num_depots, random_state=42).fit(
            self.predictions_df[[self.year, self.year_1]].values)
        self.predictions_df['nearest_depot_location_index'] = kmedoids_depots.labels_
        self.depot_indices = kmedoids_depots.medoid_indices_

        # Select rows corresponding to depot locations
        depot_locations = self.predictions_df.iloc[self.depot_indices][['Latitude', 'Longitude']].values

        # Cluster depot locations to place refineries for both years
        kmedoids_refineries = KMedoids(n_clusters=self.num_biorefineries, random_state=42).fit(depot_locations)
        self.predictions_df.loc[self.depot_indices, 'nearest_refinery_location_index'] = kmedoids_refineries.labels_
        self.refinery_indices = kmedoids_refineries.medoid_indices_

    def solve(self):
        num_sites = self.predictions_df.shape[0]
        self.flow_depots_to_sites = np.zeros((self.num_depots, num_sites))
        self.flow_depots_to_refineries = np.zeros((self.num_depots, self.num_biorefineries))

        # Distribute biomass from depots to sites
        while np.sum(self.flow_depots_to_sites) < 0.8 * np.sum(self.predictions_df[self.year]):
            for depot_index in range(self.num_depots):
                depot_capacity = self.max_depot_capacity

                depot_location_index = self.depot_indices[depot_index]
                distances_to_sites = self.distance_matrix.loc[depot_location_index]

                sorted_site_indices = distances_to_sites.argsort()
                for site_index in sorted_site_indices:
                    site_biomass = self.predictions_df.loc[site_index, self.year]

                    if site_biomass > 0 and depot_capacity > 0:
                        biomass_to_transport = min(site_biomass, depot_capacity)
                        self.flow_depots_to_sites[depot_index, site_index] = biomass_to_transport
                        depot_capacity -= biomass_to_transport

        # # Distribute biomass from depots to refineries until the depots are emtpy or the refineries are full
        for depot_index in range(self.num_depots):
            depot_capacity = self.max_depot_capacity

            for refinery_index in range(self.num_biorefineries):
                refinery_capacity = self.max_refinery_capacity

                while depot_capacity > 0 and refinery_capacity > 0:
                    biomass_to_transport = min(depot_capacity, refinery_capacity)
                    self.flow_depots_to_refineries[depot_index, refinery_index] = biomass_to_transport
                    depot_capacity -= biomass_to_transport
                    refinery_capacity -= biomass_to_transport

        # write the flow matrices to csv files
        biomass_demand_supply_matrix = self.flow_depots_to_sites
        # transpose matrix such that rows are sites and columns are depots
        biomass_demand_supply_matrix = biomass_demand_supply_matrix.T
        biomass_demand_supply_matrix_df = process_flow_matrix(biomass_demand_supply_matrix, year_1, 'biomass')
        # transform the destination indices to site location indices (using the dba.depot_indices)
        biomass_demand_supply_matrix_df['destination_index'] = self.depot_indices[
            biomass_demand_supply_matrix_df['destination_index'].values]

        pellet_demand_supply_matrix = self.flow_depots_to_refineries
        # transpose matrix such that rows are refineries and columns are depots
        # pellet_demand_supply_matrix = pellet_demand_supply_matrix.T
        pellet_demand_supply_matrix_df = process_flow_matrix(pellet_demand_supply_matrix, year_1, 'pellet')
        # transform the destination indices to refinery location indices (using the dba.refinery_indices)
        pellet_demand_supply_matrix_df['destination_index'] = self.refinery_indices[
            pellet_demand_supply_matrix_df['destination_index'].values]
        # transform the source indices to depot location indices (using the dba.depot_indices)
        pellet_demand_supply_matrix_df['source_index'] = self.depot_indices[
            pellet_demand_supply_matrix_df['source_index'].values]
        biomass_demand_supply_matrix_df.to_csv(f'dataset/3.predictions/biomass_flow_{self.year}.csv', index=False)
        pellet_demand_supply_matrix_df.to_csv(f'dataset/3.predictions/pellet_flow_{self.year}.csv', index=False)

    def print_results(self):
        print("Flow from depots to sites:")
        print(self.flow_depots_to_sites)
        print("\nFlow from depots to refineries:")
        print(self.flow_depots_to_refineries)


if __name__ == '__main__':
    # Example usage
    distance_matrix_file = 'dataset/1.initial_datasets/Distance_Matrix.csv'
    predictions_file = 'dataset/3.predictions/20230826Biomass_Predictions.csv'
    num_depots = 20
    num_biorefineries = 4
    max_depot_capacity = 20000
    max_refinery_capacity = 100000
    year_1 = '2019' # solving for 2019
    year_2 = '2018' # but also using 2018 for clustering

    dba = DecisionBasedApproach(distance_matrix_file, predictions_file, num_depots, num_biorefineries, max_depot_capacity,
                                max_refinery_capacity, '2018', '2019')
    dba.fit()
    dba.solve()

