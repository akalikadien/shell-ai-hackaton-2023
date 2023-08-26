# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adarsh Kalikadien                     #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #

import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids


class DecisionBasedApproach:
    def __init__(self, distance_matrix_file, predictions_file, num_depots, num_biorefineries, max_depot_capacity,
                 max_refinery_capacity, year):
        self.distance_matrix = pd.read_csv(distance_matrix_file, index_col=0)
        self.predictions_df = pd.read_csv(predictions_file)
        self.num_depots = num_depots
        self.num_biorefineries = num_biorefineries
        self.max_depot_capacity = max_depot_capacity
        self.max_refinery_capacity = max_refinery_capacity

        self.year = year
        self.site_labels = None
        self.depot_indices = None
        self.refinery_indices = None
        self.flow_sites_to_depots = None
        self.flow_depots_to_refineries = None

    def fit(self):
        # cluster sites into depots and refineries
        kmedoids = KMedoids(n_clusters=self.num_depots, random_state=42).fit(
            self.predictions_df[['Latitude', 'Longitude']].values)
        self.site_labels = kmedoids.labels_
        self.depot_indices = kmedoids.medoid_indices_

        kmedoids_refinery = KMedoids(n_clusters=self.num_biorefineries, random_state=42).fit(
            self.predictions_df.iloc[self.depot_indices][['Latitude', 'Longitude']].values)
        self.refinery_indices = kmedoids_refinery.medoid_indices_

    def solve(self):
        num_sites = self.predictions_df.shape[0]
        self.flow_sites_to_depots = np.zeros((num_sites, self.num_depots))
        self.flow_depots_to_refineries = np.zeros((self.num_depots, self.num_biorefineries))

        # distribute biomass from sites to depots
        for site_index, site_row in self.predictions_df.iterrows():
            site_label = self.site_labels[site_index]
            depot_index = self.depot_indices[site_label]

            biomass_at_site = site_row[str(self.year)]
            depot_capacity = min(self.max_depot_capacity, biomass_at_site)

            self.flow_sites_to_depots[site_index, depot_index] = depot_capacity

        # distribute biomass from depots to refineries
        for depot_index in range(self.num_depots):
            depot_center = self.depot_cluster_centers[depot_index]

            # calculate the distance from the current site to the current depot
            distances_to_depots = self.distance_matrix.iloc[site_index, self.depot_indices].values

            # calculate the remaining capacity of the depot
            depot_capacity = 20000 - self.flow_sites_to_depots[site_index, depot_index]

            # loop through the depots based on their distances
            for nearest_depot_index in np.argsort(distances_to_depots):
                if depot_capacity <= 0:
                    break

                max_distance = np.max(self.distance_matrix)  # You can adjust this value as needed
                if distances_to_depots[nearest_depot_index] <= max_distance:
                    biomass_to_transport = min(biomass_at_site, depot_capacity)
                    self.flow_sites_to_depots[site_index, nearest_depot_index] += biomass_to_transport
                    depot_capacity -= biomass_to_transport

                    # Update the biomass value at the site
                    site_row[str(self.year)] -= biomass_to_transport

        # distribute biomass from depots to refineries
        for depot_index in range(self.num_depots):
            depot_center = self.depot_cluster_centers[depot_index]

            # calculate the distance from the current site to the current depot
            distance_to_depot = self.distance_matrix[site_index, int(depot_center[0])]

            # calculate the remaining capacity of the depot
            depot_capacity = 20000 - self.flow_sites_to_depots[site_index, depot_index]

            max_distance = np.max(self.distance_matrix)
            # if the distance is within the maximum distance and there's remaining capacity in the depot
            if distance_to_depot <= max_distance and depot_capacity > 0:
                # calculate the biomass that can be transported to the depot
                biomass_at_site = site_row[str(self.predictions_df.columns[-1])]
                biomass_to_transport = min(biomass_at_site, depot_capacity)

                # update the flow matrix
                self.flow_sites_to_depots[site_index, depot_index] += biomass_to_transport
                depot_capacity -= biomass_to_transport

                # update the biomass value at the site
                site_row[str(self.predictions_df.columns[-1])] -= biomass_to_transport

                # update the remaining distance
                max_distance -= distance_to_depot

        # after looping through depots, if there's still biomass left at the site, distribute it among the remaining depots
        for depot_index in range(self.num_depots):
            biomass_at_site = site_row[str(self.predictions_df.columns[-1])]
            depot_capacity = 20000 - self.flow_sites_to_depots[site_index, depot_index]

            if biomass_at_site > 0 and depot_capacity > 0:
                biomass_to_transport = min(biomass_at_site, depot_capacity)

                self.flow_sites_to_depots[site_index, depot_index] += biomass_to_transport
                depot_capacity -= biomass_to_transport

                site_row[str(self.predictions_df.columns[-1])] -= biomass_to_transport

    def print_results(self):
        print("Flow from sites to depots:")
        print(self.flow_sites_to_depots)
        print("\nFlow from depots to refineries:")
        print(self.flow_depots_to_refineries)


# Example usage
distance_matrix_file = 'dataset/1.initial_datasets/Distance_Matrix.csv'
predictions_file = 'dataset/3.predictions/20230826Biomass_Predictions.csv'
num_depots = 20
num_biorefineries = 4
max_depot_capacity = 20000
max_refinery_capacity = 100000

dba = DecisionBasedApproach(distance_matrix_file, predictions_file, num_depots, num_biorefineries, max_depot_capacity,
                            max_refinery_capacity, '2018')
dba.fit()
dba.solve()
dba.print_results()
