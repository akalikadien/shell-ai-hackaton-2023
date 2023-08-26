# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adarsh Kalikadien                     #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #

import pandas as pd
from sklearn_extra.cluster import KMedoids
import numpy as np
from tqdm import tqdm
import pickle

from process_all_data_for_submission import process_flow_matrix


class BiomassGeneticAlgorithm:
    def __init__(self, biomass_history_file, distance_matrix_file, year, num_depots, num_biorefineries, genetic_algo_params_dict=None, find_depot_and_refinery_clusters=True, depot_cluster_centers=None, refinery_cluster_centers=None):
        # data and initial placement variables
        self.biomass_history_file = biomass_history_file
        self.distance_matrix_file = distance_matrix_file
        self.year = year
        self.num_depots = num_depots
        self.num_biorefineries = num_biorefineries
        self.biomass_df = None
        self.depot_cluster_centers = None
        self.refinery_cluster_centers = None
        self.depot_cluster_center_location_indices = None
        self.refinery_cluster_center_location_indices = None
        self.dist_sites_to_depots = None
        self.dist_depots_to_refineries = None
        self.biomass_df_values = None
        self.n_sites = None
        self.best_flow_solution = None
        self.flow_sites_to_depots, self.flow_depots_to_refineries = None, None
        self.final_biomass_demand_supply_df, self.final_pellet_demand_supply_df = None, None

        # genetic algorithm variables
        if genetic_algo_params_dict is None:
            self.population_size = 50
            self.num_generations = 50
            self.crossover_rate = 0.8
            self.mutation_rate = 0.2
            self.elite_size = 5
        else:
            self.population_size = genetic_algo_params_dict['population_size']
            self.num_generations = genetic_algo_params_dict['num_generations']
            self.crossover_rate = genetic_algo_params_dict['crossover_rate']
            self.mutation_rate = genetic_algo_params_dict['mutation_rate']
            self.elite_size = genetic_algo_params_dict['elite_size']

        self.find_depot_and_refinery_clusters = find_depot_and_refinery_clusters
        # add the depot and refinery cluster centers if we have them already
        if self.find_depot_and_refinery_clusters == False and depot_cluster_centers is not None and refinery_cluster_centers is not None:
            self.set_cluster_centers(depot_cluster_centers, refinery_cluster_centers)
        elif self.find_depot_and_refinery_clusters == False and (depot_cluster_centers is None or refinery_cluster_centers is None):
            raise ValueError('You must provide the depot and refinery cluster centers if you do not want to find them with k-medoids.')

    @staticmethod
    def calculate_overall_cost(transportation_cost, underutilization_cost):
        return 0.001 * transportation_cost + underutilization_cost

    @staticmethod
    def balance_mass(flow_matrix):
        # Ensure mass balance for each depot (sum of flow in = sum of flow out)
        for j in range(flow_matrix.shape[1]):
            total_inflow = np.sum(flow_matrix[:, j])
            total_outflow = np.sum(flow_matrix[j, :])
            if total_outflow != 0:  # Avoid division by zero
                if total_inflow < total_outflow:
                    # Adjust the inflow to match the outflow
                    flow_matrix[:, j] *= total_outflow / total_inflow
                elif total_inflow > total_outflow:
                    # Adjust the outflow to match the inflow
                    flow_matrix[j, :] *= total_inflow / total_outflow

        return flow_matrix

    def set_cluster_centers(self, depot_cluster_centers, refinery_cluster_centers):
        # in case we want to use the cluster centers from another year in our current year
        self.depot_cluster_centers = depot_cluster_centers
        self.refinery_cluster_centers = refinery_cluster_centers

    def check_constraints(self):
        # 1. Ensure the biomass procured for processing from each harvesting site 'i' is less than or equal to that site's biomass.
        constraint_1 = np.all(self.flow_sites_to_depots.sum(axis=1) <= self.biomass_df_values)
        # 2. Ensure the total biomass reaching each preprocessing depot 'j' is less than or equal to its yearly processing capacity (20,000).
        constraint_2 = np.all(self.flow_sites_to_depots.sum(axis=0) <= 20000)
        # 3. Ensure the total pellets reaching each refinery 'k' is less than or equal to its yearly processing capacity (100,000).
        constraint_3 = np.all(self.flow_depots_to_refineries.sum(axis=0) <= 100000)
        # 4. Verify that the number of depots is less than or equal to 25.
        constraint_4 = self.num_depots <= 25  # ToDo: better to check this via the number of clusters in the k-medoids algorithm?
        # 5. Verify that the number of refineries is less than or equal to 5.
        constraint_5 = self.num_biorefineries <= 5
        # 6. Ensure that at least 80% of the total biomass is processed by refineries each year.
        processed_biomass = self.flow_depots_to_refineries.sum()
        constraint_6 = processed_biomass >= 0.8 * self.biomass_df_values.sum()
        # 7. Verify that the total amount of biomass entering each preprocessing depot is equal to the total amount of pellets exiting that depot (within a tolerance limit of 1e-03).
        constraint_7 = all(
            np.isclose(
                self.flow_sites_to_depots[:, depot].sum(), 
                self.flow_depots_to_refineries[depot, :].sum(), 
                atol=1e-03
            )
            for depot in range(self.num_depots)
        )
        # 8. Ensure that there's only one depot per grid block/location and only one biorefinery per grid block/location.
        # This constraint is inherently satisfied by the KMeans clustering, so we can safely set it to True.
        constraint_8 = True

        # put all constraints together in a dictionary so we can easily see which constraints are violated
        constraints_dict = {'constraint_1': constraint_1,
                            'constraint_2': constraint_2,
                            'constraint_3': constraint_3,
                            'constraint_4': constraint_4,
                            'constraint_5': constraint_5,
                            'constraint_6': constraint_6,
                            'constraint_7': constraint_7,
                            'constraint_8': constraint_8}
        return constraints_dict

    def initialize_data(self):
        # load data
        biomass_history = pd.read_csv(self.biomass_history_file)
        distance_matrix = pd.read_csv(self.distance_matrix_file)
        # drop Unnamed: 0 column from distance matrix
        distance_matrix = distance_matrix.drop('Unnamed: 0', axis=1)

        # self.biomass_df = biomass_history[['Index', 'Latitude', 'Longitude', self.year]]
        self.biomass_df = biomass_history[['Index', 'Latitude', 'Longitude', self.year]].copy()
        self.biomass_df_values = self.biomass_df[self.year].values
        self.n_sites = len(self.biomass_df)

        if self.find_depot_and_refinery_clusters:
            # Location Optimization for Preprocessing Depots using K-medoids clustering
            X = self.biomass_df[['Latitude', 'Longitude', year]].copy()
            # scaling not needed for 1D k-medoids clustering
            # X[self.year] = (X[self.year] - X[self.year].min()) / (X[self.year].max() - X[self.year].min())  # Normalize biomass values
            kmedoids = KMedoids(n_clusters=self.num_depots, random_state=42).fit(X)
            # self.biomass_df.loc[:, 'depot_cluster'] = kmedoids.labels_
            self.depot_cluster_centers = kmedoids.cluster_centers_
            # save the location index of each depot cluster center for later use (first item in each cluster)
            self.depot_cluster_center_location_indices = kmedoids.medoid_indices_

            # Location Optimization for Bio-refineries using K-means clustering
            kmedoids_refinery = KMedoids(n_clusters=self.num_biorefineries, random_state=42).fit(self.depot_cluster_centers)
            # depot_centers_df.loc[:, 'refinery_cluster'] = kmedoids_refinery.labels_
            self.refinery_cluster_centers = kmedoids_refinery.cluster_centers_
            # save the location index of each refinery cluster center for later use (first item in each cluster)
            self.refinery_cluster_center_location_indices = kmedoids_refinery.medoid_indices_
        elif self.depot_cluster_centers is None or self.refinery_cluster_centers is None:
            raise ValueError('Must specify depot and refinery cluster centers if find_depot_and_refinery_clusters is False')

        # make depot centers dataframe for populating distance matrices
        depot_centers_df = pd.DataFrame(self.depot_cluster_centers, columns=['Latitude', 'Longitude', 'Normalized Biomass'])

        # Create distance matrices for harvesting sites to depots and depots to refineries
        self.dist_sites_to_depots = np.zeros((self.n_sites, self.num_depots))
        self.dist_depots_to_refineries = np.zeros((self.num_depots, self.num_biorefineries))

        depot_centers_indices = {}
        for i in range(self.num_depots):
            depot_center = tuple(self.depot_cluster_centers[i, :2])  # Extract Latitude and Longitude
            depot_indices = self.biomass_df[
                (self.biomass_df['Latitude'] == depot_center[0]) &
                (self.biomass_df['Longitude'] == depot_center[1])
                ]['Index'].values
            depot_centers_indices[depot_center] = depot_indices

        for i in range(self.n_sites):
            for j in range(self.num_depots):
                depot_indices = depot_centers_indices[tuple(self.depot_cluster_centers[j, :2])]
                self.dist_sites_to_depots[i, j] = distance_matrix.iloc[i, depot_indices].min()

        for j in range(self.num_depots):
            for k in range(self.num_biorefineries):
                refinery_indices = depot_centers_df[
                    (depot_centers_df['Latitude'] == self.refinery_cluster_centers[k, 0]) &
                    (depot_centers_df['Longitude'] == self.refinery_cluster_centers[k, 1])
                    ].index
                self.dist_depots_to_refineries[j, k] = distance_matrix.iloc[depot_indices, refinery_indices].min()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            flow_sites_to_depots = np.random.rand(self.n_sites, self.num_depots) * np.tile(self.biomass_df_values[:, None],
                                                                               (1, self.num_depots))
            flow_depots_to_refineries = np.random.rand(self.num_depots, self.num_biorefineries)
            population.append((flow_sites_to_depots, flow_depots_to_refineries))
        return population

    @staticmethod
    def calculate_underutilization_cost_for_solution(flow_sites_to_depots, flow_depots_to_refineries):
        underutilization_cost_depots = np.sum(np.maximum(0, 20000 - np.sum(flow_sites_to_depots, axis=0)))
        underutilization_cost_refineries = np.sum(np.maximum(0, 100000 - np.sum(flow_depots_to_refineries, axis=0)))
        underutilization_cost = underutilization_cost_depots + underutilization_cost_refineries
        return underutilization_cost

    # Calculate transportation cost for a given flow solution
    def calculate_transportation_cost_for_solution(self, flow_sites_to_depots, flow_depots_to_refineries):
        cost_sites_to_depots = np.sum(flow_sites_to_depots * self.dist_sites_to_depots)
        cost_depots_to_refineries = np.sum(flow_depots_to_refineries * self.dist_depots_to_refineries)
        transportation_cost = cost_sites_to_depots + cost_depots_to_refineries
        return transportation_cost

    # Fitness function to evaluate a solution
    def fitness(self, solution):
        flow_sites_to_depots, flow_depots_to_refineries = solution
        # update flow values with the new solution
        self.flow_sites_to_depots, self.flow_depots_to_refineries = flow_sites_to_depots, flow_depots_to_refineries
        # calculate transportation cost
        transportation_cost = self.calculate_transportation_cost_for_solution(flow_sites_to_depots,
                                                                              flow_depots_to_refineries)
        # calculate underutilization cost
        underutilization_cost = self.calculate_underutilization_cost_for_solution(flow_sites_to_depots,
                                                                                  flow_depots_to_refineries)
        # calculate overall cost
        overall_cost = self.calculate_overall_cost(transportation_cost, underutilization_cost)

        return overall_cost

    # Selection operation
    def select_parents(self, population):
        # Calculate fitness for each solution in the population
        fitness_values = np.array([self.fitness(solution) for solution in population])

        scaled_fitness = fitness_values - np.min(fitness_values) + 1e-10
        sum_scaled_fitness = np.sum(scaled_fitness)  # Calculate the sum of scaled fitness values

        if sum_scaled_fitness == 0:
            # Handle the case where all scaled fitness values are very small or zero
            # In this case, assign equal probabilities to all solutions
            probs = np.full(len(population), 1 / len(population))
        else:
            probs = scaled_fitness / sum_scaled_fitness

        probs = np.clip(probs, 0, 1)
        probs /= np.sum(probs)

        parents_indices = np.random.choice(len(population), size=2, p=probs)
        return population[parents_indices[0]], population[parents_indices[1]]

    # Crossover operation
    def crossover(self, parent1, parent2):
        if np.random.rand() > self.crossover_rate:
            return parent1
        child1, child2 = parent1
        child3, child4 = parent2
        # Crossover for flow from sites to depots
        crossover_point = np.random.randint(0, child1.shape[0])
        offspring1 = np.vstack((child1[:crossover_point], child3[crossover_point:]))
        offspring2 = np.vstack((child3[:crossover_point], child1[crossover_point:]))
        # Crossover for flow from depots to refineries
        crossover_point = np.random.randint(0, child2.shape[0])
        offspring3 = np.vstack((child2[:crossover_point], child4[crossover_point:]))
        offspring4 = np.vstack((child4[:crossover_point], child2[crossover_point:]))

        # # adjust flows to satisfy Constraint 7 for child1 and child3
        # for j in range(child1.shape[1]):
        #     total_exit_flow = np.sum(child2[j, :])
        #     total_entry_flow = np.sum(child1[:, j])
        #     scaling_factor = total_exit_flow / total_entry_flow
        #     child1[:, j] *= scaling_factor
        #
        # for j in range(child3.shape[1]):
        #     total_exit_flow = np.sum(child4[j, :])
        #     total_entry_flow = np.sum(child3[:, j])
        #     scaling_factor = total_exit_flow / total_entry_flow
        #     child3[:, j] *= scaling_factor

        # Ensure mass balance for the offspring
        offspring1 = self.balance_mass(offspring1)
        offspring3 = self.balance_mass(offspring3)
        return offspring1, offspring3

    def mutate(self, offspring):
        # approach as commented on issue 8 (does not work for constraint 1, 2, 6, and 7)
        if np.random.rand() > self.mutation_rate:
            return offspring

        offspring1, offspring2 = offspring

        # Mutation for flow from sites to depots
        remaining_biomass = self.biomass_df_values.copy()
        depot_capacities = np.full(self.num_depots, 20000)
        for i in range(offspring1.shape[0]):
            for j in range(offspring1.shape[1]):
                # Calculate the remaining capacity of the depot
                remaining_capacity = depot_capacities[j]
                if remaining_capacity > 0 and remaining_biomass[i] > 0:
                    # Generate a new flow within the minimum of the remaining capacity, the site's biomass, and the remaining biomass
                    new_flow = min(remaining_capacity, remaining_biomass[i])
                    offspring1[i, j] = new_flow
                    remaining_biomass[i] -= new_flow
                    depot_capacities[j] -= new_flow

        # Mutation for flow from depots to refineries
        remaining_capacity = np.full(self.num_biorefineries, 100000)
        for j in range(offspring2.shape[0]):
            for k in range(offspring2.shape[1]):
                # Calculate the remaining capacity of the refinery
                depot_remaining_capacity = depot_capacities[j]
                refinery_remaining_capacity = remaining_capacity[k]
                if depot_remaining_capacity > 0 and refinery_remaining_capacity > 0:
                    # Generate a new flow within the remaining capacity
                    new_flow = min(depot_remaining_capacity, refinery_remaining_capacity)
                    offspring2[j, k] = new_flow
                    depot_capacities[j] -= new_flow
                    remaining_capacity[k] -= new_flow

        # Ensure mass balance for the offspring
        offspring1 = self.balance_mass(offspring1)
        offspring2 = self.balance_mass(offspring2)
        return offspring1, offspring2

    # def mutate(self, offspring):
    #    # old approach (that kinda worked, but not for constraint 1, 2 and 7)
    #     if np.random.rand() > self.mutation_rate:
    #         return offspring
    #
    #     offspring1, offspring2 = offspring
    #
    #     for i in range(offspring1.shape[0]):
    #         for j in range(offspring1.shape[1]):
    #             # calculate the remaining capacity of the depot
    #             remaining_capacity = max(0, 20000 - np.sum(offspring1[:, j]))
    #
    #             # generate a new random flow within the minimum of the remaining capacity and the site's biomass
    #             new_flow = np.random.rand() * min(remaining_capacity, self.biomass_df_values[i])
    #             # new_flow = np.random.rand() * remaining_capacity
    #             offspring1[i, j] = new_flow
    #
    #     # Mutation for flow from depots to refineries
    #     for j in range(offspring2.shape[0]):
    #         # adjust the flow such that the total exit flow from the depot is equal to the total entry flow
    #         # total_exit_flow = np.sum(offspring2[j, :])
    #         # total_entry_flow = np.sum(offspring1[:, j])
    #         # scaling_factor = total_exit_flow / total_entry_flow
    #         # offspring1[:, j] *= scaling_factor
    #         for k in range(offspring2.shape[1]):
    #             # Calculate the remaining capacity of the refinery
    #             remaining_capacity = max(0, 100000 - np.sum(offspring2[j, :]))
    #
    #             # Generate a new random flow within the remaining capacity
    #             new_flow = np.random.rand() * remaining_capacity
    #             offspring2[j, k] = new_flow
    #
    #     # Ensure mass balance for the offspring
    #     offspring1 = self.balance_mass(offspring1)
    #     offspring2 = self.balance_mass(offspring2)
    #     return offspring1, offspring2

    def run_genetic_algorithm(self, print_progress=False):
        # initialize data
        self.initialize_data()
        population = self.initialize_population()

        print('Running Genetic Algorithm')
        print('Number of generations: {}'.format(self.num_generations))
        for generation in tqdm(range(self.num_generations)):
            new_population = []
            fitness_values = [self.fitness(solution) for solution in
                              population]
            elites = np.argsort(fitness_values)[:self.elite_size]
            for elite in elites:
                new_population.append(population[elite])

            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents(population)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            population = new_population

        best_solution = min(population, key=lambda solution: self.fitness(solution))
        # update values for best solution
        self.best_flow_solution = best_solution
        self.flow_sites_to_depots = best_solution[0]
        self.flow_depots_to_refineries = best_solution[1]

        # write biomass and pellet flow to csv, correct indices of depots and refineries to location indices
        biomass_flow_df = process_flow_matrix(optimizer.flow_sites_to_depots, self.year, 'biomass')
        # in biomass_flow_df the destination_index is now the depot's index, but this should be mapped to the correct
        # value from the list of self.depot_cluster_center_location_indices
        biomass_flow_df['destination_index'] = self.depot_cluster_center_location_indices[
            biomass_flow_df['destination_index'].values]
        pellet_flow_df = process_flow_matrix(optimizer.flow_depots_to_refineries, self.year, 'pellet')
        # in pellet_flow_df the destination_index is now the refinery's index, but this should be mapped to the correct
        # value from the list of self.refinery_cluster_center_location_indices
        pellet_flow_df['destination_index'] = self.refinery_cluster_center_location_indices[
            pellet_flow_df['destination_index'].values]
        # and the source index should be mapped to the correct value from the list of self.depot_cluster_center_location_indices
        pellet_flow_df['source_index'] = self.depot_cluster_center_location_indices[
            pellet_flow_df['source_index'].values]
        self.final_biomass_demand_supply_df, self.final_pellet_demand_supply_df = biomass_flow_df, pellet_flow_df
        biomass_flow_df.to_csv(f'dataset/3.predictions/biomass_flow_{self.year}.csv', index=False)
        pellet_flow_df.to_csv(f'dataset/3.predictions/pellet_flow_{self.year}.csv', index=False)

        if print_progress:
            # calulate overall costs for best solution
            print(f'Transportation cost: {self.calculate_transportation_cost_for_solution(self.flow_sites_to_depots, self.flow_depots_to_refineries)}')
            print(f'Underutilization cost: {self.calculate_underutilization_cost_for_solution(self.flow_sites_to_depots, self.flow_depots_to_refineries)}')
            print('Calculating overall costs for best solution')
            print(self.fitness(self.best_flow_solution))

            # check if the solution is feasible
            constraints_dict = self.check_constraints()
            print("Constraints:", constraints_dict)


if __name__ == "__main__":
    # biomass_history_file = 'dataset/1.initial_datasets/Biomass_History.csv'
    biomass_history_file = 'dataset/3.predictions/20230826Biomass_Predictions.csv'
    distance_matrix_file = 'dataset/1.initial_datasets/Distance_Matrix.csv'
    year = '2018'
    num_depots = 20
    num_biorefineries = 4

    genetic_algo_params = {
        'population_size': 50,
        'num_generations': 50,
        'elite_size': 5,
        'crossover_rate': 0.8,
        'mutation_rate': 0.2
    }

    # first run the genetic algo for 2018 normally
    optimizer = BiomassGeneticAlgorithm(biomass_history_file,
                                        distance_matrix_file,
                                        year,
                                        num_depots,
                                        num_biorefineries,
                                        genetic_algo_params,
                                        find_depot_and_refinery_clusters=True,
                                        depot_cluster_centers=None,
                                        refinery_cluster_centers=None)
    optimizer.run_genetic_algorithm(print_progress=True)
    depot_cluster_center_location_indices_2018 = optimizer.depot_cluster_center_location_indices
    refinery_cluster_center_location_indices_2018 = optimizer.refinery_cluster_center_location_indices

    # write optimizer object to pickle file after running the genetic algorithm
    with open(f"dataset/3.predictions/optimizer_{year}_dpts_{num_depots}_brfnrs_{num_biorefineries}_pop_{genetic_algo_params['population_size']}.pkl", 'wb') as f:
        pickle.dump(optimizer, f)

    # then use the depot and refinery cluster centers from the 2018 run to run the genetic algo for 2019
    # load the depot and refinery cluster centers from the 2018 run
    depot_cluster_centers_2018 = optimizer.depot_cluster_centers
    refinery_cluster_centers_2018 = optimizer.refinery_cluster_centers

    # run the genetic algo for 2019 using the depot and refinery cluster centers from the 2018 run
    year = '2019'
    optimizer_2019 = BiomassGeneticAlgorithm(biomass_history_file, distance_matrix_file, year, num_depots, num_biorefineries, genetic_algo_params, find_depot_and_refinery_clusters=False, depot_cluster_centers=depot_cluster_centers_2018, refinery_cluster_centers=refinery_cluster_centers_2018)
    optimizer_2019.depot_cluster_center_location_indices = depot_cluster_center_location_indices_2018
    optimizer_2019.refinery_cluster_center_location_indices = refinery_cluster_center_location_indices_2018
    optimizer_2019.run_genetic_algorithm(print_progress=True)

    # write optimizer object to pickle file after running the genetic algorithm
    with open(f"dataset/3.predictions/optimizer_{year}_dpts_{num_depots}_brfnrs_{num_biorefineries}_pop_{genetic_algo_params['population_size']}.pkl", 'wb') as f:
        pickle.dump(optimizer_2019, f)
