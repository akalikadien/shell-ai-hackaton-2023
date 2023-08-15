# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adarsh Kalikadien                     #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #

import pandas as pd
from sklearn_extra.cluster import KMedoids
import numpy as np
from tqdm import tqdm


class BiomassGeneticAlgorithm:
    def __init__(self, biomass_history_file, distance_matrix_file, year, num_depots, num_biorefineries):
        # data and initial placement variables
        self.biomass_history_file = biomass_history_file
        self.distance_matrix_file = distance_matrix_file
        self.year = year
        self.num_depots = num_depots
        self.num_biorefineries = num_biorefineries
        self.biomass_df = None
        self.depot_cluster_centers = None
        self.refinery_cluster_centers = None
        self.dist_sites_to_depots = None
        self.dist_depots_to_refineries = None
        self.biomass_df_values = None
        self.n_sites = None
        self.best_flow_solution = None
        self.flow_sites_to_depots, self.flow_depots_to_refineries = None, None

        # genetic algorithm variables
        self.population_size = 100
        self.num_generations = 500
        self.crossover_rate = 0.8
        self.mutation_rate = 0.2
        self.elite_size = 5

    @staticmethod
    def calculate_overall_cost(transportation_cost, underutilization_cost):
        return 0.001 * transportation_cost + underutilization_cost

    def calculate_transportation_cost(self):
        if self.best_flow_solution is None:
            raise ValueError('No best flow solution found. Run genetic algorithm first.')
        self.flow_sites_to_depots, self.flow_depots_to_refineries = self.best_flow_solution
        cost_sites_to_depots = np.sum(self.flow_sites_to_depots * self.dist_sites_to_depots)
        cost_depots_to_refineries = np.sum(self.flow_depots_to_refineries * self.dist_depots_to_refineries)
        transportation_cost = cost_sites_to_depots + cost_depots_to_refineries
        return transportation_cost

    def calculate_underutilization_cost(self):
        if self.best_flow_solution is None:
            raise ValueError('No best flow solution found. Run genetic algorithm first.')
        self.flow_sites_to_depots, self.flow_depots_to_refineries = self.best_flow_solution
        # ensure that only positive values are used for the underutilization cost
        # calculate underutilization cost for depots by checking how far below the capacity the depots are
        underutilization_cost_depots = np.sum(np.maximum(0, 20000 - np.sum(self.flow_sites_to_depots, axis=0)))
        # calculate underutilization cost for refineries by checking how far below the capacity the refineries are
        underutilization_cost_refineries = np.sum(np.maximum(0, 100000 - np.sum(self.flow_depots_to_refineries, axis=0)))

        # underutilization_cost_depots = np.sum(20000 - np.sum(self.flow_sites_to_depots, axis=0))
        # underutilization_cost_refineries = np.sum(100000 - np.sum(self.flow_depots_to_refineries, axis=0))
        underutilization_cost = underutilization_cost_depots + underutilization_cost_refineries
        return underutilization_cost

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
        constraint_7 = np.allclose(self.flow_sites_to_depots.sum(axis=0), self.flow_depots_to_refineries.sum(axis=1),
                                   atol=1e-03)
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

        self.biomass_df = biomass_history[['Index', 'Latitude', 'Longitude', self.year]]
        self.biomass_df_values = self.biomass_df[self.year].values
        self.n_sites = len(self.biomass_df)

        # Location Optimization for Preprocessing Depots using K-means clustering
        X = self.biomass_df[['Latitude', 'Longitude', year]].copy()
        X[self.year] = (X[self.year] - X[self.year].min()) / (X[self.year].max() - X[self.year].min())  # Normalize biomass values
        kmedoids = KMedoids(n_clusters=self.num_depots, random_state=42).fit(X)
        self.biomass_df.loc[:, 'depot_cluster'] = kmedoids.labels_
        self.depot_cluster_centers = kmedoids.cluster_centers_

        # Location Optimization for Bio-refineries using K-means clustering
        kmedoids_refinery = KMedoids(n_clusters=self.num_biorefineries, random_state=42).fit(self.depot_cluster_centers)
        depot_centers_df = pd.DataFrame(self.depot_cluster_centers, columns=['Latitude', 'Longitude', 'Normalized Biomass'])
        depot_centers_df.loc[:, 'refinery_cluster'] = kmedoids_refinery.labels_
        self.refinery_cluster_centers = kmedoids_refinery.cluster_centers_

        # Create distance matrices for harvesting sites to depots and depots to refineries
        self.dist_sites_to_depots = distance_matrix.iloc[:self.n_sites, self.biomass_df['depot_cluster'].unique() + 1].values
        self.dist_depots_to_refineries = distance_matrix.iloc[self.biomass_df['depot_cluster'].unique(), depot_centers_df['refinery_cluster'].unique() + 1].values

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            flow_sites_to_depots = np.random.rand(self.n_sites, self.num_depots) * np.tile(self.biomass_df_values[:, None],
                                                                               (1, self.num_depots))
            flow_depots_to_refineries = np.random.rand(self.num_depots, self.num_biorefineries) * 20000
            population.append((flow_sites_to_depots, flow_depots_to_refineries))
        return population

    # Fitness function to evaluate a solution
    def fitness(self, solution):
        flow_sites_to_depots, flow_depots_to_refineries = solution
        cost_sites_to_depots = np.sum(flow_sites_to_depots * self.dist_sites_to_depots)
        cost_depots_to_refineries = np.sum(flow_depots_to_refineries * self.dist_depots_to_refineries)
        return cost_sites_to_depots + cost_depots_to_refineries

    # Selection operation
    def select_parents(self, population):
        # Calculate fitness for each solution in the population
        fitness_values = np.array([self.fitness(solution) for solution in population])

        # scale fitness values to avoid very small/large probabilities
        scaled_fitness = fitness_values - np.min(fitness_values) + 1e-10

        # calculate probabilities
        probs = scaled_fitness / np.sum(scaled_fitness)
        probs = np.clip(probs, 0, 1)  # ensure probabilities are within [0, 1]
        probs /= np.sum(probs)  # normalize probabilities

        # select two parents probabilistically
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
        return offspring1, offspring3

    # Mutation operation
    def mutate(self, offspring):
        if np.random.rand() > self.mutation_rate:
            return offspring
        # Add small random values to the flow matrices
        offspring1, offspring2 = offspring
        offspring1 += np.random.randn(*offspring1.shape) * 0.05 * np.max(offspring1)
        offspring2 += np.random.randn(*offspring2.shape) * 0.05 * np.max(offspring2)
        return offspring1, offspring2

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
        self.best_flow_solution = best_solution

        # calulate overall costs
        transportation_cost = self.calculate_transportation_cost()
        underutilization_cost = self.calculate_underutilization_cost()
        overall_cost = self.calculate_overall_cost(transportation_cost, underutilization_cost)

        # check if the solution is feasible
        constraints_dict = self.check_constraints()

        if print_progress:
            print("Transportation cost:", transportation_cost)
            print("Underutilization cost:", underutilization_cost)
            print("Overall cost:", overall_cost)
            print("Constraints:", constraints_dict)


if __name__ == "__main__":
    biomass_history_file = 'dataset/1.initial_datasets/Biomass_History.csv'
    distance_matrix_file = 'dataset/1.initial_datasets/Distance_Matrix.csv'
    year = '2017'
    num_depots = 25
    num_biorefineries = 5

    optimizer = BiomassGeneticAlgorithm(biomass_history_file, distance_matrix_file, year, num_depots, num_biorefineries)
    optimizer.run_genetic_algorithm(print_progress=True)
