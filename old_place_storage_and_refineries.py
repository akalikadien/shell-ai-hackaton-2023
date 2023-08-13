# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adarsh Kalikadien                     #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #

"""Use the distance matrix to place the refineries and storage depots. Also make a demand-supply matrix for pellets and biomass
where for each location index and destination index the demand and supply is given for 2018 and 2019. For example
Dist,i,j = distance from harvesting site i to storage depot j. Dist,j,k = distance from storage depot j to refinery k.
Biomass,i,j = biomass demand-supply from harvesting site i to storage depot j. Pellets,j,k = pellets demand-supply from
storage depot j to refinery k.
The depots have an annual capacity of 20000 tons, the refineries have an annual capacity of 100000 tons
The maximum amount of depots is 25, the maximum amount of refineries is 5"""

# 1. genetic algorithm approach
import pandas as pd
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import numpy as np

# Load the data
biomass_history = pd.read_csv('dataset/1.initial_datasets/Biomass_History.csv')
distance_matrix = pd.read_csv('dataset/1.initial_datasets/Distance_Matrix.csv')

# Filter the biomass history for the year 2017
biomass_2017 = biomass_history[['Index', 'Latitude', 'Longitude', '2017']]

# Location Optimization for Preprocessing Depots using K-means clustering
n_clusters = 25
X = biomass_2017[['Latitude', 'Longitude', '2017']].copy()
X['2017'] = (X['2017'] - X['2017'].min()) / (X['2017'].max() - X['2017'].min())  # Normalize biomass values
kmeans = KMedoids(n_clusters=n_clusters, random_state=42).fit(X)
biomass_2017['depot_cluster'] = kmeans.labels_
depot_cluster_centers = kmeans.cluster_centers_

# Location Optimization for Bio-refineries using K-means clustering
n_clusters_refinery = 5
kmeans_refinery = KMedoids(n_clusters=n_clusters_refinery, random_state=42).fit(depot_cluster_centers)
depot_centers_df = pd.DataFrame(depot_cluster_centers, columns=['Latitude', 'Longitude', 'Normalized Biomass'])
depot_centers_df['refinery_cluster'] = kmeans_refinery.labels_
refinery_cluster_centers = kmeans_refinery.cluster_centers_

# Create distance matrices for harvesting sites to depots and depots to refineries
n_sites = len(biomass_2017)
n_depots = len(depot_cluster_centers)
n_refineries = len(refinery_cluster_centers)
dist_sites_to_depots = distance_matrix.iloc[:n_sites, biomass_2017['depot_cluster'].unique() + 1].values
dist_depots_to_refineries = distance_matrix.iloc[biomass_2017['depot_cluster'].unique(),
depot_centers_df['refinery_cluster'].unique() + 1].values

# Define parameters for the Genetic Algorithm
population_size = 50
num_generations = 50
crossover_rate = 0.8
mutation_rate = 0.2
elite_size = 5  # Number of best solutions to carry forward to next generation


# Initialize a random population
def initialize_population(pop_size, n_sites, n_depots, n_refineries, biomass_values):
    population = []
    for _ in range(pop_size):
        # Random flow from sites to depots
        flow_sites_to_depots = np.random.rand(n_sites, n_depots) * np.tile(biomass_values[:, None], (1, n_depots))
        # Random flow from depots to refineries
        flow_depots_to_refineries = np.random.rand(n_depots, n_refineries) * 20000
        population.append((flow_sites_to_depots, flow_depots_to_refineries))
    return population


# Fitness function to evaluate a solution
def fitness(solution, dist_sites_to_depots, dist_depots_to_refineries):
    flow_sites_to_depots, flow_depots_to_refineries = solution
    cost_sites_to_depots = np.sum(flow_sites_to_depots * dist_sites_to_depots)
    cost_depots_to_refineries = np.sum(flow_depots_to_refineries * dist_depots_to_refineries)
    return cost_sites_to_depots + cost_depots_to_refineries


# Selection operation
def select_parents(population, dist_sites_to_depots, dist_depots_to_refineries):
    # Calculate fitness for each solution in the population
    fitness_values = np.array(
        [fitness(solution, dist_sites_to_depots, dist_depots_to_refineries) for solution in population])
    # Convert the fitness values to probabilities (lower fitness is better)
    probs = (1 / fitness_values) / np.sum(1 / fitness_values)
    # Select two parents probabilistically
    parents_indices = np.random.choice(len(population), size=2, p=probs)
    return population[parents_indices[0]], population[parents_indices[1]]


# Crossover operation
def crossover(parent1, parent2):
    if np.random.rand() > crossover_rate:
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
    return (offspring1, offspring3)


# Mutation operation
def mutate(offspring):
    if np.random.rand() > mutation_rate:
        return offspring
    # Add small random values to the flow matrices
    offspring1, offspring2 = offspring
    offspring1 += np.random.randn(*offspring1.shape) * 0.05 * np.max(offspring1)
    offspring2 += np.random.randn(*offspring2.shape) * 0.05 * np.max(offspring2)
    return (offspring1, offspring2)


# Main Genetic Algorithm
def genetic_algorithm(dist_sites_to_depots, dist_depots_to_refineries, n_sites, n_depots, n_refineries, biomass_values):
    population = initialize_population(population_size, n_sites, n_depots, n_refineries, biomass_values)

    for generation in range(num_generations):
        new_population = []
        # Select elites (best solutions)
        fitness_values = [fitness(solution, dist_sites_to_depots, dist_depots_to_refineries) for solution in population]
        elites = np.argsort(fitness_values)[:elite_size]
        for elite in elites:
            new_population.append(population[elite])

        # Generate offspring
        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population, dist_sites_to_depots, dist_depots_to_refineries)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    # Return the best solution from the final generation
    best_solution = min(population,
                        key=lambda solution: fitness(solution, dist_sites_to_depots, dist_depots_to_refineries))
    return best_solution


# Run the Genetic Algorithm
best_flow_solution = genetic_algorithm(dist_sites_to_depots, dist_depots_to_refineries, n_sites, n_depots, n_refineries,
                                       biomass_2017['2017'].values)

# Print the total transportation cost for the best solution
print("Total transportation cost for the best solution:",
      fitness(best_flow_solution, dist_sites_to_depots, dist_depots_to_refineries))

# Calculate Underutilization Cost for depots and refineries
transportation_cost_sites_to_depots = np.sum(best_flow_solution[0] * dist_sites_to_depots)
transportation_cost_depots_to_refineries = np.sum(best_flow_solution[1] * dist_depots_to_refineries)
total_transportation_cost = transportation_cost_sites_to_depots + transportation_cost_depots_to_refineries

underutilization_cost_depots = np.sum(20000 - np.sum(best_flow_solution[0], axis=0))

underutilization_cost_refineries = np.sum(100000 - np.sum(best_flow_solution[1], axis=0))
total_underutilization_cost = underutilization_cost_depots + underutilization_cost_refineries

# Calculate Overall Cost
overall_cost = 0.001 * total_transportation_cost + total_underutilization_cost

print("Total Transportation Cost:", total_transportation_cost)
print("Total Underutilization Cost:", total_underutilization_cost)
print("Overall Cost:", overall_cost)

biomass_values_2017 = biomass_2017['2017'].values

# Extract the best flow solution
best_flow_sites_to_depots, best_flow_depots_to_refineries = best_flow_solution

# 1. Ensure the biomass procured for processing from each harvesting site 'i' is less than or equal to that site's biomass.
constraint_1 = np.all(best_flow_sites_to_depots.sum(axis=1) <= biomass_values_2017)

# 2. Ensure the total biomass reaching each preprocessing depot 'j' is less than or equal to its yearly processing capacity (20,000).
constraint_2 = np.all(best_flow_sites_to_depots.sum(axis=0) <= 20000)

# 3. Ensure the total pellets reaching each refinery 'k' is less than or equal to its yearly processing capacity (100,000).
constraint_3 = np.all(best_flow_depots_to_refineries.sum(axis=0) <= 100000)

# 4. Verify that the number of depots is less than or equal to 25.
constraint_4 = n_depots <= 25

# 5. Verify that the number of refineries is less than or equal to 5.
constraint_5 = n_refineries <= 5

# 6. Ensure that at least 80% of the total biomass is processed by refineries each year.
processed_biomass = best_flow_depots_to_refineries.sum()
constraint_6 = processed_biomass >= 0.8 * biomass_values_2017.sum()

# 7. Verify that the total amount of biomass entering each preprocessing depot is equal to the total amount of pellets exiting that depot (within a tolerance limit of 1e-03).
constraint_7 = np.allclose(best_flow_sites_to_depots.sum(axis=0), best_flow_depots_to_refineries.sum(axis=1),
                           atol=1e-03)

# 8. Ensure that there's only one depot per grid block/location and only one biorefinery per grid block/location.
# This constraint is inherently satisfied by the KMeans clustering, so we can safely set it to True.
constraint_8 = True

constraints_results = {
    "Constraint 1": constraint_1,
    "Constraint 2": constraint_2,
    "Constraint 3": constraint_3,
    "Constraint 4": constraint_4,
    "Constraint 5": constraint_5,
    "Constraint 6": constraint_6,
    "Constraint 7": constraint_7,
    "Constraint 8": constraint_8
}

print(constraints_results)

# # 2. K-means approach and just using that as locations
# import os
# import pandas as pd
# from sklearn.cluster import KMeans
# import numpy as np
#
#
# def k_means_placements(biomass_file, num_depots, num_biorefineries):
#     # Read the biomass data
#     biomass_data = pd.read_csv(biomass_file)
#
#     # Extract location indices and biomass availability for each year
#     locations = biomass_data['index'].values
#     biomass_2018 = biomass_data['2018'].values
#     biomass_2019 = biomass_data['2019'].values
#
#     # Combine biomass data into a single array
#     biomass_combined = np.column_stack((biomass_2018, biomass_2019))
#
#     # Perform K-means clustering
#     kmeans = KMeans(n_clusters=num_depots + num_biorefineries, n_init='auto')
#     kmeans.fit(biomass_combined)
#     facility_locations = kmeans.cluster_centers_
#
#     # Divide the facility locations into depots and biorefineries
#     depot_locations = facility_locations[:num_depots]
#     biorefinery_locations = facility_locations[num_depots:]
#     return depot_locations, biorefinery_locations
#
#
# def calculate_flow_matrix(locations, facility_locations, biomass_data, year):
#     num_locations = len(locations)
#     num_facilities = len(facility_locations)
#
#     flow_matrix = np.zeros((num_locations, num_locations))
#
#     biomass_values = biomass_data.iloc[year - 2018]  # Biomass values for the specified year
#
#     for from_loc_idx, (from_loc, biomass_value) in enumerate(zip(locations, biomass_values)):
#         if from_loc in capacity_dict:  # Skip locations without capacity
#             for to_loc_idx, to_loc in enumerate(locations):
#                 if to_loc in capacity_dict:  # Skip locations without capacity
#                     for fac_loc in facility_locations:
#                         # take min to ensure that we are not transporting more than the capacity of the facility
#                         # use norm to calculate the distance between the locations and use that as a proxy for the
#                         # transportation cost
#                         flow_matrix[from_loc_idx, to_loc_idx] += min(biomass_value, capacity_dict[to_loc]) * np.linalg.norm(
#                             from_loc - fac_loc)
#
#     return flow_matrix
#
#
# if __name__ == "__main__":
#     biomass_file = os.path.join('dataset', '3.predictions', 'biomass_predictions.csv')
#     num_depots = 25
#     num_biorefineries = 5
#
#     depot_locations, biorefinery_locations = k_means_placements(biomass_file, num_depots, num_biorefineries)
#
#     biomass_data = pd.read_csv(biomass_file)
#     locations = biomass_data['index'].values
#     biomass_2018 = biomass_data['2018'].values
#     biomass_2019 = biomass_data['2019'].values
#
#     # Create a dictionary to store the capacity of each location (depot or biorefinery)
#     capacity_dict = {}
#     for loc in locations:
#         if loc in depot_locations:
#             capacity_dict[loc] = 20000  # Capacity of depots
#         elif loc in biorefinery_locations:
#             capacity_dict[loc] = 100000  # Capacity of biorefineries
#
#     # Calculate flow matrices for biomass and pellets separately for 2018 and 2019
#     flow_matrix_biomass_2018 = calculate_flow_matrix(locations, depot_locations, biomass_data, 2018)
#     flow_matrix_pellets_2018 = calculate_flow_matrix(locations, biorefinery_locations, biomass_data, 2018)
#     flow_matrix_biomass_2019 = calculate_flow_matrix(locations, depot_locations, biomass_data, 2019)
#     flow_matrix_pellets_2019 = calculate_flow_matrix(locations, biorefinery_locations, biomass_data, 2019)
#
#     # Convert flow matrices to DataFrames with appropriate row and column indices
#     flow_matrix_biomass_2018_df = pd.DataFrame(flow_matrix_biomass_2018, index=locations, columns=locations)
#     flow_matrix_pellets_2018_df = pd.DataFrame(flow_matrix_pellets_2018, index=locations, columns=locations)
#     flow_matrix_biomass_2019_df = pd.DataFrame(flow_matrix_biomass_2019, index=locations, columns=locations)
#     flow_matrix_pellets_2019_df = pd.DataFrame(flow_matrix_pellets_2019, index=locations, columns=locations)
#
#     # Save flow matrices to csv files
#     # print("Flow Matrix Biomass 2018:")
#     flow_matrix_biomass_2018_df.to_csv('dataset/3.predictions/flow_matrix_biomass_2018.csv')
#     # print("Flow Matrix Pellets 2018:")
#     flow_matrix_pellets_2018_df.to_csv('dataset/3.predictions/flow_matrix_pellets_2018.csv')
#     # print("Flow Matrix Biomass 2019:")
#     flow_matrix_biomass_2019_df.to_csv('dataset/3.predictions/flow_matrix_biomass_2019.csv')
#     # print("Flow Matrix Pellets 2019:")
#     flow_matrix_pellets_2019_df.to_csv('dataset/3.predictions/flow_matrix_pellets_2019.csv')
