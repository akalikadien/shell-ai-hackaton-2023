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

# WIP, finding algo to place depots and refineries
# # 2. Use genetic algorithm
# import os
# import pandas as pd
# import numpy as np
# from deap import algorithms, base, creator, tools
#
# # Read the biomass data
# biomass_data = pd.read_csv(os.path.join('dataset', '3.predictions', 'biomass_predictions.csv'))
#
# # Extract location indices and biomass availability for each year
# locations = biomass_data['index'].values
# biomass_2018 = dict(zip(locations, biomass_data['2018'].values))
# biomass_2019 = dict(zip(locations, biomass_data['2019'].values))
#
# # Read the distance matrix data
# distance_matrix_data = pd.read_csv(os.path.join('dataset', '1.initial_datasets', 'Distance_Matrix.csv'), index_col=0)
# distance_matrix = np.array(distance_matrix_data)
#
# # Define the problem as a minimization problem
# creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
#
# # Define the individual (solution) as a list of binary values (0 or 1) representing depot and biorefinery locations
# creator.create("Individual", list, fitness=creator.FitnessMin)
#
# # Initialize the Genetic Algorithm toolbox
# toolbox = base.Toolbox()
#
# # Attribute generator: randomly initialize an individual
# toolbox.register("attr_bool", np.random.randint, 2)
#
# # Structure initializers: create individuals (solutions) and populations (collections of individuals)
# toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(locations) * 2)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#
# # Evaluation function: calculate the total cost and underutilization cost for an individual
# def evaluate(individual):
#     depot_indices = [idx for idx, gene in enumerate(individual) if gene == 1][:25]
#     biorefinery_indices = [idx for idx, gene in enumerate(individual) if gene == 1][25:]
#
#     # Calculate transportation costs
#     total_transport_cost = sum(distance_matrix[loc_idx-1][depot_idx-1] * biomass_2018[loc_idx+1] for loc_idx in range(len(locations)) for depot_idx in depot_indices) + \
#                           sum(distance_matrix[depot_idx-1][bio_idx-1] * 20000 for depot_idx in depot_indices for bio_idx in biorefinery_indices)
#
#     # Calculate underutilization costs
#     depot_underutilization_cost = sum((20000 - sum(biomass_2018[loc_idx+1] for loc_idx in range(len(locations)) if individual[loc_idx] == 1)) for depot_idx in depot_indices)
#     biorefinery_underutilization_cost = sum((100000 - sum(20000 for depot_idx in depot_indices if individual[depot_idx] == 1)) for bio_idx in biorefinery_indices)
#
#     return total_transport_cost, depot_underutilization_cost + biorefinery_underutilization_cost
#
#
#
#
# # Register the evaluation function to the toolbox
# toolbox.register("evaluate", evaluate)
#
# # Genetic Operators
# toolbox.register("mate", tools.cxTwoPoint)
# toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
# toolbox.register("select", tools.selNSGA2)
#
# def main():
#     # Population size
#     pop_size = 100
#
#     # Number of generations
#     ngen = 100
#
#     # Crossover probability
#     cxpb = 0.5
#
#     # Mutation probability
#     mutpb = 0.2
#
#     # Create the population and define the Hall of Fame (stores the best individuals in each generation)
#     pop = toolbox.population(n=pop_size)
#     hof = tools.ParetoFront()
#
#     # Run the Genetic Algorithm
#     algorithms.eaSimple(pop, toolbox, cxpb, mutpb, ngen, halloffame=hof)
#
#     # Print the best individuals (solutions) found by the GA
#     best_individuals = hof[0:5]  # Select the top 5 best individuals (non-dominated solutions)
#     print("Optimal locations for depots and biorefineries:")
#     for individual in best_individuals:
#         depot_indices = [idx for idx, gene in enumerate(individual) if gene == 1][:25]
#         biorefinery_indices = [idx for idx, gene in enumerate(individual) if gene == 1][25:]
#         print("Depot locations:", depot_indices)
#         print("Biorefinery locations:", biorefinery_indices)
#         print("-------------------------")
#
# if __name__ == "__main__":
#     main()


# # 1. use linear programming?
# import os
# import pandas as pd
# from pulp import *
#
# def find_optimal_locations(biomass_file, distance_matrix_file, num_depots=25, num_biorefineries=5):
#     # Read the biomass data
#     biomass_data = pd.read_csv(biomass_file)
#
#     # Extract location indices and biomass availability for each year
#     locations = biomass_data['index'].values
#     biomass_2018 = dict(zip(locations, biomass_data['2018'].values))
#     biomass_2019 = dict(zip(locations, biomass_data['2019'].values))
#
#     # Read the distance matrix data and set the first row and column as the index and columns
#     distance_matrix_data = pd.read_csv(distance_matrix_file, index_col=0)
#
#     # Create a PuLP problem
#     prob = LpProblem("Optimal_Location_Problem", LpMinimize)
#
#     # Variables: binary variables to indicate whether a depot/biorefinery is placed at a location
#     depot_vars = LpVariable.dicts("Depot", locations, cat='Binary')
#     biorefinery_vars = LpVariable.dicts("Biorefinery", locations, cat='Binary')
#
#     # Cost variables
#     # Transportation costs
#     depot_transportation_cost = LpVariable("Depot_Transport_Cost")
#     biorefinery_transportation_cost = LpVariable("Biorefinery_Transport_Cost")
#
#     # Underutilization costs
#     depot_underutilization_cost = LpVariable("Depot_Underutilization_Cost")
#     biorefinery_underutilization_cost = LpVariable("Biorefinery_Underutilization_Cost")
#
#     # Objective function: minimize the total costs (transportation and underutilization)
#     prob += depot_transportation_cost + biorefinery_transportation_cost + depot_underutilization_cost + biorefinery_underutilization_cost, "Total_Cost"
#
#     # Minimize transportation costs
#     prob += depot_transportation_cost == lpSum([
#         distance_matrix_data.at[loc, depot_loc] * biomass_2018[loc] * depot_vars[depot_loc]
#         for loc in locations for depot_loc in locations
#     ]) + lpSum([
#         distance_matrix_data.at[depot_loc, loc] * 20000 * depot_vars[depot_loc] * biorefinery_vars[loc]
#         for loc in locations for depot_loc in locations
#     ]), "Depot_Transportation_Cost"
#
#     prob += biorefinery_transportation_cost == lpSum([
#         distance_matrix_data.at[loc, biorefinery_loc] * 100000 * biorefinery_vars[loc]
#         for loc in locations for biorefinery_loc in locations
#     ]), "Biorefinery_Transportation_Cost"
#
#     # Minimize underutilization costs
#     prob += depot_underutilization_cost == lpSum([
#         (20000 - lpSum(biomass_2018[loc] * depot_vars[loc] + biomass_2019[loc] * depot_vars[loc] for loc in locations)) * depot_vars[loc]
#         for loc in locations
#     ]), "Depot_Underutilization_Cost"
#
#     prob += biorefinery_underutilization_cost == lpSum([
#         (100000 - lpSum(20000 * depot_vars[loc] * biorefinery_vars[loc] for loc in locations)) * biorefinery_vars[loc]
#         for loc in locations
#     ]), "Biorefinery_Underutilization_Cost"
#
#     # Constraints
#
#     # Constraint 1: Total number of depots and biorefineries should not exceed the given limits
#     prob += lpSum([depot_vars[loc] for loc in locations]) == num_depots, "Num_Depots"
#     prob += lpSum([biorefinery_vars[loc] for loc in locations]) == num_biorefineries, "Num_Biorefineries"
#
#     # Constraint 2: Each location should have either a depot or a biorefinery
#     for loc in locations:
#         prob += depot_vars[loc] + biorefinery_vars[loc] == 1, f"Placement_Constraint_{loc}"
#
#     # Constraint 3: At least 80% of total biomass should be processed by biorefineries each year
#     prob += lpSum([biomass_2018[loc] * biorefinery_vars[loc] + biomass_2019[loc] * biorefinery_vars[loc]
#                    for loc in locations]) >= 0.8 * lpSum([biomass_2018[loc] + biomass_2019[loc]
#                                                          for loc in locations]), "Min_Biorefinery_Biomass"
#
#     # Constraint 4: Amount of biomass leaving each harvesting site towards a depot should be less or equal to the biomass availability for that year
#     for loc in locations:
#         prob += lpSum([biomass_2018[loc] * depot_vars[loc] + biomass_2019[loc] * depot_vars[loc]
#                        for loc in locations]) <= biomass_2018[loc] + biomass_2019[loc], f"Max_Harvest_Biomass_{loc}"
#
#     # Solve the problem
#     prob.solve()
#
#     # Print the results
#     print("Status:", LpStatus[prob.status])
#     print("Optimal locations for depots:")
#     for loc in locations:
#         if depot_vars[loc].value() == 1:
#             print(f"Depot at location {loc}")
#     print("Optimal locations for biorefineries:")
#     for loc in locations:
#         if biorefinery_vars[loc].value() == 1:
#             print(f"Biorefinery at location {loc}")
#
#
# if __name__ == "__main__":
#     biomass_file = os.path.join('dataset', '3.predictions', 'biomass_predictions.csv')
#     distance_matrix_file = os.path.join('dataset', '1.initial_datasets', 'Distance_Matrix.csv')
#     find_optimal_locations(biomass_file, distance_matrix_file)
