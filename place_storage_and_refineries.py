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

# check https://scipbook.readthedocs.io/en/latest/flp.html for more info on the FLP
# WIP, 1 initial implementation via pyscipopt
import os
import multiprocessing

import pandas as pd
from pyscipopt import Model, quicksum, multidict, SCIP_PARAMSETTING


def get_optimal_thread_count(percentage=0.6):
    available_cores = multiprocessing.cpu_count()
    optimal_threads = int(available_cores * percentage)
    return optimal_threads


def calculate_transportation_cost(distance_matrix_data, depot_vars, biorefinery_vars, biomass_2018, biomass_2019, locations):
    """Transportation cost is defined as the sum of the distance times amount of pellet or biomass transported"""
    total_transport_cost = 0

    for loc_idx in range(len(locations)):
        loc = locations[loc_idx]
        for depot_idx in range(len(locations)):
            depot_loc = locations[depot_idx]
            total_transport_cost += distance_matrix_data.iloc[loc_idx, depot_idx] * (biomass_2018[loc_idx] + biomass_2019[loc_idx]) * depot_vars[depot_loc]
        for bio_idx in range(len(locations)):
            bio_loc = locations[bio_idx]
            total_transport_cost += distance_matrix_data.iloc[loc_idx, bio_idx] * (20000 * depot_vars[loc] + 100000 * biorefinery_vars[loc])

    return total_transport_cost


def calculate_underutilization_cost(depot_vars, biorefinery_vars, biomass_2018, biomass_2019, locations):
    """Underutilization cost is defined as the capacity of the depot or biorefinery minus the amount of biomass or pellets transported to the depot or biorefinery"""
    total_underutilization_cost = 0

    for loc_idx in range(len(locations)):
        loc = locations[loc_idx]
        total_underutilization_cost += max(0, 20000 - (biomass_2018[loc_idx] + biomass_2019[loc_idx])) * depot_vars[loc]
        total_underutilization_cost += max(0, 100000 - (biomass_2018[loc_idx] + biomass_2019[loc_idx])) * biorefinery_vars[loc]

    return total_underutilization_cost


def solve_optimization_problem(biomass_file, distance_matrix_file, year_1, year_2):
    # ToDo: currently everything is named after 2018/2019 but should be year1/year2 since we want the algo to take any 2 years as input
    # Read the biomass data
    biomass_data = pd.read_csv(biomass_file)

    # Extract location indices and biomass availability for each year
    locations = biomass_data['index'].values
    biomass_2018 = biomass_data[year_1].values
    biomass_2019 = biomass_data[year_2].values

    # Read the distance matrix data
    distance_matrix_data = pd.read_csv(distance_matrix_file, index_col=0)

    model = Model("Biomass_Optimization")

    # Decision variables
    depot_vars = {}
    biorefinery_vars = {}

    for loc in locations:
        depot_vars[loc] = model.addVar(vtype="B", name=f"Depot_{loc}")
        biorefinery_vars[loc] = model.addVar(vtype="B", name=f"Biorefinery_{loc}")

    # Constraints
    for loc in locations:
        # adds constraints to ensure that at each location, either a depot or a biorefinery is placed, but not both.
        model.addCons(depot_vars[loc] + biorefinery_vars[loc] <= 1, name=f"Placement_Constraint_{loc}")

    # ensure that the total number of depots is 25 and the total number of biorefineries is 5
    model.addCons(quicksum(depot_vars[loc] for loc in locations) == 25, name="Num_Depots")
    model.addCons(quicksum(biorefinery_vars[loc] for loc in locations) == 5, name="Num_Biorefineries")

    for loc in locations:
        # This loop adds constraints to ensure that the sum of biomass transported to depots from all locations (2018 and 2019 combined) is less than or equal to the total biomass available at each location (2018 and 2019 combined).
        model.addCons(quicksum(biomass_2018[loc] * depot_vars[loc] + biomass_2019[loc] * depot_vars[loc] for loc in locations) <= biomass_2018[loc] + biomass_2019[loc], name=f"Max_Harvest_Biomass_{loc}")

    # This constraint ensures that the sum of biomass transported to biorefineries from all locations (2018 and 2019 combined) is at least 80% of the total biomass available at all locations (2018 and 2019 combined).
    model.addCons(quicksum(biomass_2018[loc] * biorefinery_vars[loc] + biomass_2019[loc] * biorefinery_vars[loc] for loc in locations) >= 0.8 * quicksum(biomass_2018[loc] + biomass_2019[loc] for loc in locations), name="Min_Biorefinery_Biomass")

    # Objective function: minimize total cost
    # Calculate transportation cost and underutilization cost
    total_transport_cost = calculate_transportation_cost(distance_matrix_data, depot_vars, biorefinery_vars,
                                                         biomass_2018, biomass_2019, locations)
    total_underutilization_cost = calculate_underutilization_cost(depot_vars, biorefinery_vars, biomass_2018,
                                                                  biomass_2019, locations)
    total_cost = total_transport_cost + total_underutilization_cost

    model.setObjective(total_cost, "minimize")

    # set to concurrent mode
    model.setParam("parallel/mode", 1)
    # Configure the number of solving threads
    num_threads = get_optimal_thread_count()
    model.setParam("parallel/maxnthreads", num_threads)

    # Set the solver to optimization mode
    model.setIntParam("presolving/maxrounds", 10)  # Adjust presolving aggressiveness
    model.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)  # Use aggresive

    # Set solver's precision (number of decimal places)
    # solver_precision = 4  # Adjust this value as needed
    # model.setRealParam("numerics/feastol", 10 ** (-solver_precision))
    # model.setRealParam("numerics/dualfeastol", 10 ** (-solver_precision))
    # model.setRealParam("numerics/primalfeastol", 10 ** (-solver_precision))

    # Solve the problem
    model.optimize()

    # Create matrices to store the flow values
    flow_biomass_2018 = pd.DataFrame(index=locations, columns=locations)
    flow_biomass_2019 = pd.DataFrame(index=locations, columns=locations)
    flow_pellets_2018 = pd.DataFrame(index=locations, columns=locations)
    flow_pellets_2019 = pd.DataFrame(index=locations, columns=locations)

    # Store the flow values in the matrices
    for loc_idx in range(len(locations)):
        for depot_idx in range(len(locations)):
            flow_biomass_2018.iloc[loc_idx, depot_idx] = biomass_2018[loc_idx] * model.getVal(
                depot_vars[locations[depot_idx]])
            flow_biomass_2019.iloc[loc_idx, depot_idx] = biomass_2019[loc_idx] * model.getVal(
                depot_vars[locations[depot_idx]])
            flow_pellets_2018.iloc[depot_idx, loc_idx] = 20000 * model.getVal(depot_vars[locations[loc_idx]])
            flow_pellets_2019.iloc[depot_idx, loc_idx] = 20000 * model.getVal(depot_vars[locations[loc_idx]])

    # Print the results
    if model.getStatus() == "optimal":
        print("Optimal solution found!")
        print("Optimal locations for depots:")
        for loc in locations:
            if model.getVal(depot_vars[loc]) > 0.5:
                print(f"Depot at location {loc}")
        print("Optimal locations for biorefineries:")
        for loc in locations:
            if model.getVal(biorefinery_vars[loc]) > 0.5:
                print(f"Biorefinery at location {loc}")
        # save the optimal locations to a CSV file
        optimal_locations = pd.DataFrame(index=locations, columns=['Depot', 'Biorefinery'])
        for loc in locations:
            optimal_locations.loc[loc, 'Depot'] = model.getVal(depot_vars[loc])
            optimal_locations.loc[loc, 'Biorefinery'] = model.getVal(biorefinery_vars[loc])
        optimal_locations.to_csv('optimal_locations.csv')
    else:
        print("No optimal solution found.")

    # Save the flow matrices to CSV files
    flow_biomass_2018.to_csv(f'flow_biomass_{year_1}.csv')
    flow_biomass_2019.to_csv(f'flow_biomass_{year_2}.csv')
    flow_pellets_2018.to_csv(f'flow_pellets_{year_1}.csv')
    flow_pellets_2019.to_csv(f'flow_pellets_{year_2}.csv')


if __name__ == "__main__":
    biomass_file = os.path.join('dataset', '3.predictions', 'biomass_predictions.csv')
    distance_matrix_file = os.path.join('dataset', '1.initial_datasets', 'Distance_Matrix.csv')
    year1, year2 = '2018', '2019'
    solve_optimization_problem(biomass_file, distance_matrix_file, year1, year2)
