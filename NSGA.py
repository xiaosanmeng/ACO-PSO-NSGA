import numpy as np
import random
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import pandas as pd
import math
from tqdm import tqdm
import time

class TravelingSalesmanProblem:
    def __init__(self, city_coords, city_costs):
        self.city_coords = city_coords
        self.city_costs = city_costs
        self.num_cities = len(city_coords)
        self.dist_matrix = self._calculate_distance_matrix()
        self.cost_matrix = self._calculate_cost_matrix()

    def _calculate_distance_matrix(self):
        dist_matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                dist_matrix[i][j] = self._calculate_distance(self.city_coords.iloc[i], self.city_coords.iloc[j])
        return dist_matrix

    def _calculate_cost_matrix(self):
        cost_matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                cost_matrix[i][j] = self.city_costs.iloc[i, j]
        return cost_matrix

    def _calculate_distance(self, city1, city2):
        return math.sqrt((city1['X'] - city2['X']) ** 2 + (city1['Y'] - city2['Y']) ** 2)

    def evaluate_individual(self, individual):
        total_distance = 0
        total_cost = 0
        for i in range(len(individual)):
            if i == len(individual) - 1:
                total_distance += self.dist_matrix[individual[i]][individual[0]]
                total_cost += self.cost_matrix[individual[i]][individual[0]]
            else:
                total_distance += self.dist_matrix[individual[i]][individual[i + 1]]
                total_cost += self.cost_matrix[individual[i]][individual[i + 1]]
        return total_distance, total_cost

class NSGA2Optimizer:
    def __init__(self, tsp, pop_size, max_gen, cxpb, mutpb):
        self.tsp = tsp
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.min_distances = []
        self.min_costs = []

        # Define problem objectives
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        # Define toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", lambda: [random.randint(0, self.tsp.num_cities - 1) for _ in range(self.tsp.num_cities)])
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.attribute)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.tsp.evaluate_individual)
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("mate", tools.cxPartialyMatched)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0 / self.tsp.num_cities)

    def run(self):
        # Initialize population
        pop = self.toolbox.population(n=self.pop_size)

        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Evolution loop
        for g in tqdm(range(self.max_gen), desc="Running NSGA"):
            # Selection
            offspring = self.toolbox.select(pop, len(pop))

            # Crossover and mutation
            offspring = [self.toolbox.clone(ind) for ind in offspring]
            for i in range(1, len(offspring), 2):
                if random.random() < self.cxpb:
                    self.toolbox.mate(offspring[i - 1], offspring[i])
                    del offspring[i - 1].fitness.values, offspring[i].fitness.values

            for mutant in offspring:
                if random.random() < self.mutpb:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation
            pop = self.toolbox.select(pop + offspring, self.pop_size)

            # Update minimum distance and minimum cost
            self.min_distances.append(min(ind.fitness.values[0] for ind in pop))
            self.min_costs.append(min(ind.fitness.values[1] for ind in pop))

        return pop

def nsga_main():
    # Start time
    start_time = time.time()

    # Read data
    city_coords = pd.read_excel('city_coordinate.xlsx')
    city_costs = pd.read_excel('city_cost.xlsx')

    # Create TSP instance
    tsp = TravelingSalesmanProblem(city_coords, city_costs)

    # Run NSGA2 optimizer
    nsga2 = NSGA2Optimizer(tsp, pop_size=100, max_gen=500, cxpb=0.8, mutpb=0.2)
    final_pop = nsga2.run()

    # Get Pareto front
    pareto_front = tools.emo.sortNondominated(final_pop, len(final_pop))

    # Create a list to store the Pareto front data
    pareto_front_data = []

    # Iterate through the Pareto front and collect the data
    for individual in pareto_front[0]:
        distance, cost = tsp.evaluate_individual(individual)
        pareto_front_data.append((distance, cost, individual))

    # Convert the data to a pandas DataFrame and save to Excel
    pareto_front_df = pd.DataFrame(pareto_front_data, columns=['Distance', 'Cost', 'Path'])
    pareto_front_df.to_excel('./nsga2_result/nsga_pareto_front.xlsx', index=False)

    # Find minimum distance and minimum cost solutions
    min_distance_ind, min_distance = find_minimum_solution(tsp, pareto_front[0], 'distance')
    min_cost_ind, min_cost = find_minimum_solution(tsp, pareto_front[0], 'cost')

    # End time
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    with open('./nsga2_result/results.txt', 'w') as file:
        file.write(f"Execution time: {end_time - start_time:.2f} seconds\n")

    # Print results
    print_results(min_distance_ind, min_distance, min_cost_ind, min_cost)

    # Plot results
    plot_results(tsp, min_distance_ind, final_pop, pareto_front, nsga2)

def find_minimum_solution(tsp, pareto_front, objective):
    if objective == 'distance':
        min_value = float('inf')
        min_individual = None
    elif objective == 'cost':
        min_value = float('inf')
        min_individual = None
    else:
        raise ValueError("Invalid objective. Choose 'distance' or 'cost'.")

    for individual in pareto_front:
        total_value = tsp.evaluate_individual(individual)[['distance', 'cost'].index(objective)]
        if total_value < min_value:
            min_value = total_value
            min_individual = individual

    return min_individual, min_value

def print_results(min_distance_ind, min_distance, min_cost_ind, min_cost):
    print(f"Minimum Distance Route: {min_distance_ind}")
    print(f"Minimum Distance: {min_distance:.2f}")
    print()
    print(f"Minimum Cost Route: {min_cost_ind}")
    print(f"Minimum Cost: {min_cost:.2f}")

    with open('./nsga2_result/results.txt', 'a') as file:
        file.write(f"Minimum Distance Route: {min_distance_ind}\n")
        file.write(f"Minimum Distance: {min_distance:.2f}\n")
        file.write(f"Minimum Cost Route: {min_cost_ind}\n")
        file.write(f"Minimum Cost: {min_cost:.2f}\n")

def plot_results(tsp, min_distance_ind, final_pop, pareto_front, nsga2):
    # Plot minimum distance route
    plot_route(tsp, min_distance_ind, './nsga2_result/nsga2_solution.png', 'NSGA2 Solution')

    # Plot Pareto frontier
    plot_pareto_frontier(final_pop, pareto_front, './nsga2_result/nsga2_pareto_frontier.png', 'NSGA2 Pareto Frontier')

    # Plot minimum distance optimization history
    plot_optimization_history(nsga2.min_distances, './nsga2_result/nsga2_distance_history.png', 'NSGA2 Distance Optimization History')

    # Plot minimum cost optimization history
    plot_optimization_history(nsga2.min_costs, './nsga2_result/nsga2_cost_history.png', 'NSGA2 Cost Optimization History')

def plot_route(tsp, route, save_path, title):
    plt.figure(figsize=(8, 6))
    route_x = [tsp.city_coords.iloc[city]['X'] for city in route]
    route_y = [tsp.city_coords.iloc[city]['Y'] for city in route]
    route_x.append(route_x[0])  # Close the loop
    route_y.append(route_y[0])  # Close the loop
    plt.plot(route_x, route_y, '-o')
    for i, city in enumerate(route):
        plt.text(tsp.city_coords.iloc[city]['X'], tsp.city_coords.iloc[city]['Y'], str(city), fontsize=8)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(title)
    plt.savefig(save_path)
    plt.show()

def plot_pareto_frontier(final_pop, pareto_front, save_path, title):
    plt.figure(figsize=(8, 6))

    # Plot Pareto front
    distances = [ind.fitness.values[0] for ind in pareto_front[0]]
    costs = [ind.fitness.values[1] for ind in pareto_front[0]]
    #plt.plot(distances, costs, '-r', label='Pareto Front')

    # Plot Pareto optimal solutions
    pareto_distances = [ind.fitness.values[0] for ind in pareto_front[0]]
    pareto_costs = [ind.fitness.values[1] for ind in pareto_front[0]]
    plt.scatter(pareto_distances, pareto_costs, facecolors='none', edgecolors='r', label='Pareto Optimal Solutions')

    # Plot dominated solutions
    dominated_distances = [ind.fitness.values[0] for ind in final_pop if ind not in pareto_front[0]]
    dominated_costs = [ind.fitness.values[1] for ind in final_pop if ind not in pareto_front[0]]
    plt.scatter(dominated_distances, dominated_costs, color='gray', label='Dominated Solutions')

    plt.xlabel('Total Distance')
    plt.ylabel('Total Cost')
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.show()

def plot_optimization_history(history, save_path, title):
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(history)), history)
    plt.xlabel('Generation')
    plt.ylabel('Metric')
    plt.title(title)
    plt.savefig(save_path)
    plt.show()

nsga_main()