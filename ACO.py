import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import pandas as pd
import math
from tqdm import tqdm
import time

# 读取数据
city_coords = pd.read_excel('city_coordinate.xlsx')
city_costs = pd.read_excel('city_cost.xlsx')

# 辅助函数
def calc_dist(city1, city2):
    """计算两个城市之间的距离"""
    return math.sqrt((city1.iloc[0] - city2.iloc[0]) ** 2 + (city1.iloc[1] - city2.iloc[1]) ** 2)

# 初始化距离矩阵和花费矩阵
dist_matrix = np.zeros((len(city_coords), len(city_coords)))
cost_matrix = np.zeros((len(city_costs), len(city_costs)))

for i in range(len(city_coords)):
    for j in range(len(city_coords)):
        dist_matrix[i][j] = calc_dist(city_coords.iloc[i], city_coords.iloc[j])
        cost_matrix[i][j] = city_costs.iloc[i, j]

# 定义一个计算路径成本的函数
def tour_cost(tour, dist_matrix, cost_matrix):
    total_dist = sum(dist_matrix[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))
    total_cost = sum(cost_matrix[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))
    return total_dist, total_cost

class AntColonyOptimization:
    def __init__(self, dist_matrix, cost_matrix, num_ants=50, num_iterations=300, alpha=1.0, beta=1.0, q=100, evaporation_rate=0.1):
        self.dist_matrix = dist_matrix
        self.cost_matrix = cost_matrix
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.q = q
        self.evaporation_rate = evaporation_rate
        self.pheromone = np.ones_like(dist_matrix) / len(dist_matrix)
        self.best_solution = None
        self.best_distance = np.inf
        self.best_cost = np.inf
        self.distance_history = []
        self.cost_history = []
        self.all_solutions = []

    def select_next_city(self, current_city, unvisited_cities):
        pheromones = self.pheromone[current_city][unvisited_cities] ** self.alpha
        heuristic = 1 / (self.dist_matrix[current_city][unvisited_cities] + self.cost_matrix[current_city][unvisited_cities]) ** self.beta
        probabilities = pheromones * heuristic
        probabilities /= probabilities.sum()  # Normalize probabilities
        next_city = np.random.choice(unvisited_cities, p=probabilities)
        return next_city

    def construct_solution(self):
        solutions = []
        for ant in range(self.num_ants):
            current_city = random.randint(0, len(self.dist_matrix) - 1)
            visited_cities = [current_city]
            unvisited_cities = list(set(range(len(self.dist_matrix))) - set([current_city]))

            while unvisited_cities:
                next_city = self.select_next_city(current_city, unvisited_cities)
                visited_cities.append(next_city)
                unvisited_cities.remove(next_city)
                current_city = next_city

            visited_cities.append(visited_cities[0])  # Ensure path is closed
            distance, cost = tour_cost(visited_cities, self.dist_matrix, self.cost_matrix)
            solutions.append((visited_cities, distance, cost))

        return solutions

    def update_pheromone(self, solutions):
        delta_tau = np.zeros_like(self.pheromone)
        best_sol = min(solutions, key=lambda x: x[1] + x[2])
        for sol, dist, cost in solutions:
            for i in range(len(sol) - 1):
                delta_tau[sol[i]][sol[i + 1]] += self.q / (dist + cost)

        # 添加精英保留策略
        for i in range(len(best_sol[0]) - 1):
            delta_tau[best_sol[0][i]][best_sol[0][i + 1]] += self.q / (best_sol[1] + best_sol[2])

        self.pheromone = (1 - self.evaporation_rate) * self.pheromone + delta_tau

    def run(self):
        initial_mutpb = 0.2  # 初始变异率
        final_mutpb = 0.05  # 最终变异率
        initial_cxpb = 0.9  # 初始交叉率
        final_cxpb = 0.5  # 最终交叉率

        for iteration in tqdm(range(self.num_iterations), desc='Running ACO'):
            # 根据进化进度动态调整参数
            progress = iteration / self.num_iterations
            mutpb = initial_mutpb - progress * (initial_mutpb - final_mutpb)
            cxpb = initial_cxpb - progress * (initial_cxpb - final_cxpb)

            solutions = self.construct_solution()
            self.all_solutions.extend(solutions)
            best_sol = min(solutions, key=lambda x: x[1] + x[2])
            if best_sol[1] + best_sol[2] < self.best_distance + self.best_cost:
                self.best_solution = best_sol[0]
                self.best_distance = best_sol[1]
                self.best_cost = best_sol[2]
            self.update_pheromone(solutions)
            self.distance_history.append(self.best_distance)
            self.cost_history.append(self.best_cost)

        return self.best_solution, self.best_distance, self.best_cost

    def plot_solution(self, solution):
        plt.figure(figsize=(8, 6))
        cities_x = [city_coords.iloc[i, 0] for i in solution]
        cities_y = [city_coords.iloc[i, 1] for i in solution]
        plt.plot(cities_x, cities_y, marker='o', linestyle='-')
        # 添加城市编号
        for i, (x, y) in enumerate(zip(cities_x, cities_y)):
            plt.text(x, y, str(i), ha='center', va='bottom', fontsize=8)
        plt.title('ACO Solution')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        #保存
        plt.savefig('./aco_result/aco_solution.png')
        plt.show()

    def plot_distance_history(self):
        plt.figure(figsize=(8, 6))
        plt.plot(range(self.num_iterations), self.distance_history)
        plt.title('ACO Distance Optimization History')
        plt.xlabel('Iteration')
        plt.ylabel('Distance')
        plt.grid(True)
        #保存

        plt.savefig('./aco_result/aco_distance_history.png')
        plt.show()

    def plot_cost_history(self):
        plt.figure(figsize=(8, 6))
        plt.plot(range(self.num_iterations), self.cost_history)
        plt.title('ACO Cost Optimization History')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True)
        #保存
        plt.savefig('./aco_result/aco_cost_history.png')
        plt.show()

    def plot_pareto_frontier(self):
        distances = [sol[1] for sol in self.all_solutions]
        costs = [sol[2] for sol in self.all_solutions]
        paths = [sol[0] for sol in self.all_solutions]

        pareto_front = []
        pareto_paths = []
        for i in range(len(distances)):
            is_pareto = True
            for j in range(len(distances)):
                if distances[j] < distances[i] and costs[j] < costs[i]:
                    is_pareto = False
                    break
            if is_pareto:
                pareto_front.append((distances[i], costs[i]))
                pareto_paths.append(paths[i])

        pareto_front.sort(key=lambda x: x[0])

        # 保存pareto_front到xlsx,包括路径信息
        df = pd.DataFrame(
            {'Distance': [d for d, c in pareto_front], 'Cost': [c for d, c in pareto_front], 'Path': pareto_paths})
        df.to_excel('./aco_result/aco_pareto_front.xlsx', index=False)

        plt.figure(figsize=(8, 6))
        plt.scatter([d for d, c in pareto_front], [c for d, c in pareto_front], color='r', label='Pareto Front')
        plt.scatter(distances, costs, color='b', label='Solutions', alpha=0.5)
        plt.title('ACO Pareto Frontier')
        plt.xlabel('Distance')
        plt.ylabel('Cost')
        plt.legend()
        plt.grid(True)
        plt.savefig('./aco_result/aco_pareto_frontier.png')
        plt.show()

#if __name__ == '__main__':
def aco_main():
    #开始时间
    start = time.time()

    # 初始化并执行蚁群算法
    aco = AntColonyOptimization(dist_matrix, cost_matrix,num_iterations=100)
    best_path_ACO, best_dist_ACO, best_cost_ACO = aco.run()
    #结束时间
    end = time.time()

    print(f"蚁群算法得到的最优路径: {best_path_ACO}")
    print(f"最短距离: {best_dist_ACO}")
    print(f"最小花费: {best_cost_ACO}")
    
    print('Running time: %s Seconds'%(end-start))

    with open('./aco_result/results.txt', 'w') as file:
        # 写入蚁群算法得到的最优路径
        file.write(f"蚁群算法得到的最优路径: {best_path_ACO}\n")
        # 写入最短距离
        file.write(f"最短距离: {best_dist_ACO}\n")
        # 写入最小花费
        file.write(f"最小花费: {best_cost_ACO}\n")
        file.write('Running time: %s Seconds'%(end-start))


    # 绘制最优路径图
    aco.plot_solution(best_path_ACO)

    # 绘制距离优化图
    aco.plot_distance_history()

    # 绘制花费优化图
    aco.plot_cost_history()

    # 绘制帕累托前沿
    aco.plot_pareto_frontier()

    

aco_main()