import numpy as np
import random
import matplotlib.pyplot as plt
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
    """计算给定路径的总距离和总成本"""
    total_dist = sum(dist_matrix[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))
    total_cost = sum(cost_matrix[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))
    return total_dist, total_cost

class ParticleSwarmOptimization:
    def __init__(self, dist_matrix, cost_matrix, num_particles=50, num_iterations=300, c1=2, c2=2, w=0.9, mutation_rate=0.1, elite_rate=0.1):
        """
        初始化粒子群优化算法的参数
        :param dist_matrix: 距离矩阵
        :param cost_matrix: 成本矩阵
        :param num_particles: 粒子数量
        :param num_iterations: 迭代次数
        :param c1: 个体学习因子
        :param c2: 群体学习因子
        :param w: 惯性权重
        :param mutation_rate: 随机扰动的概率
        :param elite_rate: 精英保留的比例
        """
        self.dist_matrix = dist_matrix
        self.cost_matrix = cost_matrix
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.mutation_rate = mutation_rate
        self.elite_rate = elite_rate
        self.particles = []
        self.pbest_positions = []
        self.pbest_distances = []
        self.pbest_costs = []
        self.gbest_position = None
        self.gbest_distance = np.inf
        self.gbest_cost = np.inf
        self.distance_history = []
        self.cost_history = []
        self.all_solutions = []

    def initialize_particles(self):
        """初始化粒子群"""
        for _ in range(self.num_particles):
            particle = random.sample(range(len(self.dist_matrix)), len(self.dist_matrix))
            self.particles.append(particle)
            distance, cost = tour_cost(particle, self.dist_matrix, self.cost_matrix)
            self.pbest_positions.append(particle)
            self.pbest_distances.append(distance)
            self.pbest_costs.append(cost)

            if distance + cost < self.gbest_distance + self.gbest_cost:
                self.gbest_position = particle
                self.gbest_distance = distance
                self.gbest_cost = cost

    def update_particle(self, particle, iteration):
        """更新粒子的位置"""
        new_particle = particle.copy()
        r1 = random.random()
        r2 = random.random()

        # 动态调整参数
        curr_w = self.w * (1 - iteration / self.num_iterations)
        curr_c1 = self.c1 * (1 - iteration / self.num_iterations)
        curr_c2 = self.c2 * (1 - iteration / self.num_iterations)

        for i in range(len(new_particle)):
            # 进行随机扰动
            if random.random() < self.mutation_rate:
                idx1 = random.randint(0, len(new_particle) - 1)
                idx2 = random.randint(0, len(new_particle) - 1)
                new_particle[idx1], new_particle[idx2] = new_particle[idx2], new_particle[idx1]

            distance, cost = tour_cost(new_particle, self.dist_matrix, self.cost_matrix)
            if distance + cost < self.pbest_distances[i] + self.pbest_costs[i]:
                self.pbest_positions[i] = new_particle
                self.pbest_distances[i] = distance
                self.pbest_costs[i] = cost

            if distance + cost < self.gbest_distance + self.gbest_cost:
                self.gbest_position = new_particle
                self.gbest_distance = distance
                self.gbest_cost = cost

        # 精英保留策略
        elite_size = int(self.elite_rate * self.num_particles)
        self.pbest_positions = sorted(self.pbest_positions, key=lambda x: tour_cost(x, self.dist_matrix, self.cost_matrix)[0] + tour_cost(x, self.dist_matrix, self.cost_matrix)[1])[:self.num_particles - elite_size] + self.pbest_positions[:elite_size]
        self.pbest_distances = [tour_cost(p, self.dist_matrix, self.cost_matrix)[0] for p in self.pbest_positions]
        self.pbest_costs = [tour_cost(p, self.dist_matrix, self.cost_matrix)[1] for p in self.pbest_positions]

        return new_particle

    def run(self):
        """运行粒子群优化算法"""
        self.initialize_particles()

        for iteration in tqdm(range(self.num_iterations), desc='Running PSO'):
            new_particles = []
            for particle in self.particles:
                new_particle = self.update_particle(particle, iteration)
                new_particles.append(new_particle)
            self.particles = new_particles

            self.distance_history.append(self.gbest_distance)
            self.cost_history.append(self.gbest_cost)
            self.all_solutions.append((self.gbest_position, self.gbest_distance, self.gbest_cost))

        return self.gbest_position, self.gbest_distance, self.gbest_cost

    def plot_solution(self, solution):
        """绘制最优路径图"""
        plt.figure(figsize=(8, 6))
        cities_x = [city_coords.iloc[i, 0] for i in solution]
        cities_y = [city_coords.iloc[i, 1] for i in solution]
        plt.plot(cities_x, cities_y, marker='o', linestyle='-')

        # 添加城市编号
        for i, (x, y) in enumerate(zip(cities_x, cities_y)):
            plt.text(x, y, str(i), ha='center', va='bottom', fontsize=8)

        plt.title('PSO Solution')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.savefig('./pso_result/pso_solution.png')
        plt.show()

    def plot_distance_history(self):
        """绘制距离优化历史图"""
        plt.figure(figsize=(8, 6))
        plt.plot(range(self.num_iterations), self.distance_history)
        plt.title('PSO Distance Optimization History')
        plt.xlabel('Iteration')
        plt.ylabel('Distance')
        plt.grid(True)
        plt.savefig('./pso_result/pso_distance_history.png')
        plt.show()

    def plot_cost_history(self):
        """绘制成本优化历史图"""
        plt.figure(figsize=(8, 6))
        plt.plot(range(self.num_iterations), self.cost_history)
        plt.title('PSO Cost Optimization History')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.savefig('./pso_result/pso_cost_history.png')
        plt.show()

    def plot_pareto_frontier(self):
        """绘制帕累托前沿图并保存数据"""
        distances = [sol[1] for sol in self.all_solutions]
        costs = [sol[2] for sol in self.all_solutions]
        tours = [sol[0] for sol in self.all_solutions]

        pareto_front = []
        pareto_tours = []
        dominated_solutions = []

        for i in range(len(distances)):
            is_pareto = True
            for j in range(len(distances)):
                if distances[j] < distances[i] and costs[j] < costs[i]:
                    is_pareto = False
                    break
            if is_pareto:
                pareto_front.append((distances[i], costs[i]))
                pareto_tours.append(tours[i])
            else:
                dominated_solutions.append((distances[i], costs[i], tours[i]))

        pareto_front = sorted(pareto_front, key=lambda x: x[0])
        pareto_tours = [pareto_tours[i] for i in sorted(range(len(pareto_front)), key=lambda k: pareto_front[k][0])]

        # 将帕累托前沿数据保存到 Excel 文件
        df = pd.DataFrame({
            'Distance': [d for d, c in pareto_front],
            'Cost': [c for d, c in pareto_front],
            'Path': pareto_tours
        })
        df.to_excel('./pso_result/pso_pareto_front.xlsx', index=False)

        plt.figure(figsize=(8, 6))
        plt.scatter([d for d, c, _ in dominated_solutions], [c for d, c, _ in dominated_solutions], color='gray',
                    label='Dominated Solutions', alpha=0.5)
        plt.scatter([d for d, c in pareto_front], [c for d, c in pareto_front], color='r', label='Pareto Front',
                    marker='o', facecolors='none')
        plt.plot([d for d, c in pareto_front], [c for d, c in pareto_front], color='r', label='Pareto Frontier',
                 linewidth=2)
        plt.title('PSO Pareto Frontier')
        plt.xlabel('Distance')
        plt.ylabel('Cost')
        plt.legend()
        plt.grid(True)
        plt.savefig('./pso_result/pso_pareto_frontier.png')
        plt.show()

def pso_main():
    """主函数,运行粒子群优化算法"""
    start = time.time()
    pso = ParticleSwarmOptimization(dist_matrix, cost_matrix, num_iterations=300, mutation_rate=0.1, elite_rate=0.1)
    best_path_PSO, best_dist_PSO, best_cost_PSO = pso.run()
    end = time.time()

    print(f"粒子群算法得到的最优路径: {best_path_PSO}")
    print(f"最短距离: {best_dist_PSO}")
    print(f"最小花费: {best_cost_PSO}")
    print('Running time: %s Seconds'%(end-start))

    with open('./pso_result/results.txt', 'w') as file:
        # 写入蚁群算法得到的最优路径
        file.write(f"粒子群算法得到的最优路径: {best_path_PSO}\n")
        # 写入最短距离
        file.write(f"最短距离: {best_dist_PSO}\n")
        # 写入最小花费
        file.write(f"最小花费: {best_cost_PSO}\n")
        file.write('Running time: %s Seconds'%(end-start))

    pso.plot_solution(best_path_PSO)
    pso.plot_distance_history()
    pso.plot_cost_history()
    pso.plot_pareto_frontier()

pso_main()