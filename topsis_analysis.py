import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# 重定向stdout到文件
orig_stdout = sys.stdout
f = open('output.txt', 'w')
sys.stdout = f

city_coords = pd.read_excel('city_coordinate.xlsx')

# Normalization and Entropy Weight Functions
def normalization1(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def entropyWeight(data):
    P = np.array(data)
    P[P == 0] = 1e-10
    E = np.nansum(-P * np.log(P) / np.log(len(data)), axis=0)
    return (1 - E) / (1 - E).sum()

# TOPSIS Function
def topsis(data, weight=None):
    weight = entropyWeight(data) if weight is None else np.array(weight)
    Z = pd.DataFrame([(data * weight.T).min(), (data * weight.T).max()], index=['负理想解', '正理想解'])
    Result = data.copy()
    Result['正理想解'] = np.sqrt(((data - Z.loc['正理想解']) ** 2).sum(axis=1))
    Result['负理想解'] = np.sqrt(((data - Z.loc['负理想解']) ** 2).sum(axis=1))
    Result['综合得分指数'] = Result['负理想解'] / (Result['负理想解'] + Result['正理想解'])
    Result['排序'] = Result.rank(ascending=False)['综合得分指数']
    return Result, Z, weight

# Analysis Function
def analysis(file_path):
    data = pd.read_excel(file_path)
    data1 = data.copy()
    data1 = data1.drop(columns=['Path'])
    data1['Distance'] = normalization1(data1['Distance'])
    data1['Cost'] = normalization1(data1['Cost'])
    [result, z1, weight] = topsis(data1)
    best_solution_index = result['排序'].idxmin()
    best_solution_data = data.loc[best_solution_index]
    print("最优解的原数据:")
    print(best_solution_data)
    optimal_path = [int(x) for x in best_solution_data['Path'].strip('[]').split(', ')]
    return optimal_path

# Plot Solution Function
def plot_solution(optimal_path, file_name):
    plt.figure(figsize=(8, 6))
    cities_x = [city_coords.iloc[i, 0] for i in optimal_path]
    cities_y = [city_coords.iloc[i, 1] for i in optimal_path]
    plt.plot(cities_x, cities_y, marker='o', linestyle='-')

    for i, (x, y) in enumerate(zip(cities_x, cities_y)):
        plt.text(x, y, str(i), ha='center', va='bottom', fontsize=8)

    plt.title('Optimal Path')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    #plt.savefig(file_name)
    plt.show()

if __name__ == '__main__':
    print("ACO Pareto Front Analysis:")
    aco_optimal_path = analysis('./aco_result/aco_pareto_front.xlsx')
    print("PSO Pareto Front Analysis:")
    pso_optimal_path = analysis('./pso_result/pso_pareto_front.xlsx')
    print("NSGA2 Pareto Front Analysis:")
    nsga2_optimal_path = analysis('./nsga2_result/nsga_pareto_front.xlsx')

    print("ACO optimal path:", aco_optimal_path)
    print("PSO optimal path:", pso_optimal_path)
    print("NSGA2 optimal path:", nsga2_optimal_path)

    plot_solution(aco_optimal_path, './aco_result/aco_solution.png')
    plot_solution(pso_optimal_path, './pso_result/pso_solution.png')
    plot_solution(nsga2_optimal_path, './nsga2_result/nsga2_solution.png')

    # 恢复stdout
    sys.stdout = orig_stdout
    f.close()