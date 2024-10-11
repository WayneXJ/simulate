import os
import json
import matplotlib.pyplot as plt
import test
# 保存结果到文件
def save_results_to_file(file_path, num_node, af1, af2, af3, test):
    # 如果文件存在，读取已有内容
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            results = json.load(file)
    else:
        results = {}

    # 以test为区分，将num_node和af1, af2, af3的结果存入字典
    if test not in results:
        results[test] = {}

    results[test][str(num_node)] = {
        "af1": af1,
        "af2": af2,
        "af3": af3
    }

    # 将结果写回文件
    with open(file_path, 'w') as file:
        json.dump(results, file, indent=4)


# 读取文件并绘制图表
def load_results_and_plot(file_path, test, num_node):
    # 读取保存的文件内容
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            results = json.load(file)

        if test in results and str(num_node) in results[test]:
            af1 = results[test][str(num_node)]["af1"]
            af2 = results[test][str(num_node)]["af2"]
            af3 = results[test][str(num_node)]["af3"]

            # 绘制折线图
            plot_results(num_node, af1, af2, af3)
        else:
            print(f"No results found for test {test} and num_node {num_node}")
    else:
        print("No result file found!")


# 绘制图表函数
def plot_results(num_node, af1, af2, af3):
    x_values = range(len(af1))  # 横坐标为索引值

    plt.figure(figsize=(10, 6))

    # 绘制 af1，af2，af3 的折线图
    plt.plot(x_values, af1, label=f'af1 (num_node={num_node})', color='blue', marker='o')
    plt.plot(x_values, af2, label=f'af2 (num_node={num_node})', color='green', marker='s')
    plt.plot(x_values, af3, label=f'af3 (num_node={num_node})', color='red', marker='^')

    # 添加图例、标题和标签
    plt.legend()
    plt.title(f"Flow Results for num_node={num_node}")
    plt.xlabel("Index")
    plt.ylabel("Value")

    # 显示网格
    plt.grid(True)

    # 显示图表
    plt.show()


def main():
    # 读取的文件序号列表
    node_list = [7, 8, 9, 10]

    # 文件地址
    matrices_dir = 'matrices'
    random_flow_dir = 'random_flows'
    sensor_flow_dir = 'sink_flows'
    output_dir = 'output'

    output_file = 'position_log.json'
    output_flow_file = 'flow_results.json'

    #边容量常数
    edge_capacity = 70

    test.get_multiple_result(node_list, edge_capacity, output_file, matrices_dir, random_flow_dir,
                        sensor_flow_dir, output_dir)




if __name__ == '__main__':
    main()
