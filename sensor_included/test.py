import math
import random
from original_demo.generate_matrix import load_matrices_from_file
from original_demo.generate_flow import load_flows_from_file
import copy
import os
import json
import matplotlib.pyplot as plt
from original_demo import output
import flow_calculate as fc

'''
增加传感器节点部分，大多数流量传回到一个固定点
无人机带宽有限，存储无限  or    无人机存储有限，可以转发
传完和利用率是针对无人机而言的，利用无人机的有限窗口进行通信
'''
class Node:
    def __init__(self, node_id, x, y, x_seq, y_seq):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.x_seq = x_seq
        self.y_seq = y_seq

    def distance_to(self, other_node):
        return math.sqrt((self.x - other_node.x) ** 2 + (self.y - other_node.y) ** 2)


class Flow:
    def __init__(self, start_node, end_node, demand, start_time, end_time):
        self.start_node = start_node
        self.end_node = end_node
        self.demand = demand
        self.start_time = start_time
        self.end_time = end_time


# 生成随机流量
def generate_random_flows(num_node, num_flows, demand_range, time_range):
    flows = []
    node_ids = list(range(num_node))  # 根据节点数量生成节点编号

    for _ in range(num_flows):
        start_node = random.choice(node_ids)
        end_node = random.choice([node_id for node_id in node_ids if node_id != start_node])

        demand = random.randint(demand_range[0], demand_range[1])
        start_time = random.randint(time_range[0], time_range[1] - 1)
        end_time = random.randint(start_time + 1, time_range[1])

        flow = Flow(start_node, end_node, demand, start_time, end_time)
        flows.append(flow)

    # 排序
    flows_sorted = sorted(flows, key=lambda flow: flow.start_time)
    return flows_sorted


# 根据矩阵生成结点，返回结点字典nodes
def generate_topology(matrix, x_limit, y_limit):
    nodes = {}
    node_id = 0
    x_seq = [-1]
    y_seq = [-1]
    nodes[node_id] = Node(node_id, -1, -1, x_seq, y_seq)
    node_id += 1
    # 生成节点
    for i, row in enumerate(matrix):
        for j, cell in enumerate(row):
            if cell == 1:
                x_seq = [i]
                y_seq = [j]
                nodes[node_id] = Node(node_id, i, j, x_seq, y_seq)
                node_id += 1

    node_id = generate_sensor(x_limit, y_limit, node_id, nodes)
    return nodes, node_id


def generate_sensor(x_limit, y_limit, node_id, nodes):
    y_gap = y_limit // 3
    for i in range(2):
        x_seq = [x_limit + 2]
        y_seq = [y_gap * (i+1)]
        nodes[node_id] = Node(node_id, x_limit + 2, y_gap * (i+1), x_seq, y_seq)
        node_id += 1
    x_seq = [x_limit + 3]
    y_seq = [y_limit // 2]
    nodes[node_id] = Node(node_id, x_limit + 3, y_limit // 2, x_seq, y_seq)
    node_id += 1
    return node_id


# 移动结点，并将移动结果保存，重新生成邻居结点
def move_nodes(nodes, move_algorithm, limits, time_step, position_log, node_range):
    position_log[time_step] = {}
    for node_id, node in nodes.items():
        # 判断使得sink和sensor不会移动
        if node_range[0] < node_id < node_range[1]:
            old_position = (node.x, node.y)
            new_x, new_y = move_algorithm(node)
            if not any(n.x == new_x and n.y == new_y for n in nodes.values()) and not is_out_of_range(new_x, new_y, limits):
                nodes[node_id].x, nodes[node_id].y = new_x, new_y
                nodes[node_id].x_seq.append(new_x)
                nodes[node_id].y_seq.append(new_y)
                position_log[time_step][node.node_id] = {'position': (new_x, new_y)}
            else:
                position_log[time_step][node.node_id] = {'position': old_position}
                nodes[node_id].x_seq.append(node.x)
                nodes[node_id].y_seq.append(node.y)
        else:
            continue


def is_out_of_range(x, y, limits):
    x_min, x_max, y_min, y_max = limits
    return x < x_min or x > x_max or y < y_min or y > y_max


# 随机移动结点策略
def random_walk_algorithm(node):
    scope = 3
    list = []
    for i in range(scope):
        if i != 0:
            list.append(-i)
        list.append(i)
    return node.x + random.choice(list), node.y + random.choice(list)


def get_edge_with_UAV(nodes, edges, edge_capacity, num, time):
    # 添加节点和边
    for i in range(num):
        for j in range(i + 1, num):
            node1 = nodes[i]
            node2 = nodes[j]
            if is_in_UAV(node1, time):
                edges[(node1.node_id, node2.node_id)].append(60)
            else:
                edges[(node1.node_id, node2.node_id)].append(int(edge_capacity / node1.distance_to(node2)))
            # print(node1.node_id,"--->",node2.node_id," dis:",node1.distance_to(node2),int(edge_capacity / node1.distance_to(node2)))


def get_edge(nodes, edges, edge_capacity, num, time):
    # 添加节点和边
    for i in range(num):
        for j in range(i + 1, num):
            node1 = nodes[i]
            node2 = nodes[j]
            edges[(node1.node_id, node2.node_id)].append(int(edge_capacity / node1.distance_to(node2)))


def is_in_UAV(node, time):
    time_zones = {    # 14, 18, 22 有一个停顿取流量
        11: (0, 5, 0, 5),
        12: (5, 10, 0, 5),
        13: (10, 15, 0, 5),
        15: (10, 15, 5, 10),
        16: (5, 10, 5, 10),
        17: (0, 5, 5, 10),
        19: (0, 5, 10, 15),
        20: (5, 10, 10, 15),
        21: (10, 15, 10, 15)
    }

    # 检查当前时间是否在定义的时段内
    if time in time_zones:
        x_min, x_max, y_min, y_max = time_zones[time]
        node_x = node.x
        node_y = node.y

        # 判断节点是否在无人机的范围内
        if x_min <= node_x < x_max and y_min <= node_y < y_max:
            return True

    return False


def dict_to_list(input_dict):
    # 使用列表推导式将字典转换为列表
    return [(key[0], key[1], value) for key, value in input_dict.items()]


def init_edge(nodes, edge_capacity, num):
    edges = {}
    # 对于每两个不同的节点，生成一条边
    for i in range(num):
        for j in range(i + 1, num):
            node1 = nodes[i]
            node2 = nodes[j]
            edges[(node1.node_id, node2.node_id)] = [int(edge_capacity / node1.distance_to(node2))]
    return edges


# 创建初始的存储序列
def generate_list(num_node, h, source):
    # 创建首元素的内部列表，元素全为1
    first_element = [1] * h

    # 创建其他元素的内部列表，元素全为0
    other_elements1 = [[0] * h for _ in range(source)]
    other_elements2 = [[0] * h for _ in range(num_node - 1 - source)]

    # 将首元素和其他元素合并成一个列表
    result = other_elements1 + [first_element] + other_elements2

    return result


def flow_result_json(flow,maxflow, demand_flow, path, flag, results):
    flow_result = {
        'flow': {
            'start_node': flow.start_node,
            'end_node': flow.end_node,
            'demand': flow.demand,
            'start_time': flow.start_time,
            'end_time': flow.end_time
        },
        'result': {
            'maxflow': maxflow,
            'demand_flow': demand_flow,
            'path': path,
            'flag': flag
        }
    }
    results.append(flow_result)


# 完成率优先（传感器版本）
# 传回传感器流量过程中不涉及优先问题
def get_count_with_sensor(flows, all_flow, all_flow_to_zero, num_nodes, time_steps, graph, time_count, output_flow_file):
    print("-----------------sensor---------------")
    finish_count = [0] * time_count
    flow_count = [0] * time_count
    results = []
    fc.print_graph(graph)

    # 计算每个区间的长度
    interval_length = time_steps / time_count

    for flow in flows:
        if flow.end_node == 0 and (flow.end_time == 14 or flow.end_time == 22):
            finish_count[1] += 1
            flow_count[1] += 1
            for i in range(flow.end_time - flow.start_time):
                print("i + start_time:", i, flow.start_time)
                all_flow[i + flow.start_time] += flow.demand
                all_flow_to_zero[i + flow.start_time] += flow.demand
        else:
            NT = generate_list(num_nodes, time_steps, flow.start_node)
            maxflow, demand_flow, path, flag = fc.edmonds_karp(graph, flow.start_node, flow.end_node,
                                                               NT, flow.start_time, flow.end_time, flow.demand)
            for i in range(len(maxflow)):
                print("i + start_time:", i, flow.start_time)
                all_flow[i + flow.start_time] += maxflow[i]
                if flow.end_node == 0:
                    all_flow_to_zero[i + flow.start_time] += maxflow[i]
            flow_result_json(flow, maxflow, demand_flow, path, flag, results)

            if flag == 0:
                print("flow:", flow.start_node, flow.end_node, flow.start_time, flow.end_time)
            # 通过循环处理多个区间
            for i in range(time_count):
                start_interval = i * interval_length
                end_interval = (i + 1) * interval_length

                if start_interval < flow.end_time <= end_interval:
                    flow_count[i] += 1
                    if flag:
                        finish_count[i] += 1
                    break  # 找到对应区间后，跳出循环

    output.output_flow_results(output_flow_file, results)
    return flow_count, finish_count


# 利用率优先算法
# 优先放入大的流量需求
def get_count_usage_priority_with_random(flows_random, all_flow, all_flow_to_zero, num_nodes, time_steps, graph, time_count, output_flow_file):
    print("--------------usage_priority---------------")
    finish_count = [0] * time_count
    flow_count = [0] * time_count
    results = []
    fc.print_graph(graph)

    # 计算每个区间的长度
    interval_length = time_steps / time_count
    sorted_flow_list = sorted(flows_random, key=lambda x: (x.start_time, -x.demand))
    for flow in sorted_flow_list:
        NT = generate_list(num_nodes, time_steps, flow.start_node)
        maxflow, demand_flow, path, flag = fc.edmonds_karp(graph, flow.start_node, flow.end_node,
                                                           NT, flow.start_time, flow.end_time, flow.demand)
        for i in range(len(maxflow)):
            print("i + start_time:", i, flow.start_time)
            all_flow[i + flow.start_time] += maxflow[i]
            if flow.end_node == 0:
                all_flow_to_zero[i + flow.start_time] += maxflow[i]
        flow_result_json(flow, maxflow, demand_flow, path, flag, results)

        if flag == 0:
            print("flow:", flow.start_node, flow.end_node, flow.start_time, flow.end_time)
        # 通过循环处理多个区间
        for i in range(time_count):
            start_interval = i * interval_length
            end_interval = (i + 1) * interval_length

            if start_interval < flow.end_time <= end_interval:
                flow_count[i] += 1
                if flag:
                    finish_count[i] += 1
                break  # 找到对应区间后，跳出循环

    output.output_flow_results(output_flow_file, results)
    return flow_count, finish_count


# 完成优先
# 优先放入小的流量需求
def get_count_completed_priority_with_random(flows_random, all_flow, all_flow_to_zero, num_nodes, time_steps, graph, time_count, output_flow_file):
    print("--------------completed_priority---------------")
    finish_count = [0] * time_count
    flow_count = [0] * time_count
    results = []
    fc.print_graph(graph)

    # 计算每个区间的长度
    interval_length = time_steps / time_count
    sorted_flow_list = sorted(flows_random, key=lambda x: (x.start_time, x.demand))
    for flow in sorted_flow_list:
        NT = generate_list(num_nodes, time_steps, flow.start_node)
        maxflow, demand_flow, path, flag = fc.edmonds_karp(graph, flow.start_node, flow.end_node,
                                                           NT, flow.start_time, flow.end_time, flow.demand)
        for i in range(len(maxflow)):
            all_flow[i + flow.start_time] += maxflow[i]
            if flow.end_node == 0:
                all_flow_to_zero[i + flow.start_time] += maxflow[i]
        flow_result_json(flow, maxflow, demand_flow, path, flag, results)

        if flag == 0:
            print("flow:", flow.start_node, flow.end_node, flow.start_time, flow.end_time)
        # 通过循环处理多个区间
        for i in range(time_count):
            start_interval = i * interval_length
            end_interval = (i + 1) * interval_length

            if start_interval < flow.end_time <= end_interval:
                flow_count[i] += 1
                if flag:
                    finish_count[i] += 1
                break  # 找到对应区间后，跳出循环

    output.output_flow_results(output_flow_file, results)
    return flow_count, finish_count


def calculate_list_all(count_list):
    count = 0
    for i in count_list:
        count += i

    return count


def get_count_without_UAV(flows_random, all_flow, all_flow_to_zero, num_nodes, time_steps, graph, time_count,
                             output_flow_file):
    print("--------------Without UAV---------------")
    finish_count = [0] * time_count
    flow_count = [0] * time_count
    results = []
    fc.print_graph(graph)

    # 计算每个区间的长度
    interval_length = time_steps / time_count
    sorted_flow_list = sorted(flows_random, key=lambda x: (x.start_time))
    for flow in sorted_flow_list:
        NT = generate_list(num_nodes, time_steps, flow.start_node)
        maxflow, demand_flow, path, flag = fc.edmonds_karp(graph, flow.start_node, flow.end_node,
                                                           NT, flow.start_time, flow.end_time, flow.demand)
        for i in range(len(maxflow)):
            all_flow[i + flow.start_time] += maxflow[i]
            if flow.end_node == 0:
                all_flow_to_zero[i + flow.start_time] += maxflow[i]
        flow_result_json(flow, maxflow, demand_flow, path, flag, results)

        if flag == 0:
            print("flow:", flow.start_node, flow.end_node, flow.start_time, flow.end_time)
        # 通过循环处理多个区间
        for i in range(time_count):
            start_interval = i * interval_length
            end_interval = (i + 1) * interval_length

            if start_interval < flow.end_time <= end_interval:
                flow_count[i] += 1
                if flag:
                    finish_count[i] += 1
                break  # 找到对应区间后，跳出循环

    output.output_flow_results(output_flow_file, results)
    return flow_count, finish_count


def simulate_multiple_flows(nodes, flows_random, flows_sensor, edges, num_nodes, node_range, edge_capacity, time_steps, time_set, move_algorithm,
                            limits, output_position_log, output_flow_file):

    time_count = time_steps // time_set
    position_log = {}
    all_flow_usage_priority = [0] * (time_steps + 1)
    all_flow_completed_priority = [0] * (time_steps + 1)
    all_flow_without_UAV = [0] * (time_steps + 1)
    all_of_flows = [0] * (time_steps + 1)
    all_flow_usage_priority_to_zero = [0] * (time_steps + 1)
    all_flow_completed_priority_to_zero = [0] * (time_steps + 1)
    all_flow_without_UAV_to_zero = [0] * (time_steps + 1)
    edges_without_UAV = copy.deepcopy(edges)

    for t in range(time_steps):
        move_nodes(nodes, move_algorithm, limits, t, position_log, node_range)
        # print("nodes:")
        get_edge_with_UAV(nodes, edges, edge_capacity, num_nodes, t)
        get_edge(nodes, edges_without_UAV, edge_capacity, num_nodes, t)

    edge_list = dict_to_list(edges)
    edges_list_without_UAV = dict_to_list(edges_without_UAV)
    graph1 = fc.convert_to_directed_graph(edge_list, num_nodes)
    print("---------------graph1----------------------------")
    fc.print_graph(graph1)

    graph2 = copy.deepcopy(graph1)
    print("---------------graph2----------------------------")
    fc.print_graph(graph2)

    graph3 = fc.convert_to_directed_graph(edges_list_without_UAV, num_nodes)
    print("---------------graph3----------------------------")
    fc.print_graph(graph3)

    flow_count_sensor1, finish_count_sensor1 = get_count_with_sensor(flows_sensor, all_flow_usage_priority, all_flow_usage_priority_to_zero, num_nodes, time_steps, graph1, time_count,
                                                          output_flow_file)
    flow_count1, finish_count1 = get_count_usage_priority_with_random(flows_random, all_flow_usage_priority, all_flow_usage_priority_to_zero, num_nodes, time_steps, graph1, time_count,
                                                          output_flow_file)
    flow_count_sensor2, finish_count_sensor2 = get_count_with_sensor(flows_sensor, all_flow_completed_priority, all_flow_completed_priority_to_zero, num_nodes, time_steps, graph2, time_count,
                                                              output_flow_file)
    flow_count2, finish_count2 = get_count_completed_priority_with_random(flows_random, all_flow_completed_priority, all_flow_completed_priority_to_zero, num_nodes, time_steps, graph2, time_count,
                                                              output_flow_file)
    flow_count3, finish_count3 = get_count_without_UAV(flows_random, all_flow_without_UAV, all_flow_without_UAV_to_zero, num_nodes, time_steps, graph3, time_count,
                                                              output_flow_file)

    for flow in flows_sensor:
        if flow.end_node == 0:
            for i in range(flow.end_time - flow.start_time):
                print("i + start_time:", i, flow.start_time)
                all_of_flows[i + flow.start_time] += flow.demand

    for flow in flows_random:
        if flow.end_node == 0:
            for i in range(flow.end_time - flow.start_time):
                print("i + start_time:", i, flow.start_time)
                all_of_flows[i + flow.start_time] += flow.demand

    print("usage_priority_flow_count1:", flow_count1)
    print("usage_priority_finish_count1:", finish_count1)
    print("completed_priority_flow_count2:", flow_count2)
    print("completed_priority_finish_count2:", finish_count2)
    print("without_UAV_flow_count2:", flow_count3)
    print("without_UAV_finish_count2:", finish_count3)

    # print("流量总和")
    # print(calculate_list_all(all_of_flows))
    # print(calculate_list_all(all_flow_usage_priority_to_zero))
    # print(calculate_list_all(all_flow_completed_priority_to_zero))
    # print(calculate_list_all(all_flow_without_UAV_to_zero))


    return flow_count1, finish_count1, flow_count2, finish_count2, flow_count3, finish_count3, \
        all_flow_usage_priority, all_flow_completed_priority, all_flow_without_UAV


def get_multiple_result(node_list, edge_capacity, output_file, matrices_dir, random_flow_dir, sensor_flow_dir, output_dir):
    for num_node in node_list:
        # 文件地址
        matrices_file = os.path.join(matrices_dir, f'matrices_{num_node}.npy')
        random_flow_file = os.path.join(random_flow_dir, f'random_flows_{num_node}.json')
        sensor_flow_file = os.path.join(sensor_flow_dir, f'sink_flows_{num_node}.json')
        output_flow_file = os.path.join(output_dir, f'flow_results_{num_node}.json')
        output_result_file = os.path.join(output_dir, f'results_{num_node}.json')

        #读取矩阵
        loaded_matrices = load_matrices_from_file(matrices_file)

        # 对每个矩阵进行仿真
        for i, matrix in enumerate(loaded_matrices):
            # 得到矩阵结点移动的范围限制，
            x_limit, y_limit = matrix.shape
            limits = (0, x_limit, 0, y_limit)

            # 生成结点(包括所有节点)
            nodes, num_node = generate_topology(matrix, x_limit, y_limit)
            print("num_node:", num_node)
            node_range = (0, num_node - 3)  # (sink, sensor_start) 这里有3个sensor节点
            # 初始化边
            edges = init_edge(nodes, edge_capacity, num_node)

            # 设置流量要求

            flows_random = load_flows_from_file(random_flow_file)
            flows_sensor = load_flows_from_file(sensor_flow_file)
            # for flow in flows:
            #     print(f"Flow from Node {flow.start_node} to Node {flow.end_node} "
            #           f"with demand {flow.demand} from time {flow.start_time} to {flow.end_time}")

            flow_dis1, suc1, flow_dis2, suc2, flow_dis3, suc3, \
                af1, af2, af3 = simulate_multiple_flows(nodes,
                                                       flows_random,
                                                       flows_sensor,
                                                       edges, num_node,
                                                       node_range,
                                                       edge_capacity,
                                                       time_steps=40,
                                                       time_set=10,
                                                       move_algorithm=random_walk_algorithm,
                                                       limits=limits,
                                                       output_position_log=output_file,
                                                       output_flow_file=output_flow_file)
            print("最终结果：")

            count1 = [x / y if y != 0 else 0 for x, y in zip(suc1, flow_dis1)]
            count2 = [x / y if y != 0 else 0 for x, y in zip(suc2, flow_dis2)]
            count3 = [x / y if y != 0 else 0 for x, y in zip(suc3, flow_dis3)]

            print(count1)
            print(count2)
            print(count3)
            print("f1:", af1)
            result_af1 = calculate_list_all(af1)
            print(calculate_list_all(af1))
            result_af2 = calculate_list_all(af2)
            print("f2:", af2)
            print(calculate_list_all(af2))
            result_af3 = calculate_list_all(af3)
            print("f3:", af3)
            print(calculate_list_all(af3))

            # 保存结果到文件
            save_results_to_file(output_result_file, num_node, af1, af2, af3)

            # 绘制结果折线图
            plot_results(num_node, af1, af2, af3)


# 自动生成test编号
def get_next_test_number(file_path):
    # 如果文件不存在，返回'test1'
    if not os.path.exists(file_path):
        return "test1"

    # 读取已有文件
    with open(file_path, 'r') as file:
        results = json.load(file)

    # 获取所有的 test 编号，找出最大值并加 1
    test_numbers = [int(test.replace("test", "")) for test in results.keys() if test.startswith("test")]
    next_test_num = max(test_numbers, default=0) + 1

    return f"test{next_test_num}"


# 保存结果到文件
def save_results_to_file(file_path, num_node, af1, af2, af3):
    test = get_next_test_number(file_path)  # 动态生成test编号

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

    return test  # 返回当前保存的test编号


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

    get_multiple_result(node_list, edge_capacity, output_file, matrices_dir, random_flow_dir,
                        sensor_flow_dir, output_dir)

    output_result_file = os.path.join('output', 'results_9.json')

    load_results_and_plot(file_path='flow_results.json', test='test1', num_node=9)




if __name__ == '__main__':
    main()
