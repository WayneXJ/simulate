import os
import random
import json

# 假设 Flow 是一个类
class Flow:
    def __init__(self, start_node, end_node, demand, start_time, end_time):
        self.start_node = start_node
        self.end_node = end_node
        self.demand = demand
        self.start_time = start_time
        self.end_time = end_time

    def to_dict(self):
        return {
            "start_node": self.start_node,
            "end_node": self.end_node,
            "demand": self.demand,
            "start_time": self.start_time,
            "end_time": self.end_time
        }

    @staticmethod
    def from_dict(flow_dict):
        return Flow(
            flow_dict["start_node"],
            flow_dict["end_node"],
            flow_dict["demand"],
            flow_dict["start_time"],
            flow_dict["end_time"]
        )


def generate_random_flows(num_node, num_flows, demand_range, time_range, file_path):
    flows = []
    node_ids = list(range(num_node))  # 根据节点数量生成节点编号

    for _ in range(num_flows):
        start_node = random.choice(node_ids)
        end_node = random.choice([node_id for node_id in node_ids if node_id != start_node])

        demand = random.randint(demand_range[0], demand_range[1])
        end_time = random.randint(time_range[0] + 1, time_range[1])
        start_time = random.randint(time_range[0], end_time - 1)
        if start_node == 0:
            continue
        else:
            flow = Flow(start_node, end_node, demand, start_time, end_time)
            flows.append(flow)

    # 按开始时间排序
    flows_sorted = sorted(flows, key=lambda flow: flow.start_time)

    # 确保文件夹存在
    output_dir = os.path.dirname(file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 将排序后的流保存到文件中
    with open(file_path, 'w') as file:
        json.dump([flow.to_dict() for flow in flows_sorted], file)

    return flows_sorted


def generate_periodic_flows(node_ids, num_flows, file_path, end_node, total_demand, demand_per_flow, time_range, period, last_time):
    flows = []
    num_nodes = len(node_ids)
    total_demand_per_node = total_demand // num_nodes  # 每个节点的总流量

    for node_id in node_ids:
        current_time = time_range[0]
        generated_demand = 0

        while generated_demand < total_demand_per_node:
            if len(flows) >= num_flows:
                break

            # 计算当前周期的流量
            start_node = node_id
            demand = min(demand_per_flow, total_demand_per_node - generated_demand)
            start_time = current_time
            end_time = min(current_time + last_time, time_range[1])

            flow = Flow(start_node, end_node, demand, start_time, end_time)
            flows.append(flow)
            generated_demand += demand
            current_time += period  # 更新为下一个周期的开始时间

            if current_time > time_range[1]:  # 如果时间超出范围，结束生成
                break

    # 按开始时间排序
    flows_sorted = sorted(flows, key=lambda flow: flow.start_time)

    # 确保文件夹存在
    output_dir = os.path.dirname(file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 将排序后的流保存到文件中
    with open(file_path, 'w') as file:
        json.dump([flow.to_dict() for flow in flows_sorted], file)

    return flows_sorted


def calculate_total_flow(flows):
    total_flow = 0
    for flow in flows:
        duration = flow.end_time - flow.start_time
        flow_amount = flow.demand * duration
        total_flow += flow_amount
    return total_flow


def calculate_to_zero_flow(flows):
    total_flow = 0
    for flow in flows:
        if flow.end_node == 0:
            duration = flow.end_time - flow.start_time
            flow_amount = flow.demand * duration
            total_flow += flow_amount
    return total_flow


def load_flows_from_file(file_path):
    with open(file_path, 'r') as file:
        flows_data = json.load(file)
        return [Flow.from_dict(flow_dict) for flow_dict in flows_data]


def generate_multiple_random_flow_files(node_list, flow_multiplier, demand_range, time_range, base_dir):
    """
    根据给定的节点数列表和流量倍数，生成多个流量文件。
    :param node_list: 节点数列表，每个节点数将生成一个对应的流量文件
    :param flow_multiplier: 每个节点数对应的流量倍数，用于计算生成的流量数
    :param demand_range: 流量需求范围 (min_demand, max_demand)
    :param time_range: 时间范围 (start_time, end_time)
    :param base_dir: 存储生成流量文件的基本目录
    """
    for num_node in node_list:
        num_flows = num_node * flow_multiplier  # 流量数与节点数成正比
        file_path = os.path.join(base_dir, f'random_flows_{num_node}.json')
        print(f"------------------random_{num_node}----------------------")
        print(f"Generating {num_flows} flows for {num_node} nodes...")
        generate_random_flows(num_node, num_flows, demand_range, time_range, file_path)
        print(f"Flows saved to {file_path}")
        random_flows = load_flows_from_file(file_path)
        for flow in random_flows:
            print(flow.start_node, flow.end_node, flow.demand, flow.start_time, flow.end_time)
        print("random:", calculate_total_flow(random_flows))


def generate_multiple_sensor_flow_files(node_list, num_flows, base_dir):

    for num_node in node_list:
        node_ids = [num_node, num_node+1, num_node+2]
        file_path = os.path.join(base_dir, f'sink_flows_{num_node}.json')
        print(f"------------------sensor_{num_node}------------------------")
        print(f"Generating {num_flows} flows for {num_node} nodes...")
        generate_periodic_flows(node_ids, num_flows, file_path, end_node=0, total_demand=3600,
                                demand_per_flow=20, time_range=(0, 40), period=8, last_time=6)
        print(f"Flows saved to {file_path}")

        sink_flows = load_flows_from_file(file_path)

        for flow in sink_flows:
            print(flow.start_node, flow.end_node, flow.demand, flow.start_time, flow.end_time)
        print("sensor:", calculate_total_flow(sink_flows))

        random_file_path = os.path.join('random_flows', f'random_flows_{num_node}.json')
        random_flows = load_flows_from_file(random_file_path)
        print("sink:", calculate_to_zero_flow(random_flows) + calculate_to_zero_flow(sink_flows))


def main():
    # 节点数列表
    node_list = [7, 8, 9, 10]

    # 流量倍数（用于控制生成的流量数，流量数 = 节点数 * 流量倍数）
    flow_multiplier = 10

    # 流量需求范围
    demand_range = (1, 10)

    # 时间范围
    time_range = (0, 40)

    # sensor流的数量
    num_flows = 60

    # 生成流量文件
    generate_multiple_random_flow_files(node_list, flow_multiplier, demand_range, time_range, 'random_flows')
    generate_multiple_sensor_flow_files(node_list, num_flows, 'sink_flows')


if __name__ == '__main__':
    main()
