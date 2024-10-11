import json
import os

def output_position( output_position_log,position_log):
    # 生成新的 test 标记
    test_position_log = f"test1"

    if os.path.exists(output_position_log):
        with open(output_position_log, 'r') as f:
            try:
                existing_position_log = json.load(f)
            except json.JSONDecodeError:
                existing_position_log = {}
        test_position_log = f"test{len(existing_position_log) + 1}"
    else:
        existing_position_log = {}

    # 将新数据加入现有数据中
    existing_position_log[test_position_log] = position_log

    # 写回合并后的数据
    with open(output_position_log, 'w') as f:
        json.dump(existing_position_log, f, indent=4)



def output_flow(output_time_series_data,time_series_data):
    # 生成新的 test 标记
    test_time_series_data = f"test1"
    if os.path.exists(output_time_series_data):
        with open(output_time_series_data, 'r') as f:
            try:
                existing_time_series_data = json.load(f)
            except json.JSONDecodeError:
                existing_time_series_data = {}
        test_time_series_data = f"test{len(existing_time_series_data) + 1}"
    else:
        existing_time_series_data = {}

    # 将新数据加入现有数据中,对time_series_data序列化
    serializable_time_series_data = {}
    for flow, data in time_series_data.items():
        key = f"Flow from Node {flow.start_node.node_id} to Node {flow.end_node.node_id}"
        serializable_time_series_data[key] = data

    existing_time_series_data[test_time_series_data] = serializable_time_series_data
    with open(output_time_series_data, 'w') as f:
        json.dump(existing_time_series_data, f, indent=4)


def output_flow_results(output_file, results):
    # 生成新的 test 标记
    test_label = "test1"
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            try:
                existing_results = json.load(f)
            except json.JSONDecodeError:
                existing_results = {}
        test_label = f"test{len(existing_results) + 1}"
    else:
        existing_results = {}

    # 将新数据加入现有数据中
    serializable_results = {}
    for result in results:
        flow_info = result['flow']
        result_info = result['result']
        key = f"Flow from Node {flow_info['start_node']} to Node {flow_info['end_node']}"
        serializable_results[key] = result_info

    existing_results[test_label] = serializable_results

    # 将数据写入文件
    with open(output_file, 'w') as f:
        json.dump(existing_results, f, indent=4)


#读取位置日志中的数据
def read_position_log(file_path):
    with open(file_path, 'r') as f:
        position_log = json.load(f)
    return position_log


# 读取流量的日志
def read_time_series_data(file_path):
    with open(file_path, 'r') as f:
        time_series_data = json.load(f)
    return time_series_data


