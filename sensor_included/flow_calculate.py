import copy
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt

# r_graph 和 graph相同
# 增加了起止时间判断，可以计算任意时间段的最大流
#可以一次性输出所有的路径
class Edge:
    def __init__(self, to, capacity, reverse_edge, flag):
        self.to = to
        self.capacity = capacity
        self.reverse_edge = reverse_edge    # 该边是由残余图产生的
        self.flag = flag   # 0 表示产生的原边，1 表示从由残余图残生的边


# 输入两个点以及两个点之间的边容量来生成边，将生成的边保存在图的结点中
def add_edge(graph, u, v, capacity):
    h = len(capacity)
    forward_edge = Edge(v, capacity, None, 0)
    reverse_edge = Edge(u, [0]*h, forward_edge, 1)   # 该反向边是残余图所产生的
    forward_edge.reverse_edge = reverse_edge
    graph[u].append(forward_edge)
    graph[v].append(reverse_edge)


# 判断当前结点到目标结点在t-1时刻及其之后是否存在边容量
def is_empty(graph, u, v, t, start_time,h):
    for edges in graph[u]:
        if edges.to == v:
            for i in range(h):
                if edges.capacity[i+start_time] != 0:
                    if t-1 <= i:
                        return True
    return False

def is_UVA_in_path(graph, u, v, t, start_time,h):
    for edges in graph[u]:
        if edges.to == v:
            for i in range(h):
                if edges.capacity[i+start_time] == -1:
                    if t-1 <= i:
                        return True
    return False


def all_zero(lst):
    return all(x == 0 for x in lst)


def are_graphs_equal(graph1, graph2):
    # 如果两个图的节点数量不一样，直接返回 False
    if len(graph1) != len(graph2):
        return False

    # 比较每个节点的边集
    for i in range(len(graph1)):
        edges1 = graph1[i]
        edges2 = graph2[i]

        # 如果某个节点的边数量不一样，返回 False
        if len(edges1) != len(edges2):
            return False

        # 对每条边进行详细比较
        for e1, e2 in zip(edges1, edges2):
            # 检查 to, flag, capacity 列表是否相等
            if e1.to != e2.to or e1.flag != e2.flag or e1.capacity != e2.capacity:
                return False

            # 检查反向边是否一致
            if (e1.reverse_edge is None and e2.reverse_edge is not None) or (
                    e1.reverse_edge is not None and e2.reverse_edge is None):
                return False

            # 如果反向边存在，检查反向边的内容
            if e1.reverse_edge and e2.reverse_edge:
                if e1.reverse_edge.to != e2.reverse_edge.to or e1.reverse_edge.capacity != e2.reverse_edge.capacity or e1.reverse_edge.flag != e2.reverse_edge.flag:
                    return False

    # 所有的比较都通过，说明两个图相等
    return True


def bfs(graph, source, target, n, h, start_time, path):
    visited = [False] * len(graph)
    print("len of graph:",len(graph))
    print("source:",source)
    queue = deque()
    path_list = []
    queue.append((source, visited, [[h, source]]))
    visited[source] = True
    t_s = h
    # print(h)
    while queue:
        u, visited, path_list_u = queue.popleft()
        # print("u:",u,"visited:",visited,"path_list_u:",path_list_u)
        a_l = u

        for i in range(len(graph)):
            v = a_l

            for j in range(t_s, 1, -1):  # Iterate from h to 2
                # print(j)
                # print("进了for：")
                # print("n[v][j-2]:",n[v][j-2])
                if n[v][j-2] == 0:
                    t_s = j
                    # print("break")
                    break
                else:
                    t_s = j-1
            if visited[i] == False and is_empty(graph, a_l, i, t_s, start_time, h):
                temp_visited = visited.copy()
                temp_visited[i] = True
                temp_path_list = []
                for item in path_list_u:
                    if item[-1] == u:
                        new_item = item.copy()
                        new_item.append(i)
                        new_item[0] = t_s
                        temp_path_list.append(new_item)
                    else:
                        new_item = item.copy()
                        temp_path_list.append(new_item)
                queue.append((i, temp_visited, temp_path_list))
                if i == target:
                    path_list.append(temp_path_list[0])
    if path_list != []:
        # print("path_list",path_list)
        list_set = sorted([(item, item[0]) for item in path_list], key=lambda x: x[0])  # 以时间排序
        # print("list_set:",list_set)
        return list_set

    else:
        return []


def bfs_without_sensor(graph, source, target, n, h, start_time, path):
    visited1 = [False] * (len(graph) - 3)
    visited2 = [True] * 3
    visited = visited1 + visited2
    print("len of graph:",len(graph))
    print("source:",source)
    queue = deque()
    path_list = []
    queue.append((source, visited, [[h, source]]))
    visited[source] = True
    t_s = h
    # print(h)
    while queue:
        u, visited, path_list_u = queue.popleft()
        # print("u:",u,"visited:",visited,"path_list_u:",path_list_u)
        a_l = u

        for i in range(len(graph)):
            if 0 < i < 7:
                v = a_l

                for j in range(t_s, 1, -1):  # Iterate from h to 2
                    # print(j)
                    # print("进了for：")
                    # print("n[v][j-2]:",n[v][j-2])
                    if n[v][j-2] == 0:
                        t_s = j
                        # print("break")
                        break
                    else:
                        t_s = j-1
                if visited[i] == False and is_empty(graph, a_l, i, t_s, start_time, h):
                    temp_visited = visited.copy()
                    temp_visited[i] = True
                    temp_path_list = []
                    for item in path_list_u:
                        if item[-1] == u:
                            new_item = item.copy()
                            new_item.append(i)
                            new_item[0] = t_s
                            temp_path_list.append(new_item)
                        else:
                            new_item = item.copy()
                            temp_path_list.append(new_item)
                    queue.append((i, temp_visited, temp_path_list))
                    #  加入对 -1 的判断，使得一旦出现 -1 则视为送达
                    #  此处分为两种情况：
                    #无人机已知： 只走无人机的路，一旦遇到 -1 则可以传输所有的数据，因此一旦遇到 -1 则返回
                    #无人机未知： 只有-1那条路能够传输完所有数据，因此只能够继续传输数据
                    # if is_UVA_in_path(graph, a_l, i, t_s, start_time, h):
                    #     print("存在利用无人机")
                    #     print(temp_path_list[0])
                    #     path_list.append(temp_path_list[0])
                    #     return path_list  # 无人机已知
                    if i == target:
                        path_list.append(temp_path_list[0])
    if path_list != []:
        # print("path_list",path_list)
        list_set = sorted([(item, item[0]) for item in path_list], key=lambda x: x[0])  # 以时间排序
        # print("list_set:",list_set)
        return list_set
    else:
        return []


def edmonds_karp(graph, source, target, n, start_time, end_time, demand):
    h = end_time - start_time + 1
    max_flow = [0]*h
    r_graph = graph
    graph_copy = copy.deepcopy(graph)
    demand_flow = [demand]*h
    demand_flow[h-1] = 0
    path = []
    l_all = bfs_without_sensor(r_graph, source, target, n, h, start_time, path)
    if l_all != []:
        for l_i, time in l_all:
            # print("l_i:",l_i)
            path.append(l_i)
            tflow = get_tflow(l_i, r_graph, n, h, source, start_time, demand_flow)   # 得到一条路径的最大流
            demand_flow = get_demand_flow(tflow, demand_flow)

            r_graph = get_res_network(l_i, tflow, r_graph, n, h, source, target, graph, start_time)  # 更新残余图

            max_flow = [x + y for x, y in zip(max_flow, tflow)]  # 计算最大流
            if all_zero(demand_flow):
                clear_graph(graph)
                return max_flow, demand_flow, path, True
        clear_graph(graph)
        print("修改之后（没有达到要求）：", are_graphs_equal(graph,graph_copy))
        return max_flow, demand_flow, path, False
    else:
        clear_graph(graph)
        print("没有找到路径")
        return max_flow, demand_flow, path, False


def get_res_network(l_i, max_flow, r_graph, n, h, source, target, graph, start_time):
    a = [[0] * h for _ in range(h)]
    aflow = [0] * h
    dflow = [0] * h
    tflow = [0] * h
    # 更新容量uv
    update_edge(source, l_i[2], r_graph, max_flow, h, start_time)


    son = [None] * len(graph)  # 子节点对父节点索引
    getson(l_i, son, target)  # 得到parent列表
    # print_graph(r_graph)

    for node in l_i[2:-1]:
        tN_T = n[node].copy()  # 得到v点的存储序列拷贝
        if node == l_i[2]:
            tflow_uv = max_flow
        else:
            tflow_uv = tflow.copy()
        cap_vw = get_cap(node, son[node], r_graph, h, start_time).copy()
        for p in range(h):
            for q in range(h):
                bet = get_beta(tN_T, h)  # beta
                a[q][p] = min(cap_vw[p], bet[q][p], tflow_uv[q])

                if q > p:
                    aflow[p] += a[q][p]
                # 更新cap_uv、 tflow_vw、tN_T
                cap_vw[p] = cap_vw[p] - a[q][p]   # 这里的论文中可能存在问题（在代码中已经改正）#######################
                tflow_uv[q] = tflow_uv[q] - a[q][p]
                for i in range(p, q):
                    tN_T[i] = tN_T[i] - a[q][p]
        for p in range(h):
            if p == 0:
                dflow[p] = min(cap_vw[p] - aflow[p], sum(tflow_uv[:p+1]))

            else:
                dflow[p] = min(cap_vw[p] - aflow[p], sum(tflow_uv[:p+1]) - sum(dflow[:p]))
            tflow[p] = aflow[p] + dflow[p]


        # 更新每条边的容量
        update_edge(node, son[node], r_graph, tflow, h, start_time)
        # print_graph(r_graph)
        # print("tflow_uv:",tflow_uv,"tflow",tflow)
        for i in range(h-1):
            if i == 0:
                n[node][i] = n[node][i] + tflow_uv[i] - tflow[i]
                # print(n[node][i])
            else:
                n[node][i] = n[node][i-1] + n[node][i] + tflow_uv[i] - tflow[i]
                # print(n[node][i])

        # print("node is in",node,":",n[node])
        return r_graph
    return r_graph


def get_tflow(l_i, graph, n, h, source, start_time, flow):    # 计算最大流
    if len(l_i) == 3:
        tflow = get_cap(l_i[1],l_i[2],graph,h,start_time)
    else:
        bflow = [0]*h
        sflow = [0]*h
        tflow = [0]*h
        b = [[0] * h for _ in range(h)]
        parent = [None] * len(graph)    # 子节点对父节点索引
        # print("get_tflow(l_i)",l_i)
        getparent(l_i, parent, source)   # 得到parent列表
        # print(l_i)
        for node in l_i[-2:1:-1]:
            tN_T = n[node].copy()   # 得到v点的存储序列拷贝

            # 得到v,w两点之间的暂时可行流
            if node == l_i[-2]:
                tflow_vw = get_cap(node, l_i[-1], graph, h, start_time).copy()
            else:
                tflow_vw = tflow.copy()

            # 计算u，v两点之间的容量
            # print("parent:",parent,"node:",node)
            cap_uv = get_cap(parent[node], node, graph, h, start_time).copy()
            # 计算
            for q in range(h):
                for p in range(h):
                    bet = get_beta(tN_T, h)  # beta
                    b[q][p] = min(cap_uv[q], bet[q][p], tflow_vw[p])
                    # 计算反向可行流
                    if q > p:
                        bflow[q] += b[q][p]

                    # 更新cap_uv、 tflow_vw、tN_T
                    cap_uv[q] = cap_uv[q] - b[q][p]
                    tflow_vw[p] = tflow_vw[p] - b[q][p]
                    for i in range(p, q):
                        tN_T[i] = tN_T[i] - b[q][p]
            for q in range(h-1, -1, -1):
                if q == h-1:
                    sflow[q] = min(cap_uv[q] - bflow[q], sum(tflow_vw[q:]))
                else:
                    sflow[q] = min(cap_uv[q] - bflow[q], sum(tflow_vw[q:]) - sum(sflow[q+1:]))
                tflow[q] = bflow[q] + sflow[q]
    # 判断流量是否符合约定，即传输速度不能超过源节点的发送速度
    # print("before tflow ", tflow)
    for i in range(h):
        sum_flow = sum(flow[:i+1])
        sum_tflow = sum(tflow[:i+1])
        # print("sum_tflow", sum_tflow)
        if sum_tflow > sum_flow:
            tflow[i] = sum_flow - sum_tflow + tflow[i]
    #         print("get_flow  i", i)
    #         print("get_flow  tflow[i]", tflow[i])
    # print("after  tflow", tflow)
    # print("get_flow  flow", flow)
    return tflow


def are_lists_equal(lst, lsts):
    # 首先检查长度是否相同
    for lst2 in lsts:
        if len(lst) == len(lst2):
            # 使用 zip 函数和 all 函数逐元素比较
            if all(x == y for x, y in zip(lst, lst2)):
                return True

    return False


def get_demand_flow(tflow, demand_flow):
    res = [demand_flow[i] - tflow[i] for i in range(len(tflow))]
    pos = 0  # 第一个非零数的位置
    for i in range(len(res)):
        if res[i] != 0:
            pos = i
            break
    for i in range(len(res)):
        if res[i] < 0:
            number = 0 - res[i]
            # print("number:", number)
            res[i] = 0
            for j in range(pos, i):
                if res[j] > number:
                    res[j] = res[j] - number
                    break
                else:
                    number = number - res[j]
                    res[j] = 0
                    pos = pos + 1
    return res


def clear_graph(graph):
    for node in range(len(graph)):
        for edge in graph[node]:
            if edge.flag == 1:
                edge.capacity = [0]*len(edge.capacity)


# 更新边容量
def update_edge(v, to, graph, flow, h, start_time):
    for edge in graph[v]:
        if edge.to == to and edge.flag == 0:
            # 更新边容量以及剩余流量
            flow = trans_edge_cap(edge, flow, h, start_time)

            # 同时更新由无向边分裂的反向边容量使得容量相等
            for t_edge in graph[to]:
                if t_edge.to == v and t_edge.flag == 0:
                    t_edge.capacity = edge.capacity
        if edge.to == to and edge.flag == 1:
            flow = trans_edge_cap(edge, flow, h, start_time)

    # print_graph(graph)


# 更新边容量
def trans_edge_cap(edge, flow, h, start_time):
    trans_cap = []    # 剩余边容量
    edge_cap = [0]*h    # 记录开始时间到结束时间的边容量
    for i in range(h):
        edge_cap[i] = edge.capacity[i+start_time]
    # print("flow: ",flow)
    # print("edge.capacity", edge.capacity)
    cap = [x - y for x, y in zip(edge_cap, flow)]
    # print("cap",cap ," edge.capacity - flow :",edge.capacity,"-",flow)
    for i in cap:
        if i < 0:
            trans_cap.append(0)
        else:
            trans_cap.append(i)
    re_cap = [x - y for x, y in zip(edge_cap, trans_cap)]   # 变化的边容量
    trans_flow = [x - y for x, y in zip(flow, re_cap)]  # 剩余流量
    # print("trans_cap",trans_cap)
    # print("re_cap",re_cap)
    # print("trans_flow",trans_flow)
    flow = trans_flow.copy()
    for i in range(h):
        edge.capacity[i+start_time] = trans_cap[i]
        edge.reverse_edge.capacity[i+start_time] = re_cap[i] + edge.reverse_edge.capacity[i+start_time]

    return flow


def getparent(path, parent, source):
    # print("getparent:(path)",path)
    tem = source
    for node in path[1:]:
        if node == source:
            parent[source] = source #源点没有父节点
            tem = source
        else:
            parent[node] = tem
            tem = node


def getson(path,son,target):
    tem = target
    for node in path[-1:0:-1]:
        if node == path[-1]:
            son[target] = target # 源点没有父节点
            tem = target
        else:
            son[node] = tem
            tem = node


def get_beta(n, h):      # n是结点存储量，h为时间划分数
    beta = [[0] * h for _ in range(h)]
    for q in range(h):
        for p in range(h):
            if p < q:
                beta[q][p] = min(n[p:q])   # p<q
    return beta


def get_cap(v, to, graph, h, start_time):
    cap = [0]*h
    edge_cap = [0]*h
    for edge in graph[v]:
        if edge.to == to:
            for i in range(h):
                edge_cap[i] = edge.capacity[i+start_time]
            cap = [x + y for x, y in zip(cap, edge_cap)]
    # print(cap)
    return cap


# Convert undirected graph to directed graph
def convert_to_directed_graph(undirected_edges, num_nodes):
    graph = [[] for _ in range(num_nodes)]
    for u, v, capacity in undirected_edges:
        add_edge(graph, u, v, capacity)
        add_edge(graph, v, u, capacity)
    return graph


# 打印图
def print_graph(graph):
    for i, edges in enumerate(graph):
        print("Node", i, ": ", end="")
        for edge in edges:
            print("(to:", edge.to, ", cap:", edge.capacity, ", rc:", edge.reverse_edge.capacity, ", flag:",edge.flag,") ", end="")
        print()

#画图
def draw_graph(graph):
    G = nx.DiGraph()  # 初始化为有向图
    for i, edges in enumerate(graph):
        for edge in edges:
            G.add_edge(i, edge.to, capacity=edge.capacity)

    pos = nx.spring_layout(G)  # 定义布局
    edge_labels = {(u, v): f"{d['capacity']}/{edge.reverse_edge.capacity}" for u, v, d in G.edges(data=True)}  # 边的标签

    nx.draw(G, pos, with_labels=True, node_size=800, node_color='lightblue', font_size=10, font_weight='bold', arrows=True)  # 绘制有向图
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')  # 绘制边的标签
    plt.show()


def main():
    node = [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    # 0-S,1-A,2-B,3-D
    edges = [(0, 1, [7, 2, 6, 1, 2]),
             (1, 2, [6, 2, 3, 0, 2]),
             (0, 2, [2, 3, 2, 2, 2]),
             (1, 3, [2, 7, 2, 4, 2]),
             (2, 3, [3, 0, 2, 5, 3])]
    # edges = [(0, 1, [2, 0, 3, 0, 0]), (1, 2, [3, 0, 0, 0, 0]), (0, 2, [0, 3, 0, 0, 0]), (1, 3, [0, 0, 0, 1, 4]),
    #          (2, 3, [0, 0, 0, 2, 0])]
    num_nodes = 4
    graph = convert_to_directed_graph(edges, num_nodes)
    print_graph(graph)

    source = 0
    target = 3
    start_time = 0
    end_time = 4
    demand = 9
    # path = bfs(graph, source, target,node,5)
    max, left_flow, path, flag = edmonds_karp(graph, source, target, node, start_time,end_time,demand)
    print(max)
    print(left_flow)
    print(path)
    print_graph(graph)
    if flag :
        print("meet the requirement1")
    else:
        print("Don't reach the requirement1")

if __name__ == '__main__':
    main()
