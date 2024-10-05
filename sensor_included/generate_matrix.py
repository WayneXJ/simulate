import numpy as np
import os

def generate_custom_matrix(rows, cols, num_ones):
    """
    生成一个随机矩阵，矩阵大小为rows x cols，并且矩阵中元素为1的个数为num_ones，其余元素为0。

    参数:
    rows (int): 矩阵的行数
    cols (int): 矩阵的列数
    num_ones (int): 矩阵中元素为1的个数

    返回:
    np.ndarray: 生成的随机矩阵
    """
    if num_ones > rows * cols:
        raise ValueError("num_ones不能大于矩阵的总元素数")

    # 创建一个全零的矩阵
    matrix = np.zeros((rows, cols), dtype=int)

    # 随机选择num_ones个位置设为1
    ones_positions = np.random.choice(rows * cols, num_ones, replace=False)
    matrix[np.unravel_index(ones_positions, (rows, cols))] = 1

    return matrix


def save_matrices_to_file(filename, num_matrices, rows, cols, num_ones):
    """
    生成多个矩阵并保存到文件中。

    参数:
    filename (str): 保存矩阵的文件名
    num_matrices (int): 生成矩阵的数量
    rows (int): 矩阵的行数
    cols (int): 矩阵的列数
    num_ones (int): 每个矩阵中元素为1的个数
    """
    matrices = []
    for _ in range(num_matrices):
        matrices.append(generate_custom_matrix(rows, cols, num_ones))


    # 保存矩阵到文件中
    np.save(filename, matrices)


def load_matrices_from_file(filename):
    """
    从文件中读取矩阵。

    参数:
    filename (str): 存储矩阵的文件名

    返回:
    list: 矩阵列表
    """
    matrices = np.load(filename, allow_pickle=True)
    return matrices


def generate_multiple_matrices(node_list, num_matrices, rows, cols, base_dir):
    for num_node in node_list:
        print(f"------------------matrices_{num_node + 1}---------------------")
        file_path = os.path.join(base_dir, f'matrices_{num_node + 1}.npy')
        save_matrices_to_file(file_path, num_matrices, rows, cols, num_node)
        loaded_matrices = load_matrices_from_file( file_path)
        for i, matrix in enumerate(loaded_matrices):
            print(f"Matrix {i + 1}:\n{matrix}\n")


def main():
    # 示例用法
    node_list = [6, 7, 8, 9]
    num_matrices = 1  #矩阵个数
    rows = 12
    cols = 12

    generate_multiple_matrices(node_list, num_matrices, rows, cols, 'matrices')


if __name__ == '__main__':
    main()
