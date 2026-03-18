import numpy as np


class Temporal_Walk(object):
    def __init__(self, learn_data, inv_relation_id, transition_distr):
        """
        Initialize temporal random walk object.

        Parameters:
            learn_data (np.ndarray): data on which the rules should be learned
            inv_relation_id (dict): mapping of relation to inverse relation
            transition_distr (str): transition distribution
                                    "unif" - uniform distribution
                                    "exp"  - exponential distribution

        Returns:
            None
        """

        self.learn_data = learn_data
        self.inv_relation_id = inv_relation_id # 关系→逆关系的映射（如{1:5,5:1})
        self.transition_distr = transition_distr # 转移分布类型：unif/exp
        self.neighbors = store_neighbors(learn_data) # 节点邻居索引
        self.edges = store_edges(learn_data) # 关系边索引

    def sample_start_edge(self, rel_idx):
        """
        Define start edge distribution.

        Parameters:
            rel_idx (int): relation index

        Returns:
            start_edge (np.ndarray): start edge
        """

        rel_edges = self.edges[rel_idx] # 获取指定关系的所有边
        # 均匀采样一条边作为起始边
        start_edge = rel_edges[np.random.choice(len(rel_edges))]

        return start_edge

    def sample_next_edge(self, filtered_edges, cur_ts):
        """
        Define next edge distribution.

        Parameters:
            filtered_edges (np.ndarray): filtered (according to time) edges
            cur_ts (int): current timestamp

        Returns:
            next_edge (np.ndarray): next edge
        """

        if self.transition_distr == "unif": # 均匀分布：等概率采样所有合法边
            next_edge = filtered_edges[np.random.choice(len(filtered_edges))]
        elif self.transition_distr == "exp": # 指数分布：时间越接近当前时间，采样概率越高
            tss = filtered_edges[:, 3]
            prob = np.exp(tss - cur_ts) # 时间差越小，exp值越大
            try:
                prob = prob / np.sum(prob) # 归一化概率
                next_edge = filtered_edges[
                    np.random.choice(range(len(filtered_edges)), p=prob)
                ]
            except ValueError:  # All timestamps are far away # 概率和为0/NaN（所有时间戳过远），降级为均匀采样
                next_edge = filtered_edges[np.random.choice(len(filtered_edges))]

        return next_edge

    def transition_step(self, cur_node, cur_ts, prev_edge, start_node, step, L):
        """
        Sample a neighboring edge given the current node and timestamp.
        In the second step (step == 1), the next timestamp should be smaller than the current timestamp.
        In the other steps, the next timestamp should be smaller than or equal to the current timestamp.
        In the last step (step == L-1), the edge should connect to the source of the walk (cyclic walk).
        It is not allowed to go back using the inverse edge.

        Parameters:
            cur_node (int): current node
            cur_ts (int): current timestamp
            prev_edge (np.ndarray): previous edge
            start_node (int): start node
            step (int): number of current step
            L (int): length of random walk

        Returns:
            next_edge (np.ndarray): next edge
            单步转移（约束过滤核心)
        """

        next_edges = self.neighbors[cur_node] # 当前节点的所有出边
        # 约束1：时序约束
        if step == 1:  # The next timestamp should be smaller than the current timestamp
            # 时间戳必须严格小于当前时间（递减）
            filtered_edges = next_edges[next_edges[:, 3] < cur_ts]
        else:  # The next timestamp should be smaller than or equal to the current timestamp
            # 后续步骤：时间戳小于等于当前时间
            filtered_edges = next_edges[next_edges[:, 3] <= cur_ts]
            # Delete inverse edge
            # 约束2：禁止反向游走（删除逆关系边）
            inv_edge = [
                cur_node,
                self.inv_relation_id[prev_edge[1]],
                prev_edge[0],
                cur_ts,
            ]
            # 找到逆关系边的索引并删除
            row_idx = np.where(np.all(filtered_edges == inv_edge, axis=1))
            filtered_edges = np.delete(filtered_edges, row_idx, axis=0)
        # 约束3：循环约束（最后一步必须回到起始节点）
        if step == L - 1:  # Find an edge that connects to the source of the walk
            filtered_edges = filtered_edges[filtered_edges[:, 2] == start_node]
        # 采样下一条边（无合法边则返回空）
        if len(filtered_edges):
            next_edge = self.sample_next_edge(filtered_edges, cur_ts)
        else:
            next_edge = []

        return next_edge

    def sample_walk(self, L, rel_idx):
        """
        Try to sample a cyclic temporal random walk of length L (for a rule of length L-1).

        Parameters:
            L (int): length of random walk
            rel_idx (int): relation index

        Returns:
            walk_successful (bool): if a cyclic temporal random walk has been successfully sampled
            walk (dict): information about the walk (entities, relations, timestamps)
        """

        walk_successful = True
        walk = dict()
        # 1. 初始化：采样起始边
        prev_edge = self.sample_start_edge(rel_idx)
        start_node = prev_edge[0] # 游走起始节点
        cur_node = prev_edge[2] # 当前节点（起始边的尾节点）
        cur_ts = prev_edge[3] # 当前时间戳
        # 2. 初始化游走记录
        walk["entities"] = [start_node, cur_node] # 实体路径
        walk["relations"] = [prev_edge[1]] # 关系路径
        walk["timestamps"] = [cur_ts] # 时间戳路径

        # 3. 多步游走（总长度L，已走1步，还需走L-1步）
        for step in range(1, L):
            next_edge = self.transition_step(
                cur_node, cur_ts, prev_edge, start_node, step, L
            )
            if len(next_edge):
                # 更新当前状态
                cur_node = next_edge[2]
                cur_ts = next_edge[3]
                walk["relations"].append(next_edge[1])
                walk["entities"].append(cur_node)
                walk["timestamps"].append(cur_ts)
                prev_edge = next_edge
            else:  # No valid neighbors (due to temporal or cyclic constraints)
                # 无合法边，游走失败
                walk_successful = False
                break

        return walk_successful, walk


def store_neighbors(quads):
    """
    Store all neighbors (outgoing edges) for each node.

    Parameters:
        quads (np.ndarray): indices of quadruples

    Returns:
        neighbors (dict): neighbors for each node
    """

    neighbors = dict()
    nodes = list(set(quads[:, 0]))
    for node in nodes:
        neighbors[node] = quads[quads[:, 0] == node]
    # 输出：{节点ID: 该节点的所有出边四元组} 字典
    return neighbors


def store_edges(quads):
    """
    Store all edges for each relation.

    Parameters:
        quads (np.ndarray): indices of quadruples

    Returns:
        edges (dict): edges for each relation
    """

    edges = dict()
    relations = list(set(quads[:, 1]))
    for rel in relations:
        edges[rel] = quads[quads[:, 1] == rel]
    # 存储该关系的所有四元组
    return edges
