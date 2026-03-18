import time
import argparse
import numpy as np
from datetime import datetime
from joblib import Parallel, delayed

from grapher import Grapher
from temporal_walk import Temporal_Walk
from rule_learning import Rule_Learner, rules_statistics
import os
os.environ['JOBLIB_TEMP_FOLDER'] = r'E:\temptemp'
os.makedirs(r'E:\temptemp', exist_ok=True)
print(f"JOBLIB_TEMP_FOLDER = {os.environ.get('JOBLIB_TEMP_FOLDER', 'NOT SET')}")
print(f"TEMP = {os.environ.get('TEMP')}")
print(f"TMP = {os.environ.get('TMP')}")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", default="", type=str)
parser.add_argument("--rule_lengths", "-l", default="3", type=int, nargs="+") # 规则长度列表
parser.add_argument("--num_walks", "-n", default="100", type=int) # 每个关系的游走次数
parser.add_argument("--transition_distr", default="exp", type=str)
parser.add_argument("--num_processes", "-p", default=1, type=int) # 进程数
parser.add_argument("--seed", "-s", default=None, type=int) # 随机种子
parsed = vars(parser.parse_args())
# 参数解析与格式化
dataset = parsed["dataset"]
rule_lengths = parsed["rule_lengths"]
rule_lengths = [rule_lengths] if (type(rule_lengths) == int) else rule_lengths # 统一为列表
num_walks = parsed["num_walks"]
transition_distr = parsed["transition_distr"]
num_processes = parsed["num_processes"]
seed = parsed["seed"]

dataset_dir = "../data/" + dataset + "/"
data = Grapher(dataset_dir) # 加载数据集（Grapher类负责读取数据、构建索引、映射关系/实体ID）
# 初始化时序游走器（传入训练数据、逆关系映射、转移分布）
temporal_walk = Temporal_Walk(data.train_idx, data.inv_relation_id, transition_distr)
# 初始化规则学习器（传入边索引、ID到关系的映射、逆关系映射、数据集名）
rl = Rule_Learner(temporal_walk.edges, data.id2relation, data.inv_relation_id, dataset)
# 获取所有关系并排序（保证多进程分配的一致性）
all_relations = sorted(temporal_walk.edges)  # Learn for all relations


def learn_rules(i, num_relations):
    """
    Learn rules (multiprocessing possible).

    Parameters:
        i (int): process number
        num_relations (int): minimum number of relations for each process
        每个进程分配的最小关系数

    Returns:
        rl.rules_dict (dict): rules dictionary 该进程学习到的规则字典
    """

    if seed:
        np.random.seed(seed)
    # 计算当前进程负责的关系范围（负载均衡）
    num_rest_relations = len(all_relations) - (i + 1) * num_relations
    if num_rest_relations >= num_relations:
        # 正常分配：每个进程num_relations个关系
        relations_idx = range(i * num_relations, (i + 1) * num_relations)
    else:
        # 最后一个进程：分配剩余所有关系
        relations_idx = range(i * num_relations, len(all_relations))
    # 统计规则数量
    num_rules = [0]
    for k in relations_idx:
        rel = all_relations[k]# 当前处理的关系ID
        for length in rule_lengths: # 遍历所有规则长度
            it_start = time.time()
            # 采样num_walks次游走
            for _ in range(num_walks):
                # 采样长度为length+1的游走（节点数=length+1 → 边数=length → 规则长度=length）
                walk_successful, walk = temporal_walk.sample_walk(length + 1, rel)
                if walk_successful: # 游走成功（满足所有约束）
                    rl.create_rule(walk) # 从游走路径创建规则
            it_end = time.time()
            it_time = round(it_end - it_start, 6)
            # 统计新增规则数
            num_rules.append(sum([len(v) for k, v in rl.rules_dict.items()]) // 2)
            num_new_rules = num_rules[-1] - num_rules[-2]
            # 打印进程进度
            print(
                "Process {0}: relation {1}/{2}, length {3}: {4} sec, {5} rules".format(
                    i,
                    k - relations_idx[0] + 1,
                    len(relations_idx),
                    length,
                    it_time,
                    num_new_rules,
                )
            )

    return rl.rules_dict


start = time.time()
num_relations = len(all_relations) // num_processes
# 多进程并行执行规则学习
output = Parallel(n_jobs=num_processes)(
    delayed(learn_rules)(i, num_relations) for i in range(num_processes)
)
end = time.time()
# 合并所有进程的规则字典
all_rules = output[0]
for i in range(1, num_processes):
    all_rules.update(output[i])

total_time = round(end - start, 6)
print("Learning finished in {} seconds.".format(total_time))

# 规则后处理：排序、保存、统计
rl.rules_dict = all_rules
rl.sort_rules_dict() # 规则排序（如按支持度/置信度）
dt = datetime.now()
dt = dt.strftime("%d%m%y%H%M%S")
rl.save_rules(dt, rule_lengths, num_walks, transition_distr, seed) # 保存规则
rl.save_rules_verbalized(dt, rule_lengths, num_walks, transition_distr, seed) # 保存可视化规则
rules_statistics(rl.rules_dict) # 规则统计（如规则总数、平均长度、支持度分布等）
