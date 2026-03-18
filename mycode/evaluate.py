import json
import argparse
import numpy as np

import rule_application as ra
from grapher import Grapher
from temporal_walk import store_edges
from baseline import baseline_candidates, calculate_obj_distribution


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", default="", type=str)
parser.add_argument("--test_data", default="test", type=str)
parser.add_argument("--candidates", "-c", default="", type=str)
parsed = vars(parser.parse_args())


def filter_candidates(test_query, candidates, test_data):
    """
    Filter out those candidates that are also answers to the test query
    but not the correct answer.

    Parameters:
        test_query (np.ndarray): test_query
        candidates (dict): answer candidates with corresponding confidence scores
        test_data (np.ndarray): test dataset

    Returns:
        candidates (dict): filtered candidates

        过滤掉那些是测试查询的答案，但不是正确答案的候选（排除干扰项

    """
    # 筛选出：和当前查询有相同头实体、关系、时间，但尾实体不同的其他答案
    other_answers = test_data[
        (test_data[:, 0] == test_query[0])# 头实体ID相同
        * (test_data[:, 1] == test_query[1])# 关系ID相同
        * (test_data[:, 2] != test_query[2])# 尾实体（答案）不同
        * (test_data[:, 3] == test_query[3])# 时间戳相同（时序知识图谱
    ]

    if len(other_answers):
        objects = other_answers[:, 2]# 提取这些干扰项的尾实体ID
        for obj in objects:
            candidates.pop(obj, None)# 从候选列表中删除这些干扰项

    return candidates


def calculate_rank(test_query_answer, candidates, num_entities, setting="best"):
    """
    Calculate the rank of the correct answer for a test query.
    Depending on the setting, the average/best/worst rank is taken if there
    are several candidates with the same confidence score.

    Parameters:
        test_query_answer (int): test query answer
        candidates (dict): answer candidates with corresponding confidence scores
        num_entities (int): number of entities in the dataset
        setting (str): "average", "best", or "worst"

    Returns:
        rank (int): rank of the correct answer

        计算正确答案在候选列表中的排名（处理同分情况）
    setting可选：average（平均排名）、best（最佳排名）、worst（最差排名）

    """

    rank = num_entities # 默认排名：实体总数（表示候选中无正确答案）
    if test_query_answer in candidates:
        conf = candidates[test_query_answer] # 正确答案的置信度分数
        all_confs = list(candidates.values()) # 所有候选的置信度列表
        ranks = [idx for idx, x in enumerate(all_confs) if x == conf]
        if setting == "average":
            rank = (ranks[0] + ranks[-1]) // 2 + 1
        elif setting == "best":
            rank = ranks[0] + 1
        elif setting == "worst":
            rank = ranks[-1] + 1

    return rank


dataset = parsed["dataset"]
candidates_file = parsed["candidates"]
dir_path = "../output/" + dataset + "/"
dataset_dir = "../data/" + dataset + "/"
data = Grapher(dataset_dir)
num_entities = len(data.id2entity)  # 数据集中的实体总数
# 选择测试集/验证集
test_data = data.test_idx if (parsed["test_data"] == "test") else data.valid_idx
# 加载训练集的边信息 & 计算目标实体分布（用于基线候选生成）
learn_edges = store_edges(data.train_idx)
obj_dist, rel_obj_dist = calculate_obj_distribution(data.train_idx, learn_edges)

# 加载候选答案文件（JSON格式）
all_candidates = json.load(open(dir_path + candidates_file))
# 转换键为int类型（JSON加载的键默认是字符串，而实体ID是int）
all_candidates = {int(k): v for k, v in all_candidates.items()}
for k in all_candidates:
    all_candidates[k] = {int(cand): v for cand, v in all_candidates[k].items()}

hits_1 = 0
hits_3 = 0
hits_10 = 0
mrr = 0

num_samples = len(test_data)
print("Evaluating " + candidates_file + ":")
for i in range(num_samples):
    test_query = test_data[i] # 获取当前测试查询（格式：[头实体, 关系, 正确尾实体, 时间]）
    # 获取候选答案：有预测候选则用预测的，否则用基线方法生成
    if all_candidates[i]:
        candidates = all_candidates[i]
    else:
        candidates = baseline_candidates(
            test_query[1], learn_edges, obj_dist, rel_obj_dist
        )
    # 过滤干扰项候选
    candidates = filter_candidates(test_query, candidates, test_data)
    # 计算正确答案的排名
    rank = calculate_rank(test_query[2], candidates, num_entities)

    if rank:
        if rank <= 10:
            hits_10 += 1
            if rank <= 3:
                hits_3 += 1
                if rank == 1:
                    hits_1 += 1
        mrr += 1 / rank

hits_1 /= num_samples
hits_3 /= num_samples
hits_10 /= num_samples
mrr /= num_samples

print("Hits@1: ", round(hits_1, 6))
print("Hits@3: ", round(hits_3, 6))
print("Hits@10: ", round(hits_10, 6))
print("MRR: ", round(mrr, 6))

filename = candidates_file[:-5] + "_eval.txt"
with open(dir_path + filename, "w", encoding="utf-8") as fout:
    fout.write("Hits@1: " + str(round(hits_1, 6)) + "\n")
    fout.write("Hits@3: " + str(round(hits_3, 6)) + "\n")
    fout.write("Hits@10: " + str(round(hits_10, 6)) + "\n")
    fout.write("MRR: " + str(round(mrr, 6)))
