
import json
import time
import argparse
import itertools
import numpy as np
from joblib import Parallel, delayed

import rule_application as ra
from grapher import Grapher
from temporal_walk import store_edges
from rule_learning import rules_statistics
from score_functions import score_12

import os
os.environ['JOBLIB_TEMP_FOLDER'] = r'E:\temptemp'
os.makedirs(r'E:\temptemp', exist_ok=True)
print(f"JOBLIB_TEMP_FOLDER = {os.environ.get('JOBLIB_TEMP_FOLDER', 'NOT SET')}")
print(f"TEMP = {os.environ.get('TEMP')}")
print(f"TMP = {os.environ.get('TMP')}")
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", default="", type=str,
                    help="数据集名称，用于定位数据文件路径")
parser.add_argument("--test_data", default="test", type=str,
                    help="使用的数据类型，可选test/valid，对应测试集/验证集")
parser.add_argument("--rules", "-r", default="", type=str,
                    help="预训练规则文件的路径/名称")
parser.add_argument("--rule_lengths", "-l", default=1, type=int, nargs="+",
                    help="规则长度过滤条件，可以传入多个值，如 -l 2 3")
parser.add_argument("--window", "-w", default=-1, type=int,
                    help="时间窗口大小，-1表示使用全部历史数据")
parser.add_argument("--top_k", default=20, type=int,
                    help="每个查询返回的候选答案数量上限")
parser.add_argument("--num_processes", "-p", default=1, type=int,
                    help="并行处理的进程数")
parsed = vars(parser.parse_args())

dataset = parsed["dataset"]
rules_file = parsed["rules"]
window = parsed["window"]
top_k = parsed["top_k"]
num_processes = parsed["num_processes"]
rule_lengths = parsed["rule_lengths"]
# 确保rule_lengths始终是列表类型（处理单个整数输入的情况）
rule_lengths = [rule_lengths] if (type(rule_lengths) == int) else rule_lengths

dataset_dir = "../data/" + dataset + "/"
dir_path = "../output/" + dataset + "/"
data = Grapher(dataset_dir)
# 选择测试集或验证集作为待处理的查询数据
test_data = data.test_idx if (parsed["test_data"] == "test") else data.valid_idx
# 从JSON文件加载预学习的规则字典
rules_dict = json.load(open(dir_path + rules_file))
# 将规则字典的键转换为整数类型（JSON读取的键默认是字符串）
rules_dict = {int(k): v for k, v in rules_dict.items()}
print("Rules statistics:")
# ===================== 规则预处理 =====================
# 打印原始规则的统计信息（数量、长度分布等）
rules_statistics(rules_dict)
# 规则剪枝：过滤掉置信度低、支持度不足、长度不符合要求的规则
rules_dict = ra.filter_rules(
    rules_dict, min_conf=0.01, min_body_supp=2, rule_lengths=rule_lengths
)
print("Rules statistics after pruning:")
rules_statistics(rules_dict)
# 从训练数据中提取并存储边信息，用于后续规则匹配
learn_edges = store_edges(data.train_idx)

# ===================== 评分函数和参数配置 =====================
# 选择评分函数（score_12为自定义的评分算法

score_func = score_12
# It is possible to specify a list of list of arguments for tuning
# 评分函数的超参数列表（用于调优，这里是示例值）
# 格式为列表的列表，支持多组参数并行测试
args = [[0.1, 0.5]]


def apply_rules(i, num_queries):
    """
    对指定批次的测试查询应用规则，生成候选答案（支持多进程并行）

    Parameters:
        i (int): 当前进程编号，用于分配数据批次
        num_queries (int): 每个进程处理的查询数量（基准值）

    Returns:
        all_candidates (list): 每个查询的候选答案字典列表（对应不同参数组合）
                              结构: [参数组合1的候选字典, 参数组合2的候选字典, ...]
                              候选字典结构: {查询索引: {候选答案: 分数, ...}, ...}
        no_cands_counter (int): 没有找到候选答案的查询数量
    """

    print("Start process", i, "...")
    all_candidates = [dict() for _ in range(len(args))]
    no_cands_counter = 0

    num_rest_queries = len(test_data) - (i + 1) * num_queries
    if num_rest_queries >= num_queries:
        test_queries_idx = range(i * num_queries, (i + 1) * num_queries)
    else:
        test_queries_idx = range(i * num_queries, len(test_data))

    cur_ts = test_data[test_queries_idx[0]][3]
    edges = ra.get_window_edges(data.all_idx, cur_ts, learn_edges, window)

    it_start = time.time()
    for j in test_queries_idx:
        test_query = test_data[j]
        cands_dict = [dict() for _ in range(len(args))]

        if test_query[3] != cur_ts:
            cur_ts = test_query[3]
            edges = ra.get_window_edges(data.all_idx, cur_ts, learn_edges, window)

        if test_query[1] in rules_dict:
            dicts_idx = list(range(len(args)))
            for rule in rules_dict[test_query[1]]:
                walk_edges = ra.match_body_relations(rule, edges, test_query[0])

                if 0 not in [len(x) for x in walk_edges]:
                    rule_walks = ra.get_walks(rule, walk_edges)
                    if rule["var_constraints"]:
                        rule_walks = ra.check_var_constraints(
                            rule["var_constraints"], rule_walks
                        )

                    if not rule_walks.empty:
                        cands_dict = ra.get_candidates(
                            rule,
                            rule_walks,
                            cur_ts,
                            cands_dict,
                            score_func,
                            args,
                            dicts_idx,
                        )
                        for s in dicts_idx:
                            cands_dict[s] = {
                                x: sorted(cands_dict[s][x], reverse=True)
                                for x in cands_dict[s].keys()
                            }
                            cands_dict[s] = dict(
                                sorted(
                                    cands_dict[s].items(),
                                    key=lambda item: item[1],
                                    reverse=True,
                                )
                            )
                            top_k_scores = [v for _, v in cands_dict[s].items()][:top_k]
                            unique_scores = list(
                                scores for scores, _ in itertools.groupby(top_k_scores)
                            )
                            if len(unique_scores) >= top_k:
                                dicts_idx.remove(s)
                        if not dicts_idx:
                            break

            if cands_dict[0]:
                for s in range(len(args)):
                    # Calculate noisy-or scores
                    scores = list(
                        map(
                            lambda x: 1 - np.product(1 - np.array(x)),
                            cands_dict[s].values(),
                        )
                    )
                    cands_scores = dict(zip(cands_dict[s].keys(), scores))
                    noisy_or_cands = dict(
                        sorted(cands_scores.items(), key=lambda x: x[1], reverse=True)
                    )
                    all_candidates[s][j] = noisy_or_cands
            else:  # No candidates found by applying rules
                no_cands_counter += 1
                for s in range(len(args)):
                    all_candidates[s][j] = dict()

        else:  # No rules exist for this relation
            no_cands_counter += 1
            for s in range(len(args)):
                all_candidates[s][j] = dict()

        if not (j - test_queries_idx[0] + 1) % 100:
            it_end = time.time()
            it_time = round(it_end - it_start, 6)
            print(
                "Process {0}: test samples finished: {1}/{2}, {3} sec".format(
                    i, j - test_queries_idx[0] + 1, len(test_queries_idx), it_time
                )
            )
            it_start = time.time()

    return all_candidates, no_cands_counter


start = time.time()
num_queries = len(test_data) // num_processes
output = Parallel(n_jobs=num_processes)(
    delayed(apply_rules)(i, num_queries) for i in range(num_processes)
)
end = time.time()

final_all_candidates = [dict() for _ in range(len(args))]
for s in range(len(args)):
    for i in range(num_processes):
        final_all_candidates[s].update(output[i][0][s])
        output[i][0][s].clear()

final_no_cands_counter = 0
for i in range(num_processes):
    final_no_cands_counter += output[i][1]

total_time = round(end - start, 6)
print("Application finished in {} seconds.".format(total_time))
print("No candidates: ", final_no_cands_counter, " queries")

for s in range(len(args)):
    score_func_str = score_func.__name__ + str(args[s])
    score_func_str = score_func_str.replace(" ", "")
    ra.save_candidates(
        rules_file,
        dir_path,
        final_all_candidates[s],
        rule_lengths,
        window,
        score_func_str,
    )
