# 导入系统标准库
import json          # 用于JSON文件的读写操作
import time          # 用于计算程序运行时间
import argparse      # 用于解析命令行参数
import itertools     # 用于迭代器和组合操作（此处用于去重分数）
import os            # 用于操作系统相关操作（路径、环境变量）

# 导入第三方库
import numpy as np                           # 用于数值计算（数组、数学运算）
from joblib import Parallel, delayed         # 用于多进程并行处理

# 导入自定义模块（项目内的核心功能模块）
import rule_application as ra                # 规则应用核心模块（匹配、候选生成等）
from grapher import Grapher                  # 图数据加载与管理类
from temporal_walk import store_edges        # 提取并存储图中边信息的函数
from rule_learning import rules_statistics   # 规则统计信息打印函数
from score_functions import score_12         # 自定义的规则评分函数

# 对应论文的关键步骤
# 论文 Algorithm 2 的流程：
# 输入：查询 q、规则 TR、时序知识图谱 G
# 按时间窗口 w 取出历史子图
# 对每个规则，找到 body groundings
# 提取候选实体并计算分数
# 用 Noisy‑OR 融合多规则分数
# 返回 top‑k 候选

# ===================== 环境变量配置 =====================
# 设置joblib的临时文件夹（解决多进程时临时文件路径问题）
os.environ['JOBLIB_TEMP_FOLDER'] = r'E:\temptemp'
# 确保临时文件夹存在，不存在则创建（exist_ok=True避免重复创建报错）
os.makedirs(r'E:\temptemp', exist_ok=True)

# 打印环境变量信息，用于调试验证
print(f"JOBLIB_TEMP_FOLDER = {os.environ.get('JOBLIB_TEMP_FOLDER', 'NOT SET')}")
print(f"TEMP = {os.environ.get('TEMP')}")
print(f"TMP = {os.environ.get('TMP')}")

# ===================== 命令行参数解析 =====================
# 创建参数解析器对象
parser = argparse.ArgumentParser(description="基于规则的时序图链接预测：规则应用与候选生成")

# 添加命令行参数，均设置了默认值和说明，方便使用
parser.add_argument("--dataset", "-d", default="", type=str,
                    help="数据集名称，用于拼接数据文件的路径（如../data/[dataset]/）")
parser.add_argument("--test_data", default="test", type=str,
                    help="使用的数据类型，可选test/valid，对应测试集/验证集")
parser.add_argument("--rules", "-r", default="", type=str,
                    help="预训练规则文件的路径/名称（存储在../output/[dataset]/下）")
parser.add_argument("--rule_lengths", "-l", default=1, type=int, nargs="+",
                    help="规则长度过滤条件，可以传入多个值（如 -l 2 3 表示只保留长度为2、3的规则）")
parser.add_argument("--window", "-w", default=-1, type=int,
                    help="时间窗口大小，-1表示使用全部历史数据，正数表示仅使用最近window时间内的边")
parser.add_argument("--top_k", default=20, type=int,
                    help="每个查询返回的候选答案数量上限（只保留分数最高的top_k个）")
parser.add_argument("--num_processes", "-p", default=1, type=int,
                    help="并行处理的进程数（根据CPU核心数调整，加速候选生成）")

# 解析参数并转换为字典格式（方便后续取值）
parsed = vars(parser.parse_args())

# ===================== 参数初始化 =====================
# 从解析后的参数中提取核心变量
dataset = parsed["dataset"]          # 数据集名称
rules_file = parsed["rules"]         # 预训练规则文件路径
window = parsed["window"]            # 时间窗口大小
top_k = parsed["top_k"]              # 候选答案数量上限
num_processes = parsed["num_processes"]  # 并行进程数
rule_lengths = parsed["rule_lengths"]    # 规则长度过滤列表

# 确保rule_lengths始终是列表类型（兼容单个整数输入的情况，如-l 2 → [2]）
rule_lengths = [rule_lengths] if (type(rule_lengths) == int) else rule_lengths

# 拼接数据/输出路径（按数据集分类，保证文件结构清晰）
dataset_dir = "../data/" + dataset + "/"  # 原始数据集存储路径
dir_path = "../output/" + dataset + "/"   # 规则/候选结果输出路径

# ===================== 数据加载 =====================
# 初始化图数据对象（加载数据集的节点、边、查询等信息）
data = Grapher(dataset_dir)

# 选择待处理的查询数据：测试集或验证集
# test_idx/valid_idx是Grapher类中预加载的查询索引列表
test_data = data.test_idx if (parsed["test_data"] == "test") else data.valid_idx

# 加载预训练的规则字典（JSON格式，键为关系ID，值为该关系对应的规则列表）
rules_dict = json.load(open(dir_path + rules_file))
# JSON读取的键默认是字符串，转换为整数（匹配Grapher中关系的整数ID）
rules_dict = {int(k): v for k, v in rules_dict.items()}

# ===================== 规则预处理 =====================
print("Rules statistics (before pruning):")
# 打印原始规则的统计信息（数量、长度分布、置信度分布等）
rules_statistics(rules_dict)

# 规则剪枝：过滤低质量规则，只保留符合条件的规则
# min_conf: 最小置信度阈值（0.01）；min_body_supp: 最小体支持度（2）；rule_lengths: 规则长度过滤
rules_dict = ra.filter_rules(
    rules_dict, min_conf=0.01, min_body_supp=2, rule_lengths=rule_lengths
)

print("Rules statistics (after pruning):")
# 打印剪枝后的规则统计信息（对比剪枝前后，验证过滤效果）
rules_statistics(rules_dict)

# 从训练数据中提取并存储边信息（按时间索引，方便后续按时间窗口快速查询）
learn_edges = store_edges(data.train_idx)

# ===================== 评分函数配置 =====================
# 选择规则评分函数（score_12为自定义的混合评分算法，结合置信度、支持度等）
score_func = score_12

# 评分函数的超参数列表（用于调优，格式为列表的列表，支持多组参数并行测试）
# 示例：[[0.1, 0.5], [0.2, 0.6]] 表示测试两组超参数
args = [[0.1, 0.5]]

# ===================== 核心函数：规则应用与候选生成 =====================
def apply_rules(i, num_queries):
    """
    对指定批次的测试查询应用规则，生成候选答案（支持多进程并行）
    每个进程处理一部分查询，避免单进程处理大量数据耗时过长

    Parameters:
        i (int): 当前进程编号（从0开始），用于分配数据批次
        num_queries (int): 每个进程处理的查询数量（基准值）

    Returns:
        all_candidates (list): 每个查询的候选答案字典列表
                              结构: [参数组合1的候选字典, 参数组合2的候选字典, ...]
                              候选字典结构: {查询索引j: {候选节点ID: 最终分数, ...}, ...}
        no_cands_counter (int): 该进程中没有找到候选答案的查询数量（用于统计覆盖率）
    """
    # 打印进程启动信息（方便调试，确认多进程是否正常启动）
    print(f"Start process {i} ...")

    # 初始化候选答案存储：每个参数组合对应一个空字典
    all_candidates = [dict() for _ in range(len(args))]
    # 初始化无候选答案的查询计数器
    no_cands_counter = 0

    # ===================== 分配查询批次 =====================
    # 计算剩余未分配的查询数量
    num_rest_queries = len(test_data) - (i + 1) * num_queries
    # 确定当前进程处理的查询索引范围（最后一个进程处理剩余所有查询）
    if num_rest_queries >= num_queries:
        # 前N-1个进程：处理固定数量的查询
        test_queries_idx = range(i * num_queries, (i + 1) * num_queries)
    else:
        # 最后一个进程：处理剩余所有查询（避免数据遗漏）
        test_queries_idx = range(i * num_queries, len(test_data))

    # ===================== 按时间窗口加载边 =====================
    # 初始时间戳：当前批次第一个查询的时间戳
    cur_ts = test_data[test_queries_idx[0]][3]
    # 获取该时间戳下的边集合（按时间窗口过滤）
    edges = ra.get_window_edges(data.all_idx, cur_ts, learn_edges, window)

    # 记录批次处理开始时间（用于打印进度）
    it_start = time.time()

    # ===================== 遍历查询，生成候选 =====================
    for j in test_queries_idx:
        # 获取当前查询（格式：[头节点, 关系, 尾节点, 时间戳]）
        test_query = test_data[j]
        # 初始化当前查询的候选字典（每个参数组合对应一个字典）
        cands_dict = [dict() for _ in range(len(args))]

        # 如果当前查询的时间戳变化，更新边集合（时间窗口内的边）
        if test_query[3] != cur_ts:
            cur_ts = test_query[3]
            edges = ra.get_window_edges(data.all_idx, cur_ts, learn_edges, window)

        # ===================== 应用规则生成候选 =====================
        # 检查当前查询的关系是否有预训练规则
        if test_query[1] in rules_dict:
            # 待处理的参数组合索引（用于提前终止已满足top_k的参数组合）
            dicts_idx = list(range(len(args)))

            # 遍历该关系下的所有规则
            for rule in rules_dict[test_query[1]]:
                # 匹配规则体中的关系，找到符合条件的边路径
                walk_edges = ra.match_body_relations(rule, edges, test_query[0])

                # 确保所有规则体部分都能匹配到边（避免空路径）
                if 0 not in [len(x) for x in walk_edges]:
                    # 根据匹配到的边生成完整的路径（walks）
                    rule_walks = ra.get_walks(rule, walk_edges)

                    # 如果规则有变量约束，过滤不符合约束的路径
                    if rule["var_constraints"]:
                        rule_walks = ra.check_var_constraints(
                            rule["var_constraints"], rule_walks
                        )

                    # 如果存在有效路径，生成候选答案并计算分数
                    if not rule_walks.empty:
                        # 核心函数：根据规则路径生成候选节点，并计算每个参数组合的分数
                        cands_dict = ra.get_candidates(
                            rule,                  # 当前规则
                            rule_walks,            # 规则匹配的路径
                            cur_ts,                # 当前查询的时间戳
                            cands_dict,            # 候选答案字典（累加更新）
                            score_func,            # 评分函数
                            args,                  # 评分函数超参数
                            dicts_idx              # 待处理的参数组合索引
                        )

                        # 对每个参数组合的候选结果排序，并提前终止已满足top_k的组合
                        for s in dicts_idx:
                            # 按分数降序排序候选答案
                            cands_dict[s] = {
                                x: sorted(cands_dict[s][x], reverse=True)
                                for x in cands_dict[s].keys()
                            }
                            # 按分数从高到低重新排列字典项
                            cands_dict[s] = dict(
                                sorted(
                                    cands_dict[s].items(),
                                    key=lambda item: item[1],
                                    reverse=True,
                                )
                            )
                            # 提取前top_k个分数（去重），判断是否已满足数量要求
                            top_k_scores = [v for _, v in cands_dict[s].items()][:top_k]
                            unique_scores = list(
                                scores for scores, _ in itertools.groupby(top_k_scores)
                            )
                            # 如果去重后的分数数量≥top_k，该参数组合无需继续处理
                            if len(unique_scores) >= top_k:
                                dicts_idx.remove(s)

                        # 如果所有参数组合都已满足top_k，提前终止规则遍历（优化性能）
                        if not dicts_idx:
                            break

            # ===================== 候选结果后处理 =====================
            # 如果当前查询生成了候选答案（至少参数组合0有结果）
            if cands_dict[0]:
                for s in range(len(args)):
                    # 计算Noisy-OR分数（融合多个规则的分数，避免单一规则偏差）
                    # 公式：1 - ∏(1 - 分数) → 多个规则支持的候选分数更高
                    scores = list(
                        map(
                            lambda x: 1 - np.prod(1 - np.array(x)),
                            cands_dict[s].values(),
                        )
                    )
                    # 将候选节点与Noisy-OR分数绑定
                    cands_scores = dict(zip(cands_dict[s].keys(), scores))
                    # 按分数降序排序，保留top_k个候选
                    noisy_or_cands = dict(
                        sorted(cands_scores.items(), key=lambda x: x[1], reverse=True)
                    )
                    # 将结果存入当前进程的候选字典（按查询索引j存储）
                    all_candidates[s][j] = noisy_or_cands
            else:
                # 当前查询无候选答案，计数器+1
                no_cands_counter += 1
                # 初始化空字典（保证结果结构统一）
                for s in range(len(args)):
                    all_candidates[s][j] = dict()

        else:
            # 当前查询的关系无预训练规则，计数器+1
            no_cands_counter += 1
            # 初始化空字典
            for s in range(len(args)):
                all_candidates[s][j] = dict()

        # ===================== 进度打印 =====================
        # 每处理100个查询，打印一次进度（方便监控程序运行状态）
        if not (j - test_queries_idx[0] + 1) % 100:
            it_end = time.time()
            it_time = round(it_end - it_start, 6)
            print(
                f"Process {i}: test samples finished: {j - test_queries_idx[0] + 1}/{len(test_queries_idx)}, {it_time} sec"
            )
            it_start = time.time()

    # 返回当前进程的候选结果和无候选计数器
    return all_candidates, no_cands_counter

# ===================== 多进程并行执行 =====================
# 记录总开始时间
start = time.time()

# 计算每个进程处理的基准查询数量（均分查询）
num_queries = len(test_data) // num_processes

# 多进程执行apply_rules函数：
# n_jobs: 进程数；delayed: 包装函数，传入进程编号和基准查询数
output = Parallel(n_jobs=num_processes)(
    delayed(apply_rules)(i, num_queries) for i in range(num_processes)
)

# 记录总结束时间
end = time.time()

# ===================== 结果合并 =====================
# 初始化最终候选结果（按参数组合分桶）
final_all_candidates = [dict() for _ in range(len(args))]

# 合并所有进程的候选结果（按参数组合逐个更新）
for s in range(len(args)):
    for i in range(num_processes):
        final_all_candidates[s].update(output[i][0][s])
        # 清空进程结果（释放内存）
        output[i][0][s].clear()

# 合并所有进程的无候选计数器
final_no_cands_counter = 0
for i in range(num_processes):
    final_no_cands_counter += output[i][1]

# ===================== 结果统计与保存 =====================
# 计算总运行时间
total_time = round(end - start, 6)
print(f"\nApplication finished in {total_time} seconds.")
print(f"No candidates found for {final_no_cands_counter} queries (coverage: {1 - final_no_cands_counter/len(test_data):.4f})")

# 保存每个参数组合的候选结果
for s in range(len(args)):
    # 生成结果文件名（包含评分函数和参数，便于区分不同实验）
    score_func_str = score_func.__name__ + str(args[s])
    score_func_str = score_func_str.replace(" ", "")  # 移除空格，避免路径错误

    # 保存候选结果到指定路径（JSON格式，便于后续评估）
    ra.save_candidates(
        rules_file,          # 源规则文件名（用于关联实验）
        dir_path,            # 结果保存路径
        final_all_candidates[s],  # 当前参数组合的候选结果
        rule_lengths,        # 规则长度过滤条件（用于文件名）
        window,              # 时间窗口大小（用于文件名）
        score_func_str       # 评分函数+参数（用于文件名）
    )