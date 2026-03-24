import json
import os.path
import pickle
from grapher import Grapher
import re
import multiprocessing as mp
import numpy as np
from collections import defaultdict
import argparse
import glob
from tqdm import tqdm
# 新增scipy稀疏矩阵依赖
import scipy.sparse as sp
from scipy.sparse import csr_matrix, lil_matrix, dok_matrix


# 直接在脚本中定义需要的函数
def print_msg(msg):
    msg = "## {} ##".format(msg)
    length = len(msg)
    msg = "\n{}\n".format(msg)
    print(length * "#" + msg + length * "#")


def parse_rdf(rdf):
    """
    解析RDF格式的三元组

    参数:
        rdf (str): RDF格式的三元组，格式为 "头实体 关系 尾实体"

    返回:
        tuple: (头实体, 关系, 尾实体)
    """
    parts = rdf.strip().split()
    if len(parts) < 3:
        return None, None, None

    # 处理关系可能包含空格的情况
    h = parts[0]
    t = parts[-1]
    r = ' '.join(parts[1:-1])

    return h, r, t


def ill_rank(pred, gt, ent2idx, q_h, q_t, q_r):
    pred_ranks = np.argsort(pred)[::-1]
    truth = gt[(q_h, q_r)]
    truth = [t for t in truth if t != ent2idx[q_t]]
    filtered_ranks = []
    for i in range(len(pred_ranks)):
        idx = pred_ranks[i]
        if idx not in truth and pred[idx] > pred[ent2idx[q_t]]:
            filtered_ranks.append(idx)

    rank = len(filtered_ranks) + 1
    return rank


def harsh_rank(pred, gt, ent2idx, q_h, q_t, q_r):
    pred_ranks = np.argsort(pred)[::-1]
    truth = gt[(q_h, q_r)]
    truth = [t for t in truth]
    filtered_ranks = []
    for i in range(len(pred_ranks)):
        idx = pred_ranks[i]
        if idx not in truth and pred[idx] >= pred[ent2idx[q_t]]:
            filtered_ranks.append(idx)

    rank = len(filtered_ranks) + 1
    return rank


def balance_rank(pred, gt, ent2idx, q_h, q_t, q_r):
    if pred[ent2idx[q_t]] != 0:
        pred_ranks = np.argsort(pred)[::-1]

        truth = gt[(q_h, q_r)]
        truth = [t for t in truth if t != ent2idx[q_t]]

        filtered_ranks = []
        for i in range(len(pred_ranks)):
            idx = pred_ranks[i]
            if idx not in truth:
                filtered_ranks.append(idx)

        rank = filtered_ranks.index(ent2idx[q_t]) + 1
    else:
        truth = gt[(q_h, q_r)]

        filtered_pred = []

        for i in range(len(pred)):
            if i not in truth:
                filtered_pred.append(pred[i])
        n_non_zero = np.count_nonzero(filtered_pred)
        rank = n_non_zero + 1
    return rank


def random_rank(pred, gt, ent2idx, q_h, q_t, q_r):
    pred_ranks = np.argsort(pred)[::-1]
    truth = gt[(q_h, q_r)]
    truth = [t for t in truth if t != ent2idx[q_t]]
    truth.append(ent2idx[q_t])
    filtered_ranks = []
    for i in range(len(pred_ranks)):
        idx = pred_ranks[i]
        if idx not in truth and pred[idx] >= pred[ent2idx[q_t]]:
            if (pred[idx] == pred[ent2idx[q_t]]) and (np.random.uniform() < 0.5):
                filtered_ranks.append(idx)
            else:
                filtered_ranks.append(idx)

    rank = len(filtered_ranks) + 1
    return rank


import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

head2mrr = defaultdict(list)
head2hit_5 = defaultdict(list)
head2hit_10 = defaultdict(list)
head2hit_1 = defaultdict(list)


def sortSparseMatrix(m, r, rev=True, only_indices=False):
    """ Sort a row in matrix row and return column index
    """
    d = m.getrow(r)
    s = zip(d.indices, d.data)
    sorted_s = sorted(s, key=lambda v: v[1], reverse=rev)
    if only_indices:
        res = [element[0] for element in sorted_s]
    else:
        res = sorted_s
    return res


def remove_var(r):
    """R1(A, B), R2(B, C) --> R1, R2"""
    r = re.sub(r"\(\D?, \D?\)", "", r)
    return r


def parse_rule(r):
    """parse a rule into body and head"""
    head, body = r.split(" <-- ")
    head_list = head.split("\t")
    score = [float(s) for s in head_list[:-1]]
    head = head_list[-1]
    body = body.split(", ")
    return score, head, body


def load_rules(rule_path, all_rules, all_heads):
    for input_filepath in glob.glob(os.path.join(rule_path, "*.txt")):
        with open(input_filepath, 'r') as f:
            rules = f.readlines()
            for i_, rule in enumerate(rules):
                score, head, body = parse_rule(rule.strip('\n'))
                # Skip zero support rules
                if score[0] == 0.0:
                    continue
                if head not in all_rules:
                    all_rules[head] = []
                all_rules[head].append((head, body, score))

                if head not in all_heads:
                    all_heads.append(head)


def idx_to_rdf(idx, id2entity, id2relation):
    """将索引格式的四元组转换为RDF格式的三元组字符串"""
    h_idx, r_idx, t_idx, _ = idx
    h = id2entity[h_idx]
    r = id2relation[r_idx]
    t = id2entity[t_idx]
    return f"{h} {r} {t}"


def get_gt(grapher):
    # entity
    idx2ent, ent2idx = grapher.id2entity, grapher.entity2id
    # relation
    idx2rel = grapher.id2relation
    gt = defaultdict(list)

    # 从所有索引数据中构建ground truth
    all_idx = grapher.all_idx
    for idx in all_idx:
        try:
            rdf = idx_to_rdf(idx, idx2ent, idx2rel)
            h, r, t = parse_rdf(rdf)
            if h and r and t and h in ent2idx and t in ent2idx:
                gt[(h, r)].append(ent2idx[t])
        except Exception as e:
            # 跳过无效的三元组
            continue
    return gt


def kg_completion(rules, grapher, args):
    """
    Input a set of rules
    Complete Querys from test_rdf based on rules and fact_rdf
    """
    # entity
    idx2ent, ent2idx = grapher.id2entity, grapher.entity2id
    e_num = len(idx2ent)  # 实体总数
    # relation
    idx2rel = grapher.id2relation
    rel2idx = grapher.relation2id
    # groud truth
    gt = get_gt(grapher)

    # 从索引数据构建RDF格式的三元组
    def build_rdf_from_idx(idx_array):
        rdf_list = []
        for idx in idx_array:
            rdf = idx_to_rdf(idx, idx2ent, idx2rel)
            rdf_list.append(rdf)
        return rdf_list

    # 构建训练、验证和测试数据的RDF格式
    train_rdf = build_rdf_from_idx(grapher.train_idx)
    valid_rdf = build_rdf_from_idx(grapher.valid_idx)
    test_rdf = build_rdf_from_idx(grapher.test_idx)

    # ========== 核心优化1：用稀疏矩阵构建关系邻接矩阵 ==========
    def construct_rmat_sparse(idx2rel, ent2idx, rdf_list):
        """
        从RDF格式的三元组列表构建关系邻接稀疏矩阵

        参数:
            idx2rel (dict): 索引到关系的映射
            ent2idx (dict): 实体到索引的映射
            rdf_list (list): RDF格式的三元组列表

        返回:
            dict: 关系到CSR稀疏矩阵的映射，格式为 {关系: csr_matrix(e_num x e_num)}
        """
        r2mat = {}

        # 初始化每个关系的稀疏矩阵（DOK格式：高效赋值）
        for r_id in idx2rel:
            r = idx2rel[r_id]
            # DOK (Dictionary of Keys) 适合稀疏矩阵的快速赋值
            r2mat[r] = dok_matrix((e_num, e_num), dtype=np.float32)

        # 填充邻接矩阵
        for rdf in rdf_list:
            h, r, t = parse_rdf(rdf)
            if h in ent2idx and t in ent2idx:
                h_idx = ent2idx[h]
                t_idx = ent2idx[t]
                r2mat[r][h_idx, t_idx] = 1.0

        # 转换为CSR格式（高效矩阵乘法）
        for r in r2mat:
            r2mat[r] = r2mat[r].tocsr()

        return r2mat

    # 构建关系邻接稀疏矩阵（替代原字典实现）
    r2mat = construct_rmat_sparse(idx2rel, ent2idx, train_rdf + valid_rdf)

    # Test rdf grouped by head
    test = {}
    for rdf in test_rdf:
        query = parse_rdf(rdf)
        q_h, q_r, q_t = query
        if q_r not in test:
            test[q_r] = [query]
        else:
            test[q_r].append(query)

    mrr, hits_1, hits_5, hits_10 = [], [], [], []
    output_pred = {}

    score_name_to_id = {"support": 0, "coverage": 1, "confidence": 2, "pca_confidence": 3, "none": -1}
    score_id = score_name_to_id[args.score]
    threshold_score_id = score_name_to_id[args.threshold_score]

    for head in tqdm(test.keys()):
        if not args.rank_only:
            output_pred[head] = {}
        if head not in rules:
            continue
        _rules = rules[head]

        if not args.rank_only:
            # ========== 核心优化2：用稀疏矩阵存储路径计数 ==========
            # 初始化路径计数矩阵（LIL格式：高效逐行更新）
            path_count = lil_matrix((e_num, e_num), dtype=np.float32)

            # 规则筛选（保留原有逻辑）
            if score_id != -1:
                sorted_rules = sorted(_rules, key=lambda x: x[2][score_id], reverse=True)
                if args.top > 0:
                    _rules = sorted_rules[:args.top]
                if args.threshold > 0:
                    _rules = [rule for rule in sorted_rules if rule[2][threshold_score_id] > args.threshold]

            for rule in _rules:
                head, body, score = rule
                rule_weight = score[score_id] if score_id != -1 else 1.0

                if not body:
                    continue

                # 初始化第一个关系的路径矩阵
                first_rel = body[0]
                if first_rel not in r2mat:
                    continue
                current_matrix = r2mat[first_rel].copy()  # CSR矩阵

                # ========== 核心优化3：矩阵乘法替代嵌套循环 ==========
                # 计算规则体的路径（关系链：rel1 × rel2 × ...）
                for b_rel in body[1:]:
                    if b_rel not in r2mat:
                        current_matrix = None
                        break
                    # 稀疏矩阵乘法：替代原字典嵌套循环（O(n^3) → O(n^2)）
                    current_matrix = current_matrix @ r2mat[b_rel]
                    if current_matrix.nnz == 0:  # 无可达路径，提前终止
                        break

                # 累加当前规则的路径得分到总矩阵
                if current_matrix is not None and current_matrix.nnz > 0:
                    path_count += current_matrix * rule_weight

            # ========== 预测阶段优化：稀疏矩阵行提取 ==========
            for q_i, query_rdf in enumerate(test[head]):
                q_h, q_r, q_t = query_rdf
                if args.debug:
                    print("{}\t{}\t{}".format(q_h, q_r, q_t))

                if not args.rank_only:
                    # 提取当前头实体的预测行（稀疏矩阵 → 密集数组）
                    q_h_idx = ent2idx.get(q_h, -1)
                    pred = np.zeros(e_num, dtype=np.float32)
                    if q_h_idx != -1 and path_count.nnz > 0:
                        # 稀疏矩阵行提取：O(1) 时间复杂度
                        pred = path_count[q_h_idx].toarray().flatten()
                    output_pred[head][(q_h, q_r, q_t)] = pred
                else:
                    pred = output_pred[head][(q_h, q_r, q_t)]

                # 排名计算（保留原有逻辑）
                if args.rank_mode == 'ill':
                    rank = ill_rank(pred, gt, ent2idx, q_h, q_t, q_r)
                elif args.rank_mode == 'harsh':
                    rank = harsh_rank(pred, gt, ent2idx, q_h, q_t, q_r)
                elif args.rank_mode == 'balance':
                    rank = balance_rank(pred, gt, ent2idx, q_h, q_t, q_r)
                else:
                    rank = random_rank(pred, gt, ent2idx, q_h, q_t, q_r)

                # 指标统计
                mrr.append(1.0 / rank)
                head2mrr[q_r].append(1.0 / rank)
                hits_1.append(1 if rank <= 1 else 0)
                hits_5.append(1 if rank <= 5 else 0)
                hits_10.append(1 if rank <= 10 else 0)
                head2hit_1[q_r].append(1 if rank <= 1 else 0)
                head2hit_5[q_r].append(1 if rank <= 5 else 0)
                head2hit_10[q_r].append(1 if rank <= 10 else 0)

                if args.debug:
                    print("rank at {}: {}".format(q_i, rank))

    return mrr, hits_1, hits_5, hits_10


def load_results(head):
    input_file_name = os.path.join(args.output_path, args.dataset, args.p, 'output_predict.pkl')
    with open(input_file_name, 'rb') as f:
        pred_results_dict = pickle.load(f)
    return pred_results_dict[head]


def feq(relation, fact_rdf):
    count = 0
    for rdf in fact_rdf:
        h, r, t = parse_rdf(rdf)
        if r == relation:
            count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default='ranked_rules')
    parser.add_argument("--dataset", default="icews14")
    parser.add_argument('--output_path', default='pred_results', type=str, help='path to save pred results')
    parser.add_argument("-p", default='ranked_rules/icews14/gpt-3.5-turbo-top-50-f-3-l-2/none/all', help="rule path")
    parser.add_argument("--eval_mode", choices=['all', "test", 'fact'], default="all",
                        help="evaluate on all or only test set")
    parser.add_argument('--cpu_num', type=int, default=mp.cpu_count() // 2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--top", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--rank_mode", choices=['ill', 'harsh', 'balance'], default='harsh')
    parser.add_argument("--rank_only", action="store_true")
    parser.add_argument("--threshold_score", choices=['pca_confidence', 'confidence', 'coverage', 'support'],
                        default='support')
    parser.add_argument("--score", choices=['pca_confidence', 'confidence', 'coverage', 'support', 'none'],
                        default='pca_confidence')
    parser.add_argument("--data_path", default="../data", help="path to data directory")
    args = parser.parse_args()
    grapher = Grapher(dataset_dir=os.path.join(args.data_path, args.dataset, ''))
    all_rules = {}
    all_rule_heads = []

    print("Rule path is {}".format(args.p))
    load_rules(args.p, all_rules, all_rule_heads)

    test_mrr, test_hits_1, test_hits_5, test_hits_10 = kg_completion(all_rules, grapher, args)

    # 打印规则和测试数据的统计信息
    print(f"Loaded {len(all_rules)} rules with {len(all_rule_heads)} unique heads")


    # 构建测试RDF以便在debug模式下使用
    def build_rdf_from_idx(idx_array):
        rdf_list = []
        for idx in idx_array:
            h_idx, r_idx, t_idx, _ = idx
            h = grapher.id2entity[h_idx]
            r = grapher.id2relation[r_idx]
            t = grapher.id2entity[t_idx]
            rdf_list.append(f"{h} {r} {t}")
        return rdf_list


    test_rdf = build_rdf_from_idx(grapher.test_idx)
    train_rdf = build_rdf_from_idx(grapher.train_idx)
    valid_rdf = build_rdf_from_idx(grapher.valid_idx)

    print(f"Test data size: {len(test_rdf)}")
    print(f"Train data size: {len(train_rdf)}")
    print(f"Valid data size: {len(valid_rdf)}")

    if args.debug:
        print_msg("distribution of test query")
        for head in all_rule_heads:
            count = feq(head, test_rdf)
            print("Head: {} Count: {}".format(head, count))

        print_msg("distribution of train query")
        for head in all_rule_heads:
            count = feq(head, train_rdf + valid_rdf)
            print("Head: {} Count: {}".format(head, count))

        all_results = {"mrr": [], "hits_1": [], "hits_5": [], "hits_10": []}
        print_msg("Stat on head and hit@1")
        for head, hits in head2hit_1.items():
            print(head, np.mean(hits))
            all_results["hits_1"].append(np.mean(hits))

        print_msg("Stat on head and hit@5")
        for head, hits in head2hit_5.items():
            print(head, np.mean(hits))
            all_results["hits_5"].append(np.mean(hits))

        print_msg("Stat on head and hit@10")
        for head, hits in head2hit_10.items():
            print(head, np.mean(hits))
            all_results["hits_10"].append(np.mean(hits))

        print_msg("Stat on head and mrr")
        for head, mrr in head2mrr.items():
            print(head, np.mean(mrr))
            all_results["mrr"].append(np.mean(mrr))

    dataset_name = args.dataset + ": " + args.p

    # 处理空结果的情况
    if not test_mrr:
        print("No results to display - no rules matched test queries")
        result_dict = {"mrr": 0.0, "hits_1": 0.0, "hits_5": 0.0, "hits_10": 0.0}
    else:
        result_dict = {"mrr": np.mean(test_mrr), "hits_1": np.mean(test_hits_1), "hits_5": np.mean(test_hits_5),
                       "hits_10": np.mean(test_hits_10)}

    output_dir = args.p.replace(args.input_folder, args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "result_dict.json"), 'w') as f:
        json.dump(result_dict, f)

    # 打印结果到控制台
    print("MRR\tHit@1\tHit@5\tHit@10")
    print(
        f"{result_dict['mrr']:.4f}\t{result_dict['hits_1']:.4f}\t{result_dict['hits_5']:.4f}\t{result_dict['hits_10']:.4f}")