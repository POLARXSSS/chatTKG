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
import scipy.sparse as sp
from scipy.sparse import csr_matrix, lil_matrix, dok_matrix
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 线程锁（保证全局统计安全）
lock = threading.Lock()

def print_msg(msg):
    msg = "## {} ##".format(msg)
    length = len(msg)
    msg = "\n{}\n".format(msg)
    print(length * "#" + msg + length * "#")

def parse_rdf(rdf):
    parts = rdf.strip().split()
    if len(parts) < 3:
        return None, None, None
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
    d = m.getrow(r)
    s = zip(d.indices, d.data)
    sorted_s = sorted(s, key=lambda v: v[1], reverse=rev)
    if only_indices:
        res = [element[0] for element in sorted_s]
    else:
        res = sorted_s
    return res

def remove_var(r):
    r = re.sub(r"\(\D?, \D?\)", "", r)
    return r

def parse_rule(r):
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
                if score[0] == 0.0:
                    continue
                if head not in all_rules:
                    all_rules[head] = []
                all_rules[head].append((head, body, score))
                if head not in all_heads:
                    all_heads.append(head)

def idx_to_rdf(idx, id2entity, id2relation):
    h_idx, r_idx, t_idx, _ = idx
    h = id2entity[h_idx]
    r = id2relation[r_idx]
    t = id2entity[t_idx]
    return f"{h} {r} {t}"

def get_gt(grapher):
    idx2ent, ent2idx = grapher.id2entity, grapher.entity2id
    idx2rel = grapher.id2relation
    gt = defaultdict(list)
    all_idx = grapher.all_idx
    for idx in all_idx:
        try:
            rdf = idx_to_rdf(idx, idx2ent, idx2rel)
            h, r, t = parse_rdf(rdf)
            if h and r and t and h in ent2idx and t in ent2idx:
                gt[(h, r)].append(ent2idx[t])
        except Exception:
            continue
    return gt

# ==================== 多线程任务：处理单个关系头 ====================
def process_head(
    head, rules, test_queries, r2mat, ent2idx, e_num, gt, args
):
    mrr = []
    h1 = []
    h5 = []
    h10 = []

    score_name_to_id = {"support": 0, "coverage": 1, "confidence": 2, "pca_confidence": 3, "none": -1}
    score_id = score_name_to_id[args.score]
    threshold_score_id = score_name_to_id[args.threshold_score]

    path_count = lil_matrix((e_num, e_num), dtype=np.float32)
    _rules = rules

    if score_id != -1:
        sorted_rules = sorted(_rules, key=lambda x: x[2][score_id], reverse=True)
        if args.top > 0:
            _rules = sorted_rules[:args.top]
        if args.threshold > 0:
            _rules = [rule for rule in sorted_rules if rule[2][threshold_score_id] > args.threshold]

    for rule in _rules:
        head, body, score = rule
        rule_weight = score[score_id] if score_id != -1 else 1.0
        if not body: continue

        first_rel = body[0]
        if first_rel not in r2mat: continue
        current_matrix = r2mat[first_rel].copy()

        for b_rel in body[1:]:
            if b_rel not in r2mat:
                current_matrix = None
                break
            current_matrix = current_matrix @ r2mat[b_rel]
            if current_matrix.nnz == 0:
                break

        if current_matrix is not None and current_matrix.nnz > 0:
            path_count += current_matrix * rule_weight

    for q in test_queries:
        q_h, q_r, q_t = q
        try:
            q_h_idx = ent2idx[q_h]
            pred = path_count[q_h_idx].toarray().flatten()

            if args.rank_mode == 'ill':
                rank = ill_rank(pred, gt, ent2idx, q_h, q_t, q_r)
            elif args.rank_mode == 'harsh':
                rank = harsh_rank(pred, gt, ent2idx, q_h, q_t, q_r)
            elif args.rank_mode == 'balance':
                rank = balance_rank(pred, gt, ent2idx, q_h, q_t, q_r)
            else:
                rank = random_rank(pred, gt, ent2idx, q_h, q_t, q_r)

            mrr.append(1.0 / rank)
            h1.append(1 if rank <= 1 else 0)
            h5.append(1 if rank <= 5 else 0)
            h10.append(1 if rank <= 10 else 0)

            with lock:
                head2mrr[q_r].append(1.0 / rank)
                head2hit_1[q_r].append(1 if rank <= 1 else 0)
                head2hit_5[q_r].append(1 if rank <= 5 else 0)
                head2hit_10[q_r].append(1 if rank <= 10 else 0)
        except:
            continue

    return mrr, h1, h5, h10

def kg_completion(rules, grapher, args):
    idx2ent, ent2idx = grapher.id2entity, grapher.entity2id
    e_num = len(idx2ent)
    idx2rel = grapher.id2relation
    gt = get_gt(grapher)

    def build_rdf_from_idx(idx_array):
        rdf_list = []
        for idx in idx_array:
            rdf_list.append(idx_to_rdf(idx, idx2ent, idx2rel))
        return rdf_list

    train_rdf = build_rdf_from_idx(grapher.train_idx)
    valid_rdf = build_rdf_from_idx(grapher.valid_idx)
    test_rdf = build_rdf_from_idx(grapher.test_idx)

    def construct_rmat_sparse(idx2rel, ent2idx, rdf_list):
        r2mat = {}
        for r_id in idx2rel:
            r = idx2rel[r_id]
            r2mat[r] = dok_matrix((e_num, e_num), dtype=np.float32)
        for rdf in rdf_list:
            h, r, t = parse_rdf(rdf)
            if h in ent2idx and t in ent2idx:
                h_idx = ent2idx[h]
                t_idx = ent2idx[t]
                r2mat[r][h_idx, t_idx] = 1.0
        for r in r2mat:
            r2mat[r] = r2mat[r].tocsr()
        return r2mat

    r2mat = construct_rmat_sparse(idx2rel, ent2idx, train_rdf + valid_rdf)

    test = defaultdict(list)
    for rdf in test_rdf:
        q = parse_rdf(rdf)
        if q[0]:
            test[q[1]].append(q)

    all_mrr = []
    all_h1 = []
    all_h5 = []
    all_h10 = []

    # ==================== 多线程并行执行 ====================
    valid_heads = [r for r in test if r in rules]
    max_workers = min(16, len(valid_heads))
    print(f"\n🚀 启动多线程，线程数 = {max_workers}\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_head,
                r_head, rules[r_head], test[r_head],
                r2mat, ent2idx, e_num, gt, args
            )
            for r_head in valid_heads
        ]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            m, h1, h5, h10 = fut.result()
            all_mrr.extend(m)
            all_h1.extend(h1)
            all_h5.extend(h5)
            all_h10.extend(h10)

    return all_mrr, all_h1, all_h5, all_h10

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
    parser.add_argument('--output_path', default='pred_results', type=str)
    parser.add_argument("-p", default='ranked_rules/icews14/gpt-3.5-turbo-top-50-f-3-l-2/none/all')
    parser.add_argument("--eval_mode", choices=['all', "test", 'fact'], default="all")
    parser.add_argument('--cpu_num', type=int, default=mp.cpu_count() // 2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--top", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--rank_mode", choices=['ill', 'harsh', 'balance'], default='harsh')
    parser.add_argument("--rank_only", action="store_true")
    parser.add_argument("--threshold_score", default='support')
    parser.add_argument("--score", default='pca_confidence')
    parser.add_argument("--data_path", default="../data")
    args = parser.parse_args()

    grapher = Grapher(dataset_dir=os.path.join(args.data_path, args.dataset, ''))
    all_rules = {}
    all_rule_heads = []
    load_rules(args.p, all_rules, all_rule_heads)

    test_mrr, test_hits_1, test_hits_5, test_hits_10 = kg_completion(all_rules, grapher, args)

    print(f"Loaded {len(all_rules)} rules")

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
    print(f"Test queries: {len(test_rdf)}")

    if not test_mrr:
        result_dict = {"mrr": 0.0, "hits_1": 0.0, "hits_5": 0.0, "hits_10": 0.0}
    else:
        result_dict = {
            "mrr": np.mean(test_mrr),
            "hits_1": np.mean(test_hits_1),
            "hits_5": np.mean(test_hits_5),
            "hits_10": np.mean(test_hits_10)
        }

    output_dir = args.p.replace(args.input_folder, args.output_path)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "result_dict.json"), 'w') as f:
        json.dump(result_dict, f)

    print("\n✅ 完成！")
    print("MRR\tHit@1\tHit@5\tHit@10")
    print(f"{result_dict['mrr']:.4f}\t{result_dict['hits_1']:.4f}\t{result_dict['hits_5']:.4f}\t{result_dict['hits_10']:.4f}")