import os
import argparse
import glob
from tqdm import tqdm
from grapher import Grapher
import json
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, dok_matrix
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 线程锁（保证文件写入不乱）
file_lock = threading.Lock()

def parse_rule(r):
    head, body = r.split(" <-- ")
    body = body.split(", ")
    return head, body

def construct_fact_dict_from_idx(data_idx, id2relation, id2entity):
    fact_dict = {}
    for quad in data_idx:
        h_id, r_id, t_id, ts_id = quad
        h = id2entity[h_id]
        r = id2relation[r_id]
        t = id2entity[t_id]
        if r not in fact_dict:
            fact_dict[r] = []
        fact_dict[r].append(f"{h}\t{r}\t{t}")
    return fact_dict

# ==================== scipy稀疏矩阵构建 ====================
def construct_rmat_sparse(idx2rel, ent2idx, data_idx, e_num):
    r2mat = {}
    for r_id in idx2rel:
        r = idx2rel[r_id]
        r2mat[r] = dok_matrix((e_num, e_num), dtype=np.float32)

    for quad in data_idx:
        h_id, r_id, t_id, _ = quad
        r = idx2rel[r_id]
        r2mat[r][h_id, t_id] = 1.0

    for r in r2mat:
        r2mat[r] = r2mat[r].tocsr()
    return r2mat

def load_rules(rule_path):
    all_rules = {}
    for input_filepath in glob.glob(os.path.join(rule_path, "*_cleaned_rules.txt")):
        with open(input_filepath, 'r') as f:
            rules = f.readlines()
            for rule in rules:
                head, body = parse_rule(rule.strip('\n'))
                if head not in all_rules:
                    all_rules[head] = []
                all_rules[head].append(body)
    return all_rules

def parse_rdf(fact_str):
    parts = fact_str.split('\t')
    return parts[0], parts[1], parts[2]

# ==================== 稀疏矩阵规则评估 ====================
def evaluate_rule(rule_body, rule_head, fact_dict, r2mat, e_num, ent2idx):
    score = {}
    r_size = len(fact_dict[rule_head])
    support = 0
    pca_negative = 0

    for b_rel in rule_body:
        if b_rel not in r2mat:
            return {"support": 0., "coverage": 0., "confidence": 0., "pca_confidence": 0.}

    # 矩阵乘法路径计算（原来的三重循环 → 1行）
    path_matrix = r2mat[rule_body[0]].copy()
    for b_rel in rule_body[1:]:
        path_matrix = path_matrix @ r2mat[b_rel]
        if path_matrix.nnz == 0:
            break

    if path_matrix.nnz == 0:
        return {"support": 0., "coverage": 0., "confidence": 0., "pca_confidence": 0.}

    visted_head = set()
    for fact in fact_dict[rule_head]:
        h, _, t = parse_rdf(fact)
        h_idx = ent2idx[h]
        t_idx = ent2idx[t]
        if path_matrix[h_idx, t_idx] > 0:
            support += 1
        visted_head.add(h)

    if support == 0:
        return {"support": 0., "coverage": 0., "confidence": 0., "pca_confidence": 0.}

    for head in visted_head:
        h_idx = ent2idx[head]
        pca_negative += path_matrix[h_idx].nnz

    all_path = path_matrix.nnz

    score['support'] = support
    score['coverage'] = support / r_size
    score['confidence'] = support / all_path if all_path > 0 else 0
    score['pca_confidence'] = support / pca_negative if pca_negative > 0 else 0

    return score

# ==================== 单规则头处理函数（多线程用） ====================
def process_single_head(
    r_head,
    rule,
    fact_dict,
    r2mat,
    e_num,
    ent2idx,
    output_folder,
    debug
):
    try:
        if r_head not in fact_dict:
            return None

        rule_statics = {"support": 0., "coverage": 0., "confidence": 0., "pca_confidence": 0.}
        file_name = r_head.replace('/', '-')
        output_rule_file = os.path.join(output_folder, f"{file_name}_ranked_rules.txt")

        # 加锁写入，避免多线程混乱
        with file_lock:
            with open(output_rule_file, 'w', encoding='utf-8') as f:
                for rule_body in rule[r_head]:
                    score = evaluate_rule(rule_body, r_head, fact_dict, r2mat, e_num, ent2idx)
                    rule_str = f"{r_head} <-- {', '.join(rule_body)}"
                    f.write(f"{score['support']}\t{score['coverage']}\t{score['confidence']}\t{score['pca_confidence']}\t{rule_str}\n")
                    for k in score:
                        rule_statics[k] += score[k]

        # 统计平均
        for k in rule_statics:
            rule_statics[k] /= len(rule[r_head])

        output_stat_file = os.path.join(output_folder, f"{file_name}_rule_statics.json")
        with file_lock:
            with open(output_stat_file, 'w', encoding='utf-8') as f:
                json.dump(rule_statics, f, indent=2, ensure_ascii=False)

        return rule_statics

    except Exception as e:
        print(f"[错误] {r_head}: {e}")
        return None

# ==================== 主函数 ====================
def main(args):
    rule = load_rules(args.p)
    data_path = os.path.join(args.data_path, args.dataset) + '/'
    dataset = Grapher(data_path)

    all_idx = dataset.all_idx
    if args.eval_mode == "all":
        test_idx = all_idx
    elif args.eval_mode == "test":
        test_idx = dataset.test_idx
    else:
        test_idx = dataset.train_idx

    fact_dict = construct_fact_dict_from_idx(test_idx, dataset.id2relation, dataset.id2entity)
    ent2idx = dataset.entity2id
    idx2rel = dataset.id2relation
    e_num = len(ent2idx)

    # 构建稀疏矩阵
    r2mat = construct_rmat_sparse(idx2rel, ent2idx, all_idx, e_num)

    output_folder = args.p.replace(args.input_path, args.output_path)
    output_folder = os.path.join(output_folder, args.eval_mode)
    os.makedirs(output_folder, exist_ok=True)

    data_statics = {"support": 0., "coverage": 0., "confidence": 0., "pca_confidence": 0.}
    valid_heads = [h for h in rule if h in fact_dict]

    # ==================== 多线程并行评估 ====================
    max_workers = min(16, len(valid_heads))  # 自动设置线程数
    print(f"\n🚀 启动多线程评估，线程数 = {max_workers}\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_single_head,
                r_head, rule, fact_dict, r2mat, e_num, ent2idx,
                output_folder, args.debug
            )
            for r_head in valid_heads
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating rules"):
            res = future.result()
            if res is not None:
                for k in data_statics:
                    data_statics[k] += res[k]

    # 全局统计
    total_heads = len(valid_heads)
    if total_heads > 0:
        for k in data_statics:
            data_statics[k] /= total_heads

    # 保存全局统计
    with open(os.path.join(output_folder, "data_statics.json"), 'w', encoding='utf-8') as f:
        json.dump(data_statics, f, indent=2, ensure_ascii=False)

    print("\n✅ 完成！")
    print("support\tcoverage\tconfidence\tpca_confidence")
    print(f"{data_statics['support']:.4f}\t{data_statics['coverage']:.4f}\t{data_statics['confidence']:.4f}\t{data_statics['pca_confidence']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="icews14")
    parser.add_argument("--data_path", default="../data", type=str)
    parser.add_argument("-p", default="clean_rules/icews14/gpt-3.5-turbo-top-50-f-3-l-2/none")
    parser.add_argument("--eval_mode", choices=['all', "test", 'fact'], default="all")
    parser.add_argument("--input_path", default="clean_rules", type=str)
    parser.add_argument("--output_path", default="ranked_rules", type=str)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args)