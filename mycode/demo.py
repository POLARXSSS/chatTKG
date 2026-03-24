import argparse
import json
import math
import os
import re
from tqdm import tqdm
from functools import partial
import random
from llms import llm_client
from multiprocessing.pool import ThreadPool

RDict_PATH = "data/icews14/relation2id.json"

# 模拟缺失的依赖模块（避免运行报错）
class Dataset:
    def __init__(self, data_root, inv=True):
        self.data_root = data_root

    def get_relation_dict(self):
        # 模拟返回关系字典（从TXT中提取的关系名）
        class RDict:

            def __init__(self, json_path=RDict_PATH):
                # 核心逻辑：打开并加载 json 文件
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        self.rel2idx = json.load(f)
                except FileNotFoundError:
                    print(f"错误：未找到文件 {json_path}")
                    self.rel2idx = {}
                except json.JSONDecodeError:
                    print(f"错误：{json_path} 文件格式非有效 JSON")
                    self.rel2idx = {}

        return RDict()


def get_registed_model(model_name):
    # 模拟返回LLM类
    class MockLLM:
        @staticmethod
        def add_args(parser):
            pass

    return MockLLM


def check_prompt_length(base_prompt, samples):
    """模拟检查prompt长度，直接返回拼接的样本"""
    return "\n".join(samples)


# 做sample，对于数据量>10的,依照置信度sample一半，向上取整
def read_paths(path):
    """
    读取TXT格式的规则文件，提取规则头和规则体，并按置信度抽样
    输入：TXT文件路径
    输出：[{"head": 规则头关系名, "paths": [规则字符串1, 规则字符串2...]}]
    抽样规则：
    - 数据量>10的组：按置信度加权抽样，数量为一半（向上取整），置信度越高被抽中概率越大
    - 数据量<=10的组：不抽样，保留全部
    """
    results = []
    # 用于存储提取的规则（按规则头分组），每个元素是 {"confidence": 置信度, "rule": 规则字符串}
    rule_dict = {}

    # 正则表达式：匹配 "置信度 数字 数字 规则头 ← 规则体" 部分
    # group(1) = 置信度（如1.000000），group(2) = 规则部分（如Accede_to_demands... <- ...）
    rule_pattern = re.compile(r'(\d+\.?\d*)\s+\d+\s+\d+\s+(.*? <- .*)')

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue

            # 提取置信度和规则部分
            match = rule_pattern.match(line)
            if match:
                # 1. 提取并转换置信度（字符串转浮点数）
                confidence = float(match.group(1).strip())
                # 2. 提取完整规则字符串
                full_rule = match.group(2).strip()

                # 拆分规则头和规则体，提取规则头关系名
                head_part, body_part = full_rule.split(" <- ")
                head_rel = re.match(r'^([^(]+)', head_part).group(1).strip()

                # 按规则头分组存储（包含置信度和规则字符串）
                if head_rel not in rule_dict:
                    rule_dict[head_rel] = []
                rule_dict[head_rel].append({
                    "confidence": confidence,
                    "rule": full_rule
                })

    # 按规则头处理抽样逻辑
    for head_rel, rules in rule_dict.items():
        n = len(rules)
        if n > 10:
            # 计算抽样数量：一半，向上取整（如11→6，12→6，13→7）
            sample_size = math.ceil(n / 2)
            # 提取所有规则的置信度作为抽样权重
            weights = [r["confidence"] for r in rules]

            # 加权无放回抽样（解决random.choices有放回的问题）
            sampled_rules = []
            remaining_rules = rules.copy()
            remaining_weights = weights.copy()

            for _ in range(sample_size):
                total_weight = sum(remaining_weights)
                if total_weight == 0:
                    # 极端情况：所有权重为0，随机选
                    selected_idx = random.randint(0, len(remaining_rules) - 1)
                else:
                    # 按权重累积值选择
                    rand_val = random.uniform(0, total_weight)
                    cumulative = 0
                    selected_idx = 0
                    for i, w in enumerate(remaining_weights):
                        cumulative += w
                        if cumulative >= rand_val:
                            selected_idx = i
                            break
                # 添加选中的规则并移除（避免重复抽样）
                sampled_rules.append(remaining_rules[selected_idx])
                del remaining_rules[selected_idx]
                del remaining_weights[selected_idx]

            # 提取抽样后的规则字符串
            rule_strings = [r["rule"] for r in sampled_rules]
        else:
            # 数量<=10，不抽样，保留全部规则字符串
            rule_strings = [r["rule"] for r in rules]

        # 转换为指定输出格式
        results.append({
            "head": head_rel,
            "paths": rule_strings
        })

    return results


def build_prompt(head, candidate_rels, is_zero, k):
    instruction = (
        "Logical rules define the relationship between entities and time. Each rule is written in the form "
        "of a logical implication: Rule_head <- Rule_body. If the conditions in the rule body are satisfied, "
        "then the rule head holds true.\n\n"
    )

    if is_zero and k != 0:  # Zero-shot
        context = """For examples:
        Abduct,_hijack,_or_take_hostage(X0,X1,T3) <- Use_unconventional_violence(X0,X1,T0), _Physically_assault(X1,X2,T1), Accuse(X2,X1,T2)
        """
        predict = f'\nGiven a rule head: "{head}", please generate {k} rules that are the most important and relevant to the rule head.'
    else:  # Few-shot
        context = "Samples:\n"
        if k != 0:
            predict = f'\n\nBased on the above rules, please generate {k} rules that are most important to the rule head: "{head}".Please generate the rules based on semantic logic and temporal logic. Return the rules only without any explanations.'
        else:
            predict = f'\n\nBased on the above rules, please generate as many of the most important rules for the rule head: "{head}" as possible. Please generate the rules based on semantic logic and temporal.Return the rules only without any explanations.'
    predict += "\nPlease only select predicates form: {}. Return the rules only without any explanations.".format(
        candidate_rels
    )
    return instruction, context, predict


def modify_path_format(path, head):
    """
    适配新的TXT格式：直接返回规则字符串列表（无需格式化）
    保持函数名不变，兼容原有代码逻辑
    """
    # 因为read_paths已经返回了完整的规则字符串，这里直接返回即可
    return path


def print_prompt(prompt, head, sample_idx):
    """格式化打印prompt内容"""
    print("=" * 100)
    print(f"Prompt for head: {head} | Sample: {sample_idx}")
    print("=" * 100)
    print(prompt)
    print("\n\n")

def generate_rule(row, candidate_rels, rule_path, model, args):
    head = row["head"]
    # 核心逻辑：拼接文件路径并检查文件是否存在
    # 处理 head 中可能包含的特殊字符（如 /），避免路径错误
    safe_head = head.replace("/", "-")
    txt_file_path = os.path.join(rule_path, f"{safe_head}.txt")

    # 判断文件是否存在，若存在则直接返回
    if os.path.exists(txt_file_path):
        print(f"已检测到 {safe_head}.txt 文件存在，跳过当前 head 的规则生成。")
        return
    paths = row["paths"]
    print(f"Processing head relation: {head} (共{len(paths)}条规则)")

    # 当 k=0 时跳过生成，直接输出信息并返回
    if args.k == 0:
        print(f"参数 k=0，跳过关系 {head} 的规则生成阶段。")
        return

    # Build prompt excluding rules
    if len(paths) >= 50:
        instruction, context, predict = build_prompt(
            head, candidate_rels, args.is_zero, args.k
        )
    else:
        instruction, context, predict = build_prompt(
            head, candidate_rels, args.is_zero, 0
        )
    current_prompt = instruction + context + predict

    if args.is_zero:  # For zero-shot setting
        final_prompt = current_prompt
        print_prompt(final_prompt, head, "zero-shot")

    else:  # For few-shot setting
        path_content_list = modify_path_format(paths, head)
        file_name = head.replace("/", "-")

        few_shot_paths = check_prompt_length(
            instruction + context + predict, path_content_list
        )

        final_prompt = instruction + context + few_shot_paths + predict  # Prompt
        print_prompt(final_prompt, head, "all-data")

    rules = LLM.generate_sentence(final_prompt)
    with open(os.path.join(rule_path, f"{head}.txt"), "w", encoding="utf-8") as f:
        f.write(rules + "\n")
        f.close()





def main(args, LLM):
    data_path = os.path.join(args.data_path, args.dataset) + "/"
    dataset = Dataset(data_root=data_path, inv=True)
    sampled_path_dir = os.path.join(args.sampled_paths, args.dataset)

    # 加载TXT格式的规则文件（核心修改：读取txt而非jsonl）
    # sampled_path_file = os.path.join(sampled_path_dir, "190326001638_r[1,2,3]_n200_exp_s8_rules.txt")  # 改为你的TXT文件名
    sampled_path_file = os.path.join(sampled_path_dir, "demo.txt")  # 改为你的TXT文件名
    sampled_path = read_paths(sampled_path_file)

    # 获取数据集所有关系
    rdict = dataset.get_relation_dict()
    all_rels = list(rdict.rel2idx.keys())
    candidate_rels = ", ".join(all_rels)

    # 创建规则输出目录
    rule_path = os.path.join(
        args.rule_path,
        args.dataset,
        f"{args.prefix}{args.model_name}-top-{args.k}-f-{args.f}-l-{args.l}",
    )
    if not os.path.exists(rule_path):
        os.makedirs(rule_path)

    model = LLM
    print("Prepare to print prompt content...")


    # 处理每条规则头

    # for row in tqdm(sampled_path, total=len(sampled_path)):
    #     generate_rule(
    #         row,
    #         candidate_rels=candidate_rels,
    #         rule_path=rule_path,
    #         model=model,
    #         args=args,
    #     )
    with ThreadPool(args.n) as p:
        for _ in tqdm(
            p.imap_unordered(
                partial(
                    generate_rule,
                    candidate_rels=candidate_rels,
                    rule_path=rule_path,
                    model=model,
                    args=args,
                ),
                sampled_path,
            ),
            total=len(sampled_path),
        ):
            pass





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="datasets", help="data directory")
    parser.add_argument("--dataset", type=str, default="icews14", help="dataset name")
    parser.add_argument("--sampled_paths", type=str, default="../output", help="sampled path dir")
    parser.add_argument("--rule_path", type=str, default="gen_rules", help="path to rule file")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo", help="model name")
    parser.add_argument("--is_zero", action="store_true", help="Enable zero-shot mode")
    parser.add_argument("-k", type=int, default=50, help="Number of generated rules")
    parser.add_argument("-f", type=int, default=3, help="Few-shot sample number")
    parser.add_argument("-n", type=int, default=5, help="multi thread number")
    parser.add_argument("-l", type=int, default=2, help="sample times")
    parser.add_argument("--prefix", type=str, default="", help="prefix")
    parser.add_argument("--dry_run", action="store_true", help="dry run")

    args, _ = parser.parse_known_args()
    # LLM = get_registed_model(args.model_name)
    LLM = llm_client.LLMClient()
    # LLM.add_args(parser)
    args = parser.parse_args()

    main(args, LLM)