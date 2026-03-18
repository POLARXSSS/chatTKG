import argparse
import json
import os
import re
from tqdm import tqdm
from functools import partial
import random


# 模拟缺失的依赖模块（避免运行报错）
class Dataset:
    def __init__(self, data_root, inv=True):
        self.data_root = data_root

    def get_relation_dict(self):
        # 模拟返回关系字典（从TXT中提取的关系名）
        class RDict:
            def __init__(self):
                self.rel2idx = {
                    "Abduct,_hijack,_or_take_hostage": 0,
                    "Use_unconventional_violence": 1,
                    "_Physically_assault": 2,
                    "Accuse": 3,
                    "_Accuse": 4,
                    "Make_statement": 5,
                    "Conduct_suicide,_car,_or_other_non-military_bombing": 6,
                    "Criticize_or_denounce": 7,
                    "Use_conventional_military_force": 8,
                    "_Make_an_appeal_or_request": 9,
                    "Make_optimistic_comment": 10,
                    "_Receive_deployment_of_peacekeepers": 11,
                    "_Provide_military_protection_or_peacekeeping": 12,
                    "_Return,_release_person(s)": 13,
                    "Return,_release_person(s)": 14,
                    "Engage_in_symbolic_act": 15,
                    "Express_intent_to_cooperate": 16
                }

        return RDict()


def get_registed_model(model_name):
    # 模拟返回LLM类
    class MockLLM:
        @staticmethod
        def add_args(parser):
            pass

    return MockLLM


def check_prompt_length(base_prompt, samples, model):
    """模拟检查prompt长度，直接返回拼接的样本"""
    return "\n".join(samples)


def read_paths(path):
    """
    读取TXT格式的规则文件，提取规则头和规则体
    输入：TXT文件路径
    输出：[{"head": 规则头关系名, "paths": [规则字符串1, 规则字符串2...]}]
    """
    results = []
    # 用于存储提取的规则（按规则头分组）
    rule_dict = {}

    # 正则表达式：匹配 "规则头 ← 规则体" 部分
    # 匹配逻辑：忽略开头的数字列，提取 <- 前后的内容
    rule_pattern = re.compile(r'\d+\.?\d*\s+\d+\s+\d+\s+(.*? <- .*)')

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue

            # 提取规则部分（忽略开头的数字）
            match = rule_pattern.match(line)
            if match:
                full_rule = match.group(1).strip()
                # 拆分规则头和规则体（按 <- 分割）
                head_part, body_part = full_rule.split(" <- ")

                # 提取规则头的关系名（去掉括号和参数，如 Abduct,_hijack,_or_take_hostage(X0,X1,T3) → 关系名）
                head_rel = re.match(r'^([^(]+)', head_part).group(1).strip()

                # 按规则头分组存储完整规则字符串
                if head_rel not in rule_dict:
                    rule_dict[head_rel] = []
                rule_dict[head_rel].append(full_rule)

    # 转换为原代码需要的格式：[{"head": 关系名, "paths": [规则字符串列表]}]
    for head_rel, rules in rule_dict.items():
        results.append({
            "head": head_rel,
            "paths": rules  # 这里paths存储的是完整规则字符串，而非原有的|分隔路径
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
            predict = f'\n\nBased on the above rules, please generate {k} rules that are most important to the rule head: "{head}". Return the rules only without any explanations.'
        else:
            predict = f'\n\nBased on the above rules, please generate as many of the most important rules for the rule head: "{head}" as possible. Return the rules only without any explanations.'
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
    paths = row["paths"]
    print(f"Processing head relation: {head} (共{len(paths)}条规则)")

    # Raise an error if k=0 for zero-shot setting
    if args.k == 0 and args.is_zero:
        raise NotImplementedError(
            f"""Cannot implement for zero-shot(f=0) and generate zero(k=0) rules."""
        )
    # Build prompt excluding rules
    instruction, context, predict = build_prompt(
        head, candidate_rels, args.is_zero, args.k
    )
    current_prompt = instruction + context + predict

    if args.is_zero:  # For zero-shot setting
        print_prompt(current_prompt, head, "zero-shot")
    else:  # For few-shot setting
        path_content_list = modify_path_format(paths, head)
        file_name = head.replace("/", "-")

        for i in range(args.l):
            # 随机采样f个少样本示例
            few_shot_samples = random.sample(
                path_content_list, min(args.f, len(path_content_list))
            )
            # 检查提示词长度
            few_shot_paths = check_prompt_length(
                instruction + context + predict, few_shot_samples, model
            )

            prompt = instruction + context + few_shot_paths + predict  # Prompt
            print_prompt(prompt, head, i + 1)


def main(args, LLM):
    data_path = os.path.join(args.data_path, args.dataset) + "/"
    dataset = Dataset(data_root=data_path, inv=True)
    sampled_path_dir = os.path.join(args.sampled_paths, args.dataset)

    # 加载TXT格式的规则文件（核心修改：读取txt而非jsonl）
    sampled_path_file = os.path.join(sampled_path_dir, "120326150307_r[1,2,3]_n200_exp_s12_rules.txt")  # 改为你的TXT文件名
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

    model = None
    print("Prepare to print prompt content...")

    # 处理每条规则头
    for row in tqdm(sampled_path, total=len(sampled_path)):
        generate_rule(
            row,
            candidate_rels=candidate_rels,
            rule_path=rule_path,
            model=model,
            args=args,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="datasets", help="data directory")
    parser.add_argument("--dataset", type=str, default="icews14", help="dataset name")
    parser.add_argument("--sampled_paths", type=str, default="../output", help="sampled path dir")
    parser.add_argument("--rule_path", type=str, default="gen_rules", help="path to rule file")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo", help="model name")
    parser.add_argument("--is_zero", action="store_true", help="Enable zero-shot mode")
    parser.add_argument("-k", type=int, default=5, help="Number of generated rules")
    parser.add_argument("-f", type=int, default=3, help="Few-shot sample number")
    parser.add_argument("-n", type=int, default=5, help="multi thread number")
    parser.add_argument("-l", type=int, default=2, help="sample times")
    parser.add_argument("--prefix", type=str, default="", help="prefix")
    parser.add_argument("--dry_run", action="store_true", help="dry run")

    args, _ = parser.parse_known_args()
    LLM = get_registed_model(args.model_name)
    LLM.add_args(parser)
    args = parser.parse_args()

    main(args, LLM)