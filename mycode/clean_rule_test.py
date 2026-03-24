import argparse
import os
from grapher import Grapher
import re
from difflib import get_close_matches


# ==================== 工具函数：过滤规则长度（1~2，ICEWS14专用） ====================
def filter_rule_length(body_list):
    """
    只保留 规则体长度 = 1 或 2 的规则
    解决：路径过长 → 矩阵相乘为空 → 4个0
    """
    return 1 <= len(body_list) <= 2


# ==================== 工具函数：规则去重（字符串级） ====================
def deduplicate_rules(rule_list):
    seen = set()
    new_rules = []
    for r in rule_list:
        if r not in seen:
            seen.add(r)
            new_rules.append(r)
    return new_rules


# 模拟num_tokens_from_message函数
def num_tokens_from_message(message, model="gpt-3.5-turbo"):
    """计算消息的token数"""
    return len(message.split())


# 模拟shuffle_split_path_list函数
def shuffle_split_path_list(rule_list, prompt_len, model):
    """分割规则列表"""
    return [rule_list]


# 模拟query函数，返回空字符串，避免LLM依赖
def query(message, model="gpt-3.5-turbo"):
    """调用LLM进行查询"""
    return ""


def extract_rules(content_list):
    '''
    功能：从文本行列表中提取符合特定格式的规则，清理格式并去重。
    关键逻辑：
    用正则匹配包含 <- 的规则行；
    移除规则行开头的数字编号（如 1. father(X,Y) <-- ... → father(X,Y) <-- ...）；
    转集合去重后转回列表，保证规则唯一性。
    '''
    """ Extract the rules in the content without any explanation and the leading number if it has."""
    rule_pattern = re.compile(r".* <- .*")  # 匹配包含 <- 的规则行
    extracted_rules = [s.strip() for s in content_list if rule_pattern.match(s)]
    number_pattern = re.compile(r"^\d+\. ")
    cleaned_rules = [number_pattern.sub('', s) for s in extracted_rules]
    return list(set(cleaned_rules))  # Remove duplicates by converting to set and back to list


def summarize_rules_prompt(relname, k):
    if k != 0:
        prompt = f'\n\nPlease identify the most important {k} rules from the following rules for the rule head: "{relname}(X,Y)". '
    else:
        prompt = f'\n\nPlease identify as many of the most important rules for the rule head: "{relname}(X,Y)" as possible. '

    prompt += 'You can summarize the rules that have similar meanings as one rule, if you think they are important. ' \
              'Return the rules only without any explanations. '
    return prompt


def get_valid_rules(input_filepath, output_filepath, valid_response_filepath):
    with open(input_filepath, "r") as f:
        sum_rule_list = [line.strip() for line in f]
        f.close()
    valid_prompt = ("Logical rules define the relationship between two entities: X and Y.\n"
                    "Now please analyse this relation rule path step by step to check whether it is correct. \n"
                    "If the rules is correct please write (Correct) at the end of your analysis, otherwise please write (Incorrect).\n\n")

    with open(output_filepath, "w") as f1, open(valid_response_filepath, 'w') as f2:
        for sum_rule in sum_rule_list:
            message = valid_prompt + sum_rule
            response = query(message, model="gpt-4")
            print(response)
            f2.write("Input Rule: " + sum_rule + "\n")
            f2.write("GPT-4 Response: \n" + response + '\n')
            f2.write("\n=======================================\n")
            if "incorrect" not in response.lower():
                f1.write(sum_rule + '\n')


def check_sample_times(content_list):
    sample_times = 0
    for line in content_list:
        match = re.search(r'Sample \d+ time:', line)
        if match:
            sample_times += 1
    return sample_times == 1


def summarize_rule(file, args):
    with open(file, 'r') as f:
        content = f.read()
    content_list = content.split('\n')
    filename = os.path.basename(file)
    rel_name = os.path.splitext(filename)[0]
    is_sample_once = check_sample_times(content_list)
    rule_list = extract_rules(content_list)

    if (is_sample_once or args.model == 'none') and not args.force_summarize:
        return rule_list
    else:
        summarize_prompt = summarize_rules_prompt(rel_name, args.k)
        summarize_prompt_len = num_tokens_from_message(summarize_prompt, args.model)
        list_of_rule_lists = shuffle_split_path_list(rule_list, summarize_prompt_len, args.model)
        response_list = []
        for rule_list in list_of_rule_lists:
            message = '\n'.join(rule_list) + summarize_prompt
            response = query(message, model=args.model)
            response_list.extend(response.split('\n'))
        response_rules = extract_rules(response_list)
        return response_rules


# ==================== ✅ 核心增强：clean_rules 加入长度过滤 + 强校验 ====================
def clean_rules(summarized_file_path, all_rels):
    """
    增强版清洗规则：
    1. 修复关系拼写错误
    2. 强制规则体长度 1~2
    3. 过滤无效关系
    4. 格式标准化
    """
    with open(summarized_file_path, 'r') as f:
        input_rules = [line.strip() for line in f]

    cleaned_rules = []

    for rule in input_rules:
        if not rule:
            continue

        try:
            # === 1. 解析规则头 ===
            head_match = re.search(r'([\w\s\'-.]+)\([^)]+\)', rule)
            if not head_match:
                continue

            head = head_match.group(1).strip()
            if head not in all_rels:
                best_match = get_close_matches(head, all_rels, n=1)
                if not best_match:
                    print(f"[丢弃] 头关系不存在且无法修复: {rule}")
                    continue
                head = best_match[0].strip()

            # === 2. 解析规则体 ===
            if '<-' not in rule:
                print(f"[丢弃] 格式错误: {rule}")
                continue

            condition_string = rule.split('<-')[1].strip()
            conditions = re.split(r'\),\s*', condition_string)
            conditions = [c.strip() + ')' if not c.endswith(')') else c.strip() for c in conditions]

            body_list = []
            valid = True

            for cond in conditions:
                pred_match = re.search(r'([^()]+)\([^)]+\)', cond)
                if not pred_match:
                    valid = False
                    break

                pred = pred_match.group(1).strip()
                if pred not in all_rels:
                    best_match = get_close_matches(pred, all_rels, n=1)
                    if not best_match:
                        valid = False
                        break
                    pred = best_match[0].strip()
                body_list.append(pred)

            # === 3. ✅ 新增：强制长度 1~2（最关键改进）===
            if not filter_rule_length(body_list):
                print(f"[丢弃] 规则体过长（只保留1~2）: {rule}")
                continue

            # === 4. 只保留有效规则 ===
            if valid:
                cleaned_rules.append(f"{head} <-- {', '.join(body_list)}")

        except Exception as e:
            print(f"[丢弃] 解析失败 {rule}: {str(e)}")
            continue

    # === 5. ✅ 新增：最终去重 ===
    cleaned_rules = deduplicate_rules(cleaned_rules)
    return cleaned_rules


def write_clean_rules_to_file(cleaned_rules, output_filepath, all_rels):
    with open(output_filepath, "w") as output_file:
        for rule in cleaned_rules:
            try:
                match = re.search(r'([\w\s\'-\.]+)\(X,\s?Y\)', rule)
                if match:
                    head = match.group(1).strip()
                    if head not in all_rels:
                        raise KeyError(f"Key {head} not found in all_rels dictionary")
                else:
                    continue

                condition_string = rule.split('<--')[1].strip()
                matches = re.findall(r"([\w\s'-\.]+)\(", condition_string)
                conditions = []
                for match in matches:
                    match = match.strip()
                    if match in all_rels:
                        conditions.append(match)
                    else:
                        raise KeyError(f"Key {match} not found in all_rels dictionary")

                output_file.write(f"{head} <-- {', '.join(conditions)}\n")

            except KeyError as e:
                print(f"Skipping rule {rule} due to error: {e}")
                continue


def clean(args):
    data_path = os.path.join(args.data_path, args.dataset) + '/'
    dataset = Grapher(data_path)
    all_rels = list(dataset.relation2id_old.keys())
    input_folder = os.path.join(args.rule_path, args.dataset, args.p)
    output_folder = os.path.join(args.output_path, args.dataset, args.p, args.model)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt") and "query" not in filename:
            input_filepath = os.path.join(input_folder, filename)
            name, ext = os.path.splitext(filename)
            summarized_filepath = os.path.join(output_folder, f"{name}_summarized_rules.txt")
            clean_filename = name + '_cleaned_rules.txt'
            clean_filepath = os.path.join(output_folder, clean_filename)

            if not args.clean_only:
                print("Start summarize: ", filename)
                summarized_rules = summarize_rule(input_filepath, args)
                with open(summarized_filepath, "w") as f:
                    f.write('\n'.join(summarized_rules))

            # ✅ 使用增强版 clean_rules
            print(f"Clean file {summarized_filepath} with keeping the format")
            cleaned_rules = clean_rules(summarized_filepath, all_rels)

            with open(clean_filepath, "w") as f:
                f.write('\n'.join(cleaned_rules))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--data_path', type=str, default='../data', help='data directory')
    args.add_argument("--rule_path", default="gen_rules", type=str, help="path to rule file")
    args.add_argument("--output_path", default="clean_rules", type=str, help="path to output file")
    args.add_argument('--dataset', default='icews14')
    args.add_argument('--model', default='none', help='model name',
                      choices=['none', 'gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k'])
    args.add_argument('-p', default='gpt-3.5-turbo-top-50-f-3-l-2', help='rule prefix')
    args.add_argument('-k', type=int, default=0, help='Number of summarized rules')
    args.add_argument('--clean_only', action='store_true', help='Load summarized rules then clean rules only')
    args.add_argument('--valid_clean', action='store_true', help='gpt-4 validation for rules')
    args.add_argument('--force_summarize', action='store_true', help='force summarize rules')
    args = args.parse_args()
    clean(args)