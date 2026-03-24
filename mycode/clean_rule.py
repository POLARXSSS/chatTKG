import argparse
import os
from grapher import Grapher
import re
from difflib import get_close_matches

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
    return list(set(cleaned_rules)) # Remove duplicates by converting to set and back to list


def summarize_rules_prompt(relname, k):
    """
    Generate prompt for the relation in the content_list
    功能：生成给大模型（如 GPT）的提示词，用于规则汇总。
    逻辑：
    若 k≠0：要求提取 k 条最重要的规则；
    若 k=0：要求提取尽可能多的重要规则；
    提示词包含 “合并相似规则”“仅返回规则不解释” 的要求。
    """

    if k != 0:
        prompt = f'\n\nPlease identify the most important {k} rules from the following rules for the rule head: "{relname}(X,Y)". '
    else:  # k ==0
        prompt = f'\n\nPlease identify as many of the most important rules for the rule head: "{relname}(X,Y)" as possible. '

    prompt += 'You can summarize the rules that have similar meanings as one rule, if you think they are important. ' \
              'Return the rules only without any explanations. '
    return prompt


def get_valid_rules(input_filepath, output_filepath, valid_response_filepath):
    '''
    功能：调用 GPT-4 验证规则逻辑正确性，仅保留正确规则。
    关键步骤：
    读取待验证规则列表，拼接 “逐条分析规则正确性并标注 (Correct/Incorrect)” 的提示词；
    调用 GPT-4 接口获取分析结果；
    过滤掉结果中包含 “incorrect” 的规则，写入有效规则文件；
    同时记录原始规则 + GPT-4 回复（用于追溯）。
    '''
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
    """
    Determine the sample time, return True if only sample once
    功能：判断规则文件中是否仅采样了 1 次（采样次数影响是否需要汇总）。
    逻辑：统计包含 Sample \d+ time: 的行数，返回 “是否等于 1” 的布尔值。
    """
    sample_times = 0
    for line in content_list:
        match = re.search(r'Sample \d+ time:', line)
        if match:
            sample_times += 1
    return sample_times == 1


def summarize_rule(file, args):
    """
    Summarize the rules
    功能：核心汇总逻辑，决定是否调用大模型汇总规则。
    分支逻辑：
    不汇总场景：仅采样 1 次 或 指定模型为none 且 不强制汇总 → 直接返回提取的原始规则；
    汇总场景：
    生成汇总提示词，计算提示词 token 长度；
    将规则列表分片（适配模型 token 限制）；
    逐片调用大模型，获取汇总后的规则；
    从大模型回复中再次提取规则（过滤解释性文字）并返回。
    """
    with open(file, 'r') as f:  # Load files
        content = f.read()
    content_list = content.split('\n')
    
    # 从文件名中提取关系名称
    filename = os.path.basename(file)
    rel_name = os.path.splitext(filename)[0]
    
    is_sample_once = check_sample_times(content_list)
    rule_list = extract_rules(content_list)  # Extract rules and remove any explanations
    if (is_sample_once or args.model == 'none') and not args.force_summarize:  # just return the whole rule_list
        return rule_list
    else:  # Do summarization and correct the spelling error
        summarize_prompt = summarize_rules_prompt(rel_name, args.k)
        summarize_prompt_len = num_tokens_from_message(summarize_prompt, args.model)
        list_of_rule_lists = shuffle_split_path_list(rule_list, summarize_prompt_len, args.model)
        response_list = []
        for rule_list in list_of_rule_lists:
            message = '\n'.join(rule_list) + summarize_prompt
            print('prompt: ', message)
            response = query(message, model=args.model)
            response_list.extend(response.split('\n'))
        response_rules = extract_rules(response_list) # Extract rules and remove any explanations from summarized response
            
        return response_rules


def clean_rules(summarized_file_path, all_rels):
    """
    Clean error rules and remove rules with error relation.
    功能：修复规则中的关系名错误，过滤无效规则，统一格式。
    核心步骤：
    提取规则头（如 father(X,Y) <-- ... 中的 father），若不在合法关系列表中，
    用get_close_matches找最相似的合法关系（纠正拼写 / 语法错误）；
    提取规则体中的关系（<-- 后的条件），同理纠正错误关系名；
    校验规则的 “链式格式”（如 X→Y→Z 的实体衔接），补全inv_（逆关系）保证格式统一；
    过滤无法纠正的规则，返回清洗后的规则列表。
    """
    with open(summarized_file_path, 'r') as f:
        input_rules = [line.strip() for line in f]
    cleaned_rules = list()
    # Correct spelling error/grammar error for the relation in the rules and Remove rules with error relation.
    for rule in input_rules:
        if rule == "":
            continue
        try:
            # Get rule head
            match = re.search(r'([\w\s\'-.]+)\([^)]+\)', rule)
            if not match:
                continue

            head = match.group(1).strip()
            if head not in all_rels:
                best_match = get_close_matches(head, all_rels, n=1)
                if not best_match:
                    print("Cannot correctify this rule, head not in relation: ", rule)
                    continue
                head = best_match[0].strip()

            # Get rule conditions and check if they are in the relation list
            condition_string = rule.split('<-')[1].strip()
            # 先按逗号分割规则体，然后对每个部分提取关系名称
            # 注意：我们需要考虑关系名称中可能包含逗号的情况
            # 因此我们使用正则表达式来分割，只在关系之间的逗号处分割
            # 使用正则表达式分割规则体，只在关系之间的逗号处分割
            # 匹配模式：),( 表示一个关系的结束和另一个关系的开始
            conditions = re.split(r'\),\s*', condition_string)
            # 清理每个条件，确保它们是完整的关系表达式
            conditions = [cond.strip() + ')' if not cond.strip().endswith(')') else cond.strip() for cond in conditions]
            body_list = []
            correctyfied = True if len(conditions) > 0 else False
            for cond in conditions:
                # 提取关系名称
                match = re.search(r'([^()]+)\([^)]+\)', cond)
                if not match:
                    correctyfied = False
                    print(f"Cannot extract predicate from condition: {cond}")
                    break
                predicate = match.group(1).strip()
                if predicate not in all_rels:
                    best_match = get_close_matches(predicate, all_rels, n=1)
                    if not best_match:
                        correctyfied = False
                        print(f"Cannot correctify this rule, body: {predicate} not in relaiton: ", rule)
                        break
                    predicate = best_match[0].strip()
                body_list.append(predicate)

            # Add corrected rule to cleaned_rules if it's valid
            if correctyfied:
                cleaned_rules.append(f"{head} <-- {', '.join(body_list)}")

        except Exception as e:
            print(f"Processing {rule} failed.\n Error: {str(e)}")
    return cleaned_rules


def write_clean_rules_to_file(cleaned_rules, output_filepath, all_rels):
    """
    功能：将清洗后的规则以简化格式写入文件（如 father <-- parent, inv_mother）。
    逻辑：提取规则头和规则体的关系名，仅保留关系名（去掉(X,Y)等格式），按 头 <-- 条件1, 条件2 格式写入。
    """
    with open(output_filepath, "w") as output_file:
        for rule in cleaned_rules:
            try:
                match = re.search(r'([\w\s\'-\.]+)\(X,\s?Y\)', rule)  # Get rule head
                if match:
                    head = match.group(1).strip()
                    if head not in all_rels:
                        raise KeyError(f"Key {head} not found in all_rels dictionary")
                else:
                    continue

                # Get rule conditions and write to file in simplified format
                condition_string = rule.split('<--')[1].strip()
                matches = re.findall(r"([\w\s'-\.]+)\(", condition_string)
                conditions = []
                for match in matches:
                    match = match.strip()
                    if match in all_rels:
                        conditions.append(match)
                    else:
                        raise KeyError(f"Key {match} not found in all_rels dictionary")

                # Write to file
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
                # Step 1: Summarize rules from the input file
                print("Start summarize: ", filename)
                # Summarize rules
                summarized_rules = summarize_rule(input_filepath, args)
                print("write file", summarized_filepath)
                with open(summarized_filepath, "w") as f:
                    f.write('\n'.join(summarized_rules))

            # Step 2: Clean summarized rules and keep format
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
    args.add_argument('--model', default='none', help='model name', choices=['none', 'gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k'])
    args.add_argument('-p', default='gpt-3.5-turbo-top-50-f-3-l-2', help='rule prefix')
    args.add_argument('-k', type=int, default=0, help='Number of summarized rules')
    args.add_argument('--clean_only', action='store_true', help='Load summarized rules then clean rules only')
    args.add_argument('--valid_clean', action='store_true', help='gpt-4 validation for rules')
    args.add_argument('--force_summarize', action='store_true', help='force summarize rules')
    args = args.parse_args()
    clean(args)
