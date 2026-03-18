import json
import os
import random
from llms import LLMClient

class LLMIntegration:
    def __init__(self, model_name="Qwen/Qwen3.5-35B-A3B", api_key=None):
        """
        初始化LLM集成模块
        
        Args:
            model_name: 大模型名称
            api_key: API密钥
        """
        self.llm_client = LLMClient(model_name=model_name, api_key=api_key)
    
    def generate_rules(self, head_relation, candidate_relations, examples=None, k=5):
        """
        使用大模型生成规则
        
        Args:
            head_relation: 规则头关系
            candidate_relations: 候选关系列表
            examples: 示例规则（few-shot）
            k: 生成规则数量
            
        Returns:
            生成的规则列表
        """
        # 构建提示
        instruction = (
            "Logical rules define the relationship between two entities: X and Y. Each rule is written in the form "
            "of a logical implication, which states that if the conditions on the right-hand side (rule body) are "
            "satisfied, then the statement on the left-hand side (rule head) holds true.\n\n"
        )
        
        if examples:
            # Few-shot
            context = "Samples:\n"
            for example in examples:
                context += f"{example}\n"
            predict = f'\n\nBased on the above rules, please generate {k} rules that are most important to the rule head: "{head_relation}(X,Y)". Return the rules only without any explanations.'
        else:
            # Zero-shot
            context = "For examples:\n"
            context += "husband(X,Y) <-- father(X, Z_1) & inv_mother(Z_1, Y) // X is the husband of Y, if X is the father of Z_1, and Y is the mother of Z_1\n"
            context += "husband(X,Y) <-- father(X, Z_1) & son(Z_1, Y) // X is the husband of Y, if X is the father of Z_1, and Z_1 is the son of Y.\n"
            predict = f'\nGiven a rule head: "{head_relation}(X,Y)", please generate {k} rules that are the most important and relevant to the rule head.'
        
        predict += "\nPlease only select predicates form: {}. Return the rules only without any explanations.".format(
            ", ".join(candidate_relations)
        )
        
        prompt = instruction + context + predict
        
        # 调用大模型
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates logical rules for knowledge graphs."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm_client.chat_completion(messages)
        
        # 解析生成的规则
        rules = self._parse_rules(response, head_relation)
        return rules
    
    def optimize_rules(self, rules):
        """
        使用大模型优化规则
        
        Args:
            rules: 规则列表
            
        Returns:
            优化后的规则列表
        """
        if not rules:
            return []
        
        # 构建提示
        prompt = "Please optimize the following logical rules for knowledge graphs. Remove redundant rules, improve logical consistency, and ensure they are syntactically correct. Return only the optimized rules.\n\n"
        for rule in rules:
            prompt += f"{rule}\n"
        
        # 调用大模型
        messages = [
            {"role": "system", "content": "You are a helpful assistant that optimizes logical rules for knowledge graphs."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm_client.chat_completion(messages)
        
        # 解析优化后的规则
        optimized_rules = self._parse_rules(response)
        return optimized_rules
    
    def _parse_rules(self, response, head_relation=None):
        """
        解析大模型生成的规则
        
        Args:
            response: 大模型响应
            head_relation: 规则头关系（可选）
            
        Returns:
            解析后的规则列表
        """
        rules = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 简单解析规则格式
            if '<--' in line:
                rules.append(line)
            elif head_relation and '(' in line and ')' in line:
                # 尝试构建规则
                body = line.strip('.').strip()
                rule = f"{head_relation}(X,Y) <-- {body}"
                rules.append(rule)
        
        return rules
    
    def validate_rule(self, rule):
        """
        使用大模型验证规则的逻辑正确性
        
        Args:
            rule: 规则字符串
            
        Returns:
            验证结果（True/False）和验证信息
        """
        # 构建提示
        prompt = f"Please validate the logical correctness of the following rule for knowledge graphs. Return True if the rule is logically correct, False otherwise, along with a brief explanation.\n\nRule: {rule}"
        
        # 调用大模型
        messages = [
            {"role": "system", "content": "You are a helpful assistant that validates logical rules for knowledge graphs."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm_client.chat_completion(messages)
        
        # 解析验证结果
        is_valid = "True" in response
        return is_valid, response
def read_paths(path): #读取采样路径文件
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results

def build_prompt(head, candidate_rels, is_zero, k):
    # head = clean_symbol_in_rel(head)
    instruction = ( ## 基础指令：解释逻辑规则的格式（规则头 ← 规则体）
        "Logical rules define the relationship between two entities: X and Y. Each rule is written in the form "
        "of a logical implication, which states that if the conditions on the right-hand side (rule body) are "
        "satisfied, then the statement on the left-hand side (rule head) holds true.\n\n"
    )

    if is_zero and args.k != 0:  # Zero-shot
        context = """For examples:
        husband(X,Y) <-- father(X, Z_1) & inv_mother(Z_1, Y) // X is the husband of Y, if X is the father of Z_1, and  Y is the mother of Z_1
        husband(X,Y) <-- father(X, Z_1) & son(Z_1, Y) // X is the husband of Y, if X is the father of Z_1, and Z_1 is the son of Y.
        husband(X,Y) <-- father(X, Z_1) & sister(Z_1, Z_2) & daughter(Z_2, Y) // X is the husband of Y, if X is the father of Z_1, Z_1 is the brother of Z_2, and Z_2 is the daughter of Y.
        """
        predict = f'\nGiven a rule head: "{head}(X,Y)", please generate {k} rules that are the most important and relevant to the rule head.'
    else:  # Few-shot
        context = "Samples:\n"
        if args.k != 0:
            predict = f'\n\nBased on the above rules, please generate {k} rules that are most important to the rule head: "{head}(X,Y)". Return the rules only without any explanations.'
        else:
            predict = f'\n\nBased on the above rules, please generate as many of the most important rules for the rule head: "{head}(X,Y)" as possible. Return the rules only without any explanations.'
    predict += "\nPlease only select predicates form: {}. Return the rules only without any explanations.".format(
        candidate_rels
    )
    return instruction, context, predict

def modify_path_format(path, head):
    """
    Modify path format for prompt, return a list of path in new format
    格式化关系路径为提示词示例
    """
    path_list = []
    # head = clean_symbol_in_rel(head)
    for p in path:
        context = f"{head}(X,Y) <-- "
        # 拆分路径为单个关系（如"father|inv_mother" → ["father", "inv_mother"]）
        for i, r in enumerate(p.split("|")):
            # r = clean_symbol_in_rel(r)
            # 为每个中间节点命名：X → Z_1 → Z_2 → ... → Y
            if i == 0:
                first = "X"
            else:
                first = f"Z_{i}"
            if i == len(p.split("|")) - 1:
                last = "Y"
            else:
                last = f"Z_{i + 1}"
            context += f"{r}({first}, {last}) & "
        context = context.strip(" & ")
        path_list.append(context)
    return path_list


def generate_rule(row, candidate_rels, rule_path, model, args):
    head = row["head"]
    paths = row["paths"]
    # print("Head: ", head)

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
        with open(os.path.join(rule_path, f"{head}_zero_shot.query"), "w", encoding="utf-8") as f:
            f.write(current_prompt + "\n")
            f.close()
        if not args.dry_run:
            response = query(current_prompt, model=args.model_name)
            with open(os.path.join(rule_path, f"{head}_zero_shot.txt"), "w", encoding="utf-8") as f:
                f.write(response + "\n")
                f.close()
    else:  # For few-shot setting
        path_content_list = modify_path_format(paths, head)
        file_name = head.replace("/", "-")
        # 打开规则结果文件和提示词文件
        with open(os.path.join(rule_path, f"{file_name}.txt"), "w", encoding="utf-8") as rule_file, open(
            os.path.join(rule_path, f"{file_name}.query"), "w", encoding="utf-8"
        ) as query_file:
            rule_file.write(f"Rule_head: {head}\n")
            for i in range(args.l):
                # 随机采样f个少样本示例（f是少样本数量，不超过路径总数
                few_shot_samples = random.sample(
                    path_content_list, min(args.f, len(path_content_list))
                )
                # 检查提示词长度（避免超过LLM的上下文窗口限制），截断示例
                few_shot_paths = check_prompt_length(
                    instruction + context + predict, few_shot_samples, model
                )

                prompt = instruction + context + few_shot_paths + predict  # Prompt
                # tqdm.write("Prompt: \n{}".format(prompt))
                query_file.write(f"Sample {i + 1} time: \n")
                query_file.write(prompt + "\n")
                if not args.dry_run:
                    response = model.generate_sentence(prompt)
                    # tqdm.write("Response: \n{}".format(response))
                    rule_file.write(f"Sample {i + 1} time: \n")
                    rule_file.write(response + "\n")