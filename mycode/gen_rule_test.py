import argparse
import json
import os
import re
from tqdm import tqdm
from functools import partial
import random
from multiprocessing.pool import ThreadPool

# 你自己的 LLM 客户端
from llms import llm_client

# ==================== 固定配置 ====================
RDict_PATH = "../data/icews14/relation2id.json"

# ==================== 工具类 ====================
class Dataset:
    def __init__(self, data_root, inv=True):
        self.data_root = data_root

    def get_relation_dict(self):
        class RDict:
            def __init__(self, json_path=RDict_PATH):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        self.rel2idx = json.load(f)
                except:
                    self.rel2idx = {}
        return RDict()


# ==================== 【修复版】核心过滤函数（100% 能识别你的规则） ====================
def validate_rule_format(rule_str, valid_relations):
    try:
        rule_str = rule_str.strip()
        if not rule_str or "<-" not in rule_str:
            return False

        # 1. 分割头和体
        head_part, body_part = rule_str.split("<-", 1)
        head_part = head_part.strip()
        body_part = body_part.strip()

        # 2. 提取头关系（只拿前面的英文，忽略括号）
        # 例如：Accuse_of_espionage(X0,X1,T) → Accuse_of_espionage
        head_rel = re.match(r'^([A-Za-z0-9_,_-]+)', head_part).group(1).strip()

        # 3. 检查头关系是否存在（必须存在）
        if head_rel not in valid_relations:
            return False

        # 4. 提取体关系（兼容带括号、不带括号）
        body_rels = []
        atoms = re.split(r",\s*", body_part)
        for atom in atoms:
            atom = atom.strip()
            b_rel = re.match(r'^([A-Za-z0-9_,_-]+)', atom).group(1).strip()
            body_rels.append(b_rel)

        # 5. 只允许长度 1 或 2
        if len(body_rels) not in (1, 2):
            return False

        # 6. 体关系必须全部存在于 KG 中
        for br in body_rels:
            if br not in valid_relations:
                return False

        return True

    except Exception as e:
        # 打印错误，方便调试
        # print(f"校验失败: {rule_str}, 错误: {e}")
        return False


def clean_llm_output(rules_text, valid_relations):
    if not rules_text:
        return []
    rules = [r.strip() for r in rules_text.split("\n") if r.strip()]
    final = []
    for r in rules:
        if validate_rule_format(r, valid_relations):
            final.append(r)
    return final


# ==================== 生成规则（零样本 + 强约束） ====================
def build_prompt(head_rel, valid_rels, k=50):
    prompt = f"""
You are a logical rule generator for ICEWS14.
Follow these rules STRICTLY:
1. Output rules like:
   HeadRel(X0,X1,T) <- BodyRel(X0,X1,T0)
   HeadRel(X0,X1,T) <- BodyRel1(X0,X2,T0), BodyRel2(X2,X1,T1)
2. Body length = 1 OR 2
3. Only use relations from: {valid_rels}
4. Return ONLY rules, no other text.

Generate {k} rules for: {head_rel}
""".strip()
    return prompt


def generate_one_rule(head_rel, valid_relations, rule_path, LLM, args):
    safe_head = head_rel.replace("/", "-").replace("\\", "-")
    out_path = os.path.join(rule_path, f"{safe_head}.txt")

    if os.path.exists(out_path):
        print(f"✅ Skip {head_rel} (already exists)")
        return

    print(f"🔵 Generating rules for: {head_rel}")
    prompt = build_prompt(head_rel, valid_relations, args.k)

    try:
        rules_text = LLM.generate_sentence(prompt)
        cleaned = clean_llm_output(rules_text, valid_relations)

        if not cleaned:
            print(f"⚠️ No valid rules for {head_rel}")
            return

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(cleaned))

        print(f"✅ Saved {len(cleaned)} rules for {head_rel}")

    except Exception as e:
        print(f"❌ Fail {head_rel}: {str(e)}")


# ==================== 主函数 ====================
def main(args, LLM):
    data_path = os.path.join(args.data_path, args.dataset)
    dataset = Dataset(data_root=data_path)
    rdict = dataset.get_relation_dict()
    all_rels = list(rdict.rel2idx.keys())

    print(f"📊 Total relations in ICEWS14: {len(all_rels)}")

    rule_path = os.path.join(
        args.rule_path, args.dataset,
        f"{args.model_name}-top-{args.k}-f-{args.f}-l-{args.l}"
    )
    os.makedirs(rule_path, exist_ok=True)
    print(f"📂 Output to: {rule_path}")

    # 多线程生成
    with ThreadPool(args.n) as pool:
        results = list(tqdm(
            pool.imap_unordered(
                partial(generate_one_rule, valid_relations=all_rels, rule_path=rule_path, LLM=LLM, args=args),
                all_rels
            ),
            total=len(all_rels)
        ))

    print("🎉 All done! Rules generated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../data")
    parser.add_argument("--dataset", type=str, default="icews14")
    parser.add_argument("--rule_path", type=str, default="gen_rules")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("-k", type=int, default=30, help="number of rules per relation")
    parser.add_argument("-f", type=int, default=3)
    parser.add_argument("-n", type=int, default=5, help="threads")
    parser.add_argument("-l", type=int, default=2)
    parser.add_argument("--prefix", type=str, default="")
    args = parser.parse_args()

    LLM = llm_client.LLMClient()
    main(args, LLM)