import json
import os

# ===================== 配置参数（按你的要求设置） =====================
# 规则txt文件所在目录
TXT_RULES_DIR = "./gen_rules/icews14/gpt-3.5-turbo-top-50-f-3-l-2"
# relation2id.json路径
RELATION2ID_PATH = "../data/icews14/relation2id.json"
BODY_PATH = "../data/icews14/entity2id.json"
# 输出的JSON规则文件路径（和主代码的--rules参数对应）
OUTPUT_JSON_PATH = "../output/icews14/rules.json"
# 默认置信度（你要求的0.5）
DEFAULT_CONF = 0.5
# 默认体支持度（你要求的3）
DEFAULT_BODY_SUPP = 3

# ===================== 加载关系ID映射 =====================
# 加载relation2id.json（关系名称→ID）
with open(RELATION2ID_PATH, "r", encoding="utf-8") as f:
    relation2id = json.load(f)

with open(RELATION2ID_PATH, "r", encoding="utf-8") as f:
    body2id = json.load(f)



# 构建逆关系ID映射（和Grapher类逻辑一致）
inv_relation2id = {}
num_rels = len(relation2id)
for rel_name, rel_id in relation2id.items():
    inv_relation2id["_" + rel_name] = rel_id + num_rels  # 逆关系ID = 原ID + 总关系数

# 合并正向+逆关系的名称→ID映射
full_relation2id = relation2id.copy()
full_relation2id.update(inv_relation2id)

full_body2id = body2id.copy()

# ===================== 解析所有txt规则文件 =====================
# 最终的规则字典：{关系ID: [规则列表]}
rules_dict = {}

# 遍历gen_rules下的所有txt文件
for txt_file in os.listdir(TXT_RULES_DIR):
    if not txt_file.endswith(".txt"):
        continue

    # 提取头关系名称（文件名 = 头关系名称）
    head_rel_name = txt_file.replace(".txt", "")
    # 获取头关系ID（如果不存在则跳过）
    if head_rel_name not in relation2id:
        print(f"警告：关系 {head_rel_name} 不在relation2id.json中，跳过该文件")
        continue
    head_rel_id = relation2id[head_rel_name]

    # 读取txt文件中的规则
    txt_file_path = os.path.join(TXT_RULES_DIR, txt_file)
    with open(txt_file_path, "r", encoding="utf-8") as f:
        rule_lines = [line.strip() for line in f if line.strip()]

    # 解析每一行规则
    rule_list = []
    for line in rule_lines:
        # 拆分规则头和规则体（格式：Head(...) <- Body(...)）
        if "<-" not in line:
            print(f"警告：规则格式错误 {line}，跳过")
            continue

        head_part, body_part = line.split("<-")
        head_part = head_part.strip()
        body_part = body_part.strip()

        # 提取规则体中的关系名称（处理逆关系，如_Make_an_appeal_or_request）
        # 示例：_Make_an_appeal_or_request(X0,X1,T0) → 提取 _Make_an_appeal_or_request
        body_rels = []
        # 按逗号拆分规则体（支持多步规则，如 A(X0,X1,T0), B(X1,X2,T1)）
        body_atoms = [atom.strip() for atom in body_part.split(",")]
        for atom in body_atoms:
            # 提取关系名称（括号前的部分）
            body_rel_name = atom.split("(")[0]
            if body_rel_name not in body2id:
                print(f"警告：规则体关系 {body_rel_name} 无对应ID，跳过该规则")
                break
            body_rels.append(body_rel_name)
        else:  # 所有体关系都有ID，才继续
            # 构建规则字典（TLogic要求的格式）
            rule = {
                "head": head_rel_name,  # 头关系名称
                "body": body_rels,  # 规则体关系列表
                "conf": DEFAULT_CONF,  # 默认置信度0.5
                "body_supp": DEFAULT_BODY_SUPP,  # 默认体支持度3
                "rule_length": len(body_rels),  # 规则长度 = 体关系数量
                "var_constraints": []  # 无变量约束（可根据需要扩展）
            }
            rule_list.append(rule)

    # 将解析后的规则添加到字典中
    if rule_list:
        rules_dict[head_rel_id] = rule_list

# ===================== 保存JSON文件 =====================
# 确保输出目录存在
os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

# 保存规则字典为JSON
with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(rules_dict, f, indent=2, ensure_ascii=False)

print(f"✅ 规则转换完成！")
print(f"📁 输出JSON文件：{OUTPUT_JSON_PATH}")
print(f"📊 共解析 {len(rules_dict)} 个关系的规则")