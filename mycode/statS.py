import json
import statistics
from typing import Dict, List, Tuple


def load_rule_data(file_path: str) -> Dict[str, int]:
    """
    加载JSON文件并提取每个关系对应的规则数量
    适配格式：{"关系ID": [规则1, 规则2, ...], ...}

    Args:
        file_path: JSON文件路径

    Returns:
        字典，键为关系ID（字符串），值为对应的规则数量
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 读取文件并处理可能的截断问题
            file_content = f.read().strip()
            # 修复可能的JSON截断（比如示例中最后一行不完整）
            if not file_content.endswith('}'):
                # 找到最后一个有效的闭合括号位置
                last_brace = file_content.rfind('}')
                if last_brace != -1:
                    file_content = file_content[:last_brace + 1]
                else:
                    raise ValueError("JSON文件内容不完整，无法解析")

            data = json.loads(file_content)

        # 提取每个关系的规则数量（列表长度即为规则数）
        rule_counts = {}
        for rel_id, rules_list in data.items():
            if isinstance(rules_list, list):
                rule_counts[rel_id] = len(rules_list)
            else:
                # 兼容非列表格式（设为0）
                rule_counts[rel_id] = 0

        if not rule_counts:
            raise ValueError("未从JSON文件中提取到有效的规则数量数据")

        return rule_counts

    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON文件格式错误: {str(e)}，请检查文件完整性")
    except Exception as e:
        raise Exception(f"加载数据失败: {str(e)}")


def calculate_statistics(rule_counts: Dict[str, int]) -> Dict[str, float]:
    """
    计算规则数量的统计指标

    Args:
        rule_counts: 关系-规则数量字典

    Returns:
        包含各类统计指标的字典
    """
    # 提取所有规则数量的列表
    counts = list(rule_counts.values())

    # 计算统计指标
    stats = {
        "关系总数": len(counts),  # 关系的总数
        "规则总数": sum(counts),  # 所有关系的规则总数
        "平均每条关系的规则数": round(statistics.mean(counts), 2),  # 平均数
        "中位数": statistics.median(counts),  # 中位数
        "众数": statistics.mode(counts) if len(counts) > 1 else counts[0],  # 众数
        "标准差": round(statistics.stdev(counts) if len(counts) > 1 else 0, 2),  # 标准差
        "最小规则数": min(counts),  # 最小值
        "最大规则数": max(counts),  # 最大值
        "方差": round(statistics.variance(counts) if len(counts) > 1 else 0, 2)  # 方差
    }

    return stats


def print_statistics(stats: Dict[str, float], rule_counts: Dict[str, int]) -> None:
    """
    格式化打印统计结果
    """
    print("=" * 70)
    print("关系规则数量统计分析结果")
    print("=" * 70)

    # 打印核心统计信息
    for key, value in stats.items():
        print(f"{key:>15}: {value}")

    print("\n" + "-" * 70)
    print("各关系规则数量详情（按规则数降序）")
    print("-" * 70)

    # 按规则数量降序打印各关系详情
    sorted_relations = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)
    for idx, (rel_id, count) in enumerate(sorted_relations, 1):
        print(f"排名{idx:<3} | 关系ID: {rel_id:<6} | 规则数量: {count:>5} 条")


def main(file_path: str) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    主函数：执行完整的统计流程

    Args:
        file_path: JSON文件路径

    Returns:
        元组(关系-规则数量字典, 统计指标字典)
    """
    try:
        # 1. 加载数据
        rule_counts = load_rule_data(file_path)

        # 2. 计算统计指标
        stats = calculate_statistics(rule_counts)

        # 3. 打印结果
        print_statistics(stats, rule_counts)

        return rule_counts, stats

    except Exception as e:
        print(f"\n❌ 程序执行出错: {str(e)}")
        return {}, {}

# 示例使用
if __name__ == "__main__":
    # 替换为你的JSON文件路径
    JSON_FILE_PATH = "../output/icews14/120326150307_r[1,2,3]_n200_exp_s12_rules.json"

    # 执行统计
    rule_data, stat_data = main(JSON_FILE_PATH)

    if rule_data and stat_data:
        output_data = {
            "relation_rule_counts": rule_data,
            "statistics": stat_data,
            "top_10_relations": dict(sorted(rule_data.items(), key=lambda x: x[1], reverse=True)[:10])
        }
        with open("rule_statistics_result.json", 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print(f"\n✅ 统计结果已保存到: rule_statistics_result.json")