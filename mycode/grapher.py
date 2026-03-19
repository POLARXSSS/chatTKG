import json
import numpy as np


class Grapher(object):
    """
    时序知识图谱（TKG）数据管理核心类
    核心作用：
    1. 加载数据集的实体/关系/时间戳映射表（名称→ID）
    2. 为每个关系生成逆关系（如 "father_of" → "_father_of"），并分配唯一ID
    3. 加载训练/验证/测试集的四元组，转换为ID格式，并补充逆四元组
    4. 统一管理所有数据，为后续规则学习/应用提供标准化的索引数据
    """

    def __init__(self, dataset_dir):
        """
        初始化Grapher对象，加载图谱元数据并预处理所有数据集

        Parameters:
            dataset_dir (str): 图谱数据集目录路径（需包含entity2id.json/relation2id.json/ts2id.json等文件）
        Returns:
            None
        """
        # 保存数据集根目录
        self.dataset_dir = dataset_dir

        # ===================== 加载基础映射表（名称→ID） =====================
        # 实体名称→ID映射（如 "Merkel" → 123）
        self.entity2id = json.load(open(dataset_dir + "entity2id.json"))
        # 原始关系名称→ID映射（仅正向关系，如 "consult" → 45）
        self.relation2id_old = json.load(open(dataset_dir + "relation2id.json"))
        # 扩展关系映射表（包含正向+逆向关系）
        self.relation2id = self.relation2id_old.copy()

        # 为每个原始关系生成逆关系，并分配新ID（如 "consult" → "_consult"，ID从原始关系数量开始）
        counter = len(self.relation2id_old)
        for relation in self.relation2id_old:
            self.relation2id["_" + relation] = counter  # 逆关系名称前缀加下划线区分
            counter += 1

        # 时间戳→ID映射（如 "2020-01-01" → 789）
        self.ts2id = json.load(open(dataset_dir + "ts2id.json"))

        # ===================== 构建反向映射表（ID→名称） =====================
        # ID→实体名称（方便后续结果解析）
        self.id2entity = dict([(v, k) for k, v in self.entity2id.items()])
        # ID→关系名称（包含逆关系，如 45→"consult"，90→"_consult"）
        self.id2relation = dict([(v, k) for k, v in self.relation2id.items()])
        # ID→时间戳（方便后续时间窗口筛选）
        self.id2ts = dict([(v, k) for k, v in self.ts2id.items()])

        # ===================== 构建逆关系ID映射（核心！） =====================
        # 用于快速查找一个关系对应的逆关系ID，示例：
        # 正向关系ID 0 → 逆关系ID num_relations
        # 逆关系ID num_relations → 正向关系ID 0
        self.inv_relation_id = dict()
        num_relations = len(self.relation2id_old)  # 原始正向关系数量
        # 正向关系 → 逆关系ID
        for i in range(num_relations):
            self.inv_relation_id[i] = i + num_relations
        # 逆关系 → 正向关系ID
        for i in range(num_relations, num_relations * 2):
            self.inv_relation_id[i] = i % num_relations

        # ===================== 加载并预处理训练/验证/测试集 =====================
        # 加载训练集四元组，转换为ID格式并添加逆四元组
        self.train_idx = self.create_store("train.txt")
        # 加载验证集四元组，同上
        self.valid_idx = self.create_store("valid.txt")
        # 加载测试集四元组，同上
        self.test_idx = self.create_store("test.txt")
        # 合并所有数据集（训练+验证+测试），用于全局时间窗口筛选
        self.all_idx = np.vstack((self.train_idx, self.valid_idx, self.test_idx))

        print("Grapher initialized.")  # 初始化完成提示

    def create_store(self, file):
        """
        加载指定文件中的四元组，完成三步预处理：
        1. 分割字符串四元组 → 列表格式
        2. 名称→ID映射 → 数值索引格式
        3. 添加逆四元组 → 扩充数据集（支持双向规则匹配）

        Parameters:
            file (str): 数据集文件名（如train.txt/valid.txt/test.txt）
        Returns:
            store_idx (np.ndarray): 预处理后的四元组索引数组，shape=(N,4)，每行=[头节点ID, 关系ID, 尾节点ID, 时间戳ID]
        """
        # 1. 读取文件中的所有四元组字符串
        with open(self.dataset_dir + file, "r", encoding="utf-8") as f:
            quads = f.readlines()  # 每行格式："subject\trelation\tobject\ttimestamp\n"

        # 2. 分割四元组字符串为列表（去除换行符，按制表符分割）
        store = self.split_quads(quads)

        # 3. 将四元组中的名称（字符串）转换为ID（数值）
        store_idx = self.map_to_idx(store)

        # 4. 为每个四元组添加对应的逆四元组（核心：支持反向规则匹配）
        store_idx = self.add_inverses(store_idx)

        return store_idx

    def split_quads(self, quads):
        """
        将原始字符串四元组分割为结构化列表

        Parameters:
            quads (list): 原始四元组字符串列表，每行格式："subject\trelation\tobject\ttimestamp\n"
        Returns:
            split_q (list): 结构化四元组列表，每个元素格式：[subject, relation, object, timestamp]（均为字符串）
        """
        split_q = []
        for quad in quads:
            # 去除最后一个字符（换行符\n），按制表符\t分割为四个部分
            split_q.append(quad[:-1].split("\t"))

        return split_q

    def map_to_idx(self, quads):
        """
        将字符串格式的四元组转换为数值ID格式（核心：方便后续数值计算和索引）

        Parameters:
            quads (list): 结构化四元组列表，每个元素：[subject, relation, object, timestamp]（字符串）
        Returns:
            quads (np.ndarray): 数值ID格式的四元组数组，shape=(N,4)，类型为int
        """
        # 提取所有头节点名称，转换为ID
        subs = [self.entity2id[x[0]] for x in quads]
        # 提取所有关系名称，转换为ID（包含正向/逆向关系）
        rels = [self.relation2id[x[1]] for x in quads]
        # 提取所有尾节点名称，转换为ID
        objs = [self.entity2id[x[2]] for x in quads]
        # 提取所有时间戳名称，转换为ID
        tss = [self.ts2id[x[3]] for x in quads]

        # 将四个列表按列拼接为numpy数组（每行是一个四元组的ID）
        quads = np.column_stack((subs, rels, objs, tss))

        return quads

    def add_inverses(self, quads_idx):
        """
        为每个四元组添加对应的逆四元组（核心设计：TLogic支持双向规则匹配）
        示例：原四元组 (s, r, o, t) → 逆四元组 (o, inv_r, s, t)
        其中 inv_r 是 r 的逆关系ID

        Parameters:
            quads_idx (np.ndarray): 数值ID格式的四元组数组，shape=(N,4)
        Returns:
            quads_idx (np.ndarray): 原四元组 + 逆四元组的合并数组，shape=(2N,4)
        """
        # 逆四元组的头节点 = 原四元组的尾节点
        subs = quads_idx[:, 2]
        # 逆四元组的关系 = 原关系的逆关系ID
        rels = [self.inv_relation_id[x] for x in quads_idx[:, 1]]
        # 逆四元组的尾节点 = 原四元组的头节点
        objs = quads_idx[:, 0]
        # 逆四元组的时间戳 = 原四元组的时间戳（时间不变）
        tss = quads_idx[:, 3]

        # 拼接逆四元组数组
        inv_quads_idx = np.column_stack((subs, rels, objs, tss))
        # 合并原四元组和逆四元组（行数翻倍）
        quads_idx = np.vstack((quads_idx, inv_quads_idx))

        return quads_idx