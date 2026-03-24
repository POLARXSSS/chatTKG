from openai import OpenAI
import os
from typing import Optional, Dict, List
from dotenv import load_dotenv
import time

# 加载.env文件
load_dotenv()


class LLMClient:
    def __init__(self,
                 model_name: str = "Qwen/Qwen3.5-27B",
                 api_keys: Optional[List[str]] = None):
        """
        初始化LLM客户端（支持多API Key自动切换）

        参数:
            model_name: ModelScope模型ID（默认Qwen3.5-35B），若包含"local"则标记为本地模式
            api_keys: API密钥列表（可选），优先使用传入值，否则从环境变量读取
                      环境变量格式：MODELSCOPE_API_KEY_1, MODELSCOPE_API_KEY_2, ...
        """
        self.model_name = model_name
        self.base_url = "https://api-inference.modelscope.cn/v1/"
        self.local_mode = "local" in model_name

        # 1. 加载多API Key
        self.api_keys = self._load_api_keys(api_keys)
        if not self.api_keys and not self.local_mode:
            raise ValueError(
                "至少需要提供一个API Key（可通过参数传入，或在.env中配置MODELSCOPE_API_KEY_1、MODELSCOPE_API_KEY_2等）"
            )

        # 2. 初始化当前使用的Key索引和失败Key列表
        self.current_key_index = 0  # 当前使用的Key索引
        self.failed_keys = set()  # 记录已失败的Key，避免重复尝试

        # 3. 重试配置
        self.max_retries = len(self.api_keys)  # 最大重试次数=Key的数量
        self.retry_delay = 1  # 切换Key后的重试延迟（秒）

    def _load_api_keys(self, api_keys: Optional[List[str]]) -> List[str]:
        """加载API Key列表（优先级：传入参数 > 环境变量）"""
        # 优先使用传入的Key列表
        if api_keys and len(api_keys) > 0:
            return [key.strip() for key in api_keys if key.strip()]

        # 从环境变量加载（支持MODELSCOPE_API_KEY_1、MODELSCOPE_API_KEY_2...）
        env_keys = []
        index = 1
        while True:
            key = os.getenv(f"MODELSCOPE_API_KEY_{index}")
            if not key:
                break
            env_keys.append(key.strip())
            index += 1

        # 兼容旧版单Key配置（MODELSCOPE_API_KEY）
        if not env_keys:
            single_key = os.getenv("MODELSCOPE_API_KEY")
            if single_key:
                env_keys.append(single_key.strip())

        return env_keys

    def prepare_for_inference(self):
        """准备模型进行推理"""
        if not self.api_keys and not self.local_mode:
            raise RuntimeError("❌ ModelScope API密钥未设置，请检查.env文件或传入参数")
        print(f"✅ ModelScope模型准备完成：{self.model_name}（共加载{len(self.api_keys)}个API Key）")

    def _switch_to_next_key(self) -> bool:
        """切换到下一个可用的API Key

        返回:
            bool: 是否切换成功（有可用Key返回True，否则False）
        """
        # 标记当前Key为失败
        current_key = self.api_keys[self.current_key_index]
        self.failed_keys.add(current_key)

        # 寻找下一个未失败的Key
        for i in range(len(self.api_keys)):
            next_index = (self.current_key_index + 1) % len(self.api_keys)
            next_key = self.api_keys[next_index]
            if next_key not in self.failed_keys:
                self.current_key_index = next_index
                print(f"🔄 切换到下一个API Key（索引：{self.current_key_index + 1}）")
                return True

        # 所有Key都已失败
        print("❌ 所有API Key都已尝试且失败，无可用Key")
        return False

    def _call_api_with_retry(self, messages: list[Dict[str, str]], temperature: float = 0.1) -> str:
        """调用API并自动重试（切换Key）"""
        retries = 0
        while retries < self.max_retries:
            # 获取当前可用的Key
            current_key = self.api_keys[self.current_key_index]

            try:
                # 构建客户端并调用API
                client = OpenAI(
                    api_key=current_key,
                    base_url=self.base_url
                )
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature
                )
                if response is None:
                    raise Exception("API 返回了 None")
                if not hasattr(response, 'choices') or len(response.choices) == 0:
                    raise Exception("API 返回格式无效：choices 为空")
                # 调用成功，重置失败Key列表（后续请求可重新使用之前失败的Key）
                content = response.choices[0].message.content
                if content is None or content.strip() == "":
                    raise Exception("API 返回空内容")
                self.failed_keys = set()
                return content.strip()

            except Exception as e:
                error_msg = str(e).lower()
                # 判断是否是需要切换Key的错误（额度用完、鉴权失败、接口限流等）
                if any(keyword in error_msg for keyword in
                       ["quota", "limit", "expire", "unauthorized", "401", "403", "429"]):
                    print(f"❌ 当前API Key（索引：{self.current_key_index + 1}）调用失败：{e}")
                    # 尝试切换Key
                    if not self._switch_to_next_key():
                        raise RuntimeError("所有API Key都已失败，无法继续调用")
                    # 切换后延迟重试
                    time.sleep(self.retry_delay)
                else:
                    # 非Key相关错误（如网络问题），直接抛出
                    raise RuntimeError(f"API调用失败（非Key问题）：{str(e)}")

            retries += 1

        raise RuntimeError(f"已重试{self.max_retries}次，所有API Key均调用失败")

    def generate_sentence(self, prompt: str) -> str:
        """调用ModelScope API生成回复（单轮对话，自动重试）"""
        try:
            messages = [{"role": "user", "content": prompt}]
            result = self._call_api_with_retry(messages)
            print(f"✅ 生成结果：{result[:50]}..." if len(result) > 50 else f"✅ 生成结果：{result}")
            return result
        except Exception as e:
            print(f"❌ 生成回复最终失败：{str(e)}")
            return ""

    def chat_completion(self, messages: list[Dict[str, str]], temperature: float = 0.1) -> str:
        """通用LLM对话接口（多轮对话，自动重试）"""
        try:
            return self._call_api_with_retry(messages, temperature)
        except Exception as e:
            raise RuntimeError(f"Cloud LLM call failed after retries: {str(e)}")



# # 1. 初始化客户端（自动从.env加载多Key）
# client = LLMClient(model_name="Qwen/Qwen3.5-35B-A3B")
# client.prepare_for_inference()
#
# # 2. 单轮对话调用（自动重试+切换Key）
# prompt = "请生成10条家庭关系规则，格式为：祖父(X,Y) <-- 父亲(X,Z), 父亲(Z,Y)"
# result = client.generate_sentence(prompt)
# print("最终生成结果：", result)
#
# # 3. 多轮对话调用（自动重试+切换Key）
# messages = [
#     {"role": "system", "content": "你是一个知识图谱规则专家，只返回规则，不解释"},
#     {"role": "user", "content": "生成5条关于母子关系的规则"}
# ]
# try:
#     result = client.chat_completion(messages, temperature=0.1)
#     print("多轮对话结果：", result)
# except RuntimeError as e:
#     print(f"多轮对话最终失败：{e}")