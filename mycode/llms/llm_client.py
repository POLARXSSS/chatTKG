from openai import OpenAI
import os
from typing import Optional, Dict
# from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
load_dotenv()
class LLMClient:
    def __init__(self, model_name: str = "Qwen/Qwen3.5-35B-A3B", api_key: Optional[str] = None):
        self.model_name = model_name
        self.base_url = "https://api-inference.modelscope.cn/v1/"
        self.api_key = api_key or os.getenv("MODELSCOPE_API_KEY")
        if not self.api_key and "local" not in model_name:
            raise ValueError("API key must be provided for cloud LLM (set MODELSCOPE_API_KEY env)")
        OpenAI.api_key = self.api_key
        self.local_mode = "local" in model_name
    def prepare_for_inference(self):
        """准备模型进行推理"""
        if not self.api_key:
            raise RuntimeError("❌ ModelScope API密钥未设置，请通过 --modelscope_api_key 参数提供")
        print(f"✅ ModelScope模型准备完成：{self.model_name}")
    def generate_sentence(self, prompt):
        """调用ModelScope API生成回复"""
        # 构建API请求
        client = OpenAI(
            api_key=self.api_key,  # 请替换成您的ModelScope Access Token
            base_url=self.base_url
        )
        try:
            # 发送请求
            response = client.chat.completions.create(
                model=self.model_name,  # ModelScope Model-Id
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
            )

            result = response.choices[0].message.content.strip()  # type: ignore
            print(result)
            return result
        except Exception as e:
            print(f"❌ 生成回复时出错：{str(e)}")
            return ""


    def chat_completion(self, messages: list[Dict[str, str]], temperature: float = 0.1) -> str:
        """通用LLM对话接口（兼容云端/本地模型）"""
        client = OpenAI(
            api_key=self.api_key,  # 请替换成您的ModelScope Access Token
            base_url=self.base_url
        )
        # if self.local_mode:
        #     return self._local_chat_completion(messages, temperature)
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Cloud LLM call failed: {str(e)}")

