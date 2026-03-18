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

    # def _local_chat_completion(self, messages: list[Dict[str, str]], temperature: float = 0.1) -> str:
    #     """本地LLM（如LLaMA/ChatGLM）调用逻辑"""
    #     if not hasattr(self, "tokenizer"):
    #         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    #         self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")
    #     # 拼接prompt（适配本地模型格式）
    #     prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    #     inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
    #     outputs = self.model.generate(**inputs, temperature=temperature, max_new_tokens=1024)
    #     return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()