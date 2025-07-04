from zhipuai import ZhipuAI
from typing import List
from .base import BaseEmb


class zhipuEmb(BaseEmb):
    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.client = ZhipuAI(api_key=api_key)

    def get_emb(self, text: str) -> List[float]:
        emb = self.client.embeddings.create(
            model=self.model_name,
            input=text,
        )
        return emb.data[0].embedding
