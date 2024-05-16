import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os

model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm2-chat-7b', cache_dir='/root/autodl-tmp/', revision='master')
model_dir = snapshot_download('jinaai/jina-embeddings-v2-base-zh', cache_dir='/root/autodl-tmp/', revision='master')