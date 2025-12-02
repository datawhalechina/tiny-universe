import torch
from modelscope.msdatasets import MsDataset
from modelscope import snapshot_download
import os


# model_dir = MsDataset.load('Qwen/Qwen2.5-Omni-3B', cache_dir='/share/workspace/shared_models/02_S2S_Model/Qwen2.5-Omni-3B', subset_name='master')
# model_dir = snapshot_download('sentence-transformers/all-mpnet-base-v2', cache_dir='/share/workspace/shared_models/01_LLM-models/all-mpnet-base-v2', revision='master')



# # 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# # 下载模型
os.system('huggingface-cli download --resume-download sentence-transformers/all-mpnet-base-v2 --local-dir /share/workspace/shared_models/01_LLM-models/all-mpnet-base-v2')
 
# 下载数据集
# os.system('huggingface-cli download --resume-download swulling/gsm8k_chinese --repo-type dataset --local-dir /share/workspace/shared_datasets/LLMdatasets/gsm8k_chinese')