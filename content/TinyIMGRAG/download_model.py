# 模型下载
import modelscope
import huggingface_hub

model_dir = modelscope.snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', cache_dir='./model/', revision='master')
model_dir = modelscope.snapshot_download('Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4', cache_dir='./model/', revision='master')
model_dir = modelscope.snapshot_download('stabilityai/stable-diffusion-xl-base-1.0', cache_dir='./model/', revision='master')
model_dir = huggingface_hub.snapshot_download(repo_id="h94/IP-Adapter", local_dir="./model/", max_workers=1)

