import os
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
from Eval.model.LLM import internlm2Chat, Qwen2Chat

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen2')
    return parser.parse_args(args)

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()

    model2path = json.load(open("Eval/config/model2path.json", "r"))
    model2maxlen = json.load(open("Eval/config/model2maxlen.json", "r"))
    adapter2path = json.load(open("Eval/config/adapter2path.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name]

    # datasets = ["multi_news", "multifieldqa_zh", "trec"]
    datasets = ['test_rag']

    dataset2prompt = json.load(open("Eval/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("Eval/config/dataset2maxlen.json", "r"))
    pred_model = Qwen2Chat(model2path[model_name], model_name, adapter2path[model_name])
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")

    for dataset in datasets:
        data = load_dataset('json', data_files=f'Eval/dataset/{dataset}.jsonl',split='train')
        if not os.path.exists(f"Eval/pred/{model_name}"):
            os.makedirs(f"Eval/pred/{model_name}")
        out_path = f"Eval/pred/{model_name}/{dataset}.jsonl"
        if os.path.isfile(out_path):
            os.remove(out_path)
        prompt_format = dataset2prompt.get(dataset, dataset2prompt.get('custom_zh'))
        max_gen = dataset2maxlen.get(dataset, dataset2maxlen.get('custom_zh'))
        data_all = [data_sample for data_sample in data]

        pred_model.get_pred(data, max_length, max_gen, prompt_format, device, out_path)   

