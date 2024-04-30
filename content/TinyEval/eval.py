import os
import json
import argparse
import numpy as np


from Eval.metrics import (
    qa_f1_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    rouge_zh_score
)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen2')
    return parser.parse_args(args)

dataset2metric = {
    'multifieldqa_zh': qa_f1_zh_score,
    'multi_news': rouge_score,
    'trec': classification_score,
    'custom_zh': rouge_zh_score
}

# 计算得分
def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec"]:
            prediction = prediction.lstrip('\n').split('\n')[0]  # 格式抽取
        if dataset in ['custom_zh', 'custom_en']:
            score = max(score, dataset2metric[dataset](prediction, ground_truths, all_classes=all_classes))
        else:
            score = max(score, dataset2metric.get(dataset, dataset2metric['custom_zh'])(prediction, ground_truths, all_classes=all_classes))
            # for ground_truth in ground_truths:
            #     score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

if __name__ == '__main__':
    scores = dict()
    args = parse_args()
    path = f"Eval/pred/{args.model}/"
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for file in all_files:
        if not file.endswith(".jsonl") or file == "result.json":
            continue
        predictions, answers, lengths = [], [], []
        dataset = file.split('.')[0]
        with open(f'{path}{file}', 'r', ) as f:
            for line in f:
                data = json.loads(line)  # str转为dict
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
            
            score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score

    # 保存结果
    out_path = f"Eval/pred/{args.model}/result.json"
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
