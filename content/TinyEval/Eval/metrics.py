import re
import string
import jieba
from rouge import Rouge
from collections import Counter
jieba.setLogLevel(jieba.logging.INFO)


def normalize_zh_aswer(s):
    """小写化,删除标点,删除空格"""

    def white_space_fix(text):
        return "".join(text.split())
    
    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return ''.join(ch for ch in text if ch not in all_punctuation)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_punc(lower(s)))

def normalize_en_answer(s):
    """小写化,删除标点,删除冠词和多余空白."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def classification_score(prediction, ground_truth, **kwargs):
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:                                   # 总类别里面的类别是否在预测中出现
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:  # 如果预测中的类别在答案中出现，但是不是答案  'two step'--'step'
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        score = (1.0 / len(em_match_list))
    else:
        score = 0.0
    return score

def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]

def rouge_zh_score(prediction, ground_truth, **kwargs):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False))) 
    score = rouge_score(prediction, ground_truth)
    return score

def f1_score(prediction, ground_truth, **kwargs):
    # Counter以dict的形式存储各个句子对应的词与其对应个数,&操作符返回两个Counter中共同的元素的键值对
    common = Counter(prediction) & Counter(ground_truth)  
    num_same = sum(common.values())                       # 显示prediction与gt的共同元素的个数
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)          # 即模型预测正确的样本数量与总预测样本数量的比值
    recall = 1.0 * num_same / len(ground_truth)           # 模型正确预测的样本数量与总实际样本数量的比值
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_en_answer(prediction)
    normalized_ground_truth = normalize_en_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)

def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens_norm = [normalize_zh_aswer(t) for t in prediction_tokens]
    ground_truth_tokens_norm = [normalize_zh_aswer(t) for t in ground_truth_tokens]
    prediction_tokens = [t for t in prediction_tokens_norm if len(t) > 0]
    ground_truth_tokens = [t for t in ground_truth_tokens_norm if len(t) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)

def GAOKAO_math(prediction, ground_truth, **kwargs):
    score = 0
    # 判断是单选还是多选
    if len(ground_truth) > 1:
        # 多选
        pattern = r"[A-D]"
        matches = re.findall(pattern, prediction)
        predicted_answer = ''
        
        if matches:
            # 从后往前匹配大写字母，且满足之间长度不超过10个字符的条件
            reversed_prediction = prediction[::-1]
            if len(matches) > 1:
                # 从后往前遍历匹配项
                for i, match in enumerate(matches):
                    if i == 0:
                        predicted_answer += match
                    else:
                        # 计算当前匹配项与上一个匹配项之间的距离
                        distance = reversed_prediction.find(matches[i-1]) - reversed_prediction.find(match) - 1
                        # 如果距离大于5，则停止添加更多的选项
                        if distance > 5:
                            break
                        predicted_answer += match
                # 将预测答案排序并去重
                predicted_answer = ''.join(sorted(set(predicted_answer)))
            # 计算得分
            if predicted_answer == ground_truth:
                score = 1
            elif all(option in ground_truth for option in predicted_answer) and len(predicted_answer) < len(ground_truth):
                score = 0.5
    else:
        # 单选
        pattern = r"[A-D]"
        matches = re.findall(pattern, prediction)
        if matches and matches[-1] == ground_truth:
            score = 1
    
    return score

    
    
