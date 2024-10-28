import glob
import json
import os
from tqdm import tqdm
import requests
import sentencepiece as spm
import argparse

DATA_CACHE_DIR = 'data'

def download_file(url: str, fname: str, chunk_size=1024):
    """发送HTTP GET请求以流式方式获取文件"""
    resp = requests.get(url, stream=True)
    
    # 获取文件的总大小（以字节为单位），默认为0如果没有提供'content-length'头信息
    total = int(resp.headers.get("content-length", 0))
    
    # 以写二进制模式打开一个文件以保存下载的内容
    with open(fname, "wb") as file, tqdm(
        desc=fname,           # 进度条前面的描述信息（通常是文件名）
        total=total,          # 总的字节数，用于设置进度条的总长度
        unit="iB",            # 进度条的单位，'iB'代表二进制字节
        unit_scale=True,      # 启用单位缩放，如KB、MB等
        unit_divisor=1024,    # 设置单位换算的除数，这里为1024
    ) as bar:
        # 逐块读取响应内容并写入文件
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)  # 写入数据块到文件
            bar.update(size)         # 更新进度条

def download():
    """在DATA_CACHE_DIR中创建目录，如果目录不存在则创建"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # 定义TinyStories数据集的下载URL和保存的文件名
    data_url = "https://www.modelscope.cn/datasets/AI-ModelScope/TinyStories/resolve/master/TinyStories_all_data.tar.gz"
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
    
    # 检查数据集是否已经下载，如果没有下载则进行下载
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)  # 使用之前定义的download_file函数进行下载
    else:
        print(f"{data_filename} already exists, skipping download...")

    # 定义解压缩后的数据目录
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    
    # 检查数据目录是否存在，如果不存在则解压缩数据集
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)  # 创建数据目录
        print(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")  # 使用系统命令解压缩.tar.gz文件
    else:
        print(f"{data_dir} already exists, skipping unpacking...")

    # 查找解压后的所有JSON文件，排序后获取文件名列表
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    
    # 打开第一个JSON文件并读取内容
    with open(shard_filenames[0], "r") as f:
        data = json.load(f)  # 将JSON文件内容加载到变量data中
    
    print("Download done.")  # 下载完成信息
    print(f"Number of shards: {len(shard_filenames)}")  # 打印解压后数据分片的数量
    print(f"Example story:\n{data[0]}")  # 打印第一个分片中的一个示例故事

def load_text_from_files(path):
    path_list = glob.glob(path)
    text_data = []
    for file_path in path_list:
        with open(file_path, 'r', encoding='utf-8') as file:
            text_data.extend(file.readlines())
    return text_data

def batch_iterator(text_data, batch_size=648):
    for i in range(0, len(text_data), batch_size):
        yield text_data[i:i + batch_size]

def train_vocab(vocab_size: int=32000, num_shards: int=20):
    """
    vocab_size: int, 词汇表的大小，决定分词器的词汇量。
    num_shards: int, 用于加快词汇表训练的效率，指定要处理的分片数量。
    """
    # 确保词汇表大小为正数
    assert vocab_size > 0, "Vocab size must be positive"

    # SentencePiece 模型的前缀路径，将用于保存分词器
    prefix = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")

    # 1) 将多个分片中的文本导出为单个文本文件 tiny.txt
    tiny_file = os.path.join(DATA_CACHE_DIR, "tiny.txt")
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    # 创建 tiny.txt 文件并写入指定数量的分片中的文本
    print(f"Writing temporary file {tiny_file} with {num_shards} shards...")
    with open(tiny_file, "w", encoding="utf-8") as of:
        # 遍历前 num_shards 个分片
        for shard in tqdm(shard_filenames[:num_shards]):
            with open(shard, "r") as f:
                data = json.load(f)  # 读取分片中的JSON数据
            # 遍历每个例子，将其中的故事文本写入 tiny.txt 文件
            for example in data:
                text = example["story"]
                text = text.strip()  # 去除文本首尾的空白字符
                of.write(text + "\n")  # 每个文本写入一行

    # 输出生成的 tiny.txt 文件的大小
    print(f"Size is: {os.path.getsize(tiny_file) / 1024 / 1024:.2f} MB")

    # 2) 使用 SentencePiece 训练分词器
    print("Will now train the vocab...")
    spm.SentencePieceTrainer.train(
        input=tiny_file,         # 输入文件为之前生成的 tiny.txt
        model_prefix=prefix,     # 模型前缀路径
        model_type="bpe",        # 使用 Byte-Pair Encoding (BPE) 训练分词器
        vocab_size=vocab_size,   # 词汇表大小
        self_test_sample_size=0, # 自测样本大小设置为 0
        input_format="text",     # 输入文件格式为纯文本
        character_coverage=1.0,  # 覆盖所有字符（包括非常见字符）
        num_threads=os.cpu_count(),  # 使用 CPU 的线程数
        split_digits=True,       # 拆分数字
        allow_whitespace_only_pieces=True,  # 允许仅由空格组成的词元
        byte_fallback=True,      # 启用字节级回退
        unk_surface=r" \342\201\207 ",  # UNK token 表示未知字符的方式
        normalization_rule_name="identity"  # 使用“identity”归一化规则
    )

    # 3) 可选的清理操作，询问用户是否删除临时文件 tiny.txt
    dec = input(f"Delete the temporary file {tiny_file}? [y/N] ")
    if dec.lower() == "y":
        os.remove(tiny_file)  # 删除临时文件
        print(f"Deleted {tiny_file}")

    # 输出模型保存的路径
    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", type=bool, default=True, help="download the dataset")
    parser.add_argument("--vocab_size", type=int, default=4096, help="vocab size")
    args = parser.parse_args()
    if args.download:
        download()
    train_vocab(args.vocab_size)