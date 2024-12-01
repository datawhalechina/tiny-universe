# Tiny DDPM

## Usage

```
pip install -r requirements.txt
```

* 下载 CIFAR-10 数据集到 `datasets/` 文件夹，保持目录结构为 `datasets/cifar-10-batches-py`
* 训练模型： `python ddpm/train.py`
* 采样图像： `python ddpm/sample.py`
* 计算IS与FID： `python ddpm/metrics.py`
