import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial

import torch
from model import Transformer, ModelArgs
from preprocess import Task

# -----------------------------------------------------------------------------
# I/O 配置，用于定义输出目录和训练时的日志记录与评估设置
out_dir = "output"  # 模型输出保存路径
eval_interval = 2000  # 评估间隔步数
log_interval = 1  # 日志记录间隔步数
eval_iters = 100  # 每次评估时迭代的步数
eval_only = False  # 如果为True，脚本在第一次评估后立即退出
always_save_checkpoint = False  # 如果为True，在每次评估后总是保存检查点
init_from = "scratch"  # 可以选择从头开始训练（'scratch'）或从已有的检查点恢复（'resume'）

# 数据配置
batch_size = 8  # 每个微批次的样本数量，如果使用梯度累积，实际批次大小将更大
max_seq_len = 256  # 最大序列长度
vocab_size = 4096  # 自定义词汇表大小

# 模型配置
dim = 288  # 模型的隐藏层维度
n_layers = 8  # Transformer的层数
n_heads = 8  # 注意力头的数量
n_kv_heads = 4  # 模型分组
multiple_of = 32  # 在某些层的维度必须是该数的倍数
dropout = 0.0  # Dropout概率

# AdamW优化器配置
gradient_accumulation_steps = 4  # 梯度累积步数，用于模拟更大的批次
learning_rate = 5e-4  # 最大学习率
max_iters = 100000  # 总的训练迭代次数
weight_decay = 1e-1  # 权重衰减系数
beta1 = 0.9  # AdamW优化器的β1参数
beta2 = 0.95  # AdamW优化器的β2参数
grad_clip = 1.0  # 梯度裁剪阈值，0表示不裁剪

# 学习率衰减配置
decay_lr = True  # 是否启用学习率衰减
warmup_iters = 1000  # 学习率预热的步数

# 系统设置
device = "cuda:0"  # 设备选择：'cpu'，'cuda'，'cuda:0'等
dtype = "bfloat16"  # 数据类型：'float32'，'bfloat16'，'float16'

# -----------------------------------------------------------------------------
# 获取配置参数的键值对，便于后续的日志记录
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
config = {k: globals()[k] for k in config_keys}  # 保存配置到字典中，便于日志记录
# -----------------------------------------------------------------------------

# 固定一些超参数的默认值
lr_decay_iters = max_iters  # 学习率衰减步数，设置为等于最大迭代步数
min_lr = 0.0  # 最小学习率，建议为学习率的十分之一
vocab_source = 'custom'  # 词汇表来源
master_process = True  # 用于区分主进程
seed_offset = 0  # 随机种子偏移量
ddp_world_size = 1  # 分布式数据并行的世界大小
tokens_per_iter = batch_size * max_seq_len  # 每次迭代处理的token数

# 设置随机种子，确保可重复性
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # 允许在matmul上使用tf32
torch.backends.cudnn.allow_tf32 = True  # 允许在cudnn上使用tf32
device_type = "cuda" if "cuda" in device else "cpu"  # 用于自动选择设备类型
ptdtype = torch.float16  # 设置训练时使用的数据类型

# 混合精度训练相关
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# 为特定任务设置批次迭代器 iter_batches
iter_batches = partial(
    Task.iter_batches,  # 调用 Task 类中的 iter_batches 方法
    batch_size=batch_size,  # 每个批次的样本数量
    max_seq_len=max_seq_len,  # 每个序列的最大长度
    vocab_size=vocab_size,  # 词汇表大小
    vocab_source=vocab_source,  # 词汇表来源（如 llama2 或 custom）
    device=device,  # 运行模型的设备（如 GPU 或 CPU）
    num_workers=0,  # 用于数据加载的 worker 数量，0 表示在主线程中加载
)

# 训练迭代数初始化
iter_num = 0  # 记录当前迭代数

# 验证集上的最好损失初始值设置为一个极大值，用于后续模型验证时对比更新
best_val_loss = 1e9  # 设置初始的最佳验证损失为非常大的值，以便在训练中更新

# 模型初始化参数设置
model_args = dict(
    dim=dim,  # 模型的隐藏层维度
    n_layers=n_layers,  # Transformer 的层数
    n_heads=n_heads,  # 多头注意力机制中的头数
    n_kv_heads=n_kv_heads,  # 分组数（可能是用于并行化或其他优化目的）
    vocab_size=vocab_size,  # 词汇表大小
    multiple_of=multiple_of,  # 用于调整某些维度的参数，确保其为特定数的倍数
    max_seq_len=max_seq_len,  # 最大序列长度
    dropout=dropout,  # dropout 概率，用于防止过拟合
)

# ===========================================================
# 模型初始化
gptconf = ModelArgs(**model_args)
model = Transformer(gptconf)


model.to(device)

# 初始化 GradScaler，用于自动混合精度训练（AMP）
# 如果 enabled=False，表示禁用混合精度，scaler 将不起作用
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# 优化器初始化，调用模型的 configure_optimizers 方法
optimizer = model.configure_optimizers(
    weight_decay,  # 权重衰减（L2 正则化）
    learning_rate,  # 学习率
    (beta1, beta2),  # Adam 优化器中的 beta1 和 beta2 参数
    device_type  # 当前训练设备（如 GPU 或 CPU）
)

# 定义评估损失的流程
@torch.no_grad()  # 使用 no_grad 装饰器，确保在评估过程中不计算梯度，从而节省内存
def estimate_loss():
    out = {}  # 用于存储训练集和验证集上的平均损失
    model.eval()  # 将模型设置为评估模式，这会影响 dropout 和 batchnorm 等层的行为
    for split in ["train", "val"]:  # 分别对训练集和验证集进行评估
        batch_iter = iter_batches(split=split)  # 获取对应数据集的批次迭代器
        losses = torch.zeros(eval_iters)  # 初始化一个张量用于存储多次迭代的损失，放在 CPU 上
        for k in range(eval_iters):  # 进行多次迭代以计算平均损失
            X, Y = next(batch_iter)  # 从迭代器中获取下一个批次的输入数据 X 和标签 Y
            with ctx:  # 上下文管理器，可以是 torch.autocast()，用于自动混合精度训练
                logits = model(X, Y)  # 前向传播，计算模型的输出
                loss = raw_model.last_loss  # 从模型中获取损失值
            losses[k] = loss.item()  # 将损失值转换为 Python 标量并存储在 losses 张量中
        out[split] = losses.mean()  # 计算当前数据集上的平均损失并保存到字典中
    model.train()  # 恢复模型为训练模式
    return out  # 返回包含训练集和验证集平均损失的字典

# 定义学习率调度函数
def get_lr(it):
    """
    根据当前的训练迭代步数 it 返回当前的学习率值。
    学习率调整策略包括线性预热、余弦退火和最小学习率限制。
    """
    # 1) 线性预热阶段，在 warmup_iters 之前，学习率线性增加到目标学习率
    if it < warmup_iters:
        return learning_rate * it / warmup_iters  # 预热阶段，学习率线性增长

    # 2) 如果迭代步数超过 lr_decay_iters，返回最小学习率 min_lr
    if it > lr_decay_iters:
        return min_lr  # 训练进入尾声时，学习率达到最小值并保持不变

    # 3) 余弦退火阶段，在 warmup_iters 和 lr_decay_iters 之间，学习率逐渐降低
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1  # 确保衰减比在合法范围内
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 余弦函数计算衰减系数，范围为0到1
    return min_lr + coeff * (learning_rate - min_lr)  # 根据衰减系数调整学习率

# 初始化训练数据的迭代器
train_batch_iter = iter_batches(split="train")
X, Y = next(train_batch_iter)  # 获取第一个批次的数据
t0 = time.time()  # 记录开始时间
local_iter_num = 0  # 本进程中的迭代次数
raw_model = model  # 如果使用了分布式数据并行 (DDP)，需要解包模型
running_mfu = -1.0  # 初始化模型浮点运算利用率

os.makedirs(out_dir, exist_ok=True)

while True:
    # 或许当前step的学习率
    lr = get_lr(iter_num) if decay_lr else learning_rate
    # 更新优化器中的学习率
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # 在指定的评估间隔进行模型评估和保存检查点
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()  # 评估当前模型在训练集和验证集上的损失
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # 如果验证损失降低，或者设置为始终保存检查点，则保存模型
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                # 创建检查点字典，包含模型状态、优化器状态和其他信息
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")
                # 保存检查点到指定目录
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
    # 如果只进行评估且已经完成第一次迭代，则退出循环
    if iter_num == 0 and eval_only:
        break

    # 前向和反向传播过程，支持梯度累积
    for micro_step in range(gradient_accumulation_steps):

        with ctx:  # 混合精度训练的上下文管理器
            logits = model(X, Y)  # 前向传播，计算模型输出
            loss = raw_model.last_loss  # 获取模型的损失值
            loss = loss / gradient_accumulation_steps  # 平均损失以支持梯度累积

        X, Y = next(train_batch_iter)  # 获取下一个批次的数据
        # 反向传播，计算梯度
        scaler.scale(loss).backward()
    # 梯度处理阶段
    if grad_clip != 0.0:
        # 取消梯度缩放以进行梯度裁剪
        scaler.unscale_(optimizer)
        # 对梯度进行裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # 更新优化器和梯度缩放器（用于混合精度训练）
    scaler.step(optimizer)
    scaler.update()
    # 清空优化器的梯度，释放显存
    optimizer.zero_grad(set_to_none=True)

    # 计时和日志记录
    t1 = time.time()
    dt = t1 - t0  # 计算一次迭代所需时间
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # 获取当前损失值，并根据梯度累积步骤进行调整
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # 让训练循环先运行几个迭代再计算模型利用率
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            # 使用滑动平均更新模型浮点运算利用率（MFU）
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%"
            # mfu 表示模型浮点运算利用率
        )
    iter_num += 1  # 全局迭代次数自增
    local_iter_num += 1  # 本地迭代次数自增

    # 终止条件，达到最大迭代次数则退出循环
    if iter_num > max_iters:
        break