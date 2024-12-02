import torch
from torch import nn
import math


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, up=False):
        """UNet中的基本Block模块,包含时间嵌入和上/下采样功能"""
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        if up:
            self.conv1 = nn.Conv2d(2 * in_channels, out_channels, kernel_size=3, padding=1)
            self.transform = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.transform = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.bnorm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # 第一次卷积
        h = self.bnorm1(self.relu(self.conv1(x)))
        # 时间嵌入
        time_emb = self.relu(self.time_mlp(t))
        # 将时间信息注入特征图
        h = h + time_emb[..., None, None]
        # 第二次卷积
        h = self.bnorm2(self.relu(self.conv2(h)))
        # 上采样或下采样
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    """使用正弦位置编码实现时间步的嵌入,参考Transformer中的位置编码方法,使用正余弦函数将时间步映射到高维空间"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        # 将维度分成两半,分别用于sin和cos
        half_dim = self.dim // 2
        # 计算不同频率的指数衰减
        embeddings = math.log(10000) / (half_dim - 1)
        # 生成频率序列
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # 将时间步与频率序列相乘
        embeddings = time[:, None] * embeddings[None, :]
        # 拼接sin和cos得到最终的嵌入向量
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SimpleUnet(nn.Module):
    """简单的UNet模型,用于扩散模型的噪声预测"""
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3
        time_emb_dim = 32

        # 时间嵌入层
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # 输入层、下采样层、上采样层和输出层
        self.input = nn.Conv2d(image_channels, down_channels[0], kernel_size=3, padding=1)
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i + 1], time_emb_dim) for i in range(len(down_channels) - 1)])
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True) for i in range(len(up_channels) - 1)])
        self.output = nn.Conv2d(up_channels[-1], out_dim, kernel_size=3, padding=1)

    def forward(self, x, time_step):
        # 时间步嵌入
        t = self.time_embed(time_step)
        # 初始卷积
        x = self.input(x)
        # UNet前向传播:先下采样收集特征,再上采样恢复分辨率
        residual_stack = []
        for down in self.downs:
            x = down(x, t)
            residual_stack.append(x)
        for up in self.ups:
            residual_x = residual_stack.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)


def print_shapes(model, x, time_step):
    print("Input shape:", x.shape)
    
    # 时间步嵌入
    t = model.time_embed(time_step)
    print("Time embedding shape:", t.shape)
    
    # 初始卷积
    x = model.input(x)
    print("After input conv shape:", x.shape)
    
    #下采样过程
    residual_stack = []
    print("\nDownsampling process:")
    for i, down in enumerate(model.downs):
        x = down(x, t)
        residual_stack.append(x)
        print(f"Down block {i+1} output shape:", x.shape)
    
    # 上采样过程
    print("\nUpsampling process:")
    for i, up in enumerate(model.ups):
        residual_x = residual_stack.pop()
        x = torch.cat((x, residual_x), dim=1)
        print(f"Concatenated input shape before up block {i+1}:", x.shape)
        x = up(x, t)
        print(f"Up block {i+1} output shape:", x.shape)
    
    # 最终输出
    output = model.output(x)
    print("\nFinal output shape:", output.shape)
    return output


if __name__ == "__main__":
    model = SimpleUnet()
    x = torch.randn(1, 3, 32, 32)
    time_step = torch.tensor([10])
    print_shapes(model, x, time_step)