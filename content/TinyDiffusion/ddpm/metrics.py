import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
import numpy as np
from scipy import linalg
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset


class InceptionStatistics:
    def __init__(self, device='cuda'):
        self.device = device
        # 加载预训练的Inception v3模型
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        self.model.fc = nn.Identity()  # 移除最后的全连接层
        self.model = self.model.to(device)
        self.model.eval()
        
        # 设置图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    @torch.no_grad()
    def get_features(self, images):
        """获取Inception特征"""
        features = []
        probs = []
        
        # 将图像处理为299x299大小
        images = self.preprocess(images)
        
        # 批量处理图像
        dataset = TensorDataset(images)
        dataloader = DataLoader(dataset, batch_size=32)
        
        for (batch,) in tqdm(dataloader):
            batch = batch.to(self.device)
            
            # 获取特征和logits
            feature = self.model(batch)
            prob = F.softmax(feature, dim=1)
            
            features.append(feature.cpu().numpy())
            probs.append(prob.cpu().numpy())
            
        features = np.concatenate(features, axis=0)
        probs = np.concatenate(probs, axis=0)
        
        return features, probs

def calculate_inception_score(probs, splits=10):
    """计算Inception Score
    
    IS = exp(E[KL(p(y|x) || p(y))])
    
    其中:
    - p(y|x) 是生成图像通过Inception模型得到的条件类别分布(probs)
    - p(y) 是边缘类别分布,通过对所有图像的p(y|x)取平均得到
    - KL是KL散度,用于衡量两个分布的差异
    - E是对所有图像的期望
    
    具体步骤:
    1. 将所有图像分成splits组
    2. 对每组计算:
       - 计算边缘分布p(y)
       - 计算KL散度
       - 取指数
    3. 返回所有组得分的均值和标准差
    """
    # 存储每个split的IS分数
    scores = []
    # 计算每个split的大小
    split_size = probs.shape[0] // splits
    
    # 对每个split进行计算
    for i in tqdm(range(splits)):
        # 获取当前split的概率分布
        part = probs[i * split_size:(i + 1) * split_size]
        # 计算KL散度: KL(p(y|x) || p(y))
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, axis=0), 0)))
        # 对每个样本的KL散度求平均
        kl = np.mean(np.sum(kl, axis=1))
        # 计算exp(KL)并添加到scores列表
        scores.append(np.exp(kl))
        
    # 返回所有split的IS分数的均值和标准差
    return np.mean(scores), np.std(scores)

def calculate_fid(real_features, fake_features):
    """计算Fréchet Inception Distance (FID)分数
    
    FID = ||μ_r - μ_f||^2 + Tr(Σ_r + Σ_f - 2(Σ_r Σ_f)^(1/2))
    
    其中:
    - μ_r, μ_f 分别是真实图像和生成图像特征的均值向量
    - Σ_r, Σ_f 分别是真实图像和生成图像特征的协方差矩阵
    - Tr 表示矩阵的迹(对角线元素之和)
    - ||·||^2 表示欧几里得距离的平方
    
    FID越小表示生成图像的质量越好,分布越接近真实图像
    """
    # 计算真实图像和生成图像特征的均值向量和协方差矩阵
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    # 计算均值向量之间的欧几里得距离的平方
    ssdiff = np.sum((mu1 - mu2) ** 2)
    # 计算协方差矩阵的平方根项:(Σ_r Σ_f)^(1/2)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))  # 耗时较长
    # 如果结果包含复数,取其实部
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # 计算最终的FID分数
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def evaluate_model(model, scheduler, train_loader, num_samples, batch_size, image_size, device="cuda"):
    """评估模型的IS和FID分数"""
    # 生成样本
    fake_images = []
    num_batches = num_samples // batch_size  # 每批生成batch_size张图片
    
    print(f"生成{num_samples}张图像...")
    for _ in tqdm(range(num_batches)):
        fake_batch = sample(model, scheduler, batch_size, (3, image_size, image_size), device)
        fake_batch = ((fake_batch + 1) / 2)  # 转换到[0,1]范围
        fake_images.append(fake_batch.cpu())
    
    fake_images = torch.cat(fake_images, dim=0)

    # 收集所有真实图像
    print("收集真实图像...")
    real_images = []
    for batch in tqdm(train_loader):
        real_images.append(batch[0])
    real_images = torch.cat(real_images, dim=0)

    # 初始化Inception模型
    inception = InceptionStatistics(device=device)
    
    # 获取真实图像和生成图像的特征
    print("计算真实图像特征...")
    real_features, real_probs = inception.get_features(real_images)
    print("计算生成图像特征...")
    fake_features, fake_probs = inception.get_features(fake_images)
    
    # 计算IS分数
    print("计算IS分数...")
    is_score, is_std = calculate_inception_score(fake_probs)
    
    # 计算FID分数
    print("计算FID分数...")
    fid_score = calculate_fid(real_features, fake_features)
    
    return {
        "is_score": is_score,
        "is_std": is_std,
        "fid_score": fid_score
    }

if __name__ == "__main__":
    from unet import SimpleUnet
    from diffusion import NoiseScheduler
    from sample import sample
    from dataloader import load_transformed_dataset
    
    # 加载模型和数据
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 32
    model = SimpleUnet()
    model.load_state_dict(torch.load(f"simple-unet-ddpm-{image_size}.pth", weights_only=True))
    model = model.to(device)
    model.eval()
    
    scheduler = NoiseScheduler().to(device)
    
    # 加载真实图像数据
    train_loader, _ = load_transformed_dataset(image_size, batch_size=128)
    
    # 评估模型
    metrics = evaluate_model(
        model=model,
        scheduler=scheduler,
        train_loader=train_loader,  # 传入整个train_loader
        num_samples=10000,  # 生成10000张图片进行评估
        batch_size=100,
        image_size=image_size,
        device=device
    )
    
    print(f"Inception Score: {metrics['is_score']:.2f} ± {metrics['is_std']:.2f}")
    print(f"FID Score: {metrics['fid_score']:.2f}")
