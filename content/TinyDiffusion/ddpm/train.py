import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm

from diffusion import NoiseScheduler
from unet import SimpleUnet
from dataloader import load_transformed_dataset
from sample import sample, plot


def test_step(model, dataloader, noise_scheduler, criterion, epoch, num_epochs, device):
    """测试步骤,计算测试集上的损失"""
    model.eval()
    with torch.no_grad():
        loss_sum = 0
        num_batches = 0
        pbar = tqdm(dataloader)
        for batch in pbar:
            images, _ = batch
            images = images.to(device)
            t = torch.full((images.shape[0],), noise_scheduler.num_steps-1, device=device)
            noisy_images, noise = noise_scheduler.add_noise(images, t)

            predicted_noise = model(noisy_images, t)
            loss = criterion(noise, predicted_noise)
            loss_sum += loss.item()
            num_batches += 1
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {loss_sum/num_batches:.4f}")
        return loss_sum / len(dataloader)


def train_step(model, dataloader, noise_scheduler, criterion, optimizer, epoch, num_epochs, device):
    """训练步骤,计算训练集上的损失并更新模型参数"""
    # 设置模型为训练模式
    model.train()
    loss_sum = 0
    num_batches = 0
    pbar = tqdm(dataloader)
    for batch in pbar:
        # 获取一个batch的图像数据并移至指定设备
        images, _ = batch
        images = images.to(device)
        
        # 随机采样时间步t
        t = torch.randint(0, noise_scheduler.num_steps, (images.shape[0],), device=device)
        
        # 对图像添加噪声,获得带噪声的图像和噪声
        noisy_images, noise = noise_scheduler.add_noise(images, t)

        # 使用模型预测噪声
        predicted_noise = model(noisy_images, t)
        
        # 计算预测噪声和真实噪声之间的MSE损失
        loss = criterion(noise, predicted_noise)
        
        # 反向传播和优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 计算梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪,防止梯度爆炸
        optimizer.step()  # 更新参数

        # 累计损失并更新进度条
        loss_sum += loss.item()
        num_batches += 1
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss_sum/num_batches:.4f}")
        
    # 返回平均损失
    return loss_sum / len(dataloader)


def train(model, train_loader, test_loader, noise_scheduler, criterion, optimizer, device, num_epochs=100, img_size=32):
    """训练模型"""
    for epoch in range(num_epochs):
        train_loss = train_step(model, train_loader, noise_scheduler, criterion, optimizer, epoch, num_epochs, device)
        test_loss = test_step(model, test_loader, noise_scheduler, criterion, epoch, num_epochs, device)
        if epoch % 10 == 0:
            # 采样10张图像
            images = sample(model, noise_scheduler, 10, (3, img_size, img_size), device)
            # 将图像从[-1, 1]范围缩放到[0, 1]范围,以便可视化
            images = ((images + 1) / 2).detach().cpu()
            fig = plot(images)
            fig.savefig(f"samples/epoch_{epoch}.png")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader = load_transformed_dataset(args.img_size, args.batch_size)
    noise_scheduler = NoiseScheduler().to(device)
    model = SimpleUnet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    model = train(model, train_loader, test_loader, noise_scheduler, criterion, optimizer, device, args.epochs, args.img_size)
    torch.save(model.state_dict(), f"simple-unet-ddpm-{args.img_size}.pth")
