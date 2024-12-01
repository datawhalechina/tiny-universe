import torch


class NoiseScheduler:
    def __init__(self, beta_start=0.0001, beta_end=0.02, num_steps=1000, device="cpu"):
        """初始化噪声调度器
        Args:
            beta_start: β1,初始噪声水平
            beta_end: βT,最终噪声水平  
            num_steps: T,扩散步数
            device: 运行设备
        """
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_steps = num_steps
        self.device = device

        # β_t: 线性噪声调度
        self.betas = torch.linspace(beta_start, beta_end, num_steps).to(device)
        # α_t = 1 - β_t 
        self.alphas = 1.0 - self.betas
        # α_bar_t = ∏(1-β_i) from i=1 to t
        self.alpha_bar = torch.cumprod(self.alphas, dim=0).to(device)
        # α_bar_(t-1)
        self.alpha_bar_prev = torch.cat([torch.tensor([1.0], device=device), self.alpha_bar[:-1]], dim=0)
        # sqrt(α_bar_t)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar).to(device)
        # 1/sqrt(α_t)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(device)
        # sqrt(1-α_bar_t)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar).to(device)

        # 1/sqrt(α_bar_t)
        self.sqrt_recip_alphas_bar = torch.sqrt(1.0 / self.alpha_bar).to(device)
        # sqrt(1/α_bar_t - 1)
        self.sqrt_recipm1_alphas_bar = torch.sqrt(1.0 / self.alpha_bar - 1).to(device)
        # 后验分布方差 σ_t^2
        self.posterior_var = self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        # 后验分布均值系数1: β_t * sqrt(α_bar_(t-1))/(1-α_bar_t)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        # 后验分布均值系数2: (1-α_bar_(t-1)) * sqrt(α_t)/(1-α_bar_t)
        self.posterior_mean_coef2 = (1.0 - self.alpha_bar_prev) * torch.sqrt(self.alphas) / (1.0 - self.alpha_bar)
    
    def get(self, var, t, x_shape):
        """获取指定时间步的变量值并调整形状
        Args:
            var: 要查询的变量
            t: 时间步
            x_shape: 目标形状
        Returns:
            调整后的变量值
        """
        # 从变量张量中收集指定时间步的值
        out = var[t]
        # 调整形状为[batch_size, 1, 1, 1],以便进行广播
        return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

    def add_noise(self, x, t):
        """向输入添加噪声
        实现公式: x_t = sqrt(α_bar_t) * x_0 + sqrt(1-α_bar_t) * ε, ε ~ N(0,I)
        Args:
            x: 输入图像 x_0
            t: 时间步
        Returns:
            (noisy_x, noise): 加噪后的图像和使用的噪声
        """
        # 获取时间步t对应的sqrt(α_bar_t)
        sqrt_alpha_bar = self.get(self.sqrt_alpha_bar, t, x.shape)
        # 获取时间步t对应的sqrt(1-α_bar_t)
        sqrt_one_minus_alpha_bar = self.get(self.sqrt_one_minus_alpha_bar, t, x.shape)
        # 从标准正态分布采样噪声 ε ~ N(0,I)
        noise = torch.randn_like(x)
        # 实现前向扩散过程: x_t = sqrt(α_bar_t) * x_0 + sqrt(1-α_bar_t) * ε
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise, noise


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dataloader import load_transformed_dataset, show_tensor_image

    train_loader, test_loader = load_transformed_dataset()
    image, _ = next(iter(train_loader))
    noise_scheduler = NoiseScheduler()
    noisy_image, noise = noise_scheduler.add_noise(image, torch.randint(0, noise_scheduler.num_steps, (image.shape[0],)))
    plt.imshow(show_tensor_image(noisy_image))
    plt.show()
