import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertedResidualBlock(nn.Module):
    """
    [移植] 来自 MMIF-CDDFuse 的反向残差块
    """
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim), 
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)

class DetailNode(nn.Module):
    """
    [移植] 来自 MMIF-CDDFuse 的 DetailNode (基于 INN 耦合层)
    """
    def __init__(self, dim=64):
        super(DetailNode, self).__init__()
        half_dim = dim // 2
        
        self.theta_phi = InvertedResidualBlock(inp=half_dim, oup=half_dim, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=half_dim, oup=half_dim, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=half_dim, oup=half_dim, expand_ratio=2)
        
        self.shffleconv = nn.Conv2d(dim, dim, kernel_size=1,
                                    stride=1, padding=0, bias=True)

    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2

    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
            
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        
        return z1, z2

class DistortionCNN(nn.Module):
    """
    [显存优化版] 专门用于提取失真特征的 CNN
    优化点：先 Global Pooling 再 Projection，大幅降低显存占用。
    """
    def __init__(self, in_chans=3, feature_dim=768, base_dim=64, num_layers=3):
        super(DistortionCNN, self).__init__()
        
        # 1. Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, base_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True)
        )
        
        # 2. Detail Feature Extraction
        self.detail_layers = nn.ModuleList([DetailNode(dim=base_dim) for _ in range(num_layers)])
        
        # 3. Projection (修改点：使用 Linear 层代替 Conv2d)
        # 之前的 Conv2d(64, 768) 在 224x224 尺寸下显存消耗极大
        self.proj = nn.Linear(base_dim, feature_dim)
        
    def forward(self, x):
        # x: [B, 3, H, W] (这里的 B 实际上是 Batch * Frames)
        
        # Stem
        x = self.stem(x) # [B, 64, H, W]
        
        # Detail Extraction
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        for layer in self.detail_layers:
            z1, z2 = layer(z1, z2)
        x_detail = torch.cat((z1, z2), dim=1) # [B, 64, H, W]
        
        # --- 显存优化操作 ---
        
        # 1. 先进行全局平均池化 (GAP)
        # [B, 64, H, W] -> [B, 64]
        x_pooled = x_detail.mean(dim=[2, 3]) 
        
        # 2. 再进行投影 (Projection)
        # [B, 64] -> [B, 768]
        x_out = self.proj(x_pooled)
        
        return x_out