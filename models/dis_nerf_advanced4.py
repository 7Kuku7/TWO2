import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_content_encoder, get_distortion_encoder
from .mi_estimator import MIEstimator

class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive Feature Fusion using Gated Attention.
    Learns to dynamically weight Content and Distortion features.
    """
    def __init__(self, feature_dim=768):
        super().__init__()
        # Attention weights
        self.attn_fc = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 2), # Output 2 weights (alpha, beta)
            nn.Softmax(dim=1)
        )
        
    def forward(self, feat_c, feat_d):
        combined = torch.cat([feat_c, feat_d], dim=1)
        weights = self.attn_fc(combined) # [B, 2]
        
        alpha = weights[:, 0].unsqueeze(1) # Weight for content
        beta = weights[:, 1].unsqueeze(1)  # Weight for distortion
        
        feat_c_weighted = feat_c * (1 + alpha) 
        feat_d_weighted = feat_d * (1 + beta)
        
        return torch.cat([feat_c_weighted, feat_d_weighted], dim=1)

class DisNeRFQA_Advanced(nn.Module):
    # [修复] 增加 use_multitask 和 use_decoupling 参数，匹配 main2.py 的调用
    def __init__(self, num_subscores=4, use_fusion=True, use_multitask=True, use_decoupling=True):
        super().__init__()
        self.use_fusion = use_fusion
        self.use_multitask = use_multitask
        self.use_decoupling = use_decoupling
        
        # 1. Backbones
        self.content_encoder = get_content_encoder(pretrained=True)
        self.distortion_encoder = get_distortion_encoder(pretrained=True)
        
        # Feature dimensions
        self.feat_dim = 768 
        
        # 2. MI Estimator (开关控制)
        self.mi_estimator = None
        if self.use_decoupling:
            self.mi_estimator = MIEstimator(self.feat_dim)
        
        # 3. Adaptive Fusion (Innovation)
        if self.use_fusion:
            self.fusion = AdaptiveFeatureFusion(self.feat_dim)
        
        # 4. Heads
        # Main Quality Regressor (0-1)
        self.regressor = nn.Sequential(
            nn.Linear(self.feat_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )
        
        # Auxiliary Sub-score Regressor (Multi-task Innovation)
        # [修复] 开关控制
        self.num_subscores = num_subscores
        self.subscore_head = None
        if self.use_multitask:
            self.subscore_head = nn.Sequential(
                # 注意：这里输入维度依然取决于你是否用了 Fusion 后的特征
                # Advanced4 版本使用的是 feat_fused (768*2)
                nn.Linear(self.feat_dim * 2, 256), 
                nn.ReLU(),
                nn.Linear(256, num_subscores),
                nn.Sigmoid()
            )
        
        # Contrastive Projectors
        self.proj_c = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.proj_d = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x_content, x_distortion):
        b, t, c, h, w = x_content.shape
        
        # --- Feature Extraction ---
        x_c_flat = x_content.view(b * t, c, h, w)
        x_d_flat = x_distortion.view(b * t, c, h, w)
        
        # Content Branch
        feat_c_raw = self.content_encoder.forward_features(x_c_flat)
        if hasattr(self.content_encoder, 'global_pool'):
             feat_c_seq = self.content_encoder.forward_head(feat_c_raw, pre_logits=True)
        else:
             feat_c_seq = feat_c_raw[:, 0]
             
        # Distortion Branch
        feat_d_raw = self.distortion_encoder.forward_features(x_d_flat)
        feat_d_seq = feat_d_raw.mean(dim=[1, 2])
        
        # Reshape back
        feat_c_seq = feat_c_seq.view(b, t, -1)
        feat_d_seq = feat_d_seq.view(b, t, -1)
        
        # Temporal Pooling
        feat_c = feat_c_seq.mean(dim=1)
        feat_d = feat_d_seq.mean(dim=1)
        
        # --- Adaptive Fusion ---
        if self.use_fusion and hasattr(self, 'fusion'):
            feat_fused = self.fusion(feat_c, feat_d)
        else:
            # Simple Concatenation (Ablation)
            feat_fused = torch.cat([feat_c, feat_d], dim=1)
        
        # --- Predictions ---
        # 1. Quality Score
        score = self.regressor(feat_fused)
        
        # 2. Sub-scores (Auxiliary)
        # [修复] 检查 head 是否存在
        sub_scores = None
        if self.subscore_head is not None:
            sub_scores = self.subscore_head(feat_fused)
        
        # 3. Projections
        proj_c = self.proj_c(feat_c)
        proj_d = self.proj_d(feat_d)
        
        return score, sub_scores, proj_c, proj_d, feat_c, feat_d

class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks=4):
        super(MultiTaskLoss, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, input_losses):
        loss_sum = 0
        for i, loss in enumerate(input_losses):
            precision = torch.exp(-self.log_vars[i])
            loss_sum += 0.5 * precision * loss + 0.5 * self.log_vars[i]
        return loss_sum