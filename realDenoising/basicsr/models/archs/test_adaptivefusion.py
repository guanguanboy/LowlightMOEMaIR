import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveGateFusion(nn.Module):
    """
    双分支动态自适应门控融合模块 (Dynamic Adaptive Gate Fusion Module)

    输入: F1 (例如 Mamba 特征), F2 (例如 Frequency/CNN 特征)
    输出: 融合后的特征 F_Fusion = W * F1 + (1 - W) * F2
    """
    def __init__(self, in_channels, reduction_ratio=4):
        """
        :param in_channels: 输入特征的通道数 (C)
        :param reduction_ratio: 用于权重生成网络的降维比率
        """
        super(AdaptiveGateFusion, self).__init__()
        
        # 降维后的通道数
        mid_channels = in_channels // reduction_ratio
        
        # 1. 权重生成网络 (Gate Generation Network)
        # 这是一个轻量级的网络，用于从聚合特征中学习动态权重 W
        self.gate_generator = nn.Sequential(
            # 初始卷积：通常使用 1x1 卷积进行通道降维
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            # 最终卷积：恢复通道数，用于生成门控权重 W
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
            # 注意：最后使用 Sigmoid 激活在 forward 中完成
        )

    def forward(self, F_mamba: torch.Tensor, F_freq: torch.Tensor):
        """
        :param F_mamba: Mamba 分支的输出特征 (B, C, H, W)
        :param F_freq: 频率处理分支的输出特征 (B, C, H, W)
        :return: 融合特征 (B, C, H, W)
        """
        
        # 确保输入特征维度相同
        assert F_mamba.shape == F_freq.shape, "两个输入分支的特征形状必须相同才能进行融合。"
        
        # --- Step 1: 特征聚合 ---
        # 简单的元素级相加作为聚合特征的输入
        F_agg = F_mamba + F_freq
        
        # --- Step 2 & 3: 权重生成与激活 ---
        # 通过 Gate Generator 学习初始权重 W_raw
        W_raw = self.gate_generator(F_agg)
        
        # Sigmoid 激活：将权重 W 限制在 [0, 1]，作为 F_mamba 的门控权重
        # W 的形状为 (B, C, H, W)
        W = torch.sigmoid(W_raw)
        
        # --- Step 4 & 5: 自适应融合 ---
        # F_mamba 获得权重 W
        # F_freq 获得互补权重 (1 - W)
        F_fusion = W * F_mamba + (1 - W) * F_freq
        
        return F_fusion

# 示例使用：
if __name__ == '__main__':
    # 假设通道 C=64, 分辨率 H=32, W=32, 批次大小 B=4
    C, H, W = 64, 32, 32
    B = 4
    
    # 模拟 Mamba 特征 (F1) 和 Frequency 特征 (F2)
    F1 = torch.randn(B, C, H, W)
    F2 = torch.randn(B, C, H, W)
    
    # 初始化融合模块
    fusion_module = AdaptiveGateFusion(in_channels=C, reduction_ratio=4)
    
    # 执行融合
    F_fused = fusion_module(F1, F2)
    
    print(f"输入特征 F1/F2 的形状: {F1.shape}")
    print(f"融合特征 F_fused 的形状: {F_fused.shape}")