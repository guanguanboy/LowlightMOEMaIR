import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import networks
import torch.backends.cudnn as cudnn
import pytorch_ssim

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class HybridLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(HybridLoss, self).__init__()
        self.eps = eps

        # load VGG19 function
        self.VGG = networks.VGG19(init_weights='./pre_trained_VGG19_model/vgg19.pth', feature_mode=True)
        self.VGG.cuda() 
        self.VGG.eval()       

    def forward(self, x, y):

        loss_l1 = 5*F.smooth_l1_loss(x, y) 

        result_feature = self.VGG(x)
        target_feature = self.VGG(y) 
        loss_per = 0.001*self.L2(result_feature, target_feature) 
        loss_ssim=0.002*(1-pytorch_ssim.ssim(x, 7))
        loss_final = loss_l1+loss_ssim+loss_per  
        return loss_final


import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierLoss(nn.Module):
    """
    频域损失函数：计算预测值与真实值在傅里叶空间中幅度和相位的 L1 Loss
    """
    def __init__(self, loss_weight=1.0, alpha=1.0, beta=1.0):
        super(FourierLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha  # 幅度损失的权重
        self.beta = beta    # 相位损失的权重

    def forward(self, x, y):
        # 1. 执行快速傅里叶变换 (FFT)
        # 使用 rfft2 处理实数输入，dim=(-2, -1) 表示对最后两个维度（高和宽）变换
        x_fft = torch.fft.rfft2(x, norm='backward')
        y_fft = torch.fft.rfft2(y, norm='backward')

        # 2. 提取幅度谱 (Amplitude)
        # 为了数值稳定性，添加微小的 eps
        x_mag = torch.abs(x_fft)
        y_mag = torch.abs(y_fft)

        # 3. 提取相位谱 (Phase)
        x_phi = torch.angle(x_fft)
        y_phi = torch.angle(y_fft)

        # 4. 计算 L1 Loss
        # 幅度损失：捕捉对比度和亮度分布
        loss_mag = F.l1_loss(x_mag, y_mag)
        
        # 相位损失：捕捉结构和边缘信息
        loss_phi = F.l1_loss(x_phi, y_phi)

        # 5. 组合最终损失
        loss_final = self.loss_weight * (self.alpha * loss_mag + self.beta * loss_phi)
        
        return loss_final

# 示例：整合进你的 HybridLoss
class EnhancedHybridLoss(nn.Module):
    def __init__(self):
        super(EnhancedHybridLoss, self).__init__()
        self.fourier_loss = FourierLoss(alpha=1.0, beta=0.5) # 给相位稍微小一点权重，视任务而定
        # ... 其他初始化 (VGG等)

        # load VGG19 function
        self.VGG = networks.VGG19(init_weights='./pre_trained_VGG19_model/vgg19.pth', feature_mode=True)
        self.VGG.cuda() 
        self.VGG.eval()    

    def forward(self, x, y):
        # 原有的损失项
        loss_l1 = 5 * F.smooth_l1_loss(x, y)
        # ... loss_per, loss_ssim 等
        result_feature = self.VGG(x)
        target_feature = self.VGG(y) 
        loss_per = 0.001*self.L2(result_feature, target_feature) 
        loss_ssim=0.002*(1-pytorch_ssim.ssim(x, 7))
        loss_spatial = loss_l1+loss_ssim+loss_per  
        
        # 频域损失项
        loss_fft = 0.1 * self.fourier_loss(x, y) # 权重可调
        
        return loss_spatial + loss_fft # + ...
    