# The Code Implementatio of MambaIR model for Real Image Denoising task
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from pdb import set_trace as stx
import numbers
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange
import math
from typing import Optional, Callable
from einops import rearrange, repeat
from functools import partial
from torch.nn import init as init


from basicsr.models.archs.shift_scanf_util import mair_ids_generate, mair_ids_scan, mair_ids_inverse, mair_shift_ids_generate


NEG_INF = -1000000


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)


##########################################################################
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MlpFromMaIR(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., input_resolution=(64,64)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.input_resolution = input_resolution
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution

        flops += 2 * H * W * self.in_features * self.hidden_features
        flops += H * W * self.hidden_features

        return flops

class IG_MoE(nn.Module):
    """
    完全符合 LaTeX 描述的 Illumination-Guided Mixture of Experts (IG-MoE)
    """
    def __init__(self, in_channels, expansion_ratio=2, num_experts=3, 
                 top_k=0, temperature=1.0, illum_channels=3):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.tau = temperature
        hidden_channels = int(in_channels * expansion_ratio)
        
        # --- 1. Expert Feature Extraction ---
        # phi_in: Point-wise convolution
        self.phi_in = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        
        # E_i: Depth-wise convolutional experts with different kernels/receptive fields
        self.experts = nn.ModuleList([
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, 
                      padding=1, groups=hidden_channels)
            for _ in range(num_experts)
        ])
        
        # --- 2. Illumination-Aware Routing ---
        # psi: Lightweight convolutional router
        self.router = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(8, num_experts, kernel_size=1)
        )
        
        # --- 3. Expert Fusion and Modulation ---
        # D: Final depth-wise convolution
        self.D = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, 
                           padding=1, groups=hidden_channels)
        self.delta = nn.GELU()
        # phi_out: Final point-wise projection
        self.phi_out = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)

    def _apply_topk_mask(self, logits, k):
        if k <= 0 or k >= self.num_experts:
            return logits
        # logits shape: (B, N, H, W)
        topk_vals, _ = torch.topk(logits, k=k, dim=1) # 在专家维度取top-k
        min_val = topk_vals[:, -1:, :, :] # 取第k个最大值作为阈值
        mask = logits < min_val
        return logits.masked_fill(mask, float('-inf'))

    def forward(self, x, illum_prior=None):
        """
        x: (B, C, H, W)
        illum_prior (I): (B, C_I, H, W)
        """
        # --- 1. 专家特征提取 ---
        # F = phi_in(X)
        x =  x.permute(0, 3, 1, 2).contiguous() # convert b h w c to b c h w

        F = self.phi_in(x)
        
        # F_i = E_i(F)
        expert_outputs = []
        for i in range(self.num_experts):
            expert_outputs.append(self.experts[i](F))
        # 堆叠为 (B, N, C_h, H, W)
        F_stack = torch.stack(expert_outputs, dim=1)
        
        # --- 2. 光照感知路由 ---
        # I_tilde = mean(I) over channels
        if illum_prior == None:
            illum_prior = x
        I_tilde = torch.mean(illum_prior, dim=1, keepdim=True)
        
        # A = psi(I_tilde)
        A = self.router(I_tilde)
        
        # Top-K Sparse Routing (Optional)
        if self.top_k > 0:
            A = self._apply_topk_mask(A, self.top_k)
            
        # alpha = softmax(A / tau)
        #alpha = F.softmax(A / self.tau, dim=1) # (B, N, H, W)
        alpha = torch.softmax(A / self.tau, dim=1)
        # --- 3. 专家融合与调制 ---
        # F_moe = sum(alpha_i * F_i)
        # 通过 einsum 或 unsqueeze 广播实现空间点对点的专家加权
        F_moe = (F_stack * alpha.unsqueeze(2)).sum(dim=1) # (B, C_h, H, W)
        
        # F_out = F + F_moe * F (Residual Gating)
        F_out = F + (F_moe * F)
        
        # Y = phi_out(delta(D(F_out)))
        Y = self.phi_out(self.delta(self.D(F_out)))
        
        Y = Y.permute(0, 2, 3, 1).contiguous() # convert b h w c to b c h w

        return Y



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x

class PatchEmbed(nn.Module):
    r""" transfer 2D feature map into 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" return 2D feature map from 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self):
        flops = 0
        return flops

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x, H, W):
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x, H, W):
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x

class ShuffleAttn(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=None, group=4, act_layer=nn.GELU, input_resolution=(64,64)):
        super().__init__()
        self.group = group
        self.input_resolution = input_resolution
        self.in_features = in_features
        self.out_features = out_features
        
        self.gating = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features, out_features, groups=self.group, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group
        
        x = x.reshape(batchsize, group_channels, self.group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x
    
    def channel_rearrange(self,x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group
        
        x = x.reshape(batchsize, self.group, group_channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x

    def forward(self, x):
        x = self.channel_shuffle(x)
        x = self.gating(x)
        x = self.channel_rearrange(x)

        return x
    
    def flops(self):
        flops = 0
        H, W = self.input_resolution
        
        # nn.AdaptiveAvgPool2d(1),
        flops += H * W * self.in_features

        # nn.Conv2d(in_features, out_features, groups=self.group, kernel_size=1, stride=1, padding=0),
        flops += H * W * self.in_features * self.out_features // self.group

        # nn.Sigmoid()
        flops += H * W * self.out_features * 4
        return flops

import torch.fft as fft

class SaSSM_LoSh2D(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, ssm_ratio=2.,dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            input_resolution=(64, 64),
            **kwargs,
        ):
        
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        # 继承原有的初始化参数
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.ssm_ratio = ssm_ratio

        self.d_inner = int(self.ssm_ratio * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.input_resolution = input_resolution

        # --- [新增] 结构感知 (Phase-based) 调制器 ---
        self.phase_encoder = nn.Sequential(
            nn.Conv2d(self.d_inner, self.d_inner // 4, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.d_inner // 4, 4, 1),
            nn.Sigmoid()
        )

        # --- [新增] 照度引导 (Illumination-guided) SSA 权重生成器 ---
        self.illum_gate = nn.Sequential(
            nn.Conv2d(1, self.d_inner // 4, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.d_inner // 4, 4, 1), # 为4个扫描方向生成权重
            nn.Softmax(dim=1) # 保证四个方向权重之和为1
        )

        # 保留原有的 Mamba 参数 (in_proj, conv2d, x_proj, dt_projs, A_logs, Ds等)
        # ... [此处省略原有初始化代码，保持不变] ...
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        # print(self.x_proj_weight.shape)

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        
        # ... [此处省略 A_logs, Ds 等初始化] ...
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        #self.gating = ShuffleAttn(in_features=self.d_inner*4, out_features=self.d_inner*4, group=self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, losh_ids, structure_mod, x_proj_bias: torch.Tensor=None):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        
        # 修复点 1: 显式获取单个方向的通道数，不要从 A_logs 获取合并后的维度
        D_inner = self.d_inner  # 112
        N_state = self.d_state  # 16
        R_rank = self.dt_rank
        
        # 获取扫描后的序列
        xs_scan_ids, xs_inverse_ids = losh_ids
        xs = mair_ids_scan(x, xs_scan_ids) # [B, K, C, L]

        # 修复点 2: 权重 reshape。确保第二维是单方向通道数 D_inner
        # reshape 后的形状应为 [K * (R+N*2), D_inner, 1]
        weight = self.x_proj_weight.reshape(-1, D_inner, 1)
        bias = x_proj_bias.reshape(-1) if x_proj_bias is not None else None
        
        # 执行分组卷积投影
        # input: [B, K*C, L], weight: [K*out, C, 1], groups=K
        x_dbl = F.conv1d(xs.reshape(B, -1, L), weight, bias=bias, groups=K)
        
        # 修复点 3: 结构感知调制 (structure_mod)
        # structure_mod 形状是 [B, C, H, W]，需要展平并广播到 x_dbl 的 K*out 维度上
        # 我们将其作为选择性扫描的一个“注意力权重”
        # 如果 structure_mod 是为了调节 dt，它应该作用在 dts 所在的部分
        
        # 首先将 x_dbl 分解回四个扫描方向
        # x_dbl 形状: [B, K, (R + N*2), L]
        x_dbl = x_dbl.reshape(B, K, -1, L)
        
        # 结构调制：利用相位提取的结构图来增强/减弱不同区域的扫描响应
        # 将 struct_map 展平为 [B, 1, 1, L] 进行空间上的权重缩放
        #struct_mod_flat = structure_mod.mean(dim=1, keepdim=True).view(B, 1, 1, L)
        #x_dbl = x_dbl * struct_mod_flat 

        # --- [优化点] 方向特异性调制 ---
        # structure_mod 形状从 [B, 4, H, W] 转换为 [B, 4, 1, L]
        # 这样每一组扫描方向 (K) 都有自己专属的结构权重图
        struct_mod_scanned = structure_mod.view(B, K, 1, L) 
        
        # 执行调制：直接作用于对应的扫描路径
        x_dbl = x_dbl * struct_mod_scanned

        # 继续原有的 Mamba 流程
        dts, Bs, Cs = torch.split(x_dbl, [R_rank, N_state, N_state], dim=2)
        
        # 重新合并 K 维度进行 dt 的线性映射
        dts = F.conv1d(dts.reshape(B, -1, L), self.dt_projs_weight.reshape(K * D_inner, -1, 1), groups=K)
        
        # 准备 selective scan 的参数
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, 
            delta_bias=dt_projs_bias, 
            delta_softplus=True
        ).view(B, K, -1, L)

        return mair_ids_inverse(out_y, xs_inverse_ids, shape=(B, -1, H, W))

    def forward(self, x: torch.Tensor, losh_ids, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_in = x_in.permute(0, 3, 1, 2).contiguous()
        x_feat = self.act(self.conv2d(x_in)) # B, C_inner, H, W

        # --- 步骤 1: 提取相位结构图 (Structure-aware NSS 准备) ---
        x_fft = fft.rfft2(x_feat.float(), norm='ortho')
        phase = torch.angle(x_fft)
        # 将相位信息映射回空间域作为调制器
        struct_map = self.phase_encoder(fft.irfft2(torch.exp(1j * phase), s=(H, W)).type_as(x_feat))

        # --- 步骤 2: 照度感知权重生成 (SSA 准备) ---
        # 照度估计：通常取通道均值。在暗处，赋予全局性扫描更高权重；在亮处，侧重细节。
        illum_map = torch.mean(x_feat, dim=1, keepdim=True)
        ssa_weights = self.illum_gate(illum_map) # B, 4, H, W

        # --- 步骤 3: 核心扫描 ---
        # y_multi: (B, 4, C_inner, H, W)
        y_multi = self.forward_core(x_feat, losh_ids, structure_mod=struct_map)

        # --- 步骤 4: 照度引导的自适应聚合 (SSA 核心) ---
        # 将原有的 y = y1 + y2 + y3 + y4 替换为加权聚合
        # 在第 627 行之前添加
        #print(f"DEBUG: y_multi shape: {y_multi.shape}")
        #print(f"DEBUG: ssa_weights shape: {ssa_weights.shape}")
        
        y = (y_multi.view(B, 4, -1, H, W) * ssa_weights.unsqueeze(2)).view(B, -1, H, W) # B, C_inner, H, W
        #y = (y_multi * ssa_weights).view(B, -1, H, W)
        #print(f"DEBUG: y_multi shape: {y.shape}")
        y1, y2, y3, y4 = torch.chunk(y, 4, dim=1)
        y = y1 + y2 + y3 + y4

        # 后续残差与输出
        y = y.permute(0, 2, 3, 1).contiguous()
        y = self.out_norm(y) * F.silu(z)
        out = self.out_proj(y)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

class PhaseBranch(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, bias=False):
        super(PhaseBranch, self).__init__()
        # 深度可分离卷积结构
        self.conv = nn.Sequential(
            # 1. Depthwise: 提取局部空间/频率结构
            nn.Conv2d(in_dim, in_dim, kernel_size=kernel_size, padding=kernel_size//2, 
                      groups=in_dim, bias=bias),
            nn.BatchNorm2d(in_dim),
            nn.GELU(),
            
            # 2. Pointwise: 通道间的特征投影 (等价于之前的 @ 操作)
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

# 整合进 polar_frequency_selection 的 forward 逻辑
# ... 在 __init__ 中 ...
#self.pha_conv1 = PhaseBranch(dim, hid_dim)
#self.pha_conv2 = PhaseBranch(hid_dim, dim)

class polar_frequency_selection(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.GELU, window_size=None, bias=False):
        super(polar_frequency_selection, self).__init__()
        self.act_fft_amp = act_method()
        self.act_fft_pha = act_method()

        self.window_size = window_size
        hid_dim = dim * dw
        
        # 保持原有的复数权重参数化方式，但物理意义变为：
        # weight1.real 用于处理幅度，weight1.imag 用于处理相位
        self.complex_weight1_real = nn.Parameter(torch.Tensor(dim, hid_dim))
        #self.complex_weight1_imag = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(hid_dim, dim))
        #self.complex_weight2_imag = nn.Parameter(torch.Tensor(hid_dim, dim))
        
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        #init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
        #init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(16))
        
        self.bias = bias
        self.norm = norm
        if bias:
            self.b1_mag = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)))
            #self.b1_pha = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)))
            self.b2_mag = nn.Parameter(torch.zeros((1, 1, 1, dim)))
            #self.b2_pha = nn.Parameter(torch.zeros((1, 1, 1, dim)))

        self.pha_conv1 = PhaseBranch(dim, hid_dim)
        self.pha_conv2 = PhaseBranch(hid_dim, dim)

    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        
        y = torch.fft.rfft2(x, norm=self.norm)
        y = rearrange(y, 'b c h w -> b h w c')

        # --- 核心改动：极坐标分解处理 ---
        mag = torch.abs(y) + 1e-8
        pha = torch.angle(y)

        # 第一层映射：分别作用于幅度和相位
        mag = mag @ self.complex_weight1_real
        #pha = pha @ self.complex_weight1_imag
        
        if self.bias:
            mag = mag + self.b1_mag
            #pha = pha + self.b1_pha

        # 相位分支：深度可分离卷积提取局部几何结构 (Z_p')
        pha = rearrange(pha, 'b h w c -> b c h w').contiguous()
        pha = self.pha_conv1(pha)
        pha = rearrange(pha, 'b c h w -> b h w c').contiguous()


        # 模拟原代码的非线性拼接逻辑
        # 原代码是将实部虚部拼在一起过激活，这里我们将 mag 和 pha 拼在一起
        #combined = torch.cat([mag, pha], dim=1) # dim=1 对应原代码中的位置
        #combined = self.act_fft(combined)
        #mag, pha = torch.chunk(combined, 2, dim=1)
        mag = self.act_fft_amp(mag)
        # 第二层映射
        mag = mag @ self.complex_weight2_real
        #pha = pha @ self.complex_weight2_imag
        
        if self.bias:
            mag = mag + self.b2_mag
            #pha = pha + self.b2_pha
        
        pha = self.act_fft_pha(pha)
        pha = rearrange(pha, 'b h w c -> b c h w').contiguous()
        pha = self.pha_conv2(pha)
        pha = rearrange(pha, 'b c h w -> b h w c').contiguous()

        # --- 核心改动：重组为复数 ---
        y = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
        
        y = rearrange(y, 'b h w c -> b c h w')

        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y

class RMB(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            ssm_ratio: float = 2.,
            input_resolution= (64, 64),
            is_light_sr: bool = False,
            shift_size=0,
            mlp_ratio=1.5,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SaSSM_LoSh2D(d_model=hidden_dim, d_state=d_state,expand=ssm_ratio,dropout=attn_drop_rate, input_resolution=input_resolution, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        #self.conv_blk = FeedForwardMoE(in_features=hidden_dim, hidden_features=mlp_hidden_dim,input_resolution=input_resolution)
        self.conv_blk = IG_MoE(in_channels=hidden_dim, num_experts=3, top_k=2)

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.hidden_dim = hidden_dim
        self.input_resolution = input_resolution

        self.shift_size = shift_size
        
        self.frequency = polar_frequency_selection(hidden_dim)
        #self.ln_3 = nn.LayerNorm(hidden_dim)
        self.skip_scale3 = nn.Parameter(torch.ones(hidden_dim))

        # 1. 局部增强分支
        self.local_conv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        
        # 2. 频域与空间的融合门控
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # 3. 改进 LayerScale 为更稳定的初始化
        self.skip_scale = nn.Parameter(1e-5 * torch.ones(hidden_dim))
        self.skip_scale2 = nn.Parameter(1e-5 * torch.ones(hidden_dim))

    def forward(self, input, mair_ids, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        identity = input

        x = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]

        # --- 空间分支 ---
        # 局部结构提取
        x_local = x.permute(0, 3, 1, 2).contiguous()
        x_local = self.local_conv(x_local).permute(0, 2, 3, 1)
        
        # Mamba 长程建模 (VMM)
        x_norm = self.ln_1(x)
        if self.shift_size > 0:
            x_mamba = self.self_attention(x_norm, (mair_ids[2], mair_ids[3])) 
        else:
            x_mamba = self.self_attention(x_norm, (mair_ids[0], mair_ids[1])) 
        
        # --- 频域分支 ---
        x_freq = self.frequency(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        
        # --- 深度融合 ---
        # 使用Mamba特征生成的门控来调制频域信息
        gate = self.fusion_gate(x_mamba)
        #print(x_mamba.shape, x_local.shape)
        combined_op = x_mamba + x_local*self.skip_scale3 + (x_freq * gate)
        
        combined_op = combined_op.reshape(B, -1, C)
        x = identity + self.drop_path(self.skip_scale * combined_op)
        
        # --- MoE / FeedForward ---
        x = x.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = x + self.drop_path(self.skip_scale2 * self.conv_blk(self.ln_2(x)))

        x = x.reshape(B, -1, C)
        return x
    
    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # flops of norm1 self.ln_1 -> layer_norm1
        flops += self.hidden_dim * H * W
        # flops of SS2D
        flops += self.self_attention.flops()
        # flops of input * self.skip_scale and residual
        flops += self.hidden_dim * H * W * 2 
        # flops of norm2 self.ln_2 -> layer_norm2
        flops += self.hidden_dim * H * W 
        # flops of MLP
        flops += self.conv_blk.flops()
        # flops of input * self.skip_scale2 and residual
        flops += self.hidden_dim * H * W * 2 
        
        return flops
    

class SFIMLLIE(nn.Module):
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 ssm_ratio=1.5,
                 scan_len = 8,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 mlp_ratio=2.,
                 num_refinement_blocks=4,
                 drop_path_rate=0.,
                 bias=False,
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(SFIMLLIE, self).__init__()
        self.mlp_ratio = mlp_ratio
        self.scan_len = scan_len

        img_size_ids = to_2tuple(img_size)
        self._generate_ids((1, 1, img_size_ids[0], img_size_ids[1]))
        self.conv_first = nn.Conv2d(inp_channels, dim, 3, 1, 1)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=dim,
            embed_dim=dim,
            norm_layer=nn.LayerNorm)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        #####
        self.is_light_sr = False
        base_d_state = 4
        self.ssm_ratio = ssm_ratio
        self.encoder_level1 = nn.ModuleList([
            RMB(
                hidden_dim=dim,
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=base_d_state* 2 ** 2,
                ssm_ratio=self.ssm_ratio,
                input_resolution=(patches_resolution[0],patches_resolution[1]),
                is_light_sr=self.is_light_sr,
                shift_size=0 if (i % 2 == 0) else self.scan_len // 2,
                mlp_ratio=self.mlp_ratio                
            )
            for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.ModuleList([
            RMB(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=int(base_d_state * 2 ** 2),
                ssm_ratio=self.ssm_ratio,
                input_resolution=(patches_resolution[0],patches_resolution[1]),
                is_light_sr=self.is_light_sr,
                shift_size=0 if (i % 2 == 0) else self.scan_len // 2,
                mlp_ratio=self.mlp_ratio 
            )
            for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.ModuleList([
            RMB(
                hidden_dim=int(dim * 2 ** 2),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=int(base_d_state * 2 ** 2),
                ssm_ratio=self.ssm_ratio,
                input_resolution=(patches_resolution[0],patches_resolution[1]),
                is_light_sr=self.is_light_sr,
                shift_size=0 if (i % 2 == 0) else self.scan_len // 2,
                mlp_ratio=self.mlp_ratio 
            )
            for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.ModuleList([
            RMB(
                hidden_dim=int(dim * 2 ** 3),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=int(base_d_state / 2 * 2 ** 3),
                ssm_ratio=self.ssm_ratio,
                input_resolution=(patches_resolution[0],patches_resolution[1]),
                is_light_sr=self.is_light_sr,
                shift_size=0 if (i % 2 == 0) else self.scan_len // 2,
                mlp_ratio=self.mlp_ratio 
            )
            for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.ModuleList([
            RMB(
                hidden_dim=int(dim * 2 ** 2),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=int(base_d_state * 2 ** 2),
                ssm_ratio=self.ssm_ratio,
                input_resolution=(patches_resolution[0],patches_resolution[1]),
                is_light_sr=self.is_light_sr,
                shift_size=0 if (i % 2 == 0) else self.scan_len // 2,
                mlp_ratio=self.mlp_ratio 
            )
            for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.ModuleList([
            RMB(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=int(base_d_state * 2 ** 2),
                ssm_ratio=self.ssm_ratio,
                input_resolution=(patches_resolution[0],patches_resolution[1]),
                is_light_sr=self.is_light_sr,
                shift_size=0 if (i % 2 == 0) else self.scan_len // 2,
                mlp_ratio=self.mlp_ratio 
            )
            for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.ModuleList([
            RMB(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=int(base_d_state * 2 ** 2),
                ssm_ratio=self.ssm_ratio,
                input_resolution=(patches_resolution[0],patches_resolution[1]),
                is_light_sr=self.is_light_sr,
                shift_size=0 if (i % 2 == 0) else self.scan_len // 2,
                mlp_ratio=self.mlp_ratio 
            )
            for i in range(num_blocks[0])])

        self.refinement = nn.ModuleList([
            RMB(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=int(base_d_state * 2 ** 2),
                ssm_ratio=self.ssm_ratio,
                input_resolution=(patches_resolution[0],patches_resolution[1]),
                is_light_sr=self.is_light_sr,
                shift_size=0 if (i % 2 == 0) else self.scan_len // 2,
                mlp_ratio=self.mlp_ratio 
            )
            for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def _generate_ids(self, inp_shape):
        B,C,H,W = inp_shape

        xs_scan_ids, xs_inverse_ids = mair_ids_generate(inp_shape=(1, 1, H, W), scan_len=self.scan_len)# [B,H,W,C]
        if torch.cuda.is_available():
            self.xs_scan_ids = xs_scan_ids.cuda()
            self.xs_inverse_ids = xs_inverse_ids.cuda()
        else:
            self.xs_scan_ids = xs_scan_ids
            self.xs_inverse_ids = xs_inverse_ids

        xs_shift_scan_ids, xs_shift_inverse_ids = mair_shift_ids_generate(inp_shape=(1, 1, H, W), scan_len=self.scan_len, shift_len=self.scan_len//2)# [B,H,W,C]
        if torch.cuda.is_available():
            self.xs_shift_scan_ids = xs_shift_scan_ids.cuda()
            self.xs_shift_inverse_ids = xs_shift_inverse_ids.cuda()
        else:
            self.xs_shift_scan_ids = xs_shift_scan_ids
            self.xs_shift_inverse_ids = xs_shift_inverse_ids

        del xs_scan_ids, xs_inverse_ids, xs_shift_scan_ids, xs_shift_inverse_ids

    def generate_ids_with_shape(self, Height, Width):
        xs_scan_ids, xs_inverse_ids = mair_ids_generate(inp_shape=(1, 1, Height, Width), scan_len=self.scan_len)# [B,H,W,C]
        xs_shift_scan_ids, xs_shift_inverse_ids = mair_shift_ids_generate(inp_shape=(1, 1, Height, Width), scan_len=self.scan_len, shift_len=self.scan_len//2)# [B,H,W,C]
        if torch.cuda.is_available():
            xs_scan_ids, xs_inverse_ids = xs_scan_ids.cuda(), xs_inverse_ids.cuda()
            xs_shift_scan_ids, xs_shift_inverse_ids = xs_shift_scan_ids.cuda(), xs_shift_inverse_ids.cuda()

        return (xs_scan_ids, xs_inverse_ids, xs_shift_scan_ids, xs_shift_inverse_ids)

    def forward(self, inp_img):

        _, _, H, W = inp_img.shape
        #mair_ids = (self.xs_scan_ids, self.xs_inverse_ids, self.xs_shift_scan_ids, self.xs_shift_inverse_ids)
        mair_ids = self.generate_ids_with_shape(H, W)

        shollow_feats = self.conv_first(inp_img)

        inp_enc_level1 = self.patch_embed(shollow_feats)  # b,hw,c
        out_enc_level1 = inp_enc_level1
        for layer in self.encoder_level1:
            out_enc_level1 = layer(out_enc_level1, mair_ids, [H, W])

        mair_ids_level2 = self.generate_ids_with_shape(H // 2, W // 2)
        mair_ids_level3 = self.generate_ids_with_shape(H // 4, W // 4)
        mair_ids_level4 = self.generate_ids_with_shape(H // 8, W // 8)

        inp_enc_level2 = self.down1_2(out_enc_level1, H, W)  # b, hw//4, 2c
        out_enc_level2 = inp_enc_level2
        for layer in self.encoder_level2:
            out_enc_level2 = layer(out_enc_level2, mair_ids_level2, [H // 2, W // 2])

        inp_enc_level3 = self.down2_3(out_enc_level2, H // 2, W // 2)  # b, hw//16, 4c
        out_enc_level3 = inp_enc_level3
        for layer in self.encoder_level3:
            out_enc_level3 = layer(out_enc_level3, mair_ids_level3, [H // 4, W // 4])

        inp_enc_level4 = self.down3_4(out_enc_level3, H // 4, W // 4)  # b, hw//64, 8c
        latent = inp_enc_level4
        for layer in self.latent:
            latent = layer(latent, mair_ids_level4, [H // 8, W // 8])

        inp_dec_level3 = self.up4_3(latent, H // 8, W // 8)  # b, hw//16, 4c
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 2)
        inp_dec_level3 = rearrange(inp_dec_level3, "b (h w) c -> b c h w", h=H // 4, w=W // 4).contiguous()
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        inp_dec_level3 = rearrange(inp_dec_level3, "b c h w -> b (h w) c").contiguous()  # b, hw//16, 4c
        out_dec_level3 = inp_dec_level3
        for layer in self.decoder_level3:
            out_dec_level3 = layer(out_dec_level3, mair_ids_level3, [H // 4, W // 4])

        inp_dec_level2 = self.up3_2(out_dec_level3, H // 4, W // 4)  # b, hw//4, 2c
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b (h w) c -> b c h w", h=H // 2, w=W // 2).contiguous()
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b c h w -> b (h w) c").contiguous()  # b, hw//4, 2c
        out_dec_level2 = inp_dec_level2
        for layer in self.decoder_level2:
            out_dec_level2 = layer(out_dec_level2, mair_ids_level2, [H // 2, W // 2])

        inp_dec_level1 = self.up2_1(out_dec_level2, H // 2, W // 2)  # b, hw, c
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 2)
        out_dec_level1 = inp_dec_level1
        for layer in self.decoder_level1:
            out_dec_level1 = layer(out_dec_level1, mair_ids, [H, W])

        for layer in self.refinement:
            out_dec_level1 = layer(out_dec_level1, mair_ids, [H, W])

        out_dec_level1 = rearrange(out_dec_level1, "b (h w) c -> b c h w", h=H, w=W).contiguous()

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1

if __name__ == '__main__':
    height = 64
    width = 64
    model = SFIMLLIE(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        mlp_ratio=2.,
        bias=False,
        dual_pixel_task=False
    ).cuda()
    # print(model)
    x = torch.randn((1, 3, height, width)).cuda()
    print(x.shape)
    y = model(x)
    print(y.shape)
