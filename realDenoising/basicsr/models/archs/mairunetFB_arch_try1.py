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

"""

class FeedForwardMoE(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0., input_resolution=(64,64),
                 num_experts: int = 3, top_k: int = 0, temperature: float = 1.0, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.input_resolution = input_resolution
        self.act = act_layer()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature

        self.project_in = nn.Linear(in_features, hidden_features, bias=bias)
        
        # 创建专家
        experts = []
        for i in range(num_experts):
            experts.append(nn.Sequential(
                nn.Linear(hidden_features, hidden_features, bias=bias),
                self.act,
                nn.Linear(hidden_features, hidden_features, bias=bias)
            ))
        self.experts = nn.ModuleList(experts)
        
        # 路由网络
        self.illum_to_logits = nn.Linear(hidden_features, num_experts, bias=bias)
        
        self.project_out = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor, illum: torch.Tensor = None):
        # 处理输入形状：支持 (B, H, W, C) 和 (B, L, C)
        if x.dim() == 4:
            # (B, H, W, C) -> (B, L, C)
            B, H, W, C = x.shape
            L = H * W
            x_reshaped = x.reshape(B, L, C)
        elif x.dim() == 3:
            # (B, L, C)
            B, L, C = x.shape
            x_reshaped = x
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        
        # 输入投影
        x_hidden = self.project_in(x_reshaped)
        
        # 路由依据
        if illum is None:
            routing_input = x_hidden
        else:
            # 处理illum形状
            if illum.dim() == 4:
                # (B, H, W, C) -> (B, L, C)
                B_illum, H_illum, W_illum, C_illum = illum.shape
                L_illum = H_illum * W_illum
                if L_illum != L:
                    raise ValueError(f"illum spatial size {H_illum}x{W_illum}={L_illum} doesn't match input length {L}")
                illum_reshaped = illum.reshape(B, L_illum, C_illum)
            elif illum.dim() == 3:
                # (B, L, C)
                if illum.shape != x_reshaped.shape:
                    raise ValueError(f"illum shape {illum.shape} must match input shape {x_reshaped.shape}")
                illum_reshaped = illum
            else:
                raise ValueError(f"Unsupported illum shape: {illum.shape}")
            
            routing_input = self.project_in(illum_reshaped)
        
        # 计算路由权重
        logits = self.illum_to_logits(routing_input) / self.temperature
        alphas = F.softmax(logits, dim=-1)  # (B, L, num_experts)
        
        # 所有专家处理
        expert_outs = []
        for expert in self.experts:
            expert_out = expert(x_hidden)
            expert_outs.append(expert_out)
        
        # 堆叠专家输出
        expert_stack = torch.stack(expert_outs, dim=2)  # (B, L, num_experts, hidden_features)
        
        # 专家融合
        fused = (expert_stack * alphas.unsqueeze(-1)).sum(dim=2)  # (B, L, hidden_features)
        
        # 输出投影
        x_out = self.project_out(fused)
        
        # 恢复原始形状
        if x.dim() == 4:
            # (B, L, C) -> (B, H, W, C)
            x_out = x_out.reshape(B, x.shape[1], x.shape[2], -1)
        
        return x_out

"""
class FeedForwardMoE(nn.Module):
    """优化版本的MoE模块，保持与MlpFromMaIR的兼容性"""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0., input_resolution=(64,64),
                 num_experts: int = 3, top_k: int = 0, temperature: float = 1.0, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.input_resolution = input_resolution
        self.act = act_layer()
        
        self.num_experts = num_experts
        self.top_k = max(1, min(top_k, num_experts))  # 确保top_k在合理范围内
        self.temperature = temperature

        # 输入投影
        self.project_in = nn.Linear(in_features, hidden_features, bias=bias)
        
        # 创建专家 - 使用不同的配置增加多样性
        experts = []
        # 为不同专家使用不同的隐藏层维度（可选）
        expert_hidden_factors = [1.0, 0.8, 1.2][:num_experts]  # 不同专家的隐藏层缩放
        for i in range(num_experts):
            factor = expert_hidden_factors[i] if i < len(expert_hidden_factors) else 1.0
            expert_hidden = int(hidden_features * factor)
            
            experts.append(nn.Sequential(
                nn.Linear(hidden_features, expert_hidden, bias=bias),
                self.act,
                nn.Linear(expert_hidden, hidden_features, bias=bias)
            ))
        self.experts = nn.ModuleList(experts)
        
        # 路由网络
        self.illum_to_logits = nn.Linear(hidden_features, num_experts, bias=bias)
        
        # 混合后的处理层（可选）
        self.mix_proj = nn.Sequential(
            nn.Linear(hidden_features, hidden_features, bias=bias),
            self.act
        )
        
        # 输出投影
        self.project_out = nn.Linear(hidden_features, out_features, bias=bias)
        
        # Dropout（为了兼容性）
        self.drop = nn.Dropout(drop) if drop > 0. else nn.Identity()

    def _apply_topk_mask(self, logits: torch.Tensor, k: int):
        """应用Top-K掩码，只保留前K个专家"""
        if k <= 0 or k >= logits.size(-1):
            return logits
        
        B, L, N = logits.shape
        # 获取每个位置的前k个专家
        topk_vals, topk_idx = torch.topk(logits, k=k, dim=-1)
        
        # 创建mask
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(-1, topk_idx, True)
        
        # 应用mask
        logits_masked = torch.full_like(logits, float('-inf'))
        logits_masked[mask] = logits[mask]
        
        return logits_masked

    def forward(self, x: torch.Tensor, illum: torch.Tensor = None):
        """
        x: 输入特征，支持 (B, H, W, C) 或 (B, L, C)
        illum: 光照特征，形状应与x相同
        """
        original_shape = x.shape
        original_dim = x.dim()
        
        # 处理输入形状
        if x.dim() == 4:
            # (B, H, W, C) -> (B, L, C)
            B, H, W, C = x.shape
            L = H * W
            x_reshaped = x.reshape(B, L, C)
        elif x.dim() == 3:
            # (B, L, C)
            B, L, C = x.shape
            x_reshaped = x
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        
        # 输入投影
        x_hidden = self.project_in(x_reshaped)
        
        # 路由依据
        if illum is None:
            routing_input = x_hidden
        else:
            # 处理illum形状
            if illum.dim() == 4:
                B_illum, H_illum, W_illum, C_illum = illum.shape
                L_illum = H_illum * W_illum
                if L_illum != L:
                    # 如果不匹配，使用插值
                    if H_illum * W_illum == 1:  # 全局特征
                        illum_reshaped = illum.reshape(B, 1, C_illum).expand(B, L, C_illum)
                    else:
                        # 需要调整形状
                        illum_resized = F.interpolate(
                            illum.permute(0, 3, 1, 2), 
                            size=(H, W), 
                            mode='bilinear'
                        ).permute(0, 2, 3, 1)
                        illum_reshaped = illum_resized.reshape(B, L, C_illum)
                else:
                    illum_reshaped = illum.reshape(B, L, C_illum)
            elif illum.dim() == 3:
                if illum.shape[1] == 1:  # 全局特征 (B, 1, C)
                    illum_reshaped = illum.expand(B, L, C)
                else:
                    illum_reshaped = illum
            else:
                raise ValueError(f"Unsupported illum shape: {illum.shape}")
            
            # 确保通道数一致
            if illum_reshaped.shape[-1] != C:
                if illum_reshaped.shape[-1] == 1:
                    illum_reshaped = illum_reshaped.expand(-1, -1, C)
                else:
                    # 使用投影统一通道数
                    if not hasattr(self, 'illum_proj'):
                        self.illum_proj = nn.Linear(illum_reshaped.shape[-1], C).to(x.device)
                    illum_reshaped = self.illum_proj(illum_reshaped)
            
            routing_input = self.project_in(illum_reshaped)
        
        # 计算路由权重
        logits = self.illum_to_logits(routing_input) / (self.temperature + 1e-8)
        
        # 应用Top-K稀疏化
        if self.top_k > 0 and self.top_k < self.num_experts:
            logits = self._apply_topk_mask(logits, self.top_k)
        
        alphas = F.softmax(logits, dim=-1)  # (B, L, num_experts)
        
        # 所有专家处理
        expert_outs = []
        for expert in self.experts:
            expert_out = expert(x_hidden)
            expert_outs.append(expert_out)
        
        # 堆叠专家输出
        expert_stack = torch.stack(expert_outs, dim=2)  # (B, L, num_experts, hidden_features)
        
        # 专家融合
        fused = (expert_stack * alphas.unsqueeze(-1)).sum(dim=2)  # (B, L, hidden_features)
        
        # 混合处理 + 残差连接
        mixed = self.mix_proj(fused)
        x_out = x_hidden + mixed
        
        # 输出投影 + Dropout
        x_out = self.project_out(x_out)
        x_out = self.drop(x_out)
        
        # 恢复原始形状
        if original_dim == 4:
            # (B, L, C) -> (B, H, W, C)
            x_out = x_out.reshape(B, original_shape[1], original_shape[2], -1)
        
        return x_out

class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops


#########################################
class Attention(nn.Module):
    r""" Multi-head self attention module with dynamic position bias.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W, mask=None):
        """
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, Gh*Gw, Gh*Gw) or None
            H: height of each group
            W: width of each group
        """
        group_size = (H, W)
        B_, N, C = x.shape
        assert H * W == N
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1).contiguous()  # (B_, self.num_heads, N, N), N = H*W

        if self.position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)
            position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Gh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()  # (2h-1)*(2w-1) 2

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(group_size[0], device=attn.device)
            coords_w = torch.arange(group_size[1], device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Gh, Gw
            coords_flatten = torch.flatten(coords, 1)  # 2, Gh*Gw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Gh*Gw, Gh*Gw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Gh*Gw, Gh*Gw, 2
            relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Gh*Gw, Gh*Gw

            pos = self.pos(biases)  # 2Gh-1 * 2Gw-1, heads
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                group_size[0] * group_size[1], group_size[0] * group_size[1], -1)  # Gh*Gw,Gh*Gw,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Gh*Gw, Gh*Gw
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nP = mask.shape[0]
            attn = attn.view(B_ // nP, nP, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(
                0)  # (B, nP, nHead, N, N)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
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
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

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

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

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

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


    def forward(self, input, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = input*self.skip_scale + self.drop_path(self.self_attention(x))
        x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x


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
    
class VMM(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
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
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.input_resolution = input_resolution

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

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self.gating = ShuffleAttn(in_features=self.d_inner*4, out_features=self.d_inner*4, group=self.d_inner)

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

    def forward_core(self, x: torch.Tensor, 
                     mair_ids,
                     x_proj_bias: torch.Tensor=None,
                     ):
        # print(x.shape) C=360
        B, C, H, W = x.shape
        L = H * W
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        K=4
        # print("hello")
        xs = mair_ids_scan(x, mair_ids[0])

        x_dbl = F.conv1d(xs.reshape(B, -1, L), self.x_proj_weight.reshape(-1, D, 1), bias=(x_proj_bias.reshape(-1) if x_proj_bias is not None else None), groups=K)
        dts, Bs, Cs = torch.split(x_dbl.reshape(B, K, -1, L), [R, N, N], dim=2)
        dts = F.conv1d(dts.reshape(B, -1, L), self.dt_projs_weight.reshape(K * D, -1, 1), groups=K)
        
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        out_y = self.selective_scan(
            xs, dts,
            -torch.exp(self.A_logs.float()).view(-1, self.d_state), Bs, Cs, self.Ds.float().view(-1), z=None,
            delta_bias=self.dt_projs_bias.float().view(-1),
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return mair_ids_inverse(out_y, mair_ids[1], shape=(B, -1, H, W)) #B, C, L

    def forward(self, x: torch.Tensor, mair_ids, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x, mair_ids)
        assert y.dtype == torch.float32
        y = y * self.gating(y)
        y1, y2, y3, y4 = torch.chunk(y, 4, dim=1)
        y = y1 + y2 + y3 + y4
        y = y.permute(0, 2, 3, 1).contiguous()
        
        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        if self.dropout is not None:
            y = self.dropout()
        return y

    def flops_forward_core(self, H, W):
        flops = 0
        # flops of x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) in Core
        flops += 4 * (H * W) * self.d_inner * (self.dt_rank + self.d_state * 2)
        # flops of dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dt_rank=12, d_inner=360
        flops += 4 * (H * W) * self.dt_rank * self.d_inner
        # print(flops/1e6, (4 * H * W) * (self.d_state * self.d_state * 2)/1e6)
        # 610.46784 M 8.388608 M

        # Flops of discretization
        flops += (4 * H * W) * (self.d_state * self.d_state * 2)

        # Flops of Vmamba selective_scan
        # # h' = Ah(t) + Bx(t)
        # flops += (4 * H * W) * (self.d_state * self.d_state + self.d_inner * self.d_state)
        # # y = Ch(t) + DBx(t)
        # flops += (4 * H * W) * (self.d_inner * self.d_inner + self.d_inner * self.d_state)
        # 640*360*36*90*16/1e9=11.94G 
        flops += 4 * 9 * H * W * self.d_inner * self.d_state
        # print(4 * 9 * H * W * self.d_inner * self.d_state/1e9)
        return flops
    
    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # flop of in_proj
        flops += H * W * self.d_model * self.d_inner * 2
        # flops of x = self.act(self.conv2d(x))
        flops += H * W * self.d_inner * 3 * 3 + H * W * self.d_inner
        # print(H, W, self.d_state, self.d_inner)
        flops += self.flops_forward_core(H, W)
        # 64 64 16 360
        flops += self.gating.flops()
        # y = y1 + y2 + y3 + y4
        flops += 4 * H * W * self.d_inner
        # flops of y = self.out_norm(y)
        flops += H * W * self.d_inner
        # flops of y = y * F.silu(z)
        flops += 2 * H * W * self.d_inner

        # flops of out = self.out_proj(y)
        flops += H * W * self.d_inner * self.d_model

        return flops

class FB(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.select1 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.select2 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):  # 输入 BHWC 输出 BHWC
        B, H, W, C = x.shape
        skip = x
        x = x.permute(0, 3, 1, 2).contiguous()
        y = torch.fft.rfft2(x) + 1e-8
        a = torch.abs(y)
        p = torch.angle(y)
        a = self.select1(a)
        p = self.select2(p)
        real = a * torch.cos(p)
        imag = a * torch.sin(p)
        out = torch.complex(real, imag) + 1e-8
        out = torch.fft.irfft2(out, s=(H, W), norm='backward') + 1e-8
        out = torch.abs(out) + 1e-8

        out = out.permute(0, 2, 3, 1).contiguous()
        out = out + skip

        return out

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
        self.self_attention = VMM(d_model=hidden_dim, d_state=d_state,expand=ssm_ratio,dropout=attn_drop_rate, input_resolution=input_resolution, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.conv_blk = FeedForwardMoE(in_features=hidden_dim, hidden_features=mlp_hidden_dim,input_resolution=input_resolution)
        
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.hidden_dim = hidden_dim
        self.input_resolution = input_resolution

        self.shift_size = shift_size
        
        self.frequency = FB(hidden_dim)
        self.ln_3 = nn.LayerNorm(hidden_dim)
        #self.skip_scale3 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input, mair_ids, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]

        x = self.ln_1(input)
        if self.shift_size > 0:
            x = self.drop_path(self.self_attention(x, (mair_ids[2], mair_ids[3])) + self.frequency(self.ln_3(input)))
        else:
            x = self.drop_path(self.self_attention(x, (mair_ids[0], mair_ids[1])) + self.frequency(self.ln_3(input)))

        x = input + self.conv_blk(self.ln_2(x))
        
        #frequency branch
        #x_frequency = input*self.skip_scale3 + self.frequency(self.ln_3(input))

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
    

class MaIRUNetFB(nn.Module):
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

        super(MaIRUNetFB, self).__init__()
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

class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  #__init__部分是内部属性，而forward的输入才是外部输入
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w
        
        mean_c = img.mean(dim=1).unsqueeze(1)
        # stx()
        input = torch.cat([img,mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map

class MambaIRUNetRetinex(nn.Module):
    def __init__(self,
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

        super(MambaIRUNetRetinex, self).__init__()
        self.mlp_ratio = mlp_ratio
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        base_d_state = 4
        self.encoder_level1 = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=base_d_state* 2 ** 2,
            )
            for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
            )
            for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 2),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
            )
            for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 3),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state / 2 * 2 ** 3),
            )
            for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 2),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
            )
            for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
            )
            for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
            )
            for i in range(num_blocks[0])])

        self.refinement = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
            )
            for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.estimator = Illumination_Estimator(dim)

    def forward(self, inp_img):

        illu_fea, illu_map = self.estimator(inp_img)
        inp_img = inp_img * illu_map + inp_img

        _, _, H, W = inp_img.shape
        inp_enc_level1 = self.patch_embed(inp_img)  # b,hw,c
        out_enc_level1 = inp_enc_level1
        for layer in self.encoder_level1:
            out_enc_level1 = layer(out_enc_level1, [H, W])

        inp_enc_level2 = self.down1_2(out_enc_level1, H, W)  # b, hw//4, 2c
        out_enc_level2 = inp_enc_level2
        for layer in self.encoder_level2:
            out_enc_level2 = layer(out_enc_level2, [H // 2, W // 2])

        inp_enc_level3 = self.down2_3(out_enc_level2, H // 2, W // 2)  # b, hw//16, 4c
        out_enc_level3 = inp_enc_level3
        for layer in self.encoder_level3:
            out_enc_level3 = layer(out_enc_level3, [H // 4, W // 4])

        inp_enc_level4 = self.down3_4(out_enc_level3, H // 4, W // 4)  # b, hw//64, 8c
        latent = inp_enc_level4
        for layer in self.latent:
            latent = layer(latent, [H // 8, W // 8])

        inp_dec_level3 = self.up4_3(latent, H // 8, W // 8)  # b, hw//16, 4c
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 2)
        inp_dec_level3 = rearrange(inp_dec_level3, "b (h w) c -> b c h w", h=H // 4, w=W // 4).contiguous()
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        inp_dec_level3 = rearrange(inp_dec_level3, "b c h w -> b (h w) c").contiguous()  # b, hw//16, 4c
        out_dec_level3 = inp_dec_level3
        for layer in self.decoder_level3:
            out_dec_level3 = layer(out_dec_level3, [H // 4, W // 4])

        inp_dec_level2 = self.up3_2(out_dec_level3, H // 4, W // 4)  # b, hw//4, 2c
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b (h w) c -> b c h w", h=H // 2, w=W // 2).contiguous()
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b c h w -> b (h w) c").contiguous()  # b, hw//4, 2c
        out_dec_level2 = inp_dec_level2
        for layer in self.decoder_level2:
            out_dec_level2 = layer(out_dec_level2, [H // 2, W // 2])

        inp_dec_level1 = self.up2_1(out_dec_level2, H // 2, W // 2)  # b, hw, c
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 2)
        out_dec_level1 = inp_dec_level1
        for layer in self.decoder_level1:
            out_dec_level1 = layer(out_dec_level1, [H, W])

        for layer in self.refinement:
            out_dec_level1 = layer(out_dec_level1, [H, W])

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
    model = MaIRUNetFB(
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
