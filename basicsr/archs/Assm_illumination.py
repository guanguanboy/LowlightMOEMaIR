import torch
import torch.nn as nn
import torch.nn.functional as F

def index_reverse(indices):
    """正确实现的逆排序索引生成"""
    # 创建与indices相同形状的全零张量
    inv_indices = torch.zeros_like(indices)
    # 生成顺序索引 [0,1,2,...,n-1]
    arange = torch.arange(indices.size(1), device=indices.device).view(1, -1)
    # 使用scatter将顺序索引放到正确位置
    inv_indices.scatter_(1, indices, arange)
    return inv_indices

class Selective_Scan_Illum(nn.Module):
    """光照感知的选择性扫描模块"""
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state+1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x, illum_prompt):
        B, L, _ = x.shape
        # 光照条件调制
        gamma = 0.3 + 0.7 * torch.sigmoid(illum_prompt[..., :1])  # [B,L,1]
        beta = -0.5 + illum_prompt[..., 1:2]  # [B,L,1]
        x = gamma * x + beta
        
        # 简化的选择性扫描实现
        A = -torch.exp(self.A_log.float())
        h = torch.zeros(B, self.d_state, device=x.device)
        outputs = []
        for i in range(L):
            h = h * torch.exp(A * illum_prompt[:, i, 2:3]) + x[:, i]
            outputs.append(h @ self.A_log.T + self.D * x[:, i])
        return torch.stack(outputs, dim=1)

class ASSM_Illumination(nn.Module):
    def __init__(self, dim, d_state, input_resolution, num_tokens=8):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.d_state = d_state
        self.H, self.W = input_resolution
        
        # 光照特征提取
        self.illum_proj = nn.Sequential(
            nn.Conv2d(1, dim//4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim//4, dim, kernel_size=3, padding=1)
        )
        
        # 显式光照字典
        self.register_buffer('illum_embedding', 
                           torch.linspace(0, 1, num_tokens).view(-1, 1))
        
        # 路由网络
        self.route = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.GELU(),
            nn.Linear(dim//2, num_tokens),
            nn.Softmax(dim=-1)
        )
        
        # 核心处理模块
        self.selective_scan = Selective_Scan_Illum(dim, d_state)
        self.norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, C = x.shape
        # 从RGB提取光照强度
        rgb = x.view(B, self.H, self.W, C).permute(0, 3, 1, 2)
        intensity = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
        intensity = intensity.unsqueeze(1)  # [B,1,H,W]
        
        # 光照特征提取
        illum_feat = self.illum_proj(intensity)  # [B,dim,H,W]
        illum_feat = illum_feat.view(B, self.dim, -1).permute(0, 2, 1)  # [B,L,dim]
        
        # 光照感知路由
        route_weights = self.route(x + 0.3 * illum_feat)
        flat_intensity = intensity.view(B, -1, 1)  # [B,L,1]
        illum_similarity = 1 - torch.abs(self.illum_embedding.T - flat_intensity)
        cls_policy = F.gumbel_softmax(route_weights * illum_similarity, hard=True, dim=-1)
        
        # 生成光照提示
        prompt = torch.cat([
            torch.matmul(cls_policy, self.illum_embedding),
            route_weights
        ], dim=-1)
        
        # 分组处理
        _, sort_idx = torch.sort(prompt[..., 0], dim=1)
        rev_idx = index_reverse(sort_idx)
        
        # 选择性扫描
        x_sorted = torch.gather(x, 1, sort_idx.unsqueeze(-1).expand(-1, -1, self.dim))
        y = self.selective_scan(x_sorted, torch.gather(prompt, 1, sort_idx.unsqueeze(-1).expand(-1, -1, prompt.size(-1))))
        
        # 恢复顺序并输出
        return torch.gather(self.output_proj(self.norm(y)), 1, rev_idx.unsqueeze(-1).expand(-1, -1, self.dim))

# 测试代码
if __name__ == "__main__":
    # 配置参数
    dim = 64
    d_state = 16
    H, W = 32, 32
    num_tokens = 8
    batch_size = 2
    
    # 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASSM_Illumination(dim, d_state, (H,W), num_tokens).to(device)
    
    # 测试数据
    x = torch.randn(batch_size, H*W, dim).to(device)
    
    # 运行测试
    try:
        output = model(x)
        print(f"Success! Input: {x.shape}, Output: {output.shape}")
        
        # 验证反向传播
        loss = output.mean()
        loss.backward()
        print("Backward pass successful!")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Debug info:")
        print(f"Device: {device}")
        print(f"Torch version: {torch.__version__}")