import torch
import torch.nn as nn

import transformers
print(transformers.__version__)

def verify_mamba_installation():
    print("--- 正在开始验证 Mamba 环境 ---")
    
    # 1. 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("❌ 错误: 未检测到可用 GPU。Mamba 需要 CUDA 环境。")
        return
    print(f"✅ CUDA 检测成功: {torch.version.cuda}")

    # 2. 检查 causal_conv1d
    try:
        import causal_conv1d
        from causal_conv1d import causal_conv1d_fn
        print(f"✅ causal_conv1d 导入成功 (版本: {causal_conv1d.__version__})")
    except ImportError as e:
        print(f"❌ causal_conv1d 导入失败: {e}")
        return

    # 3. 检查 mamba_ssm
    try:
        from mamba_ssm import Mamba
        print(f"✅ mamba_ssm 导入成功")
    except ImportError as e:
        print(f"❌ mamba_ssm 导入失败: {e}")
        return

    # 4. 运行简单的前向测试 (验证算子兼容性)
    print("--- 正在运行算子测试 (前向传播) ---")
    try:
        device = "cuda"
        batch, length, dim = 2, 64, 128
        x = torch.randn(batch, length, dim).to(device)
        
        # 初始化一个简单的 Mamba 层
        model = Mamba(
            d_model=dim,    # 嵌入维度
            d_state=16,     # SSM 状态维度
            d_conv=4,       # 本地卷积宽度
            expand=2,       # 扩展因子
        ).to(device)
        
        y = model(x)
        
        if y.shape == x.shape:
            print("🚀 验证通过! Mamba 算子在 GPU 上运行正常。")
        else:
            print(f"⚠️ 警告: 输出形状不匹配 {y.shape} vs {x.shape}")
            
    except Exception as e:
        print(f"❌ 运行测试失败! 可能是因为 whl 与当前的 PyTorch/CUDA 版本不兼容。")
        print(f"具体错误信息: {e}")

if __name__ == "__main__":
    verify_mamba_installation()