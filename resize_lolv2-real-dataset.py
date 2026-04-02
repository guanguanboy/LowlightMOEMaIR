import os
from pathlib import Path
from PIL import Image

# ================= 配置区域 =================
# 原始图片文件夹路径
INPUT_DIR = '/data/lgl/datasets/LowLight/LOL-v2/Real_captured/Test' 
# 处理后的图片存放路径
OUTPUT_DIR = '/data/lgl/datasets/LowLight/LOL-v2/Real_captured/Test384'
# 目标尺寸
TARGET_SIZE = (384, 384)
# 支持的图片后缀
EXTENSIONS = {'.png'}
# ===========================================

def resize_with_pilllow():
    src_root = Path(INPUT_DIR)
    dst_root = Path(OUTPUT_DIR)

    if not src_root.exists():
        print(f"错误：找不到输入目录 {INPUT_DIR}")
        return

    print(f"开始处理图片 (使用 Pillow)...")
    count = 0
    
    # 递归遍历所有文件
    for path in src_root.rglob('*'):
        # 只处理指定后缀的文件
        if path.suffix.lower() in EXTENSIONS:
            # 1. 计算输出路径并创建子目录
            relative_path = path.relative_to(src_root)
            save_path = dst_root / relative_path
            save_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                # 2. 打开图片
                with Image.open(path) as img:
                    # 3. 转换模式（防止 PNG 转 JPEG 报错或颜色异常）
                    # 如果是 RGBA (透明) 转 JPEG，需要转为 RGB
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    
                    # 4. Resize
                    # Image.LANCZOS 是 Pillow 中质量最高的缩小算法
                    resized_img = img.resize(TARGET_SIZE, Image.LANCZOS)
                    
                    # 5. 保存
                    # 自动根据后缀识别格式，如果是 JPEG 可以增加 quality 参数
                    resized_img.save(save_path, quality=95)
                    
                count += 1
                if count % 10 == 0: # 每10张打印一次进度
                    print(f"进度: 已处理 {count} 张")
                    
            except Exception as e:
                print(f"无法处理图片 {path}: {e}")

    print("-" * 30)
    print(f"任务完成！总计处理: {count} 张图片")
    print(f"输出目录: {dst_root.absolute()}")

if __name__ == "__main__":
    resize_with_pilllow()