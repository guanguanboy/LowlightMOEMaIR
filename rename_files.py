import os

def batch_rename_remove_prefix(folder_path, prefix):
    """
    重命名指定文件夹下的所有 .png 文件，移除指定的前缀。
    """
    # 确保路径存在
    if not os.path.exists(folder_path):
        print(f"Error: 文件夹路径 '{folder_path}' 不存在。")
        return

    count = 0
    # 遍历文件夹
    for filename in os.listdir(folder_path):
        # 仅处理以指定前缀开头且后缀为 .png 的文件
        if filename.startswith(prefix) and filename.lower().endswith('.png'):
            # 构建新的文件名
            new_name = filename[len(prefix):]
            
            # 获取完整路径
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)

            # 执行重命名
            try:
                os.rename(old_path, new_path)
                print(f"成功: {filename} -> {new_name}")
                count += 1
            except Exception as e:
                print(f"失败: 无法重命名 {filename}. 错误: {e}")

    print(f"\n处理完成！共重命名了 {count} 个文件。")

# --- 使用示例 ---
if __name__ == "__main__":
    # 请根据您的实际路径和需要删除的前缀进行修改
    target_dir = r'/data/lgl/datasets/LowLight/LOL-v2/Real_captured/Test/Normal' 
    prefix_to_remove = 'normal'
    
    batch_rename_remove_prefix(target_dir, prefix_to_remove)