
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def fourier_transform_rgb(img_rgb):
    """ 对RGB图像的每个通道进行傅里叶变换 """
    f_channels = []
    mag_channels = []
    phase_channels = []

    for i in range(3): # R, G, B
        channel = img_rgb[:, :, i]
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)
        
        f_channels.append(fshift)
        mag_channels.append(20 * np.log(np.abs(fshift) + 1))
        phase_channels.append(np.angle(fshift))
        
    return f_channels, np.mean(mag_channels, axis=0), np.mean(phase_channels, axis=0)

def inverse_fourier_transform_rgb(f_shifted_channels):
    """ 对每个通道的傅里叶频谱进行逆变换，重构RGB图像 """
    reconstructed_channels = []
    for f_shifted in f_shifted_channels:
        f_ishift = np.fft.ifftshift(f_shifted)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back) # 取绝对值
        
        # 归一化到0-255，并转为uint8
        reconstructed_channels.append(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))
    
    return cv2.merge(reconstructed_channels) # 合并R,G,B通道

def process_and_swap_fourier_rgb(low_light_path, normal_light_path, output_dir='fourier_swap_output_rgb'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 读取并转换色彩空间
    img_low_bgr = cv2.imread(low_light_path)
    img_normal_bgr = cv2.imread(normal_light_path)
    
    if img_low_bgr is None or img_normal_bgr is None:
        print("错误：无法读取图片，请检查路径。")
        return

    img_low_rgb = cv2.cvtColor(img_low_bgr, cv2.COLOR_BGR2RGB)
    img_normal_rgb = cv2.cvtColor(img_normal_bgr, cv2.COLOR_BGR2RGB)

    # 2. 傅里叶变换
    f_low_ch, mag_low_avg, phase_low_avg = fourier_transform_rgb(img_low_rgb)
    f_normal_ch, mag_normal_avg, phase_normal_avg = fourier_transform_rgb(img_normal_rgb)

    # 3. 交换幅度谱重构
    # 第一行结果：低光相位 (结构) + 正常光幅度 (亮度)
    re_low_struct_ch = [np.abs(f_normal_ch[i]) * np.exp(1j * np.angle(f_low_ch[i])) for i in range(3)]
    # 第二行结果：正常光相位 (结构) + 低光幅度 (亮度)
    re_normal_struct_ch = [np.abs(f_low_ch[i]) * np.exp(1j * np.angle(f_normal_ch[i])) for i in range(3)]

    res_1 = inverse_fourier_transform_rgb(re_low_struct_ch)
    res_2 = inverse_fourier_transform_rgb(re_normal_struct_ch)

    # ==========================================
    # 4. 2x4 布局可视化
    # 列1: 原图 | 列2: 幅度 | 列3: 相位 | 列4: 重建
    # ==========================================
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor='white')
    
    # --- 第一行：低光图像流程 ---
    axes[0, 0].imshow(img_low_rgb)
    axes[0, 0].set_title("Input: Low Light", fontsize=14)
    
    axes[0, 1].imshow(mag_low_avg, cmap='viridis')
    axes[0, 1].set_title("Amplitude (Low)", fontsize=14)
    
    axes[0, 2].imshow(phase_low_avg, cmap='viridis')
    axes[0, 2].set_title("Phase (Low)", fontsize=14)
    
    axes[0, 3].imshow(res_1)
    axes[0, 3].set_title("Reconstructed\n(Low Phase + Normal Amp)", fontsize=14, color='darkred')

    # --- 第二行：正常光图像流程 ---
    axes[1, 0].imshow(img_normal_rgb)
    axes[1, 0].set_title("Input: Normal Light", fontsize=14)
    
    axes[1, 1].imshow(mag_normal_avg, cmap='viridis')
    axes[1, 1].set_title("Amplitude (Normal)", fontsize=14)
    
    axes[1, 2].imshow(phase_normal_avg, cmap='viridis')
    axes[1, 2].set_title("Phase (Normal)", fontsize=14)
    
    axes[1, 3].imshow(res_2)
    axes[1, 3].set_title("Reconstructed\n(Normal Phase + Low Amp)", fontsize=14, color='darkblue')

    # 统一移除刻度并优化布局
    for ax in axes.flatten():
        ax.axis('off')

    plt.tight_layout()
    
    # 保存结果
    save_path = os.path.join(output_dir, 'fourier_comparison_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"矩阵对比图已保存至: {save_path}")


# ==============================================================================
# 使用示例 - 请替换为你的图像路径
# ==============================================================================
# 替换为你的实际图片路径
# 确保这两张图片是彩色的
low_light_image_path = 'LOLdataset/eval15/low/778.png' # 例如：一张暗的照片
normal_light_image_path = 'LOLdataset/eval15/high/778.png' # 例如：一张正常亮度的照片

# 如果没有测试图片，可以取消注释下面几行代码来生成简单的测试图片
'''
# 创建一个简单的彩色图片 (低光照)
test_low = np.zeros((200, 200, 3), dtype=np.uint8)
cv2.circle(test_low, (70, 70), 30, (0, 0, 100), -1) # 深蓝色圆
cv2.rectangle(test_low, (100, 100), (160, 160), (0, 80, 0), -1) # 深绿色方块
cv2.imwrite('low_light_color_image.jpg', test_low)

# 创建一个复杂的彩色图片 (正常光照)
test_normal = np.zeros((200, 200, 3), dtype=np.uint8) + 150 # 浅灰色背景
cv2.ellipse(test_normal, (100, 100), (60, 30), 0, 0, 360, (255, 0, 0), -1) # 红色椭圆
cv2.putText(test_normal, "Hello", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2) # 黄色文字
cv2.imwrite('normal_light_color_image.jpg', test_normal)

low_light_image_path = 'low_light_color_image.jpg'
normal_light_image_path = 'normal_light_color_image.jpg'
'''

# 执行函数
process_and_swap_fourier_rgb(low_light_image_path, normal_light_image_path)