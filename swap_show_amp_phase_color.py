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
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 读取并处理两张RGB图像
    img_low_rgb = cv2.imread(low_light_path)
    img_normal_rgb = cv2.imread(normal_light_path)

    if img_low_rgb is None or img_normal_rgb is None:
        print("无法加载一张或多张图像，请检查路径。")
        return

    # 将BGR转换为RGB (OpenCV默认读取为BGR)
    img_low_rgb = cv2.cvtColor(img_low_rgb, cv2.COLOR_BGR2RGB)
    img_normal_rgb = cv2.cvtColor(img_normal_rgb, cv2.COLOR_BGR2RGB)

    # 获取文件名作为前缀
    base_low = os.path.splitext(os.path.basename(low_light_path))[0]
    base_normal = os.path.splitext(os.path.basename(normal_light_path))[0]

    # 可视化原始图像 (RGB)
    plt.figure(figsize=(12, 6), facecolor='white')
    plt.subplot(121), plt.imshow(img_low_rgb), plt.title(f'Original: {base_low} (RGB)')
    plt.axis('off')
    plt.subplot(122), plt.imshow(img_normal_rgb), plt.title(f'Original: {base_normal} (RGB)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'0_original_rgb_images.png'), dpi=300)
    plt.show()

    # 2. 对两张图像进行傅里叶变换 (R, G, B 各通道)
    f_low_channels, mag_low_avg, phase_low_avg = fourier_transform_rgb(img_low_rgb)
    f_normal_channels, mag_normal_avg, phase_normal_avg = fourier_transform_rgb(img_normal_rgb)

    # 3. 可视化并保存幅度图和相位图 (显示平均值)
    plt.figure(figsize=(18, 12), facecolor='white')

    plt.subplot(231), plt.imshow(img_low_rgb), plt.title(f'{base_low} Original (RGB)')
    plt.axis('off')
    plt.subplot(232), plt.imshow(mag_low_avg, cmap='viridis'), plt.title(f'{base_low} Amplitude (Avg)')
    plt.axis('off')
    plt.subplot(233), plt.imshow(phase_low_avg, cmap='viridis'), plt.title(f'{base_low} Phase (Avg)')
    plt.axis('off')

    plt.subplot(234), plt.imshow(img_normal_rgb), plt.title(f'{base_normal} Original (RGB)')
    plt.axis('off')
    plt.subplot(235), plt.imshow(mag_normal_avg, cmap='viridis'), plt.title(f'{base_normal} Amplitude (Avg)')
    plt.axis('off')
    plt.subplot(236), plt.imshow(phase_normal_avg, cmap='viridis'), plt.title(f'{base_normal} Phase (Avg)')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'1_amplitude_phase_visualization_rgb.png'), dpi=300)
    plt.show()

    # 另外保存单独的幅度图和相位图 (不带标题，直接imsave)
    plt.imsave(os.path.join(output_dir, f'{base_low}_amplitude_avg.png'), mag_low_avg, cmap='viridis')
    plt.imsave(os.path.join(output_dir, f'{base_low}_phase_avg.png'), phase_low_avg, cmap='viridis')
    plt.imsave(os.path.join(output_dir, f'{base_normal}_amplitude_avg.png'), mag_normal_avg, cmap='viridis')
    plt.imsave(os.path.join(output_dir, f'{base_normal}_phase_avg.png'), phase_normal_avg, cmap='viridis')

    # 4. 交换幅度谱并重构傅里叶变换后的频谱 (逐通道操作)
    reconstructed_low_channels = []
    reconstructed_normal_channels = []

    for i in range(3): # 对R, G, B每个通道进行交换
        # img_low 的新频谱：用 img_normal 的幅度，img_low 的相位
        f_low_new_channel_spectrum = np.abs(f_normal_channels[i]) * np.exp(1j * np.angle(f_low_channels[i]))
        reconstructed_low_channels.append(f_low_new_channel_spectrum)
        
        # img_normal 的新频谱：用 img_low 的幅度，img_normal 的相位
        f_normal_new_channel_spectrum = np.abs(f_low_channels[i]) * np.exp(1j * np.angle(f_normal_channels[i]))
        reconstructed_normal_channels.append(f_normal_new_channel_spectrum)

    # 5. 使用逆傅里叶变换生成新图像 (合并通道)
    reconstructed_img_low_with_normal_amp_rgb = inverse_fourier_transform_rgb(reconstructed_low_channels)
    reconstructed_img_normal_with_low_amp_rgb = inverse_fourier_transform_rgb(reconstructed_normal_channels)

    # 6. 可视化并保存最终生成的图像 (RGB)
    plt.figure(figsize=(12, 6), facecolor='white')
    plt.subplot(121), plt.imshow(reconstructed_img_low_with_normal_amp_rgb)
    plt.title(f'Reconstructed image with normal amp: {base_low} (with {base_normal} Amp)')
    plt.axis('off')
    plt.subplot(122), plt.imshow(reconstructed_img_normal_with_low_amp_rgb)
    plt.title(f'Reconstructed image with low amp: {base_normal} (with {base_low} Amp)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'2_reconstructed_rgb_images.png'), dpi=300)
    plt.show()
    
    # 额外保存重建图像的单独文件
    # 注意：matplotlib的imshow自动处理RGB，但cv2.imwrite默认期望BGR，所以需要转换
    cv2.imwrite(os.path.join(output_dir, f'{base_low}_reconstructed_with_{base_normal}_amp_rgb.png'), cv2.cvtColor(reconstructed_img_low_with_normal_amp_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, f'{base_normal}_reconstructed_with_{base_low}_amp_rgb.png'), cv2.cvtColor(reconstructed_img_normal_with_low_amp_rgb, cv2.COLOR_RGB2BGR))

    print("-" * 30)
    print("RGB傅里叶变换与幅度交换处理完成！")
    print(f"所有结果已保存至目录: {output_dir}")
    print("-" * 30)

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