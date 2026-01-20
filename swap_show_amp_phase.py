import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def fourier_transform_image(image_channel):
    """ 对单个图像通道进行傅里叶变换并返回傅里叶谱、幅值谱和相位谱 """
    f = np.fft.fft2(image_channel)
    fshift = np.fft.fftshift(f)
    
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    phase_spectrum = np.angle(fshift)
    
    return fshift, magnitude_spectrum, phase_spectrum

def inverse_fourier_transform_image(fourier_shifted_spectrum):
    """ 对傅里叶变换后的频谱进行逆傅里叶变换，返回重构图像 """
    f_ishift = np.fft.ifftshift(fourier_shifted_spectrum)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back) # 取绝对值，因为可能包含复数部分
    
    # 归一化到0-255，以便显示
    img_back = np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))
    return img_back

def process_and_swap_fourier(low_light_path, normal_light_path, output_dir='fourier_swap_output'):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 读取并处理两张图像
    img_low = cv2.imread(low_light_path, cv2.IMREAD_GRAYSCALE) # 为简化处理，使用灰度图
    img_normal = cv2.imread(normal_light_path, cv2.IMREAD_GRAYSCALE)

    if img_low is None or img_normal is None:
        print("无法加载一张或多张图像，请检查路径。")
        return

    # 获取文件名作为前缀
    base_low = os.path.splitext(os.path.basename(low_light_path))[0]
    base_normal = os.path.splitext(os.path.basename(normal_light_path))[0]

    # 可视化原始图像
    plt.figure(figsize=(12, 6), facecolor='white')
    plt.subplot(121), plt.imshow(img_low, cmap='gray'), plt.title(f'Original: {base_low}')
    plt.axis('off')
    plt.subplot(122), plt.imshow(img_normal, cmap='gray'), plt.title(f'Original: {base_normal}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'0_original_images.png'), dpi=300)
    plt.show()

    # 2. 对两张图像进行傅里叶变换
    f_low, mag_low, phase_low = fourier_transform_image(img_low)
    f_normal, mag_normal, phase_normal = fourier_transform_image(img_normal)

    # 3. 可视化并保存幅度图和相位图
    plt.figure(figsize=(18, 12), facecolor='white')

    plt.subplot(231), plt.imshow(img_low, cmap='gray'), plt.title(f'{base_low} Original')
    plt.axis('off')
    plt.subplot(232), plt.imshow(mag_low, cmap='viridis'), plt.title(f'{base_low} Amplitude')
    plt.axis('off')
    plt.subplot(233), plt.imshow(phase_low, cmap='viridis'), plt.title(f'{base_low} Phase')
    plt.axis('off')

    plt.subplot(234), plt.imshow(img_normal, cmap='gray'), plt.title(f'{base_normal} Original')
    plt.axis('off')
    plt.subplot(235), plt.imshow(mag_normal, cmap='viridis'), plt.title(f'{base_normal} Amplitude')
    plt.axis('off')
    plt.subplot(236), plt.imshow(phase_normal, cmap='viridis'), plt.title(f'{base_normal} Phase')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'1_amplitude_phase_visualization.png'), dpi=300)
    plt.show()

    # 另外保存单独的幅度图和相位图 (不带标题，直接imsave)
    plt.imsave(os.path.join(output_dir, f'{base_low}_amplitude.png'), mag_low, cmap='viridis')
    plt.imsave(os.path.join(output_dir, f'{base_low}_phase.png'), phase_low, cmap='viridis')
    plt.imsave(os.path.join(output_dir, f'{base_normal}_amplitude.png'), mag_normal, cmap='viridis')
    plt.imsave(os.path.join(output_dir, f'{base_normal}_phase.png'), phase_normal, cmap='viridis')

    # 4. 交换幅度谱并重构傅里叶变换后的频谱
    # 新的 f_low_reconstructed 使用 img_normal 的幅度 和 img_low 的相位
    # 由于幅度是 np.abs(F) = 20 * np.log(np.abs(fshift) + 1)，所以要从原始fshift中提取
    # 然后用 np.abs(F_normal) * np.exp(1j * np.angle(F_low)) 来构建新的傅里叶频谱
    
    # 原始的傅里叶频谱 (中心化后的)
    f_low_shifted_original = np.fft.fftshift(np.fft.fft2(img_low))
    f_normal_shifted_original = np.fft.fftshift(np.fft.fft2(img_normal))

    # 构建交换后的频谱
    # img_low 的新频谱：用 img_normal 的幅度，img_low 的相位
    f_low_new_spectrum = np.abs(f_normal_shifted_original) * np.exp(1j * np.angle(f_low_shifted_original))
    
    # img_normal 的新频谱：用 img_low 的幅度，img_normal 的相位
    f_normal_new_spectrum = np.abs(f_low_shifted_original) * np.exp(1j * np.angle(f_normal_shifted_original))

    # 5. 使用逆傅里叶变换生成新图像
    reconstructed_img_low_with_normal_amp = inverse_fourier_transform_image(f_low_new_spectrum)
    reconstructed_img_normal_with_low_amp = inverse_fourier_transform_image(f_normal_new_spectrum)

    # 6. 可视化并保存最终生成的图像
    plt.figure(figsize=(12, 6), facecolor='white')
    plt.subplot(121), plt.imshow(reconstructed_img_low_with_normal_amp, cmap='gray')
    plt.title(f'Reconstructed: {base_low} (with {base_normal} Amplitude)')
    plt.axis('off')
    plt.subplot(122), plt.imshow(reconstructed_img_normal_with_low_amp, cmap='gray')
    plt.title(f'Reconstructed: {base_normal} (with {base_low} Amplitude)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'2_reconstructed_images.png'), dpi=300)
    plt.show()
    
    # 额外保存重建图像的单独文件
    cv2.imwrite(os.path.join(output_dir, f'{base_low}_reconstructed_with_{base_normal}_amp.png'), reconstructed_img_low_with_normal_amp)
    cv2.imwrite(os.path.join(output_dir, f'{base_normal}_reconstructed_with_{base_low}_amp.png'), reconstructed_img_normal_with_low_amp)

    print("-" * 30)
    print("傅里叶变换与幅度交换处理完成！")
    print(f"所有结果已保存至目录: {output_dir}")
    print("-" * 30)

# ==============================================================================
# 使用示例 - 请替换为你的图像路径
# ==============================================================================
# 创建一些假图像用于测试，或者使用你自己的图片路径
# 如果你没有图片，可以取消注释下面几行代码来生成测试图片
'''
# 生成一个简单的圆形图像 (模拟低光照)
img_low_test = np.zeros((200, 200), dtype=np.uint8)
cv2.circle(img_low_test, (100, 100), 50, 100, -1) # 灰色圆

# 生成一个带有文本的图像 (模拟正常光照)
img_normal_test = np.zeros((200, 200), dtype=np.uint8) + 50 # 背景色
cv2.putText(img_normal_test, "TEXT", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 200, 3) # 白色文字

cv2.imwrite('test_low_light.png', img_low_test)
cv2.imwrite('test_normal_light.png', img_normal_test)

low_light_image_path = 'test_low_light.png'
normal_light_image_path = 'test_normal_light.png'
'''

# 替换为你的实际图片路径
low_light_image_path = 'LOLdataset/eval15/low/778.png' # 例如：一张暗的照片
normal_light_image_path = 'LOLdataset/eval15/high/778.png' # 例如：一张正常亮度的照片

# 执行函数
process_and_swap_fourier(low_light_image_path, normal_light_image_path)