import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def process_image_fourier(image_path, output_dir='output'):
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print("无法加载图像，请检查路径。")
        return
    
    # 将BGR转换为RGB (OpenCV默认读取为BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 准备存储结果的容器
    magnitude_spectrum_all = []
    phase_spectrum_all = []

    # 2. 对每个通道 (R, G, B) 分别进行傅里叶变换
    for i in range(3):
        channel = img_rgb[:, :, i]
        
        # 执行快速傅里叶变换 (FFT)
        f = np.fft.fft2(channel)
        
        # 将低频部分移到频谱中心
        fshift = np.fft.fftshift(f)
        
        # 计算幅值 (Magnitude) - 使用对数变换以便可视化
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        # 计算相位 (Phase)
        phase_spectrum = np.angle(fshift)
        
        magnitude_spectrum_all.append(magnitude_spectrum)
        phase_spectrum_all.append(phase_spectrum)

    # 3. 可视化与保存
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原图
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original RGB Image')
    axes[0].axis('off')
    
    # 幅值图 (取三个通道的平均值展示，或只展示一个代表性通道)
    # 这里为了直观，合并三个通道的幅值
    mag_combined = np.mean(magnitude_spectrum_all, axis=0)
    axes[1].imshow(mag_combined, cmap='viridis')
    axes[1].set_title('Amplitude (Magnitude) Spectrum')
    axes[1].axis('off')
    
    # 相位图
    phase_combined = np.mean(phase_spectrum_all, axis=0)
    axes[2].imshow(phase_combined, cmap='viridis')
    axes[2].set_title('Phase Spectrum')
    axes[2].axis('off')

    # 保存结果图
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'low778_fourier_analysis.png'), dpi=300)
    plt.show()

    print(f"处理完成！结果已保存至: {output_dir}")

# 使用示例
process_image_fourier('LOLdataset/eval15/low/778.png')