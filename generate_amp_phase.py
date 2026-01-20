import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def process_and_save_fourier(image_path, output_dir='output'):
    # 1. 获取文件名（不带扩展名）
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 2. 读取并转换图像
    img = cv2.imread(image_path)
    if img is None:
        print("无法加载图像，请检查路径。")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3. 对每个通道进行变换
    mags, phases = [], []
    for i in range(3):
        f = np.fft.fft2(img_rgb[:, :, i])
        fshift = np.fft.fftshift(f)
        mags.append(20 * np.log(np.abs(fshift) + 1))
        phases.append(np.angle(fshift))

    # 取平均值得到最终的可视化矩阵
    mag_combined = np.mean(mags, axis=0)
    phase_combined = np.mean(phases, axis=0)

    # 4. 分别保存为单独的图片
    # 使用 viridis 映射并直接保存
    
    # 保存 Amplitude (幅值图)
    amp_filename = os.path.join(output_dir, f"{base_name}_amplitude.png")
    plt.imsave(amp_filename, mag_combined, cmap='viridis')
    
    # 保存 Phase (相位图)
    phase_filename = os.path.join(output_dir, f"{base_name}_phase.png")
    plt.imsave(phase_filename, phase_combined, cmap='viridis')

    print("-" * 30)
    print(f"原始图像: {os.path.basename(image_path)}")
    print(f"已保存幅值图: {amp_filename}")
    print(f"已保存相位图: {phase_filename}")
    print("-" * 30)

# 使用示例
process_and_save_fourier('LOLdataset/eval15/high/778.png')