import os
import cv2

# 输入图片文件夹（你的 HR 或训练集）
input_folder = r"J:/MaskedDenoising-main/PAtestset/HR"        # ← 这里改成你的输入目录
output_folder = r"J:/MaskedDenoising-main/PAtestset/HR1"    # ← 输出目录

target_size = (1600,1600)  # 目标分辨率

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        img = cv2.imread(input_path)
        if img is None:
            print(f"读取失败，跳过：{input_path}")
            continue

        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_path, img_resized)

        print(f"已处理：{filename}")

print("全部图片已成功 resize 到 64×64 !")
