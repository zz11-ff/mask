import os
import cv2

folder = r"J:\MaskedDenoising-main\testset\MCM\HR"

for name in os.listdir(folder):
    path = os.path.join(folder, name)

    # 跳过非图像文件
    if not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        continue

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 读成灰度 (H,W)
    if img is None:
        print("无法读取图像:", path)
        continue

    # 保存为单通道灰度图（自动覆盖原图）
    cv2.imwrite(path, img)

    print("已转换:", name)

print("全部图像已经成功转换为灰度！")
import cv2
img = cv2.imread(r"J:\MaskedDenoising-main\PAtestset\HR1\532_OR_44_index0.jpeg", cv2.IMREAD_UNCHANGED)
print(img.shape)
