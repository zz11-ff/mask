import cv2
import numpy as np
import matplotlib.pyplot as plt

def show(img, title):
    plt.figure(figsize=(5,5))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# 读取你的图像（替换路径）
img = cv2.imread("J:/MaskedDenoising-main/testset/HR/1.jpeg")
show(img, "Original")
def add_gaussian_noise(image, mean=0, sigma=50):
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

noisy_gaussian = add_gaussian_noise(img, mean=0, sigma=50)
show(noisy_gaussian, "Gaussian Noise")
def add_poisson_noise(image):
    img_float = image.astype(np.float32) / 255.0
    noisy = np.random.poisson(img_float * 255) / 255.0
    noisy = np.clip(noisy * 255, 0, 255).astype(np.uint8)
    return noisy

noisy_poisson = add_poisson_noise(img)
show(noisy_poisson, "Poisson Noise")
def add_speckle_noise(image):
    gauss = np.random.randn(*image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + image.astype(np.float32) * gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

noisy_speckle = add_speckle_noise(img)
show(noisy_speckle, "Speckle Noise")

