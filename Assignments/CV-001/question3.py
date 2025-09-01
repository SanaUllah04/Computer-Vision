import cv2
import numpy as np
import matplotlib.pyplot as plt

# ✅ Correct file path (choose one of the below styles)
image = cv2.imread(r"D:\7th\CV\Github\Computer-Vision\General\Elements\fast.jpeg", cv2.IMREAD_GRAYSCALE)
# image = cv2.imread("D:\\7th\\CV\\Github\\Computer-Vision\\General\\Elements\\fast.jpeg", cv2.IMREAD_GRAYSCALE)
# image = cv2.imread("D:/7th/CV/Github/Computer-Vision/General/Elements/fast.jpeg", cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("Image not found. Check the file path.")

# --- Add Gaussian Noise ---
def add_gaussian_noise(img, mean=0, var=20):
    noise = np.random.normal(mean, var**0.5, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

# --- Add Salt & Pepper Noise ---
def add_salt_pepper_noise(img, prob=0.02):
    noisy = img.copy()
    rnd = np.random.rand(*img.shape)
    noisy[rnd < prob/2] = 0       # pepper
    noisy[rnd > 1 - prob/2] = 255 # salt
    return noisy

# Noisy images
gaussian_noisy = add_gaussian_noise(image)
sp_noisy = add_salt_pepper_noise(image)

# --- Filtering ---
gaussian_filtered  = cv2.GaussianBlur(gaussian_noisy, (5,5), 1)
median_filtered    = cv2.medianBlur(sp_noisy, 5)
bilateral_filtered = cv2.bilateralFilter(gaussian_noisy, 9, 75, 75)

# --- Visualization ---
titles = ["Original", "Gaussian Noise", "Salt & Pepper Noise", 
          "Gaussian Filter", "Median Filter", "Bilateral Filter"]
images = [image, gaussian_noisy, sp_noisy, 
          gaussian_filtered, median_filtered, bilateral_filtered]

plt.figure(figsize=(12,6))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis("off")
plt.tight_layout()
plt.show()
