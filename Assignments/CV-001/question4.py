import cv2
import numpy as np
import matplotlib.pyplot as plt

# ✅ Correct path (raw string or double slashes or forward slashes)
image = cv2.imread(r"D:\7th\CV\Github\Computer-Vision\General\Elements\fast.jpeg", cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("Image not found. Check the file path.")

# --- Sobel Operator ---
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel = np.sqrt(sobel_x**2 + sobel_y**2)

# --- Prewitt Operator ---
prewitt_kernel_x = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]], dtype=np.float32)

prewitt_kernel_y = np.array([[-1,-1,-1],
                             [ 0, 0, 0],
                             [ 1, 1, 1]], dtype=np.float32)

prewitt_x = cv2.filter2D(image, -1, prewitt_kernel_x)
prewitt_y = cv2.filter2D(image, -1, prewitt_kernel_y)
prewitt = np.sqrt(prewitt_x.astype(np.float32)**2 + prewitt_y.astype(np.float32)**2)

# --- Visualization ---
titles = ["Original", "Sobel Edge Map", "Prewitt Edge Map"]
images = [image, sobel, prewitt]

plt.figure(figsize=(10,4))
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis("off")
plt.tight_layout()
plt.show()
