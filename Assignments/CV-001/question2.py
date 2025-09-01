import numpy as np

# Example image (3x3)
image = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

# Non-symmetric kernel
kernel = np.array([[1, 0],
                   [0, -1]])

# Manual correlation (no flip of kernel)
def correlation2d(img, ker):
    h, w = ker.shape
    out = np.zeros((img.shape[0]-h+1, img.shape[1]-w+1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(img[i:i+h, j:j+w] * ker)
    return out

# Manual convolution (flip kernel before applying)
def convolution2d(img, ker):
    ker_flipped = np.flipud(np.fliplr(ker))  # rotate 180°
    return correlation2d(img, ker_flipped)

# Run both
corr_out = correlation2d(image, kernel)
conv_out = convolution2d(image, kernel)

print("Correlation:\n", corr_out)
print("Convolution:\n", conv_out)
