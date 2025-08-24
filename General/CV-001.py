from PIL import Image, ImageFilter, ImageOps
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = Image.open(r"D:\7th\CV\Github\Computer-Vision\General\Elements\fast.jpeg")

# Convert to numpy array for manipulation
image_array = np.array(image)

# Create manipulated image (invert colors)
manipulated_image_array = 255 - image_array

# Convert back to PIL Image
manipulated_image = Image.fromarray(manipulated_image_array)

# Create rotated image
rotated_image = image.rotate(90)

# Create resized image
resized_image = image.resize((image.width // 2, image.height // 2))

# Create blurred image
blurred_image = image.filter(ImageFilter.GaussianBlur(radius=2))

# Create contour image
contour_image = ImageOps.invert(image.filter(ImageFilter.CONTOUR))

# Create subplot figure
fig, axes = plt.subplots(2, 3, figsize=(10, 10))
fig.suptitle("Image Manipulations")

# Display original image
axes[0, 0].imshow(image)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis("off")

# Display rotated image
axes[0, 1].imshow(rotated_image)
axes[0, 1].set_title("Rotated Image")
axes[0, 1].axis("off")

# Display resized image
axes[0, 2].imshow(resized_image)
axes[0, 2].set_title("Resized Image")
axes[0, 2].axis("off")

# Display blurred image
axes[1, 0].imshow(blurred_image)
axes[1, 0].set_title("Blurred Image")
axes[1, 0].axis("off")

# Display manipulated image
axes[1, 1].imshow(manipulated_image)
axes[1, 1].set_title("Manipulated Image")
axes[1, 1].axis("off")

# Display contour image
axes[1, 2].imshow(contour_image)
axes[1, 2].set_title("Contour Image")
axes[1, 2].axis("off")

# Show the plot
plt.show(block=True)