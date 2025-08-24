import cv2
import numpy as np
import matplotlib.pyplot as plt

# Open an image file
image = cv2.imread(r"D:\7th\CV\Github\Computer-Vision\General\Elements\random people.jpg")
if image is None:
    raise FileNotFoundError("Image not found. Check the path again.")

orig_image = image.copy()

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints in the image
keypoints = sift.detect(gray_image, None)

# Draw keypoints on the original image
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None,
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Apply thresholding to the image
_, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Rotate the image by 90 degrees clockwise
height, width = image.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 90, 1)
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

# Add text on top of the rotated image
image_with_text = cv2.putText(rotated_image, 'Sample Text', (50, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Calculate histogram of the grayscale image
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

# Apply face detection to the image
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw bounding boxes around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# ---------------- DISPLAY RESULTS ----------------
plt.figure(figsize=(15,10))

plt.subplot(2,3,1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))

plt.subplot(2,3,2)
plt.title("SIFT Keypoints")
plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))

plt.subplot(2,3,3)
plt.title("Thresholded Image")
plt.imshow(thresholded_image, cmap="gray")

plt.subplot(2,3,4)
plt.title("Rotated + Text")
plt.imshow(cv2.cvtColor(image_with_text, cv2.COLOR_BGR2RGB))

plt.subplot(2,3,5)
plt.title("Face Detection")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(2,3,6)
plt.title("Histogram")
plt.plot(hist)
plt.xlim([0,256])

plt.show()
