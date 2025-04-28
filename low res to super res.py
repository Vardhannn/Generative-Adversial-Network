import cv2
import matplotlib.pyplot as plt

# Load the original image
image_path = "G:/Damn/Project/data/image_29.png"  # Change this to your image path
original = cv2.imread(image_path)
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Resize to low-resolution (downscale)
scale_factor = 0.2 # Adjust as needed
low_res = cv2.resize(original, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

# Upscale back to original size (simulated super-resolution)
super_res =  cv2.resize(original, (0, 0), fx=0.38, fy=0.38, interpolation=cv2.INTER_LINEAR)

# Plot the images
plt.figure(figsize=(12, 4))
titles = ["Low Resolution", "Super Resolution", "Original"]
images = [low_res, super_res, original]

for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
