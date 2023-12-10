import cv2
import numpy as np
# Function to calculate image properties
def get_image_properties(image_path):
    image = cv2.imread(image_path)
    height, width, channels = image.shape

    # Calculating aspect ratio
    aspect_ratio = width / height if height != 0 else 0

    # Calculating average intensity per channel (Red, Green, Blue)
    avg_intensity_per_channel = np.mean(image, axis=(0, 1))
    avg_red_intensity = avg_intensity_per_channel[0]
    avg_green_intensity = avg_intensity_per_channel[1]
    avg_blue_intensity = avg_intensity_per_channel[2]

    # Calculating brightness and contrast
    brightness = np.mean(avg_intensity_per_channel)
    contrast = np.std(image)

    # Blurriness
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()

    return width, height, aspect_ratio, avg_red_intensity, avg_green_intensity, avg_blue_intensity, brightness, contrast, laplacian_var