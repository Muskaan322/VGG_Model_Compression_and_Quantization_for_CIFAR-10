
"""
**Pillow is the main library used for handling image zooming in and zoom out, We can install it using pip install pillow**
##Task1
"""

from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

import matplotlib.pyplot as plt
from PIL import Image

def resize_image_handler(original_image, target_width, target_height):
    original_width, original_height = original_image.size
    resized_image = Image.new("RGB", (target_width, target_height))

    for new_y in range(target_height):
        for new_x in range(target_width):
            x_ratio = (original_width - 1) * (new_x / (target_width - 1))
            y_ratio = (original_height - 1) * (new_y / (target_height - 1))

            x0 = int(x_ratio)
            x1 = min(x0 + 1, original_width - 1)
            y0 = int(y_ratio)
            y1 = min(y0 + 1, original_height - 1)

            pixel_00 = original_image.getpixel((x0, y0))
            pixel_01 = original_image.getpixel((x1, y0))
            pixel_10 = original_image.getpixel((x0, y1))
            pixel_11 = original_image.getpixel((x1, y1))

            x_weight = x_ratio - x0
            y_weight = y_ratio - y0

            red = (pixel_00[0] * (1 - x_weight) * (1 - y_weight) +
                   pixel_01[0] * x_weight * (1 - y_weight) +
                   pixel_10[0] * (1 - x_weight) * y_weight +
                   pixel_11[0] * x_weight * y_weight)
            green = (pixel_00[1] * (1 - x_weight) * (1 - y_weight) +
                     pixel_01[1] * x_weight * (1 - y_weight) +
                     pixel_10[1] * (1 - x_weight) * y_weight +
                     pixel_11[1] * x_weight * y_weight)
            blue = (pixel_00[2] * (1 - x_weight) * (1 - y_weight) +
                    pixel_01[2] * x_weight * (1 - y_weight) +
                    pixel_10[2] * (1 - x_weight) * y_weight +
                    pixel_11[2] * x_weight * y_weight)

            resized_image.putpixel((new_x, new_y), (int(red), int(green), int(blue)))

    return resized_image

def zoom_out(image_path, original_dpi, target_dpi):
    original_image = Image.open(image_path)
    shrink_factor = original_dpi / target_dpi
    new_width = int(original_image.width / shrink_factor)
    new_height = int(original_image.height / shrink_factor)
    return resize_image_handler(original_image, new_width, new_height)

def zoom_in(shrunk_image, original_width, original_height):
    return resize_image_handler(shrunk_image, original_width, original_height)

# Display both zoomed-out and zoomed-in images using matplotlib
def display_images(original_image, zoomed_out_image, zoomed_in_image):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Display original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Display zoomed-out image
    axes[1].imshow(zoomed_out_image)
    axes[1].set_title("Zoomed-Out Image")
    axes[1].axis('off')

    # Display zoomed-in image
    axes[2].imshow(zoomed_in_image)
    axes[2].set_title("Zoomed-In Image")
    axes[2].axis('off')

    plt.show()

# Example usage
image_path_zoomout = '/content/drive/MyDrive/main_image.jpg'  # Replace with the path to your image

# Zoom out to 100 dpi
shrunk_image = zoom_out(image_path_zoomout, original_dpi=1250, target_dpi=100)

# Zoom back to original size (1250 dpi)
original_image = Image.open(image_path_zoomout)  # Reopen the original image
zoomed_image = zoom_in(shrunk_image, original_width=original_image.width, original_height=original_image.height)

# Display images using matplotlib
display_images(original_image, shrunk_image, zoomed_image)

"""**following code is used to apply The log transformation plus A power-law transformation**
##Task2
"""

# Provide the path to your grayscale image
input_image_path = "/content/drive/MyDrive/task2-grayscale.jpg"

# Read the image in grayscale mode
grayscale_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded correctly
if grayscale_image is None:
    raise Exception("Could not read the image. Make sure the image path is correct and the image is in grayscale format.")

# Normalize the image to [0, 1] range for transformation operations
normalized_image = grayscale_image / 255.0

# Perform Log transformation
log_scaling_constant = 1  # Scaling constant for log transformation
log_transformed_image = log_scaling_constant * np.log(1 + normalized_image)
log_transformed_image_rescaled = np.uint8(255 * log_transformed_image / np.max(log_transformed_image))

# Perform Power-law (Gamma) transformation
gamma_scaling_constant = 1  # Scaling constant for power-law transformation
gamma_value = 0.5  # Gamma value, can be adjusted for different effects
power_law_transformed_image = gamma_scaling_constant * np.power(normalized_image, gamma_value)
power_law_transformed_image_rescaled = np.uint8(255 * power_law_transformed_image / np.max(power_law_transformed_image))

# Display the original and transformed images using matplotlib
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# Display the original grayscale image
axes[0].imshow(grayscale_image, cmap='gray')
axes[0].set_title("Original Grayscale Image")
axes[0].axis('off')

# Display the image after log transformation
axes[1].imshow(log_transformed_image_rescaled, cmap='gray')
axes[1].set_title("Log Transformation")
axes[1].axis('off')

# Display the image after power-law transformation
axes[2].imshow(power_law_transformed_image_rescaled, cmap='gray')
axes[2].set_title(f"Power-law Transformation (Gamma={gamma_value})")
axes[2].axis('off')

# Show all images
plt.show()

"""**Reducing the number of intensity levels in an image**
##Task3
"""

# Function to quantize intensity levels of a grayscale image
def quantize_intensity_levels(grayscale_img, num_intensity_levels):
    """
    Quantize the intensity levels of a grayscale image to reduce the number of levels.

    Parameters:
    grayscale_img (ndarray): Grayscale image input with 256 intensity levels.
    num_intensity_levels (int): The desired number of intensity levels (must be a power of 2).

    Returns:
    quantized_image (ndarray): Image with reduced intensity levels.
    """
    # Calculate step size based on the desired number of intensity levels
    intensity_step_size = 256 // num_intensity_levels

    # Quantize the image by grouping pixel values into reduced intensity levels
    quantized_image = (grayscale_img // intensity_step_size) * intensity_step_size

    return quantized_image

# Provide the path to your grayscale image
image_file_path = "/content/drive/MyDrive/task3-img.jpg"

# Read the image in grayscale mode
original_grayscale_image = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded correctly
if original_grayscale_image is None:
    raise Exception("Could not read the image. Make sure the image path is correct and the image is in grayscale format.")

# Input: Desired number of intensity levels (must be a power of 2)
desired_intensity_levels = int(input("Enter the desired number of intensity levels (e.g., 2, 4, 8, 16): "))

# Check if the input is a power of 2 and valid
if desired_intensity_levels not in [2, 4, 8, 16, 32, 64, 128, 256]:
    raise ValueError("The desired number of intensity levels must be an integer power of 2 (e.g., 2, 4, 8, 16, 32, etc.)")

# Perform intensity level quantization
quantized_intensity_image = quantize_intensity_levels(original_grayscale_image, desired_intensity_levels)

# Display the original and quantized intensity images side by side using matplotlib
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Display the original grayscale image
axes[0].imshow(original_grayscale_image, cmap='gray')
axes[0].set_title("Original Image (256 levels)")
axes[0].axis('off')

# Display the image with reduced intensity levels
axes[1].imshow(quantized_intensity_image, cmap='gray')
axes[1].set_title(f"Quantized Image ({desired_intensity_levels} levels)")
axes[1].axis('off')

# Show the images
plt.show()

"""**Region Based Histogram Equalization**
##Task4

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# Function to apply histogram equalization to a selected region
def apply_histogram_equalization(image, roi):
    x1, y1, x2, y2 = roi
    # Ensure coordinates are integers
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Crop the ROI from the image
    roi_image = image[y1:y2, x1:x2]

    # Apply histogram equalization (ensure that the ROI is not empty)
    if roi_image.size > 0:
        equalized_roi = cv2.equalizeHist(roi_image)
        # Replace the ROI in the original image with the equalized ROI
        image[y1:y2, x1:x2] = equalized_roi

    return image

# Function to handle the selection of the ROI
def onselect(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    if None not in (x1, y1, x2, y2):  # Ensure valid ROI coordinates
        # Apply histogram equalization to the selected ROI
        result_image = apply_histogram_equalization(gray_image.copy(), (x1, y1, x2, y2))
        # Display the result
        plt.imshow(result_image, cmap='gray')
        plt.title("Histogram Equalized Image")
        plt.show()

# Load the grayscale image
image_path = '/content/drive/MyDrive/task2-grayscale.jpg'  # Replace with your image file
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Ensure the image is loaded properly
if gray_image is None:
    print("Error: Could not load the image.")
else:
    # Display the image and let the user select an ROI
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(gray_image, cmap='gray')
    ax.set_title("Select ROI using the mouse")

    # Create a rectangle selector for ROI selection
    rect_selector = RectangleSelector(ax, onselect, useblit=True,
                                      button=[1],  # Left mouse button
                                      minspanx=5, minspany=5,
                                      spancoords='pixels', interactive=True)

    plt.show()

"""###Note: Due to the limitations of Google Colab, which does not support GUI-based interaction for tasks such as mouse-driven ROI selection, we performed this task using VS Code on a local machine. Image 2 represents the region of interest (ROI) where histogram equalization was applied, showing the enhanced contrast in the selected area.

##Here is the code for Task:4
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# Function to apply histogram equalization to a selected region


def apply_histogram_equalization(image, roi):
    x1, y1, x2, y2 = roi
    # Ensure coordinates are integers
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Crop the ROI from the image
    roi_image = image[y1:y2, x1:x2]

    # Apply histogram equalization (ensure that the ROI is not empty)
    if roi_image.size > 0:
        equalized_roi = cv2.equalizeHist(roi_image)
        # Replace the ROI in the original image with the equalized ROI
        image[y1:y2, x1:x2] = equalized_roi

    return image

# Function to handle the selection of the ROI


def onselect(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    if None not in (x1, y1, x2, y2):  # Ensure valid ROI coordinates
        # Apply histogram equalization to the selected ROI
        result_image = apply_histogram_equalization(
            gray_image.copy(), (x1, y1, x2, y2))
        # Display the result
        plt.imshow(result_image, cmap='gray')
        plt.title("Histogram Equalized Image")
        plt.show()


# Load the grayscale image
# Replace with the actual path to your local image
image_path = 'task2-grayscale.jpg'
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Ensure the image is loaded properly
if gray_image is None:
    print(f"Error: Could not load the image at {image_path}.")
else:
    # Display the image and let the user select an ROI
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(gray_image, cmap='gray')
    ax.set_title("Select ROI using the mouse")

    # Create a rectangle selector for ROI selection
    rect_selector = RectangleSelector(ax, onselect, useblit=True,
                                      button=[1],  # Left mouse button
                                      minspanx=5, minspany=5,
                                      spancoords='pixels', interactive=True)

    plt.show()