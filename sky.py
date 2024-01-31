import cv2
import numpy as np
import matplotlib.pyplot as plt

def combine_masks(color_mask, edge_mask):
    """
    Combine color thresholding mask and edge detection mask.
    Logic: Keep the regions identified as sky in the color_mask
           and refine edges using edge_mask.
    """
    # Invert edge mask to get regions without edges
    inverted_edge_mask = cv2.bitwise_not(edge_mask)

    # Combine masks: retain color_mask areas where there are no strong edges
    combined_mask = cv2.bitwise_and(color_mask, inverted_edge_mask)

    return combined_mask

# Load the image
image_path = '/Users/mac2022/Desktop/b.jpeg'
image = cv2.imread(image_path)

# Convert the image to the HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Apply Gaussian blur to the image before edge detection (adjust kernel size as needed)
blurred_image = cv2.GaussianBlur(hsv_image, (5, 5), 3)

# Perform Canny edge detection
edges = cv2.Canny(blurred_image, threshold1=30, threshold2=100)  # Adjust thresholds as needed

# Define the range of blue color in HSV
# These values can be adjusted based on the actual blue sky range in the image
lower_blue = np.array([95, 50, 50])
upper_blue = np.array([135, 255, 255])

# Threshold the HSV image to get only blue colors (sky)
mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
refined_mask = combine_masks(mask, edges)

# Bitwise-AND mask and original image to extract the sky
sky = cv2.bitwise_and(image, image, mask=refined_mask)

# Display the original image and the extracted sky
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(sky, cv2.COLOR_BGR2RGB))
plt.title('Sky Detected')
plt.axis('off')

plt.tight_layout()
plt.show()
