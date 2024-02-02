import cv2
import numpy as np
import gradio as gr
from PIL import Image

def detect_sky(input_image):
    # Convert PIL Image to OpenCV format
    input_image = np.array(input_image)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    # Convert the image to the HSV color space for color thresholding
    hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([95, 50, 50])
    upper_blue = np.array([135, 255, 255])

    # Threshold the image to get a mask of blue colors
    color_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Apply Canny edge detection
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    edge_mask = cv2.Canny(gray_image, threshold1=30, threshold2=150)

    # Combine color mask and edge mask
    combined_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(edge_mask))

    # Extract sky using the combined mask
    sky = cv2.bitwise_and(input_image, input_image, mask=combined_mask)

    # Convert back to RGB (PIL format) for display
    sky_rgb = cv2.cvtColor(sky, cv2.COLOR_BGR2RGB)
    sky_pil = Image.fromarray(sky_rgb)

    return sky_pil

def gradio_interface(image):
    # Process image and detect sky
    sky_image = detect_sky(image)
    return sky_image

# Set up Gradio interface
# Set up Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Image(image_mode='RGB'), # Corrected input
    outputs=gr.Image(type="pil"), # Corrected output
    title="Sky Detection Demo",
    description="Upload an image to detect the sky."
)

# Run the Gradio app (launch in the current IPython notebook if possible, else open in a new tab)
iface.launch()
