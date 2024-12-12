import argparse
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
from matplotlib import pyplot as plt

# Load pre-defined colors for segmentation masks
COLOURS = np.loadtxt("assets/colours.txt")

def detect_cup(image):
    """Detect the cup region from the image."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray_image)

    _, bright_regions = cv2.threshold(enhanced_image, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bright_regions_refined = cv2.morphologyEx(bright_regions, cv2.MORPH_CLOSE, kernel)

    masked_image = cv2.bitwise_and(enhanced_image, enhanced_image, mask=bright_regions_refined)
    _, cup_segment = cv2.threshold(masked_image, 220, 255, cv2.THRESH_BINARY)

    cup_refined = cv2.morphologyEx(cup_segment, cv2.MORPH_OPEN, kernel)
    return cup_refined

def fit_ellipse_to_cup(cup_image):
    """Fit an ellipse to the detected cup region."""
    contours, _ = cv2.findContours(cup_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            ellipse_mask = np.zeros_like(cup_image, dtype=np.uint8)
            cv2.ellipse(ellipse_mask, ellipse, 255, -1)
            return ellipse_mask
    return np.zeros_like(cup_image)

def process_frame(request, imx500, picam2):
    """Process each frame using the AI engine and overlay segmentation masks."""
    np_outputs = imx500.get_outputs(metadata=request.get_metadata())
    input_w, input_h = imx500.get_input_size()

    if np_outputs is None:
        print("No AI output found.")
        return

    mask = np_outputs[0]
    output_shape = [input_h, input_w, 4]
    overlay = np.zeros(output_shape, dtype=np.uint8)

    found_indices = np.unique(mask)
    for idx in found_indices:
        if idx == 0:  # Skip background
            continue
        color = COLOURS[int(idx)].tolist() + [150]  # Adding alpha channel
        overlay[mask == idx] = color

    # Convert AI mask to image for cup detection and ellipse fitting
    ai_image = (mask * 255).astype(np.uint8)
    cup_image = detect_cup(cv2.cvtColor(ai_image, cv2.COLOR_GRAY2BGR))
    ellipse_mask = fit_ellipse_to_cup(cup_image)

    # Add the ellipse mask to the overlay
    overlay[..., 0] = np.maximum(overlay[..., 0], ellipse_mask)

    picam2.set_overlay(overlay)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="/usr/share/imx500-models/imx500_network_deeplabv3plus.rpk",
        help="Path to the AI model",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for inference",
    )
    args = parser.parse_args()

    # Initialize the IMX500 AI engine
    imx500 = IMX500(args.model)

    # Configure network intrinsics
    intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
    intrinsics.task = "segmentation"
    intrinsics.inference_rate = args.fps
    intrinsics.update_with_defaults()

    # Initialize the Picamera2 instance
    picam2 = Picamera2(imx500.camera_num)

    # Configure the camera for preview
    config = picam2.create_preview_configuration(
        controls={"FrameRate": intrinsics.inference_rate}, buffer_count=6
    )

    # Set up the AI inference callback
    picam2.pre_callback = lambda request: process_frame(request, imx500, picam2)

    # Start the camera
    picam2.start(config, show_preview=True)

    # Run indefinitely
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping the camera...")
        picam2.stop()

if __name__ == "__main__":
    main()
