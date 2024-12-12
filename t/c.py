import cv2
import numpy as np
from matplotlib import pyplot as plt
from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

# Define colors for segmentation
COLOURS = np.array([[0, 0, 0], [255, 0, 0]], dtype=np.uint8)  # Background and Cup color

def process_segmentation_output(segmentation_mask, original_image):
    """Process the segmentation output from the IMX500 AI camera."""
    # Convert segmentation mask to binary for the optic cup
    cup_mask = (segmentation_mask == 1).astype(np.uint8) * 255

    # Morphological operations to refine the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    refined_mask = cv2.morphologyEx(cup_mask, cv2.MORPH_OPEN, kernel)

    # Fit ellipse to the detected cup
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            ellipse_mask = np.zeros_like(refined_mask, dtype=np.uint8)
            cv2.ellipse(ellipse_mask, ellipse, 255, -1)
            return ellipse_mask, refined_mask

    return np.zeros_like(refined_mask), refined_mask

def display_results(original_image, segmentation_mask, refined_mask, ellipse_mask):
    """Display the segmentation and processed results."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 6))

    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(segmentation_mask, cmap="gray")
    axes[1].set_title("Segmentation Output")
    axes[1].axis("off")

    axes[2].imshow(refined_mask, cmap="gray")
    axes[2].set_title("Refined Mask")
    axes[2].axis("off")

    axes[3].imshow(ellipse_mask, cmap="gray")
    axes[3].set_title("Fitted Ellipse")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Initialize the IMX500 camera
    model_path = "/usr/share/imx500-models/imx500_network_deeplabv3plus.rpk"
    imx500 = IMX500(model_path)
    picam2 = Picamera2(imx500.camera_num)

    # Load network intrinsics for segmentation
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "segmentation"

    # Start the camera with segmentation configuration
    config = picam2.create_preview_configuration(buffer_count=12)
    picam2.start(config, show_preview=False)

    # Capture a frame
    frame = picam2.capture_array()

    # Process AI segmentation
    outputs = imx500.get_outputs(metadata=picam2.capture_metadata())
    segmentation_mask = outputs[0] if outputs is not None else np.zeros_like(frame[:, :, 0])

    # Process segmentation results
    ellipse_mask, refined_mask = process_segmentation_output(segmentation_mask, frame)

    # Display results
    display_results(frame, segmentation_mask, refined_mask, ellipse_mask)

    picam2.stop()
