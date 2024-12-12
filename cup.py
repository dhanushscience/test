
import cv2
from matplotlib import pyplot as plt
import numpy as np

def detect_cup(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray_image)

    # Focus on the brightest regions (optic disc and cup)
    _, bright_regions = cv2.threshold(enhanced_image, 200, 255, cv2.THRESH_BINARY)

    # Morphological operations to isolate the optic disc and cup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bright_regions_refined = cv2.morphologyEx(bright_regions, cv2.MORPH_CLOSE, kernel)

    # Mask the original enhanced image to isolate the cup within the bright regions
    masked_image = cv2.bitwise_and(enhanced_image, enhanced_image, mask=bright_regions_refined)

    # Threshold within the optic disc to segment the cup (lighter center part)
    _, cup_segment = cv2.threshold(masked_image, 220, 255, cv2.THRESH_BINARY)

    # Further refine the cup region
    cup_refined = cv2.morphologyEx(cup_segment, cv2.MORPH_OPEN, kernel)

    return cup_refined

def fit_ellipse_to_cup(cup_image):
    # Find contours of the segmented cup region
    contours, _ = cv2.findContours(cup_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour, assuming it's the cup region
        largest_contour = max(contours, key=cv2.contourArea)
        
        if len(largest_contour) >= 5:  # At least 5 points are needed to fit an ellipse
            # Fit an ellipse to the largest contour
            ellipse = cv2.fitEllipse(largest_contour)
            
            # Create a blank mask
            ellipse_mask = np.zeros_like(cup_image, dtype=np.uint8)
            
            # Draw the fitted ellipse on the mask
            cv2.ellipse(ellipse_mask, ellipse, 255, -1)
            
            return ellipse_mask
    
    return np.zeros_like(cup_image)  # Return a blank mask if no valid contours are found

# Load the fundus image
image = cv2.imread("002.jpg")

# Detect the cup region
cup_image = detect_cup(image)

# Apply further morphological operations
opening = cv2.morphologyEx(cup_image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12)))
merged_image = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100)))

# Fit an ellipse to the detected cup region
ellipse_mask = fit_ellipse_to_cup(merged_image)

# Display the results
fig, axes = plt.subplots(1, 5, figsize=(15, 6))

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Fundus Image")
axes[0].axis('off')

axes[1].imshow(cup_image, cmap='gray')
axes[1].set_title("Detected Cup Region")
axes[1].axis('off')

axes[2].imshow(opening, cmap='gray')
axes[2].set_title("Open Cup Region")
axes[2].axis('off')

axes[3].imshow(merged_image, cmap='gray')
axes[3].set_title("Merged Cup Region")
axes[3].axis('off')

axes[4].imshow(ellipse_mask, cmap='gray')
axes[4].set_title("Cup Region")
axes[4].axis('off')

plt.tight_layout()
plt.show()
