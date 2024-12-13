import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

# Load image
image = cv2.imread('fundus_dhanush.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load processor and model
processor = AutoImageProcessor.from_pretrained("pamixsun/segformer_for_optic_disc_cup_segmentation")
model = SegformerForSemanticSegmentation.from_pretrained("pamixsun/segformer_for_optic_disc_cup_segmentation")

# Preprocess the image
inputs = processor(image, return_tensors="pt")

# Get model predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits.cpu()

# Upsample logits to original image size
upsampled_logits = nn.functional.interpolate(
    logits,
    size=image.shape[:2],
    mode="bilinear",
    align_corners=False,
)

# Get predicted segmentation mask (optic disc and cup)
pred_disc_cup = upsampled_logits.argmax(dim=1)[0].numpy().astype(np.uint8)

# Display original image and prediction side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Display the original image
axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[0].axis('off')

# Display the segmentation result (optic disc and cup mask)
axes[1].imshow(pred_disc_cup, cmap='jet')
axes[1].set_title('Predicted Segmentation (Disc/Cup)')
axes[1].axis('off')

plt.show()

#python3 -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
