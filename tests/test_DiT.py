import numpy as np
from PIL import Image
from transformers import AutoFeatureExtractor

i = Image.new("RGB", (2, 1))

def empty_image(height=2, width=2):
    i = np.ones((height, width, 3), np.uint8) * 255  # whitepage
    return i
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
res = feature_extractor(images=[i], return_tensors="pt")
