import requests
import torch
import numpy as np
from PIL import Image
from transformers import PromptDepthAnythingForDepthEstimation, PromptDepthAnythingImageProcessor
from promptda.utils.io_wrapper import load_image, load_depth, save_depth

#url = "https://github.com/DepthAnything/PromptDA/blob/main/assets/example_images/image.jpg?raw=true"
#image = Image.open(requests.get(url, stream=True).raw)
img_path = "../Realtime-Depth-Estimation-Nconv/tmp/color_rgb.png"
image = Image.open(img_path)

image_processor = PromptDepthAnythingImageProcessor.from_pretrained("depth-anything/prompt-depth-anything-vitl-hf")
model = PromptDepthAnythingForDepthEstimation.from_pretrained("depth-anything/prompt-depth-anything-vitl-hf")

#prompt_depth_url = "https://github.com/DepthAnything/PromptDA/blob/main/assets/example_images/arkit_depth.png?raw=true"
#prompt_depth = Image.open(requests.get(prompt_depth_url, stream=True).raw)
depth_path = "../Realtime-Depth-Estimation-Nconv/tmp/depth_output.npy"
#prompt_depth = Image.open(depth_path)
prompt_depth = np.load(depth_path)
#prompt_depth = np.resize(prompt_depth, (image.width//2, image.height//2))

inputs = image_processor(images=image, return_tensors="pt", prompt_depth=prompt_depth)
with torch.no_grad():
    outputs = model(**inputs)
post_processed_output = image_processor.post_process_depth_estimation(
    outputs,
    target_sizes=[(image.height, image.width)],
)

predicted_depth = post_processed_output[0]["predicted_depth"]


breakpoint()
predicted_depth = predicted_depth.unsqueeze(0)
predicted_depth = predicted_depth.unsqueeze(0)
save_depth(predicted_depth)

