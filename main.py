import requests
import torch
import numpy as np
from PIL import Image
from transformers import PromptDepthAnythingForDepthEstimation, PromptDepthAnythingImageProcessor
from promptda.utils.io_wrapper import load_image, load_depth, save_depth

from scipy.ndimage import distance_transform_edt


def load_depth_from_binary(file_path, width, height):
    with open(file_path, 'rb') as f:
        # Read the first 4 bytes to get the data length (int32)
        length_bytes = f.read(4)
        if len(length_bytes) < 4:
            raise ValueError(f'File {file_path} is too short to contain a valid header.')

        data_length = np.frombuffer(length_bytes, dtype=np.int32)[0]

        # Expected data length
        expected_size = 640*480 #width * height

        if data_length != expected_size:
            raise ValueError(
                f'File {file_path} has data length {data_length}, expected {expected_size}.'
            )

        # Read the float32 depth data
        float_data = np.fromfile(f, dtype=np.float32, count=data_length)

        if len(float_data) != expected_size:
            raise ValueError(
                f'File {file_path} contains {len(float_data)} float values, expected {expected_size}.'
            )

        # Reshape to 2D depth map
        float_data = float_data.reshape((480, 640))
        #float_data = float_data[44:-44, 24:-24]
        #float_data = cv2.resize(float_data, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
        float_data = np.rot90(float_data, -1).copy() # copy to get rid of strides

        depth_map  = float_data.reshape((height, width))

        return depth_map



#url = "https://github.com/DepthAnything/PromptDA/blob/main/assets/example_images/image.jpg?raw=true"
#image = Image.open(requests.get(url, stream=True).raw)
img_path = "../datasets/new_spot_data/1/color/0.png"
image = Image.open(img_path)

image_processor = PromptDepthAnythingImageProcessor.from_pretrained("depth-anything/prompt-depth-anything-vitl-hf")
model = PromptDepthAnythingForDepthEstimation.from_pretrained("depth-anything/prompt-depth-anything-vitl-hf")

#prompt_depth_url = "https://github.com/DepthAnything/PromptDA/blob/main/assets/example_images/arkit_depth.png?raw=true"
#prompt_depth = Image.open(requests.get(prompt_depth_url, stream=True).raw)
depth_path = "../datasets/new_spot_data/1/depth/0"
prompt_depth = load_depth_from_binary(depth_path, image.width, image.height)

# depth_path = "results/arkit_depth.png"
# prompt_depth = Image.open(depth_path)
#prompt_depth = Image.open(depth_path, mode='r').convert('L')
#prompt_depth = np.resize(prompt_depth, (image.width//2, image.height//2))

# Load mask
#mask_path = "results/0.npy"
#mask = np.load(mask_path)
#mask = np.resize(mask, (prompt_depth.size[1], prompt_depth.size[0]))
#mask = mask[0:prompt_depth.height, 0:prompt_depth.width]  # Resize mask to match image dimensions

inputs = image_processor(images=image, return_tensors="pt", prompt_depth=prompt_depth)
# Hack: input depth is already good
inputs['prompt_depth'] = torch.from_numpy(np.array(prompt_depth)).unsqueeze(0).unsqueeze(0)
#mask = np.expand_dims(np.expand_dims(mask, axis=0), axis=0)  # Add channel and batch dimensions

processed_depth = inputs['prompt_depth']
minval = processed_depth.mean()
maxval = processed_depth.max()
meanval = processed_depth.mean()
# Strategy 1: Fill masked pixels with a random value bounded by the min and the max of the prompt depth
#processed_depth[mask==0] = 0 #torch.from_numpy(np.random.uniform(minval, maxval, size=processed_depth[mask==0].shape)).float()
# Interpolate the prompt depth where each masked pixel takes the value of the nearest non-masked pixel

# # Interpolate the prompt depth where each masked pixel takes the value of the nearest non-masked pixel
# from scipy.ndimage import distance_transform_edt

# downscale depth by 4x
# processed_depth = torch.nn.functional.interpolate(
#     processed_depth,
#     scale_factor=0.25,
#     mode='bilinear',
#     align_corners=False
# )

# Convert to numpy for processing
depth_np = processed_depth.squeeze().numpy()
#mask_np = mask.squeeze()

# Find valid (non-masked) pixels
valid_mask = depth_np > 0.01

if np.any(~valid_mask):  # If there are masked pixels to interpolate
    # Get indices of valid pixels
    indices = np.indices(depth_np.shape)
    valid_indices = np.column_stack([indices[0][valid_mask], indices[1][valid_mask]])

    # For each masked pixel, find nearest valid pixel
    masked_indices = np.column_stack([indices[0][~valid_mask], indices[1][~valid_mask]])

    # Use distance transform to find nearest valid pixels
    _, nearest_indices = distance_transform_edt(~valid_mask, return_indices=True)

    # Interpolate masked pixels
    depth_np[~valid_mask] = depth_np[nearest_indices[0][~valid_mask], nearest_indices[1][~valid_mask]]

    # Convert back to tensor
    processed_depth = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)


# from scipy.spatial.distance import cdist

# if np.any(~valid_mask):  # If there are masked pixels to interpolate
#     # Get coordinates of valid and masked pixels
#     valid_coords = np.column_stack(np.where(valid_mask))
#     masked_coords = np.column_stack(np.where(~valid_mask))
#     valid_values = depth_np[valid_mask]

#     # For each masked pixel, find 3 nearest valid neighbors and average them
#     for i, masked_coord in enumerate(masked_coords):
#         # Calculate distances to all valid pixels
#         distances = cdist([masked_coord], valid_coords, metric='euclidean')[0]

#         # Find indices of 3 nearest neighbors
#         k = min(3, len(distances))  # In case there are fewer than 3 valid pixels
#         nearest_indices = np.argpartition(distances, k-1)[:k]

#         # Average the depth values of the k nearest neighbors
#         nearest_values = valid_values[nearest_indices]
#         averaged_depth = np.mean(nearest_values)

#         # Assign the averaged value to the masked pixel
#         depth_np[masked_coord[0], masked_coord[1]] = averaged_depth

#     # Convert back to tensor
#     processed_depth = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)


# Update inputs with interpolated depth
inputs['prompt_depth'] = processed_depth



with torch.no_grad():
    outputs = model(**inputs)
post_processed_output = image_processor.post_process_depth_estimation(
    outputs,
    target_sizes=[(image.height, image.width)],
)

predicted_depth = post_processed_output[0]["predicted_depth"]

#breakpoint()
predicted_depth = predicted_depth.unsqueeze(0)
predicted_depth = predicted_depth.unsqueeze(0)
save_depth(predicted_depth, prompt_depth=inputs['prompt_depth'], image=image, output_path="results/example_depth.png")
