from promptda.promptda import PromptDA
from promptda.utils.io_wrapper import load_image, load_depth, save_depth
import torch

DEVICE = 'cpu'
image_path = "../../datasets/nyuv2/train/img/img_1.png"
prompt_depth_path = "../../datasets/nyuv2/train/gt/gt_1.npy"
mask_path = "../../datasets/nyuv2/mask/0.npy"

image = load_image(image_path)[:,:3,:,:].to(DEVICE)
prompt_depth = load_depth(prompt_depth_path).to(DEVICE) # 192x256, ARKit LiDAR depth in meters
mask = load_depth(mask_path).to(DEVICE)
# invert mask
mask = torch.logical_not(mask.bool())

image = image[:,:,2:,:]
image = image[:,:,:-2,:]
image = image[:,:,:,5:]
image = image[:,:,:,:-5]

prompt_depth = prompt_depth[:,:,2:,:]
prompt_depth = prompt_depth[:,:,:-2,:]
prompt_depth = prompt_depth[:,:,:,5:]
prompt_depth = prompt_depth[:,:,:,:-5]

padding = (22, 21, 52, 53)
mask = torch.nn.functional.pad(mask, padding, 'constant', 0)

#prompt_depth[mask] = 0

model = PromptDA.from_pretrained("depth-anything/prompt-depth-anything-vitl.ckpt").to(DEVICE).eval()
depth = model.predict(image, prompt_depth) # HxW, depth in meters

save_depth(depth, prompt_depth=prompt_depth, image=image)