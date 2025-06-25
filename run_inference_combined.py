#!/usr/bin/env python
import argparse
import sys
import cv2
from pathlib import Path
import numpy as np
import os
import time
import matplotlib
import torch
import torch.nn as nn

# Local project imports
from nconv.step2 import SETP2_BP_TRAIN
from promptda.promptda import PromptDA
from promptda.my_utils import save_rgb, save_depth
# If you have a specialized "save_depth" in your code, import that instead:
# from promptda.my_utils import save_depth

from transformers import PromptDepthAnythingImageProcessor

from scipy.ndimage import distance_transform_edt

###############################################################################
#                           Utility Functions                                 #
###############################################################################
# def save_depth(depth_map: np.ndarray, path: str, save_npy: bool = False):
#     """
#     Example utility to save depth as an 8-bit color-coded image.
#     If save_npy=True, also save the raw depth to .npy
#     """
#     # Optionally save raw depth
#     if save_npy:
#         np.save(path + ".npy", depth_map)

#     # Normalize & convert for visualization
#     d_min, d_max = depth_map.min(), depth_map.max()
#     if d_max - d_min < 1e-6:
#         # Avoid division by zero if the image is nearly constant
#         vis_depth = np.zeros_like(depth_map, dtype=np.uint8)
#     else:
#         vis_depth = (depth_map - d_min) / (d_max - d_min) * 255.0
#         vis_depth = vis_depth.astype(np.uint8)

#     # Use a colormap
#     cmap = matplotlib.colormaps.get_cmap('Spectral')
#     colored_depth = cmap(vis_depth)[:, :, :3]  # RGBA -> RGB
#     colored_depth = (colored_depth * 255).astype(np.uint8)
#     # Convert RGB -> BGR for OpenCV
#     colored_depth = colored_depth[..., ::-1]

#     # Save final .png
#     cv2.imwrite(path + ".png", colored_depth)

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

        depth_map  = float_data.reshape((1, 1, height, width))
        
        return depth_map

def load_local_checkpoint(
        model: nn.Module,
        checkpoint_path: str,
        device: torch.device,
        strict=False) -> None:
    """
    Loads a local checkpoint into the model's state_dict.
    (Merges the logic from both scripts.)
    """
    print(f"Attempting to load checkpoint from {checkpoint_path} with strict={strict}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "state_dict" in ckpt:
        model_sd = ckpt["state_dict"]
    elif "model_state" in ckpt:
        model_sd = ckpt["model_state"]
    else:
        # assume it's a raw state_dict
        model_sd = ckpt

    # Remove the "module." prefix if it's present (module is for DataParallel)
    new_state_dict = {}
    for k, v in model_sd.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    missing, unexpected = model.load_state_dict(new_state_dict, strict=strict)
    if missing:
        print(f"Missing keys in state_dict: {missing}")
    if unexpected:
        print(f"Unexpected keys in state_dict: {unexpected}")
    print("Checkpoint loaded.")


###############################################################################
#                           Main Inference Function                           #
###############################################################################
@torch.no_grad()
def run_inference_on_images(
    files: list,    # 0: rgb, 1: depth
    step2_model: nn.Module,
    promptda_model: nn.Module,
    device: torch.device,
    outdir: str,
    input_width: int,
    input_height: int,
    save_numpy: bool,
    pred_only: bool,
    grayscale: bool
):
    """
    For each image in `file_list`:
      1) Read with cv2
      2) Convert to tensor
      3) Run step2_model → refined depth
      4) Pass refined depth into promptda_model
      5) Save result (color-coded or grayscale) in `outdir`.
    """
    os.makedirs(outdir, exist_ok=True)

    # Optional colormap for grayscale usage
    cmap = matplotlib.colormaps.get_cmap('Spectral')


    image_processor = PromptDepthAnythingImageProcessor.from_pretrained("depth-anything/prompt-depth-anything-vitl-hf")


    for idx, (rgb_file, depth_file) in enumerate(files):
        print(f"[{idx+1}/{len(files)}] Processing: {rgb_file} // {depth_file}")

        # 1) Read the image
        raw_image = cv2.imread(rgb_file)
        if raw_image is None:
            print(f"Could not read {rgb_file}. Skipping...")
            continue

        # BGR -> RGB if your model expects RGB
        rgb_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

        # 2) Preprocess for Step2 (resize or other transform as needed):
        #    Example: simple cv2.resize
        # if (input_width > 0) and (input_height > 0):
        #     rgb_resized = cv2.resize(rgb_image, (input_width, input_height), interpolation=cv2.INTER_AREA)
        # else:
        rgb_resized = rgb_image[4:-4, ...]
        rgb_resized = np.pad(rgb_resized, ((0,0),(1,1),(0,0)), mode='constant', constant_values=0)

        # Convert to tensor
        rgb_tensor   = torch.from_numpy(rgb_resized).float().permute(2, 0, 1).unsqueeze(0).to(device)
        depth_sparse = torch.from_numpy(load_depth_from_binary(depth_file, input_width, input_height)).to(device)

        inputs = image_processor(images=rgb_image, return_tensors="pt", prompt_depth=depth_sparse.squeeze())
        rgb_tensor = inputs['pixel_values'].to(device)
        depth_np = depth_sparse.detach().cpu().squeeze().numpy()
        valid_mask = (depth_np > 0) & (depth_np < 1000)  # Assuming valid depth is in range [0, 1000]

        # if np.any(~valid_mask):
        #     _, nearest_indices = distance_transform_edt(
        #         ~valid_mask, return_indices=True
        #     )

        #     depth_np[~valid_mask] = depth_np[nearest_indices[0][~valid_mask], nearest_indices[1][~valid_mask]]

        #     depth_sparse = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).to(device)

        # 3) Step2 inference
        t_start = time.time()
        #step2_outputs = step2_model(rgb_tensor, depth_sparse)
        step2_outputs = step2_model.step1(depth_sparse)
        refined_depth = step2_outputs #[-1]  # Typically the final scale

        #refined_depth = depth_sparse #HACK
        # shape: [1, 1, H, W]

        # 4) PromptDA inference
        promptda_pred = promptda_model(rgb_tensor, refined_depth)
        #promptda_pred = refined_depth
        # shape: [1, 1, H, W], presumably

        print(f"Inference time: {(time.time() - t_start)*1000:4.4f}ms")

        # 5) Save the results:
        # Convert promptda_pred to numpy
        pred_depth_np = promptda_pred[0, 0].detach().cpu().numpy()

        # Save the raw depth or color-coded
        base_name = os.path.splitext(os.path.basename(rgb_file))[0]
        save_path = os.path.join(outdir, base_name)

        if save_numpy:
            np.save(save_path + "_raw_depth_meter.npy", pred_depth_np)

        save_rgb(rgb_tensor[0].detach().cpu().numpy(), save_path + "_rgb.png")
        save_depth(refined_depth[0,0].detach().cpu().numpy(), save_path , save_npy=False)
        save_depth(pred_depth_np, save_path + "_final", save_npy=save_numpy)

        # # Create a color or grayscale output
        # depth_vis = (pred_depth_np - pred_depth_np.min()) / (pred_depth_np.max() - pred_depth_np.min() + 1e-8)
        # depth_vis = (depth_vis * 255).astype(np.uint8)

        # if grayscale:
        #     depth_vis_3c = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
        # else:
        #     # Apply colormap
        #     colored = cmap(depth_vis)[:, :, :3]  # RGBA -> RGB
        #     colored = (colored * 255).astype(np.uint8)
        #     depth_vis_3c = colored[:, :, ::-1]  # RGB -> BGR for cv2

        # if pred_only:
        #     # Save only the depth
        #     out_pred_name = save_path + "_pred.png"
        #     cv2.imwrite(out_pred_name, depth_vis_3c)
        # else:
        #     # Concat side-by-side
        #     # resize depth_vis_3c to match original raw_image height
        #     depth_vis_3c_resized = cv2.resize(depth_vis_3c, (raw_image.shape[1], raw_image.shape[0]))
        #     # Create a vertical or horizontal split
        #     split_region = np.ones((raw_image.shape[0], 10, 3), dtype=np.uint8) * 255
        #     combined = cv2.hconcat([raw_image, split_region, depth_vis_3c_resized])
        #     out_pred_name = save_path + "_combined.png"
        #     cv2.imwrite(out_pred_name, combined)


###############################################################################
#                            Main Entry Point                                 #
###############################################################################
def gather_files(
    rgb_path: str,
    depth_path: str
):
    """
    Gather files from an rgb and a depth path:
      - If it's a file, return [file].
      - If it's a .txt, read lines as file paths.
      - Otherwise treat it as a directory.
    """
    if os.path.isfile(rgb_path):
        assert os.path.isfile(depth_path), "Depth path must be a file if RGB is a file."  
        return [(rgb_path, depth_path)]
    else:
        rgb_path   = Path(rgb_path)
        depth_path = Path(depth_path)
        rgb_list = list(rgb_path.glob('*'))
        depth_list = list(depth_path.glob('*'))
        
        try:
            # Try sorting based on the numeric value of the filename (excluding extension)
            sorted_rgb   = sorted(rgb_list, key=lambda p: int(p.stem))
            sorted_depth = sorted(depth_list, key=lambda p: int(p.stem))
        except ValueError:
            # If conversion fails, fall back to lexicographical sort
            sorted_rgb   = sorted(rgb_list)
            sorted_depth = sorted(depth_list)
        
        return list(zip(sorted_rgb, sorted_depth))


def pick_device(device_str: str = 'cpu'):
    """
    Decide whether to use CPU, CUDA, or MPS (Apple Silicon).
    """
    if device_str == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    elif device_str == 'mps' and getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser(description='Combined Step2 -> PromptDA Inference Example')

    # Inputs
    parser.add_argument('--img-path', type=str, required=True,
                        help='Path to an image, text file listing images, or a directory.')
    parser.add_argument('--depth-path', type=str, required=True,
                        help='Path to the sparse depth map.')
    parser.add_argument('--outdir', type=str, default='./vis_depth',
                        help='Output directory for predictions.')

    # Step2
    parser.add_argument('--step2-ckpt', type=str, default='',
                        help='Path to a Step2 checkpoint file.')
    # parser.add_argument('--step2-base', type=str, default='baseline2',
    #                     help='Name for the step1 checkpoint inside Step2 if needed (e.g. "baseline2")')

    # PromptDA (DepthAnything) – example argument if you want to load from path:
    parser.add_argument('--promptda-path', type=str, default='depth-anything/prompt-depth-anything-vitl',
                        help='Path or name to load PromptDA weights from')

    # Resizing
    parser.add_argument('--input-width', type=int, default=640)
    parser.add_argument('--input-height', type=int, default=480)

    # Visualization
    parser.add_argument('--save-numpy', action='store_true',
                        help='Also save raw depth predictions as .npy')
    parser.add_argument('--pred-only', action='store_true',
                        help='Save only the predicted depth (no side-by-side with original).')
    parser.add_argument('--grayscale', action='store_true',
                        help='Render the output in grayscale instead of color.')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cpu, cuda, mps).')

    args = parser.parse_args()

    # 1) Device
    device = pick_device(args.device)
    print(f"Using device: {device}")

    # 2) Prepare Step2 model
    step2_model = SETP2_BP_TRAIN(step1_checkpoint_name=args.step2_ckpt)
    step2_model.to(device)

    if args.step2_ckpt and os.path.isfile(args.step2_ckpt):
        load_local_checkpoint(step2_model, args.step2_ckpt, device, strict=False)
    else:
        if args.step2_ckpt:
            print(f"Warning: Step2 checkpoint not found at {args.step2_ckpt}. Proceeding with random weights...")

    step2_model.eval()

    # 3) Prepare PromptDA model
    promptda_model = PromptDA.from_pretrained(args.promptda_path)
    promptda_model.to(device)
    promptda_model.eval()

    # 4) Gather file list
    file_list = gather_files(args.img_path, args.depth_path)
    print(f"Found {len(file_list)} files in {args.img_path}")

    # 5) Inference
    run_inference_on_images(
        files=file_list,
        step2_model=step2_model,
        promptda_model=promptda_model,
        device=device,
        outdir=args.outdir,
        input_width=args.input_width,
        input_height=args.input_height,
        save_numpy=args.save_numpy,
        pred_only=args.pred_only,
        grayscale=args.grayscale
    )

    print("All done!")


if __name__ == '__main__':
    main()
