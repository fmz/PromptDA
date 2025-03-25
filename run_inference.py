import os
import sys
import time
import copy
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity

# Local project imports
from promptda.promptda import PromptDA
from promptda.nyuloader_v2 import NYUDepthDataset

from promptda.debug import Break
from promptda.my_utils import (
    save_depth,
    save_rgb,
    torch_pick_device
)


###############################################################################
#                        Logging & Misc Setup                                 #
###############################################################################

def setup_logger(log_level=logging.INFO):
    """
    Configure the root logger format and level.

    Args:
        log_level (int): logging.DEBUG, logging.INFO, etc.
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger()
    logger.handlers = []  # Clear existing handlers (especially in notebooks or repeated runs)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)

    logger.setLevel(log_level)
    logger.addHandler(handler)

    return logger


###############################################################################
#                    Run inference                                            #
###############################################################################

@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    debug: bool = False,
    profiler=None
):
    """
    Run inference the model for a single epoch.

    Args:
        model (nn.Module): The depth model.
        loader (DataLoader): Data loader.
        device (torch.device): GPU or CPU device.
        debug (bool): Whether to log debug info.
        profiler: PyTorch profiler (optional).

    Returns:
        Nothing (for now)
    """
    model.eval()
    t_start = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            rgb = batch['rgb'].to(device)
            depth = batch['gt'].to(device)
            # mask = batch.get('mask', None)
            # if mask is not None:
            #     mask = torch.logical_not(mask.bool().to(device))
            #     depth[mask] = 0

            depth = depth.unsqueeze(1)

            if profiler:
                # FIXME
                with record_function("train_batch"):
                    pred_depth = model(rgb, depth)
            else:
                pred_depth = model(rgb, depth)

            if profiler:
                profiler.step()

            # Quick debug saves every 10 steps
            if debug and (batch_idx % 1 == 0):
                file_id = batch['id'][0]

                detached_pred = pred_depth[0,0].detach().cpu().numpy()
                detached_prompt_depth = depth[0,0].detach().cpu().numpy()
                detached_rgb  = rgb[0].detach().cpu().numpy()

                save_depth(detached_pred, 'tmp/polished_depth_' + file_id, save_npy=True)
                save_depth(detached_prompt_depth,   'tmp/gt_depth_' + file_id)
                save_rgb(detached_rgb,    'tmp/input_rgb_' + file_id)

    t_end = time.time()

    logging.info(
        f"Inference finished for {len(loader)} images in {t_end - t_start:.2f}s "
    )
    return


def load_local_checkpoint(
        model: nn.Module,
        checkpoint_path: str,
        device: torch.device,
        strict=False) -> None:
    """
    Loads a local checkpoint into the model's state_dict.

    Args:
        model (nn.Module): The model to load weights into.
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): The device for loading.
        strict (bool): Whether to enforce that all keys match exactly.
    """
    logging.info(f"Attempting to load checkpoint from {checkpoint_path} with strict={strict}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # If the checkpoint was saved with a dictionary containing "model_state" or similar
    if "model_state" in ckpt:
        model_sd = ckpt["model_state"]
    else:
        model_sd = ckpt  # assume it's a direct state_dict

    missing, unexpected = model.load_state_dict(model_sd, strict=strict)
    if missing:
        logging.warning(f"Missing keys in state_dict: {missing}")
    if unexpected:
        logging.warning(f"Unexpected keys in state_dict: {unexpected}")
    logging.info("Checkpoint loaded.")

def main(args):
    """
    Main function for training/fine-tuning a DPT model with a YAML config.

    Args:
        config_path (str): Path to the YAML config file.
    """
    Break.start()

    # 1) Logging
    log_level = "INFO"
    logger = setup_logger(getattr(logging, log_level, logging.INFO))

    debug = args.debug

    # 2) Device
    device = torch_pick_device(args.device)
    logging.info(f"Using device: {device}")
    if device == torch.device("cuda"):
        torch.backends.cudnn.enabled   = True
        torch.backends.cudnn.benchmark = True

    # 3) Dataset & Dataloader
    dataset_path = args.data
    train_dataset = NYUDepthDataset(
        data_dir=dataset_path,
        mode="train",
        use_mask=True,
        add_noise=True,
        height=480,
        width=640,
        resize=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True
    )

    model = PromptDA.from_pretrained("depth-anything/prompt-depth-anything-vitl.ckpt")
    model.to(device)

    # 6) Profiling
    enable_profiling = False
    # profiling_cfg = cfg.get("profiling", {})
    # enable_profiling = profiling_cfg.get("enable", False)
    # steps_to_profile = profiling_cfg.get("steps_to_profile", 5)
    export_trace_path = "profile_trace.json"

    # -------------------------------------------------------------------------
    # Launch inference loop
    # -------------------------------------------------------------------------
    if enable_profiling:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=0, warmup=1, active=1, repeat=1
            ),
            on_trace_ready=lambda p: p.export_chrome_trace(export_trace_path),
            record_shapes=True,
            profile_memory=True
        ) as prof:
            run_inference(
                model=model,
                loader=train_loader,
                device=device,
                debug=debug,
                profiler=prof
            )
    else:
        run_inference(
            model=model,
            loader=train_loader,
            device=device,
            debug=debug
        )

###############################################################################
#                           Script Entry Point                                #
###############################################################################
def read_cmd_line_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="A sample command-line application demonstrating argparse usage."
    )

    parser.add_argument(
        '-m', '--model',
        type=str,
        help='The path to the model file'
    )

    parser.add_argument(
        '-d', '--data',
        type=str,
        help='The location of the nyuv2 data',
        required=True
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='The device on which to run (only a hint)'
    )
    # Flag to enable debug mode
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} /path/to/train_config.yaml")
        sys.exit(1)

    args = read_cmd_line_args()

    main(args)
