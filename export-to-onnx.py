import argparse
import sys
import torch
import os
import onnx
from onnx import helper, numpy_helper

from promptda.promptda import PromptDA

def export_promptda(
    model,
    output_path,
    batch_size=1,
    height=640,
    width=480,
    depth_height=640,
    depth_width=480,
    dynamic=False,
    fp16=False,
    opset=21,
    device=torch.device("cpu")
):
    model.eval()

    # Determine dummy input shapes and dtype
    bs = batch_size if batch_size and batch_size > 0 else 1
    rgb_h = height if height and height > 0 else 518
    rgb_w = width if width and width > 0 else 518
    depth_h = depth_height if depth_height and depth_height > 0 else rgb_h
    depth_w = depth_width if depth_width and depth_width > 0 else rgb_w
    
    # Check aspect ratio consistency
    rgb_aspect = rgb_w / rgb_h
    depth_aspect = depth_w / depth_h
    if abs(rgb_aspect - depth_aspect) > 0.01:  # Allow small floating point differences
        print(f"Warning: RGB aspect ratio ({rgb_aspect:.3f}) differs from depth aspect ratio ({depth_aspect:.3f})")
    
    dtype = torch.float32
    dummy_rgb = torch.randn(bs, 3, rgb_h, rgb_w, dtype=dtype).to(device)
    dummy_depth = torch.randn(bs, 1, depth_h, depth_w, dtype=dtype).to(device)
    
    # Configure dynamic axes if requested
    if dynamic or batch_size == 0 or height == 0 or width == 0:
        dynamic_axes = {
            "rgb":   {0: "batch_size", 2: "rgb_height", 3: "rgb_width"},
            "depth": {0: "batch_size", 2: "depth_height", 3: "depth_width"},
            "output": {0: "batch_size", 2: "output_height", 3: "output_width"}
        }
    else:
        dynamic_axes = None

    print(f"Exporting with batch_size={bs}, RGB: {rgb_h}x{rgb_w}, depth: {depth_h}x{depth_w}, opset={opset}, dynamic={dynamic}")
    onnx_program = torch.onnx.export(
        model,
        (dummy_rgb, dummy_depth),
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["rgb", "depth"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        dynamo=True
    )
    print(f"Model exported to {output_path} as FP32")

    onnx_program.optimize()
    # Save the ONNX model
    optimized_output_path = output_path.replace(".onnx", "_optimized.onnx")
    onnx_program.save(optimized_output_path)
    print(f"Optimized ONNX model saved to {optimized_output_path}")

    # If fp16 option is requested, convert the ONNX model to mixed precision:
    if fp16:
        try:
            print("Converting model to mixed precision (FP16)...")
            # Load the exported FP32 model
            model_fp32 = onnx.load(output_path)
            
            # Convert model to FP16 while keeping inputs/outputs as FP32
            from onnx import version_converter, helper
            
            # Convert weights to FP16
            for initializer in model_fp32.graph.initializer:
                if initializer.data_type == onnx.TensorProto.FLOAT:
                    # Convert float32 weights to float16
                    float32_data = numpy_helper.to_array(initializer)
                    float16_data = float32_data.astype('float16')
                    new_initializer = numpy_helper.from_array(float16_data, initializer.name)
                    new_initializer.data_type = onnx.TensorProto.FLOAT16
                    initializer.CopyFrom(new_initializer)
            
            # Update intermediate value types to FP16 (keep inputs/outputs as FP32)
            input_names = {inp.name for inp in model_fp32.graph.input}
            output_names = {out.name for out in model_fp32.graph.output}
            
            for value_info in model_fp32.graph.value_info:
                if (value_info.name not in input_names and 
                    value_info.name not in output_names and
                    value_info.type.tensor_type.elem_type == onnx.TensorProto.FLOAT):
                    value_info.type.tensor_type.elem_type = onnx.TensorProto.FLOAT16
            
            # Save the FP16 model
            output_fp16_path = output_path.replace(".onnx", "_fp16.onnx")
            onnx.save(model_fp32, output_fp16_path)
            print(f"Model converted to mixed precision (FP16 weights, FP32 I/O) and saved to {output_fp16_path}.")
            
        except Exception as e:
            print(f"Failed to convert model to FP16: {e}")
            print("FP32 model is still available at:", output_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PromptDA to ONNX")
    parser.add_argument("--load-from", type=str, required=True,
                        help="Path to the .pth checkpoint for PromptDA.")
    parser.add_argument("--encoder", type=str, default='vitl',
                        choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help="Type of ViT encoder in the PromptDA model.")
    parser.add_argument("--max-depth", type=float, default=20.0,
                        help="Max depth value for PromptDA.")

    # ONNX arguments
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for dummy input (use 0 for dynamic batch)")
    parser.add_argument("--height", type=int, default=518, help="Input RGB image height (use 0 for dynamic height)")
    parser.add_argument("--width", type=int, default=518, help="Input RGB image width (use 0 for dynamic width)")
    parser.add_argument("--depth-width", type=int, default=518, help="Input depth width (use 0 to match RGB width)") 
    parser.add_argument("--depth-height", type=int, default=518, help="Input depth height (use 0 to match RGB height)")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic axes for batch, height, and width")
    parser.add_argument("--fp16", action="store_true", help="Export model in FP16 half-precision")
    parser.add_argument("--opset", type=int, default=21, help="ONNX opset version to use")
    parser.add_argument("--output", type=str, default="PromptDA.onnx", help="Output ONNX file path")
    args = parser.parse_args()
    # set device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # load checkpoint
    if os.path.isfile(args.load_from):
        model = PromptDA.from_pretrained(args.load_from)
    else:
        print(f"model_path={args.load_from} not found!")
        sys.exit(1)

    model.to(DEVICE)

    # Export
    export_promptda(
        model=model,
        output_path=args.output,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        depth_height=args.depth_height,
        depth_width=args.depth_width,
        dynamic=args.dynamic,
        fp16=args.fp16,
        opset=args.opset,
        device=DEVICE
    )