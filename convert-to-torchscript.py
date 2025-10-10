#!/usr/bin/env python3
"""
PyTorch .ckpt to LibTorch converter for PromptDA.
Uses PyTorch 2.7 optimizations.
"""

import argparse
import torch
import torch.nn as nn
import os
import sys
import time
import logging
from typing import Dict, List
import numpy as np
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from promptda.promptda import PromptDA
    PROMPTDA_AVAILABLE = True
except ImportError:
    PROMPTDA_AVAILABLE = False
    logger.error("PromptDA not available - please install promptda")
    sys.exit(1)

# PyTorch 2.7 optimizations
try:
    torch._dynamo.reset()
    torch._dynamo.config.automatic_dynamic_shapes = True
    torch._dynamo.config.cache_size_limit = 256
    logger.info("PyTorch 2.7 Dynamo optimizations enabled")
except AttributeError:
    logger.warning("Some PyTorch 2.7 features not available in this version")


@contextmanager
def inference_mode_context():
    """Context manager for optimal inference settings."""
    old_grad_enabled = torch.is_grad_enabled()
    old_deterministic = torch.backends.cudnn.deterministic
    old_benchmark = torch.backends.cudnn.benchmark
    old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else None

    try:
        # Prefer inference_mode over manual grad toggling
        with torch.inference_mode():
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                # Guard optional SDP toggles
                try:
                    torch.backends.cuda.enable_flash_sdp(True)
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                    torch.backends.cuda.enable_math_sdp(True)
                except Exception:
                    pass
            yield
    finally:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = old_deterministic
            torch.backends.cudnn.benchmark = old_benchmark
            if old_allow_tf32 is not None:
                torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32


class OptimizedTracingWrapper(nn.Module):
    """Wrapper that makes PromptDA models tracing-friendly with memory optimizations."""

    def __init__(self, model: nn.Module, fixed_height: int = 480, fixed_width: int = 640):
        super().__init__()
        self.model = model
        self.fixed_height = fixed_height
        self.fixed_width = fixed_width
        self._optimize_memory_format()

    def _optimize_memory_format(self):
        """Optimize memory layout for better performance."""
        try:
            for module in self.model.modules():
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    # if hasattr(module, 'weight') and module.weight.dim() == 4:
                    #     module.weight.data = module.weight.data.to(memory_format=torch.channels_last)
                    if hasattr(module, 'bias') and module.bias is not None:
                        module.bias.data = module.bias.data.contiguous()
        except Exception as e:
            logger.warning(f"Memory format optimization failed: {e}")

    def forward(self, rgb: torch.Tensor, depth_prompt: torch.Tensor, rotate : bool) -> torch.Tensor:
        """Forward pass with optimized memory format for PromptDA."""
        # # Convert to channels_last for better performance
        # if rgb.dim() == 4 and rgb.device.type == 'cuda':
        #     rgb = rgb.to(memory_format=torch.channels_last)

        # Ensure RGB input is the expected size
        if rgb.shape[-2:] != (self.fixed_height, self.fixed_width):
            rgb = torch.nn.functional.interpolate(
                rgb, size=(self.fixed_height, self.fixed_width),
                mode='bilinear', align_corners=False, antialias=True
            )

        # Ensure depth input is the expected size
        if depth_prompt.shape[-2:] != (self.fixed_height, self.fixed_width):
            depth_prompt = torch.nn.functional.interpolate(
                depth_prompt, size=(self.fixed_height, self.fixed_width),
                mode='bilinear', align_corners=False, antialias=True
            )

        return self.model(rgb, depth_prompt, rotate)


class MixedPrecisionWrapper(nn.Module):
    """Wrapper to handle mixed precision inference using PyTorch AMP."""

    def __init__(self, model: nn.Module, device_type: str = 'cuda'):
        super().__init__()
        self.model = model
        self.device_type = device_type
        self.use_amp = device_type == 'cuda' and torch.cuda.is_available()

    def forward(self, rgb: torch.Tensor, depth_prompt: torch.Tensor) -> torch.Tensor:
        if self.use_amp:
            with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                return self.model(rgb, depth_prompt)
        else:
            return self.model(rgb, depth_prompt)


class BenchmarkWrapper:
    """Unified wrapper for benchmarking both compiled and TorchScript models."""

    def __init__(self, model):
        self.model = model

    def eval(self):
        return self

    def __call__(self, inputs):
        if isinstance(inputs, tuple) and len(inputs) == 2:
            return self.model(inputs[0], inputs[1])
        else:
            raise ValueError("PromptDA requires both RGB and depth inputs as tuple")


class AdvancedModelConverter:
    """Enhanced model converter with PyTorch 2.7 optimizations for PromptDA."""

    def __init__(self):
        self._setup_compilation_cache()

    def _setup_compilation_cache(self):
        """Setup PyTorch 2.7 compilation cache for better performance."""
        try:
            os.environ.setdefault('TORCH_COMPILE_DEBUG', '0')
            os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', '/tmp/torch_cache')
            cache_dir = os.environ.get('TORCHINDUCTOR_CACHE_DIR')
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Compilation cache setup at: {cache_dir}")
        except Exception as e:
            logger.warning(f"Cache setup failed: {e}")

    def load_checkpoint(self, checkpoint_path: str, device: torch.device) -> nn.Module:
        """Load PromptDA checkpoint with optimizations."""
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading PromptDA checkpoint from {checkpoint_path}")

        try:
            model = PromptDA.from_pretrained(checkpoint_path).to(device)
            logger.info("PromptDA model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load PromptDA checkpoint: {e}")
            raise

        model = model.to(device).eval()

        # Disable gradients and apply memory optimizations
        for param in model.parameters():
            param.requires_grad_(False)

        logger.info(f"PromptDA model loaded on {device}")
        return model

    def prepare_inputs(self, batch_size: int, height: int, width: int,
                      device: torch.device, use_fp16: bool = False) -> tuple:
        """Prepare example inputs for PromptDA."""
        # Always use FP32 inputs for mixed precision - let AMP handle the conversion
        dtype = torch.float32

        rgb_input = torch.randn(batch_size, 3, height, width, device=device, dtype=dtype)
        depth_input = torch.randn(batch_size, 1, height, width, device=device, dtype=dtype)

        # Optimize memory format for CUDA
        # if device.type == 'cuda':
        #     rgb_input = rgb_input.to(memory_format=torch.channels_last)

        logger.info(f"RGB input: {rgb_input.shape}, {rgb_input.dtype}, {rgb_input.device}")
        logger.info(f"Depth input: {depth_input.shape}, {depth_input.dtype}, {depth_input.device}")

        return rgb_input, depth_input

    def apply_optimizations(self, model: nn.Module, use_wrapper: bool, use_fp16: bool,
                          use_int8: bool, height: int, width: int,
                          device: torch.device, for_torchscript: bool = True) -> nn.Module:
        """Apply all optimizations in sequence."""
        logger.info("Applying optimizations...")

        # Apply tracing wrapper if requested
        if use_wrapper:
            model = OptimizedTracingWrapper(model, height, width)
            logger.info(f"Applied optimized tracing wrapper for {height}x{width} inputs")

        # Apply quantization
        if use_int8 and (device.type == 'cpu'):
            try:
                model = torch.ao.quantization.quantize_dynamic(
                    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                )
                logger.info("✓ Applied INT8 dynamic quantization")
            except Exception as e:
                logger.warning(f"INT8 quantization failed: {e}")

        # Apply mixed precision optimization
        if use_fp16 and device.type == 'cuda':
            try:
                model = MixedPrecisionWrapper(model, device.type)
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
                logger.info("✓ Applied mixed precision optimization with AMP")
            except Exception as e:
                logger.warning(f"Mixed precision optimization failed: {e}")

        # Apply torch.compile if not converting to TorchScript
        if not for_torchscript:
            model = self._apply_torch_compile(model)

        return model

    def _apply_torch_compile(self, model: nn.Module) -> nn.Module:
        """Apply torch.compile optimizations."""
        try:
            backends = ['inductor', 'aot_eager'] if torch.cuda.is_available() else ['aot_eager']

            for backend in backends:
                try:
                    compiled_model = torch.compile(
                        model, backend=backend, mode='max-autotune', fullgraph=False, dynamic=False
                    )
                    logger.info(f"✓ Model compiled with {backend} backend")
                    return compiled_model
                except Exception as e:
                    logger.warning(f"Compilation with {backend} failed: {e}")
                    continue

            logger.warning("All compilation attempts failed, using uncompiled model")
        except Exception as e:
            logger.warning(f"torch.compile not available or failed: {e}")

        return model

    def convert_to_torchscript(self, model: nn.Module, rgb_input: torch.Tensor,
                             depth_input: torch.Tensor, method: str = "trace") -> torch.jit.ScriptModule:
        """Convert PromptDA model to TorchScript."""
        logger.info(f"Converting PromptDA to TorchScript using {method} method...")

        try:
            with inference_mode_context():
                if method == "script":
                    try:
                        scripted = torch.jit.script(model)
                        logger.info("✓ Scripting completed successfully")
                    except Exception as script_error:
                        logger.warning(f"Scripting failed: {script_error}, falling back to tracing")
                        method = "trace"

                if method == "trace":
                    try:
                        torch._dynamo.reset()
                    except:
                        pass

                    logger.info("Starting TorchScript tracing for PromptDA...")
                    scripted = torch.jit.trace(
                        model, (rgb_input, depth_input), strict=False, check_trace=False
                    )
                    logger.info("✓ Advanced tracing completed successfully")

                # Optimize the scripted model
                scripted = self._optimize_scripted_model(scripted)
                self._validate_trace_consistency(model, scripted, rgb_input, depth_input)

                return scripted

        except Exception as e:
            logger.error(f"TorchScript conversion failed: {e}")
            raise

    def _optimize_scripted_model(self, scripted_model: torch.jit.ScriptModule) -> torch.jit.ScriptModule:
        """Apply optimizations to scripted model."""
        logger.info("Applying TorchScript optimizations...")

        try:
            scripted_model = torch.jit.freeze(scripted_model)
            scripted_model = torch.jit.optimize_for_inference(scripted_model)

            # Apply graph optimizations
            try:
                torch.jit.run_unused_elimination(scripted_model.graph)
                torch.jit.run_dead_code_elimination(scripted_model.graph)
                torch.jit.run_constant_folding(scripted_model.graph)
                torch.jit.run_constant_propagation(scripted_model.graph)
                torch.jit.run_algebraic_simplification(scripted_model.graph)
                torch.jit.run_peephole(scripted_model.graph, addmm_fusion_enabled=True)
                logger.info("✓ Advanced graph optimizations applied")
            except Exception as e:
                logger.warning(f"Some graph optimizations failed: {e}")

        except Exception as e:
            logger.warning(f"Model optimization failed: {e}")

        return scripted_model

    def _validate_trace_consistency(self, original_model: nn.Module, scripted_model: torch.jit.ScriptModule,
                                  rgb_input: torch.Tensor, depth_input: torch.Tensor):
        """Validate trace consistency."""
        logger.info("Validating trace consistency...")

        def _normalize_output(out):
            if isinstance(out, dict):
                # Compare first value deterministically
                try:
                    return next(iter(out.values()))
                except StopIteration:
                    return None
            if isinstance(out, (list, tuple)):
                return out[0] if len(out) == 1 else out
            return out

        try:
            with inference_mode_context():
                original_out = _normalize_output(original_model(rgb_input, depth_input))
                traced_out = _normalize_output(scripted_model(rgb_input, depth_input))
                if original_out is None or traced_out is None:
                    logger.info("Skipping diff: empty outputs")
                    return
                if isinstance(original_out, (list, tuple)) or isinstance(traced_out, (list, tuple)):
                    logger.info("Skipping diff: non-tensor outputs")
                    return
                diff = torch.max(torch.abs(original_out.float() - traced_out.float())).item()
                logger.info(f"  Max difference: {diff:.6f}")
        except Exception as e:
            logger.warning(f"Validation failed: {e}")

    def benchmark_model(self, model, inputs: tuple, num_warmup: int = 20,
                       num_runs: int = 100) -> Dict[str, float]:
        """Comprehensive benchmarking with mixed precision support."""
        logger.info(f"Running benchmark ({num_warmup} warmup + {num_runs} runs)...")

        wrapper = BenchmarkWrapper(model)

        # Memory profiling
        if inputs[0].is_cuda:
            torch.cuda.empty_cache()
            start_memory = torch.cuda.memory_allocated()

        # Warmup
        with inference_mode_context():
            for _ in range(num_warmup):
                _ = wrapper(inputs)

        if inputs[0].is_cuda:
            torch.cuda.synchronize()

        # Benchmark
        times = []
        with inference_mode_context():
            for _ in range(num_runs):
                if inputs[0].is_cuda:
                    torch.cuda.synchronize()

                start_time = time.perf_counter()
                _ = wrapper(inputs)

                if inputs[0].is_cuda:
                    torch.cuda.synchronize()

                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)

        results = {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'fps': 1000.0 / np.mean(times),
        }

        if inputs[0].is_cuda:
            current_memory = torch.cuda.memory_allocated()
            results['memory_mb'] = (current_memory - start_memory) / 1024 / 1024

        logger.info(f"Performance: {results['mean_ms']:.2f} ± {results['std_ms']:.2f} ms ({results['fps']:.1f} FPS)")
        if 'memory_mb' in results:
            logger.info(f"Memory usage: {results['memory_mb']:.1f} MB")

        return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert PromptDA .ckpt to optimized LibTorch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("checkpoint", help="Path to input PromptDA .ckpt checkpoint")
    parser.add_argument("output", help="Path to output TorchScript .pt file")

    parser.add_argument("--method", choices=["script", "trace"], default="trace",
                       help="TorchScript conversion method")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--height", type=int, default=480, help="Input height")
    parser.add_argument("--width", type=int, default=640, help="Input width")

    parser.add_argument("--compile-only", action="store_true",
                       help="Use torch.compile instead of TorchScript")
    parser.add_argument("--use-wrapper", dest="use_wrapper", action="store_true",
                       help="Use optimized tracing wrapper")
    parser.add_argument("--no-wrapper", dest="use_wrapper", action="store_false",
                       help="Disable optimized tracing wrapper")
    parser.set_defaults(use_wrapper=True)

    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--int8", action="store_true", help="Apply INT8 quantization (CPU)")

    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--benchmark-runs", type=int, default=100, help="Number of benchmark runs")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    converter = AdvancedModelConverter()

    try:
        # Load model
        logger.info("=== Loading PromptDA Model ===")
        model = converter.load_checkpoint(args.checkpoint, device)

        # Prepare inputs
        rgb_input, depth_input = converter.prepare_inputs(
            args.batch, args.height, args.width, device, args.fp16
        )

        # Apply optimizations
        logger.info("=== Applying Optimizations ===")
        optimized_model = converter.apply_optimizations(
            model, args.use_wrapper, args.fp16, args.int8,
            args.height, args.width, device, for_torchscript=not args.compile_only
        )

        if args.compile_only:
            logger.info("torch.compile optimization completed!")
            if args.benchmark:
                converter.benchmark_model(optimized_model, (rgb_input, depth_input), num_runs=args.benchmark_runs)
            return

        # Convert to TorchScript
        logger.info("=== Converting to TorchScript ===")
        scripted_model = converter.convert_to_torchscript(
            optimized_model, rgb_input, depth_input, args.method
        )

        # Save model
        logger.info("=== Saving Model ===")
        torch.jit.save(scripted_model, args.output)
        logger.info(f"Saved optimized model to: {args.output}")

        # Benchmark if requested
        if args.benchmark:
            logger.info("=== Benchmarking ===")
            converter.benchmark_model(scripted_model, (rgb_input, depth_input), num_runs=args.benchmark_runs)

        # Summary
        file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
        logger.info(f"Conversion completed! Output size: {file_size_mb:.1f} MB")

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()