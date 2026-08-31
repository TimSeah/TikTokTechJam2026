from __future__ import annotations

import argparse
import time

import open_clip
import torch

MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED_CHECKPOINT = "openai"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the detector environment.")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--benchmark-clip", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def resolve_device(requested: str, require_gpu: bool) -> torch.device:
    gpu_available = torch.cuda.is_available()
    if require_gpu and not gpu_available:
        raise RuntimeError("A GPU was required, but PyTorch cannot access one.")
    if requested == "cuda" and not gpu_available:
        raise RuntimeError("--device cuda was requested, but no GPU is available.")
    if requested == "auto":
        requested = "cuda" if gpu_available else "cpu"
    return torch.device(requested)


def validate_tensor_math(device: torch.device) -> None:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    left = torch.randn((1024, 1024), device=device, dtype=dtype)
    right = torch.randn((1024, 1024), device=device, dtype=dtype)
    result = left @ right
    if device.type == "cuda":
        torch.cuda.synchronize()
    if not torch.isfinite(result).all():
        raise RuntimeError("Tensor validation produced non-finite values.")


def benchmark_clip(device: torch.device, batch_size: int) -> None:
    if batch_size < 1:
        raise ValueError("--batch-size must be positive.")

    model = (
        open_clip.create_model(
            MODEL_NAME,
            pretrained=PRETRAINED_CHECKPOINT,
        )
        .eval()
        .to(device)
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count >= 2_000_000_000:
        raise RuntimeError(f"Model has too many parameters: {parameter_count:,}")

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    images = torch.randn((batch_size, 3, 224, 224), device=device, dtype=dtype)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                features = model.encode_image(images)
        else:
            features = model.encode_image(images)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if features.shape != (batch_size, 512):
        raise RuntimeError(f"Unexpected CLIP feature shape: {tuple(features.shape)}")
    if not torch.isfinite(features).all():
        raise RuntimeError("CLIP produced non-finite features.")

    print(f"clip_model={MODEL_NAME}:{PRETRAINED_CHECKPOINT}")
    print(f"clip_parameters={parameter_count:,}")
    print(f"clip_feature_shape={tuple(features.shape)}")
    print(f"clip_seconds={elapsed:.3f}")
    print(f"clip_images_per_second={batch_size / elapsed:.1f}")
    if device.type == "cuda":
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3
        print(f"clip_peak_vram_gib={peak_gib:.2f}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device, args.require_gpu)

    print(f"torch={torch.__version__}")
    print(f"hip={torch.version.hip}")
    print(f"open_clip={open_clip.__version__}")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
        total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"gpu_vram_gib={total_gib:.1f}")

    validate_tensor_math(device)
    print("tensor_math=ok")
    if args.benchmark_clip:
        benchmark_clip(device, args.batch_size)


if __name__ == "__main__":
    main()
