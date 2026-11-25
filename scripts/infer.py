#!/usr/bin/env python
"""Run IVIF inference with SwinFusion and a trained EMA checkpoint.

This wrapper intentionally mirrors the official test_swinfusion.py pipeline:
grayscale uint8 -> [0, 1] tensor, reflection-style flip padding to the next
window boundary, strict loading of 10000_E.pth['params'], output cropping, and
the repository's tensor2uint/imsave conversion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run IVIF inference with SwinFusion and a trained EMA checkpoint."
    )
    parser.add_argument("--repo_root", type=Path, required=True,
                        help="SwinFusion repository root containing models/ and utils/.")
    parser.add_argument("--ir_dir", type=Path, required=True)
    parser.add_argument("--vi_dir", type=Path, required=True,
                        help="Visible luminance (VI_Y) image directory.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True,
                        help="Path to a step-10000 EMA checkpoint.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--expected_count", type=int, default=None,
                        help="Fail unless this many complete source pairs are discovered.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N sorted pairs (pilot test).")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output_ext", choices=("png", "bmp", "tif", "tiff"), default="png")
    parser.add_argument("--reference_dir", type=Path, default=None,
                        help="Optional official-output folder; every processed image must match exactly.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacement of already existing output images.")
    return parser.parse_args()


def natural_key(value: str) -> Tuple[object, ...]:
    return tuple(int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", value))


def image_map(folder: Path) -> Dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {folder}")
    result: Dict[str, Path] = {}
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = path.stem.lower()
        if key in result:
            raise RuntimeError(
                f"Duplicate image stem in {folder}: {result[key].name!r} and {path.name!r}"
            )
        result[key] = path.resolve()
    return result


def discover_pairs(ir_dir: Path, vi_dir: Path) -> List[Tuple[str, Path, Path]]:
    ir = image_map(ir_dir)
    vi = image_map(vi_dir)
    only_ir = sorted(set(ir) - set(vi), key=natural_key)
    only_vi = sorted(set(vi) - set(ir), key=natural_key)
    if only_ir or only_vi:
        raise RuntimeError(
            "IR/VI pairing mismatch. "
            f"only_ir={only_ir[:10]} (total {len(only_ir)}), "
            f"only_vi={only_vi[:10]} (total {len(only_vi)})"
        )
    return [(ir[stem].stem, ir[stem], vi[stem]) for stem in sorted(ir, key=natural_key)]


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError):
        pass


def load_checkpoint(path: Path):
    return torch.load(str(path), map_location="cpu", weights_only=True)


def extract_state_dict(checkpoint) -> Tuple[dict, str]:
    """Resolve common checkpoint containers without weakening strict loading."""
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f"Unsupported checkpoint object type: {type(checkpoint).__name__}. Expected a dictionary."
        )

    for key in ("params", "params_ema", "state_dict", "model_state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict) and candidate:
            return candidate, key

    # The official SwinFusion 10000_E.pth distributed with some repository
    # versions is itself an OrderedDict[str, Tensor], not {'params': ...}.
    if checkpoint and all(
        isinstance(value, (torch.Tensor, torch.nn.Parameter))
        for value in checkpoint.values()
    ):
        return checkpoint, "direct_state_dict"

    keys = list(checkpoint.keys())[:20]
    raise RuntimeError(
        "Unable to locate a model state_dict in the checkpoint. "
        f"Top-level keys (first 20): {keys}"
    )


def to_model_tensor(util, path: Path, device: torch.device) -> torch.Tensor:
    image = util.imread_uint(str(path), n_channels=1)
    image = util.uint2single(image)
    image = util.single2tensor3(image).unsqueeze(0)
    return image.to(device, non_blocking=False)


def official_flip_pad(image: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, int, int]:
    _, _, old_h, old_w = image.size()
    # Keep the official formula exactly: divisible dimensions receive one full
    # extra window, matching test_swinfusion.py rather than a generic pad-to-8.
    h_pad = (old_h // window_size + 1) * window_size - old_h
    w_pad = (old_w // window_size + 1) * window_size - old_w
    image = torch.cat([image, torch.flip(image, [2])], 2)[:, :, : old_h + h_pad, :]
    image = torch.cat([image, torch.flip(image, [3])], 3)[:, :, :, : old_w + w_pad]
    return image, old_h, old_w


def version_or_none(module_name: str):
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "unknown")
    except Exception:
        return None


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    ir_dir = args.ir_dir.resolve()
    vi_dir = args.vi_dir.resolve()
    output_dir = args.output_dir.resolve()
    model_path = args.model_path.resolve()
    reference_dir = args.reference_dir.resolve() if args.reference_dir else None

    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {model_path}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The requested CUDA device is unavailable.")
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())

    pairs = discover_pairs(ir_dir, vi_dir)
    discovered_count = len(pairs)
    if args.expected_count is not None and discovered_count != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} source pairs, but discovered {discovered_count}."
        )
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be a positive integer.")
        pairs = pairs[: args.limit]
    if not pairs:
        raise RuntimeError("No paired source images were discovered.")

    output_dir.mkdir(parents=True, exist_ok=True)
    planned_outputs = [output_dir / f"{stem}.{args.output_ext}" for stem, _, _ in pairs]
    existing = [p for p in planned_outputs if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{len(existing)} planned output image(s) already exist, for example {existing[0]}. "
            "Use a new output folder or pass --overwrite."
        )

    sys.path.insert(0, str(repo_root))
    original_cwd = Path.cwd()
    os.chdir(repo_root)
    try:
        from models.network_swinfusion1 import SwinFusion
        from utils import utils_image as util
    finally:
        os.chdir(original_cwd)

    seed_everything(args.seed)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    window_size = 8
    model = SwinFusion(
        upscale=1,
        in_chans=1,
        img_size=128,
        window_size=window_size,
        img_range=1.0,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler=None,
        resi_connection="1conv",
    )
    checkpoint = load_checkpoint(model_path)
    state_dict, checkpoint_format = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    reference_map = image_map(reference_dir) if reference_dir else None
    timing_rows: List[dict] = []
    inference_seconds: List[float] = []
    output_mins: List[int] = []
    output_maxs: List[int] = []
    exact_reference_count = 0

    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    total_start = time.perf_counter()

    for index, (stem, ir_path, vi_path) in enumerate(pairs, start=1):
        ir_tensor = to_model_tensor(util, ir_path, device)
        vi_tensor = to_model_tensor(util, vi_path, device)
        ir_tensor, old_h, old_w = official_flip_pad(ir_tensor, window_size)
        vi_tensor, vi_h, vi_w = official_flip_pad(vi_tensor, window_size)
        if (old_h, old_w) != (vi_h, vi_w):
            raise RuntimeError(
                f"Spatial-size mismatch for {stem}: IR={(old_h, old_w)}, VI={(vi_h, vi_w)}"
            )

        if use_cuda:
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            fused = model(ir_tensor, vi_tensor)
            fused = fused[..., :old_h, :old_w]
        if use_cuda:
            torch.cuda.synchronize(device)
        inference_time = time.perf_counter() - start

        if not torch.isfinite(fused).all().item():
            raise RuntimeError(f"Non-finite model output detected for {stem}.")
        fused_uint = util.tensor2uint(fused.detach()[0].float().cpu())
        output_path = output_dir / f"{stem}.{args.output_ext}"
        util.imsave(fused_uint, str(output_path))

        output_min = int(np.min(fused_uint))
        output_max = int(np.max(fused_uint))
        output_mins.append(output_min)
        output_maxs.append(output_max)
        inference_seconds.append(inference_time)

        reference_exact = None
        reference_mae = None
        reference_max_diff = None
        if reference_map is not None:
            if stem.lower() not in reference_map:
                raise FileNotFoundError(f"Reference output missing for stem {stem!r}: {reference_dir}")
            reference = np.squeeze(util.imread_uint(str(reference_map[stem.lower()]), n_channels=1))
            generated = np.squeeze(fused_uint)
            if reference.shape != generated.shape:
                raise RuntimeError(
                    f"Reference shape mismatch for {stem}: reference={reference.shape}, generated={generated.shape}"
                )
            difference = np.abs(reference.astype(np.int16) - generated.astype(np.int16))
            reference_mae = float(difference.mean())
            reference_max_diff = int(difference.max())
            reference_exact = reference_max_diff == 0
            if not reference_exact:
                raise RuntimeError(
                    f"Official-reference mismatch for {stem}: MAE={reference_mae:.8f}, "
                    f"MAX_DIFF={reference_max_diff}."
                )
            exact_reference_count += 1

        timing_rows.append(
            {
                "index": index,
                "stem": stem,
                "height": old_h,
                "width": old_w,
                "inference_seconds": f"{inference_time:.8f}",
                "output_min": output_min,
                "output_max": output_max,
                "reference_exact": reference_exact,
                "reference_mae": reference_mae,
                "reference_max_diff": reference_max_diff,
            }
        )
        print(
            f"[{index:04d}/{len(pairs):04d}] {stem} "
            f"{old_h}x{old_w} time={inference_time:.4f}s "
            f"range=[{output_min},{output_max}]"
            + (" reference=EXACT" if reference_map is not None else ""),
            flush=True,
        )

    total_elapsed = time.perf_counter() - total_start
    after_first = inference_seconds[1:]
    summary = {
        "status": "PASS",
        "created_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "repo_root": str(repo_root),
        "ir_dir": str(ir_dir),
        "vi_dir": str(vi_dir),
        "output_dir": str(output_dir),
        "reference_dir": str(reference_dir) if reference_dir else None,
        "discovered_pairs": discovered_count,
        "processed_pairs": len(pairs),
        "expected_count": args.expected_count,
        "limit": args.limit,
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "checkpoint_format": checkpoint_format,
        "parameters": parameter_count,
        "strict_load": True,
        "model_eval": True,
        "seed": args.seed,
        "output_ext": args.output_ext,
        "output_global_min": min(output_mins),
        "output_global_max": max(output_maxs),
        "reference_exact_count": exact_reference_count if reference_map is not None else None,
        "reference_all_exact": exact_reference_count == len(pairs) if reference_map is not None else None,
        "elapsed_seconds": round(total_elapsed, 6),
        "mean_inference_seconds_all": round(mean(inference_seconds), 8),
        "mean_inference_seconds_after_first": round(mean(after_first), 8) if after_first else None,
        "std_inference_seconds_after_first": round(pstdev(after_first), 8) if len(after_first) > 1 else 0.0 if after_first else None,
        "peak_cuda_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1024**2, 3) if use_cuda else None,
        "peak_cuda_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 1024**2, 3) if use_cuda else None,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": version_or_none("torchvision"),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if use_cuda else None,
        "pid": os.getpid(),
    }

    with (output_dir / "per_image_timing.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timing_rows[0].keys()))
        writer.writeheader()
        writer.writerows(timing_rows)
    (output_dir / "inference_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if reference_map is not None:
        print("OFFICIAL_PIXEL_EXACT_MATCH_PASS")
    print("SWINFUSION_WINDOWS_INFERENCE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
