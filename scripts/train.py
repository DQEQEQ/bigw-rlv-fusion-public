"""Run the paper's single-GPU training protocol with an official SwinFusion checkout."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, DistributedSampler

from bigw_rlv import SwinFusionSupervision
from scripts.infer import discover_pairs, seed_everything


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PARAMETERS = 973_681


def update(model, batch, accumulation_steps):
    model.G_optimizer.zero_grad()
    average = 0.0
    # The reference sampler assigns every third observation to the same rank.
    for rank in range(accumulation_steps):
        micro = {key: batch[key][rank::accumulation_steps] for key in ("A", "B")}
        model.feed_data(micro, need_GT=False)
        model.netG_forward()
        loss, *_ = model.G_lossfn(model.A, model.B, model.E)
        loss = model.G_lossfn_weight * loss
        (loss / accumulation_steps).backward()
        average += float(loss.detach()) / accumulation_steps
    if not math.isfinite(average):
        raise RuntimeError("Non-finite training loss")
    model.G_optimizer.step()
    model.update_E(model.opt_train["E_decay"])
    return average


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--ir-dir", type=Path, required=True)
    parser.add_argument("--vi-dir", type=Path, required=True, help="Prepared visible luminance (vi_y)")
    parser.add_argument("--output-dir", type=Path, required=True, help="A fresh folder for one run")
    parser.add_argument("--variant", choices=SwinFusionSupervision.variants, required=True)
    parser.add_argument("--seed", type=int, choices=(123, 456, 789), default=123)
    parser.add_argument("--steps", type=int, default=10000, help="Lower for a GPU smoke test")
    parser.add_argument("--dry-run", action="store_true", help="Check input files and sampling without CUDA")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    protocol = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
    settings = protocol["training"]
    wanted = protocol["datasets"]["MSRS"]["training_pairs"]
    pairs = discover_pairs(args.ir_dir, args.vi_dir)
    if len(pairs) != wanted:
        parser.error(f"Expected {wanted} MSRS training pairs, found {len(pairs)}")

    upstream = args.repo_root.resolve()
    if not (upstream / "models/network_swinfusion1.py").is_file():
        parser.error("--repo-root must point to a SwinFusion source checkout")
    sys.path.insert(0, str(upstream))
    from data.select_dataset import define_Dataset
    from utils import utils_option

    options = utils_option.parse(str(ROOT / "configs/swinfusion_vif.json"), is_train=True)
    options["datasets"]["train"].update(
        dataroot_A=str(args.ir_dir.resolve()), dataroot_B=str(args.vi_dir.resolve())
    )
    options["train"]["manual_seed"] = args.seed
    options["gpu_ids"] = [0]
    options["num_gpu"] = 1
    options["rank"] = 0
    options["world_size"] = 1
    if not args.dry_run and not torch.cuda.is_available():
        parser.error("Training requires one CUDA GPU. Use --dry-run to check the inputs on CPU.")
    seed_everything(args.seed)
    options = utils_option.dict_to_nonedict(options)
    dataset = define_Dataset(options["datasets"]["train"])
    paths_a = [Path(path).stem.casefold() for path in dataset.paths_A]
    paths_b = [Path(path).stem.casefold() for path in dataset.paths_B]
    if paths_a != paths_b or set(paths_a) != {stem.casefold() for stem, *_ in pairs}:
        raise RuntimeError("Upstream dataset order does not preserve the audited IR/VI pairs")
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, drop_last=True, seed=args.seed)
    loader = DataLoader(
        dataset, batch_size=settings["effective_batch_size"], sampler=sampler,
        num_workers=0, pin_memory=not args.dry_run, drop_last=True,
    )
    print(f"{len(dataset)} training pairs, {len(loader)} batches per epoch, seed {args.seed}")
    if args.dry_run:
        first = next(iter(loader))
        expected = (settings["effective_batch_size"], 1, settings["patch_size"], settings["patch_size"])
        if any(tuple(first[key].shape) != expected for key in ("A", "B")):
            raise RuntimeError("Unexpected first-batch dimensions")
        print(f"Verified the first batch and paired augmentation: {expected}")
        return

    from models.loss_ssim import create_window
    from models.select_model import define_Model

    run_dir = args.output_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    options["path"].update(root=str(run_dir), models=str(run_dir), pretrained_optimizerG=None)
    torch.cuda.set_device(0)
    model = define_Model(options)
    try:
        model.init_train()
        params = model.get_bare_model(model.netG)
        if sum(t.numel() for t in params.parameters()) != EXPECTED_PARAMETERS:
            raise RuntimeError("The trainable network does not match the paper architecture")
        metadata = {
            "variant": args.variant,
            "seed": args.seed,
            "training_pairs": len(dataset),
            "requested_steps": args.steps,
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "initial_network_sha256": state_hash(params.state_dict()),
        }
        (run_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        model.G_lossfn = SwinFusionSupervision(model.G_lossfn, create_window, args.variant).to(model.device)

        step = 0
        with (run_dir / "training.csv").open("w", encoding="utf-8") as log:
            log.write("step,total_loss\n")
            # The reference never advances the sampler's epoch. Augmentation RNG still advances.
            while step < args.steps:
                for batch in loader:
                    step += 1
                    model.update_learning_rate(step)
                    loss = update(model, batch, settings["gradient_accumulation_steps"])
                    log.write(f"{step},{loss:.10g}\n")
                    if step % 200 == 0:
                        log.flush()
                        print(f"step {step:5d}   loss {loss:.6f}", flush=True)
                    if step % 1000 == 0 or step == args.steps:
                        model.save(step)
                    if step == args.steps:
                        break
    finally:
        model.writer.close()


if __name__ == "__main__":
    main()
