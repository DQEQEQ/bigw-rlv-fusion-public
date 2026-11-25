"""Compute the four-way RLV comparison from aligned source and fused PNG images."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats
import torch

from bigw_rlv import relative_local_variance_error
from scripts.evaluate_ivif import collect_tasks, write_csv


METHODS = ("baseline", "bigw", "rlv_only", "bigw_rlv")
COMPARISONS = (
    ("bigw", "baseline"),
    ("rlv_only", "baseline"),
    ("bigw_rlv", "bigw"),
    ("bigw_rlv", "rlv_only"),
    ("bigw_rlv", "baseline"),
)


def _load_gray(path):
    with Image.open(path) as image:
        values = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(values)[None, None]


def _bootstrap_ci(values, rng, repetitions=20000):
    means = []
    for start in range(0, repetitions, 1000):
        count = min(1000, repetitions - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        means.append(values[indices].mean(axis=1))
    return np.percentile(np.concatenate(means), [2.5, 97.5])


def _holm(values):
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    ordered = np.asarray(values)[order] * np.arange(len(values), 0, -1)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(ordered))
    return adjusted


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir-dir", type=Path, required=True)
    parser.add_argument("--vi-dir", type=Path, required=True)
    for method in METHODS:
        parser.add_argument(f"--{method.replace('_', '-')}-dir", dest=method, type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=361)
    args = parser.parse_args()

    directories = [(method, str(getattr(args, method))) for method in METHODS]
    tasks = collect_tasks(args.ir_dir, args.vi_dir, directories, 0, args.expected_count)
    per_image = []
    for filename, ir_path, vi_path, methods in tasks:
        infrared, visible = _load_gray(ir_path), _load_gray(vi_path)
        row = {"image": filename}
        for method, directory in methods:
            fused = _load_gray(Path(directory) / filename)
            if infrared.shape != visible.shape or infrared.shape != fused.shape:
                raise ValueError(f"Mismatched image shapes for {filename}, method {method}")
            row[method] = relative_local_variance_error(infrared, visible, fused).item()
        per_image.append(row)

    values = {method: np.array([row[method] for row in per_image]) for method in METHODS}
    rng = np.random.default_rng(123)
    comparisons = []
    for candidate, reference in COMPARISONS:
        c, r = values[candidate], values[reference]
        delta = c - r
        low, high = _bootstrap_ci(delta, rng)
        p = 1.0 if np.allclose(delta, 0) else stats.wilcoxon(
            delta, alternative="two-sided", zero_method="wilcox", method="auto"
        ).pvalue
        comparisons.append({
            "candidate": candidate, "reference": reference, "n": len(delta),
            "candidate_mean": c.mean(), "reference_mean": r.mean(),
            "error_reduction_percent": 100.0 * (r.mean() - c.mean()) / r.mean() if r.mean() else np.nan,
            "mean_error_difference": delta.mean(), "ci95_low": low, "ci95_high": high,
            "win_rate_percent": 100.0 * (c < r).mean(), "wilcoxon_p": p,
        })
    for row, adjusted in zip(comparisons, _holm([row["wilcoxon_p"] for row in comparisons])):
        row["holm_p"] = adjusted

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "rlv_mechanism_per_image.csv", per_image, ["image", *METHODS])
    write_csv(args.output_dir / "rlv_mechanism_paired_statistics.csv", comparisons, list(comparisons[0]))


if __name__ == "__main__":
    main()
