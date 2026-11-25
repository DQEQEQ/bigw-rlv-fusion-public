"""Regenerate the five manuscript tables from archived experimental scores."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import mean, stdev


REFERENCE = Path(__file__).resolve().parents[1] / "reference" / "scores"
METRICS = ("MI", "NCIE", "Qabf", "Qp", "Qy", "VIF")
CORE_NAMES = {
    "Baseline_20_20_10": "Baseline",
    "BI_GW_20_20_10": "BI-GW",
    "RLV_ONLY_20_20_10": "RLV-only",
    "BI_GW_RLV100_20_20_10": "BI-GW+RLV",
}
COMPARISON_METHODS = (
    ("CDDFuse", "CDDFuse"),
    ("LUTFuse", "LUT-Fuse"),
    ("PIAFusion", "PIAFusion"),
    ("SFDFusion", "SFDFusion"),
    ("FSATFusion", "FSATFusion"),
    ("SwinFusionOfficial", "SwinFusion"),
    ("BI_GW", "BI-GW"),
    ("BI-GW+RLV", "BI-GW+RLV"),
)


def read(name):
    with (REFERENCE / name).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def means(name, size):
    grouped = defaultdict(dict)
    for row in read(name):
        method = CORE_NAMES.get(row["method"], row["method"])
        stem = Path(row["image"]).stem.casefold()
        if stem in grouped[method]:
            raise ValueError(f"Duplicate score for {method} and {stem} in {name}")
        grouped[method][stem] = tuple(float(row[m]) for m in METRICS)
    reference = set(next(iter(grouped.values())))
    if len(reference) != size or any(set(rows) != reference for rows in grouped.values()):
        raise ValueError(f"Unpaired inputs or incorrect sample count in {name}")
    return {method: tuple(map(mean, zip(*rows.values(), strict=True))) for method, rows in grouped.items()}


def gains(baseline, candidate):
    return tuple(100 * (new - old) / old for old, new in zip(baseline, candidate, strict=True))


def scores(values):
    return tuple(f"{value:.4f}" for value in values)


def write(output, stem, header, rows):
    with (output / f"{stem}.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def build(output):
    output.mkdir(parents=True, exist_ok=False)
    msrs = means("msrs-fourway.csv", 361)
    initial_design = means("msrs-design-ablation.csv", 361)
    rows = []
    for method in ("Baseline", "BI-EQ", "BILS", "BI-GW", "RLV-only", "BI-GW+RLV"):
        numbers = initial_design["BI_EQ" if method == "BI-EQ" else method] if method in ("BI-EQ", "BILS") else msrs[method]
        gain = "0.000" if method == "Baseline" else f"{mean(gains(msrs['Baseline'], numbers)):+.3f}"
        rows.append((method, *scores(numbers), "--" if method in ("BI-EQ", "BILS") else gain))
    write(output, "table1-ablation", ("Method", *METRICS, "Mean gain (%)"), rows)

    external = (
        ("TNO", means("tno49.csv", 49)),
        ("RoadScene", means("roadscene221.csv", 221)),
        ("LLVIP", means("llvip3463.csv", 3463)),
    )
    rows = []
    for dataset, summary in external:
        for method in ("Baseline", "BI-GW", "BI-GW+RLV"):
            rows.append((dataset, method, *scores(summary[method])))
        delta = gains(summary["Baseline"], summary["BI-GW+RLV"])
        rows.append((dataset, "Relative change (%)", *(f"{value:+.3f}" for value in delta)))
    write(output, "table2-cross-dataset", ("Dataset", "Method", *METRICS), rows)

    seed_results = [(msrs["Baseline"], msrs["BI-GW+RLV"])]
    for seed in (456, 789):
        values = means(f"msrs-seed{seed}.csv", 361)
        seed_results.append((values[f"Baseline_seed{seed}"], values[f"BI_GW_RLV100_seed{seed}"]))
    seed_gains = [gains(base, joint) for base, joint in seed_results]
    rows = []
    for i, metric in enumerate(METRICS):
        baselines = [base[i] for base, _ in seed_results]
        combinations = [joint[i] for _, joint in seed_results]
        improvements = [per_seed[i] for per_seed in seed_gains]
        baseline = f"{mean(baselines):.4f} ± {stdev(baselines):.4f}"
        joint = f"{mean(combinations):.4f} ± {stdev(combinations):.4f}"
        gain = f"{mean(improvements):+.3f} ± {stdev(improvements):.3f}"
        improved = f"{sum(value > 0 for value in improvements)}/{len(seed_results)}"
        rows.append((metric, baseline, joint, gain, improved))
    write(output, "table3-three-seed", ("Metric", "Baseline", "BI-GW+RLV", "Relative gain (%)", "Seeds improved"), rows)

    profiles = read("complexity.csv")
    baseline_profile, = [row for row in profiles if row["method"] == "Baseline"]
    if len({(row["parameters"], row["flops_256"]) for row in profiles}) != 1:
        raise ValueError("Archived baseline and BI-GW profiles differ")
    rows = [(name, baseline_profile["parameters"], f"{float(baseline_profile['gflops_256']):.4f}", "No")
            for name in CORE_NAMES.values()]
    write(output, "table4-complexity", ("Method", "Parameters", "GFLOPs", "Extra branches"), rows)

    rows = []
    for dataset, size, primary in (("MSRS", 361, msrs), ("TNO", 49, external[0][1])):
        summary = means(f"comparison-{dataset.lower()}.csv", size)
        summary["BI-GW+RLV"] = primary["BI-GW+RLV"]
        for method, label in COMPARISON_METHODS:
            rows.append((dataset, label, *scores(summary[method])))
    write(output, "table5-representative-methods", ("Dataset", "Method", *METRICS), rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.output_dir)


if __name__ == "__main__":
    main()
