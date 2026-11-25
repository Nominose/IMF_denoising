"""Compute the K ablation from per-case MAE, SSIM and LPIPS measurements."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys


def summarize(path, dataset, protocol):
    study = protocol["averaging_study"]
    n_cases = protocol["datasets"][dataset]["test_cases"]
    expected_k = study["k_per_input"]
    cases_by_k = {k: {} for k in expected_k}
    with path.open(encoding="utf-8-sig", newline="") as src:
        for row in csv.DictReader(src):
            if row["dataset"] != dataset or int(row["nfe"]) != study["nfe"]:
                continue
            if dataset == "brain" and row["model"] != "flow":
                continue
            k = int(row["k"])
            if k not in cases_by_k:
                continue
            case = row["patient"].strip()
            values = [float(row[key]) for key in ("mae", "ssim", "lpips")]
            if not case or case in cases_by_k[k] or not all(map(math.isfinite, values)):
                raise ValueError(f"{dataset}, K={k}: empty or duplicate case key, or nonfinite metric")
            cases_by_k[k][case] = values
    cohort = cases_by_k[expected_k[0]].keys()
    if len(cohort) != n_cases:
        raise ValueError(f"{dataset}: expected {n_cases} cases, found {len(cohort)}")
    for k, cases in cases_by_k.items():
        if cases.keys() != cohort:
            raise ValueError(f"{dataset}, K={k}: incomplete or inconsistent cohort")
        metrics = []
        for values in zip(*cases.values()):
            metrics.extend((f"{statistics.mean(values):.8f}", f"{statistics.stdev(values):.8f}"))
        yield [dataset, study["nfe"], k, n_cases, *metrics]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain", type=Path, required=True)
    parser.add_argument("--abdominal", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
    rows = [*summarize(args.brain, "brain", protocol), *summarize(args.abdominal, "abdominal", protocol)]
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["dataset", "nfe", "k_per_input", "test_cases", "mae_hu_mean", "mae_hu_sd",
                     "ssim_mean", "ssim_sd", "lpips_mean", "lpips_sd"])
    writer.writerows(rows)


if __name__ == "__main__":
    main()
