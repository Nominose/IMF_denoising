"""Print the sampling comparison table and optionally plot both ablations."""
import argparse
import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
METRICS = {"mae_hu_mean": "MAE (HU)", "ssim_mean": "SSIM", "lpips_mean": "LPIPS"}


def read_results(name):
    with (HERE / "results" / f"{name}.csv").open(newline="") as src:
        return list(csv.DictReader(src))


def print_table(rows):
    print("Sampling methods at NFE = 3 and K = 10 (mean ± SD across cases)")
    print(f"{'Dataset':11} {'Method':15} {'MAE (HU)':15} {'SSIM':16} LPIPS")
    for r in rows:
        if (r["nfe"], r["k_per_input"]) == ("3", "10"):
            cells = [f"{r[key]} ± {r[key.replace('_mean', '_sd')]}" for key in METRICS]
            print(f"{r['dataset']:11} {r['method']:15} {cells[0]:15} {cells[1]:16} {cells[2]}")


def plot_ablation(rows, settings, x_key, file):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sampling = x_key == "nfe"
    methods = settings.get("methods", ["N2N-iMF"])
    figure, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    for axis_row, dataset in zip(axes, settings["datasets"]):
        for axis, (metric, label) in zip(axis_row, METRICS.items()):
            for method in methods:
                selected = sorted(
                    (r for r in rows if r["dataset"] == dataset and r.get("method", "N2N-iMF") == method),
                    key=lambda r: int(r[x_key]),
                )
                x = [int(r[x_key]) for r in selected]
                if x != settings[x_key]:
                    raise ValueError(f"Incomplete series: {dataset}, {method}, {metric}")
                axis.plot(x, [float(r[metric]) for r in selected], marker=".", label=method)
            if sampling:
                axis.set_xscale("log")
                axis.set_xticks(settings["nfe"], labels=settings["nfe"])
                if metric in ("mae_hu_mean", "lpips_mean"):
                    axis.set_yscale("log")
            else:
                axis.set_xticks([1, 5, 10, 15, 20])
            axis.set(xlabel="NFE per output" if sampling else "K per conditioning input",
                     ylabel=label, title=f"{dataset.title()} CT")
    if sampling:
        axes[0, 0].legend(fontsize="small")
    figure.savefig(file)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots", type=Path, help="directory for the NFE and K plots (requires matplotlib)")
    args = parser.parse_args()
    rows = read_results("sampler_comparison")
    print_table(rows)
    if args.plots:
        args.plots.mkdir(parents=True, exist_ok=True)
        protocol = json.loads((HERE / "protocol.json").read_text())
        plot_ablation(rows, protocol["sampler_comparison"], "nfe", args.plots / "metrics_vs_nfe.svg")
        plot_ablation(read_results("averaging_study"), protocol["averaging_study"], "k_per_input",
                      args.plots / "metrics_vs_k.svg")


if __name__ == "__main__":
    main()
