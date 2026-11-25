"""Run the existing N2N-iMF inference scripts with the main study's sampling settings."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    protocol = json.loads((HERE / "protocol.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    datasets = parser.add_subparsers(dest="dataset", required=True)
    for name in protocol["datasets"]:
        run = datasets.add_parser(name)
        run.add_argument("--dry-run", action="store_true", help="print the commands without running inference")
        if name in ("brain", "abdominal"):
            run.add_argument("--trial", required=True, help="model directory in the study folder")
            run.add_argument("--epoch", required=True, type=int)
        if name in ("brain", "pcct"):
            run.add_argument("--weights", required=True, choices=("ema", "raw"), help="checkpoint weights to evaluate")
        if name == "pcct":
            run.add_argument("--checkpoint", required=True, type=Path)
            run.add_argument("--data", required=True, type=Path, help="prepared PCCT test set")
            run.add_argument("--output", required=True, type=Path, help="output directory for this checkpoint")
            run.add_argument("--work", required=True, type=Path, help="sample cache dedicated to this checkpoint and weight selection")
    args = parser.parse_args()
    preset = protocol["datasets"][args.dataset]
    nfe, k = str(preset["nfe"]), str(preset["k_per_input"])
    if args.dataset == "pcct":
        script = ROOT / "PCCT_experiments/predict_2D_imf.py"
        flags = ["--ckpt", str(args.checkpoint.resolve()), "--data", str(args.data.resolve()),
                 "--out", str(args.output.resolve()), "--work", str(args.work.resolve()),
                 "--weights", args.weights, "--nfe", nfe, "--k", k, "--k_save", k]
        commands = [[sys.executable, str(script), *flags]]
    else:
        flags = ["--trial_name", args.trial, "--epoch", str(args.epoch),
                 "--num_steps", nfe, "--iteration_num", k, "--solver", "euler"]
        if args.dataset == "brain":
            script = ROOT / "Thinslice_experiments/predict_2D_imf_v2.py"
            flags.extend(["--weights", args.weights, "--k_save", k])
        else:
            script = ROOT / "CT_experiments/predict_2D_imf.py"
            flags.extend(["--input", "both"])
        commands = [[sys.executable, str(script), "--mode", mode, *flags] for mode in ("pred", "avg")]

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT.parent), env.get("PYTHONPATH"))))
    for command in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
