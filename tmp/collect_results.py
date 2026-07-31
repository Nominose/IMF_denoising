"""Collect a trial's predictions into a flat, shareable result tree.

A trial directory nests differently per dataset -- brain CT carries an extra sub-ID level
(<patient>/<subid>/random_N/), Mayo does not (<patient>/random_N/) -- and each case folder holds the
per-sample volumes, the K-averages, gt, condition and a std map. This flattens all of that to the
one layout the result folders use:

    <out>/pred_images_NFE{n}/<patient>/<epoch>avg/pred_img_scans{1,10,20}.nii.gz

Only the K values actually reported are copied (1, 10, 20 by default); gt/condition/std and the
per-sample volumes stay behind. Files are hardlinked when the destination is on the same drive, so
a full tree costs no extra disk; pass --copy to force real copies (needed when moving to another
drive or when the destination must survive the source being deleted).

Two traps this handles, both of which have silently corrupted collected trees before:
  * Stray epoch*avg folders from side experiments sit next to the deployed one for SOME patients
    only (e.g. Mayo nfe3 L310 once had epoch15avg beside epoch200avg). Picking whichever the glob
    returned first mixed checkpoints across patients. --epoch pins it; otherwise the highest epoch
    number wins, which is the deployed one in every trial here.
  * A patient whose K-average is missing is reported rather than skipped quietly.

Examples
    python collect_results.py --trial imf_v2_unsupervised_gaussian_mayo \
        --nfe 2 3 5 10 20 30 50 --out D:/research/projects/denoising/results/mayo/imf_v2
    python collect_results.py --trial imf_gan_unsupervised_gaussian_brainCT --epoch 28 \
        --nfe 3 --k 1 10 20 --out D:/share/gan_brain --copy
"""
import argparse, glob, os, re, shutil, sys

sys.stdout.reconfigure(encoding="utf-8")
DEFAULT_MODELS = os.environ.get(
    "MODELS_ROOT", r"D:\research\projects\denoising\models")


def epoch_of(path):
    m = re.search(r"epoch(\d+)avg", path.replace("\\", "/"))
    return int(m.group(1)) if m else -1


def find_cases(trial_dir, nfe):
    """-> {patient_id: avg_dir}. Handles both the brain (extra sub-ID level) and Mayo layouts."""
    pats = {}
    for sub in (f"pred_images_nfe{nfe}", f"pred_images_NFE{nfe}",
                f"pred_images_input_both_nfe{nfe}"):
        base = os.path.join(trial_dir, sub)
        if os.path.isdir(base):
            break
    else:
        return None, None
    for pdir in sorted(glob.glob(os.path.join(base, "*"))):
        if not os.path.isdir(pdir):
            continue
        pid = os.path.basename(pdir)
        avgs = glob.glob(os.path.join(pdir, "**", "epoch*avg"), recursive=True)
        if avgs:
            pats[pid] = avgs
    return base, pats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trial", required=True, help="trial folder name under the models root")
    ap.add_argument("--out", required=True, help="destination root")
    ap.add_argument("--nfe", type=int, nargs="+", default=[2, 3, 5, 10, 20, 30, 50])
    ap.add_argument("--k", type=int, nargs="+", default=[1, 10, 20],
                    help="which K-averages to collect (default 1 10 20)")
    ap.add_argument("--epoch", type=int, default=None,
                    help="pin the epoch; default = highest epoch*avg present, per patient")
    ap.add_argument("--models_root", default=DEFAULT_MODELS)
    ap.add_argument("--copy", action="store_true",
                    help="copy instead of hardlink (needed across drives)")
    args = ap.parse_args()

    trial_dir = os.path.join(args.models_root, args.trial)
    if not os.path.isdir(trial_dir):
        sys.exit(f"trial not found: {trial_dir}")

    place = shutil.copy2 if args.copy else os.link
    total, missing, epochs_used = 0, [], set()
    for nfe in args.nfe:
        base, pats = find_cases(trial_dir, nfe)
        if base is None:
            print(f"  NFE{nfe}: 源文件夹不存在 -- 跳过")
            continue
        got = 0
        for pid, avgs in pats.items():
            if args.epoch is not None:
                sel = [a for a in avgs if epoch_of(a) == args.epoch]
                if not sel:
                    missing.append(f"NFE{nfe}/{pid}: 无 epoch{args.epoch}avg")
                    continue
                avg = sel[0]
            else:
                avg = max(avgs, key=epoch_of)
            epochs_used.add(epoch_of(avg))
            for k in args.k:
                src = os.path.join(avg, f"pred_img_scans{k}.nii.gz")
                if not os.path.isfile(src):
                    missing.append(f"NFE{nfe}/{pid}: 缺 scans{k}")
                    continue
                dst = os.path.join(args.out, f"pred_images_NFE{nfe}", pid,
                                   os.path.basename(avg), f"pred_img_scans{k}.nii.gz")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if os.path.exists(dst):
                    os.remove(dst)
                try:
                    place(src, dst)
                except OSError:                       # e.g. hardlink across drives
                    shutil.copy2(src, dst)
                got += 1
        total += got
        print(f"  NFE{nfe}: {len(pats)} 患者, {got} 个文件")

    print(f"\n{'复制' if args.copy else '硬链接'} {total} 个文件 -> {args.out}")
    print(f"epoch: {sorted(epochs_used) if epochs_used else '(无)'}   K: {args.k}")
    if missing:
        print(f"\n缺失 {len(missing)} 项:")
        for m in missing[:20]:
            print(f"  {m}")
        if len(missing) > 20:
            print(f"  ... 另有 {len(missing) - 20} 项")


if __name__ == "__main__":
    main()
