#!/usr/bin/env python3
"""
cleanup_pred_samples.py — reclaim disk after the brain-CT iMF+GAN NFE sweep.

The sweep writes, per NFE and per case:
    pred_images_nfe{N}/<pid>/<subid>/random_{r}/
        epoch{E}_1 .. epoch{E}_K/pred_img.nii.gz    <- K raw stochastic samples (the bulk of the disk)
        epoch{E}avg/pred_img_scans10.nii.gz          <- K=10 average   (KEEP)
        epoch{E}avg/pred_img_scans20.nii.gz          <- K=20 average   (KEEP)
        epoch{E}avg/gt_img.nii.gz                    <- ground truth   (KEEP)
        epoch{E}avg/sample_std.npy                   <- per-pixel std  (kept unless --drop-std)

eval_gan_nfe.py reads ONLY epoch{E}avg/{pred_img_scans10,pred_img_scans20,gt_img}.nii.gz, so the
raw epoch{E}_* sample folders are disposable once averaged. This script deletes them, leaving just
the K10/K20 averages + gt per case.  ** IRREVERSIBLE: afterwards you can no longer re-average at
other K, nor recompute sample_std. **

SAFETY — a case+epoch's raw folders are deleted ONLY IF its epoch{E}avg already holds valid
K10 + K20 + gt. If any is missing (that NFE's `avg` step never ran, or the job died mid-sweep),
the case is SKIPPED and its raw samples are kept, so nothing that can't be regenerated is lost.

DRY-RUN BY DEFAULT: run once to preview + see the space it would free, then re-run with --execute.

    python gan/cleanup_pred_samples.py                          # preview brain-CT GAN trial
    python gan/cleanup_pred_samples.py --execute                # actually delete
    python gan/cleanup_pred_samples.py --execute --drop-std     # also drop sample_std.npy
    python gan/cleanup_pred_samples.py --nfe 5 20 --execute     # limit to some NFEs
    python gan/cleanup_pred_samples.py --trial <name> --execute # a different trial (e.g. Mayo)
"""
import os
import sys
import glob
import shutil
import argparse

MIN_VALID_BYTES = 1024   # a real .nii.gz volume is MBs; guards against 0-byte / half-written files


def _detect_base():
    for b in ('/gpfs/work/aac/xingyiyao23', '/host/d/research', '/host/d'):
        if os.path.isdir(os.path.join(b, 'projects/denoising/models')):
            return b
    return '/gpfs/work/aac/xingyiyao23'


def _valid(path):
    return os.path.isfile(path) and os.path.getsize(path) >= MIN_VALID_BYTES


def _dir_bytes(d):
    total = 0
    for root, _, files in os.walk(d):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _fmt(n):
    n = float(n)
    for u in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024 or u == 'TB':
            return f'{n:.1f}{u}'
        n /= 1024


def main():
    ap = argparse.ArgumentParser('delete raw per-sample preds, keep only K10/K20/gt per case')
    ap.add_argument('--study', default=os.path.join(_detect_base(), 'projects/denoising/models'))
    ap.add_argument('--trial', default='imf_gan_unsupervised_gaussian_brainCT')
    ap.add_argument('--nfe', type=int, nargs='*', default=None,
                    help='limit to these NFEs (default: every pred_images_nfe* folder found)')
    ap.add_argument('--k_keep', type=int, nargs='+', default=[10, 20],
                    help='avg-of-K volumes that must exist before a case is cleaned (default 10 20)')
    ap.add_argument('--drop-std', dest='drop_std', action='store_true',
                    help='also delete epoch{E}avg/sample_std.npy (~50MB/case; NOT recomputable after)')
    ap.add_argument('--execute', action='store_true',
                    help='actually delete (default: dry-run preview, deletes nothing)')
    args = ap.parse_args()

    trial_dir = os.path.join(args.study, args.trial)
    if not os.path.isdir(trial_dir):
        sys.exit(f'trial folder not found: {trial_dir}')

    if args.nfe:
        nfe_dirs = [d for d in (os.path.join(trial_dir, f'pred_images_nfe{n}') for n in args.nfe)
                    if os.path.isdir(d)]
    else:
        nfe_dirs = sorted(glob.glob(os.path.join(trial_dir, 'pred_images_nfe*')))
    if not nfe_dirs:
        sys.exit(f'no pred_images_nfe* folders under {trial_dir}')

    print('EXECUTE (deleting)' if args.execute
          else 'DRY-RUN (nothing deleted; add --execute to delete)')
    print('trial      :', trial_dir)
    print('NFE folders:', ', '.join(os.path.basename(d) for d in nfe_dirs))
    print('keep/case  : ' + ', '.join(f'scans{k}' for k in args.k_keep) + ', gt'
          + ('' if args.drop_std else ', sample_std') + '\n')

    total_dirs = total_bytes = n_cleaned = n_skipped = 0
    for nfe_dir in nfe_dirs:
        for case in sorted(glob.glob(os.path.join(nfe_dir, '*', '*', 'random_*'))):
            # epochs that actually have raw sample folders (epoch{E}_<iter>) in this case
            epochs = set()
            for rd in glob.glob(os.path.join(case, 'epoch*_*')):
                if os.path.isdir(rd):
                    tag = os.path.basename(rd)[len('epoch'):].split('_', 1)[0]
                    if tag.isdigit():
                        epochs.add(tag)
            for E in sorted(epochs, key=int):
                avg = os.path.join(case, f'epoch{E}avg')
                keep = [os.path.join(avg, f'pred_img_scans{k}.nii.gz') for k in args.k_keep]
                keep.append(os.path.join(avg, 'gt_img.nii.gz'))
                missing = [os.path.basename(f) for f in keep if not _valid(f)]
                raw = sorted(r for r in glob.glob(os.path.join(case, f'epoch{E}_*')) if os.path.isdir(r))
                rel = os.path.relpath(case, trial_dir)
                if missing:
                    n_skipped += 1
                    reason = f'no epoch{E}avg' if not os.path.isdir(avg) else f'missing {missing}'
                    print(f'[SKIP] {rel} epoch{E}: {reason} -> keeping {len(raw)} raw folders')
                    continue
                freed = sum(_dir_bytes(r) for r in raw)
                if args.drop_std:
                    std = os.path.join(avg, 'sample_std.npy')
                    if os.path.isfile(std):
                        freed += os.path.getsize(std)
                n_cleaned += 1
                total_dirs += len(raw)
                total_bytes += freed
                print(f'[{"DEL " if args.execute else "PREV"}] {rel} epoch{E}: '
                      f'{len(raw)} raw folders, {_fmt(freed)}'
                      + (' (+std)' if args.drop_std else ''))
                if args.execute:
                    for r in raw:
                        shutil.rmtree(r, ignore_errors=True)
                    if args.drop_std:
                        std = os.path.join(avg, 'sample_std.npy')
                        if os.path.isfile(std):
                            os.remove(std)

    print(f'\ncases cleaned: {n_cleaned} | cases skipped (avg incomplete): {n_skipped}')
    print(f'raw folders removed: {total_dirs} | '
          f'space {"freed" if args.execute else "to free"}: {_fmt(total_bytes)}')
    if not args.execute and total_dirs:
        print('\npreview only — re-run with --execute to delete.')


if __name__ == '__main__':
    main()
