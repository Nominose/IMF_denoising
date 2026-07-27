"""
fix_pcct_xlsx.py — rewrite the PCCT patient list so its paths point at the uploaded data.

The shipped PCCT_split.xlsx was written on the old docker box and is stale in THREE ways, all of
which must be fixed together (fixing only the directory silently finds nothing):

  1. directory : /host/d/Data/PCCT/data/soft_thins/<case>/  ->  <base>/Data/PCCT/soft_thins_xy/<case>/
  2. FILENAME  : soft_thins_0_noblank.nii.gz  ->  soft_thins_0_noblank_sliced.nii.gz
                 (the uploaded data is the 50-slice "sliced" build, a different file)
  3. slice_num : the column says 138/100/96/... (the FULL volumes). Every uploaded volume is
                 actually 50 slices. Left uncorrected, anything trusting that column picks slices
                 that do not exist.

Writes a NEW file (PCCT_split_hpc.xlsx) and never touches the original. Every rewritten path is
checked on disk, and the true slice count is read from each volume's NIfTI header, so a wrong
result is reported here rather than 20 hours into a training run.

    python PCCT_experiments/fix_pcct_xlsx.py            # dry-run: show what would change
    python PCCT_experiments/fix_pcct_xlsx.py --write    # write <base>/Data/PCCT/Patient_lists/PCCT_split_hpc.xlsx
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import nibabel as nb


def _detect_base():
    for b in ('/gpfs/work/aac/xingyiyao23', '/host/d/research', '/host/d'):
        if os.path.isdir(os.path.join(b, 'Data')):
            return b
    return '/gpfs/work/aac/xingyiyao23'


_BASE = _detect_base()
_PCCT = os.path.join(_BASE, 'Data', 'PCCT')
OLD_NAME, NEW_NAME = 'soft_thins_0_noblank.nii.gz', 'soft_thins_0_noblank_sliced.nii.gz'


def remap(p, case, data_dir):
    """Old docker path -> the uploaded location. Keyed on the CASE NUMBER (from the sheet) rather
    than on string surgery, so a differently-shaped stale path still resolves."""
    cand = os.path.join(data_dir, str(case), NEW_NAME)
    if os.path.isfile(cand):
        return cand
    # fall back to the un-sliced name, in case the full volumes were uploaded instead
    alt = os.path.join(data_dir, str(case), OLD_NAME)
    if os.path.isfile(alt):
        return alt
    return cand          # return the expected path; the caller reports it as missing


def main():
    ap = argparse.ArgumentParser('rewrite the PCCT patient list for the cluster')
    ap.add_argument('--xlsx_in', default=os.path.join(_PCCT, 'Patient_lists', 'PCCT_split.xlsx'),
                    help='source list; also tries the repo copy and Data/PCCT/ if absent')
    ap.add_argument('--xlsx_out', default=os.path.join(_PCCT, 'Patient_lists', 'PCCT_split_hpc.xlsx'))
    ap.add_argument('--data_dir', default=os.path.join(_PCCT, 'soft_thins_xy'))
    ap.add_argument('--roi_dir', default=os.path.join(_PCCT, 'ROI'))
    ap.add_argument('--write', action='store_true', help='actually write (default: dry-run preview)')
    args = ap.parse_args()

    src = args.xlsx_in
    if not os.path.isfile(src):
        for c in (os.path.join(_PCCT, 'PCCT_split.xlsx'),
                  os.path.join(_BASE, 'Data', 'PCCT_split.xlsx'),
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PCCT_split.xlsx')):
            if os.path.isfile(c):
                src = c
                break
    if not os.path.isfile(src):
        sys.exit(f'patient list not found: {args.xlsx_in}\n(upload PCCT_split.xlsx, or pass --xlsx_in)')

    print('base     :', _BASE)
    print('list  in :', src)
    print('list out :', args.xlsx_out, '' if args.write else '  (DRY-RUN, nothing written)')
    print('data dir :', args.data_dir)
    if not os.path.isdir(args.data_dir):
        sys.exit(f'data dir missing: {args.data_dir}\nrun PCCT_experiments/unpack_pcct_hpc.sh first')

    d = pd.read_excel(src)
    print(f'\nrows: {len(d)} | batches: {dict(d["batch"].value_counts().sort_index())}'
          '   (0=train, 1=val, 2=test)')

    new_paths, new_slices, missing = [], [], []
    for _, r in d.iterrows():
        case = r['Patient_ID']
        p = remap(r['noise_file'], case, args.data_dir)
        new_paths.append(p)
        if os.path.isfile(p):
            try:
                new_slices.append(int(nb.load(p).shape[2]))
            except Exception as e:
                new_slices.append(-1)
                missing.append((case, f'unreadable: {str(e)[:40]}'))
        else:
            new_slices.append(-1)
            missing.append((case, 'file not found'))

    out = d.copy()
    out['noise_file'] = new_paths
    old_slices = d['slice_num'].tolist() if 'slice_num' in d.columns else [None] * len(d)
    out['slice_num'] = new_slices

    print('\n--- per case (old slice_num -> actual) ---')
    for i, (_, r) in enumerate(d.iterrows()):
        ok = 'OK ' if os.path.isfile(new_paths[i]) else 'MISS'
        chg = '' if old_slices[i] == new_slices[i] else f'   slice_num {old_slices[i]} -> {new_slices[i]}'
        if ok == 'MISS' or chg:
            print(f'  [{ok}] batch {r["batch"]} case {str(r["Patient_ID"]):>3}{chg}')

    print(f'\nresolved {len(d) - len(missing)}/{len(d)} volumes')
    uniq = sorted(set(s for s in new_slices if s > 0))
    print(f'actual slice counts present: {uniq}'
          + ('   <- all volumes identical, slice_range=None covers everything' if len(uniq) == 1 else ''))

    # ROI coverage: CNR needs BOTH masks for every test case (batch 2)
    test_cases = [str(c) for c in d.loc[d['batch'] == 2, 'Patient_ID'].tolist()]
    print(f'\n--- ROI check (test batch 2: {len(test_cases)} cases) ---')
    roi_bad = []
    for c in test_cases:
        g = os.path.join(args.roi_dir, c, 'GM_ROI.nii.gz')
        w = os.path.join(args.roi_dir, c, 'WM_ROI.nii.gz')
        if not (os.path.isfile(g) and os.path.isfile(w)):
            roi_bad.append(c)
    print('all test cases have GM+WM ROI' if not roi_bad else f'*** MISSING ROI for cases: {roi_bad} ***')

    if missing:
        print(f'\n*** {len(missing)} volume(s) unresolved: {missing[:6]}{" ..." if len(missing) > 6 else ""}')
        print('*** fix the data before training (run unpack_pcct_hpc.sh); NOT writing the list.')
        sys.exit(1)

    if args.write:
        os.makedirs(os.path.dirname(args.xlsx_out), exist_ok=True)
        out.to_excel(args.xlsx_out, index=False)
        print(f'\nwritten: {args.xlsx_out}')
        print('next: sbatch PCCT_experiments/run_train_imf_pcct.sh')
    else:
        print('\ndry-run only — re-run with --write to save the corrected list.')


if __name__ == '__main__':
    main()
