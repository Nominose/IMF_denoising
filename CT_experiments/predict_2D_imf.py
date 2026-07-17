"""
predict_2D_imf.py — Mayo Low-Dose CT inference with improved MeanFlow (iMF). HPC/gpfs build.

Self-supervised N2N model: the condition is a half-projection recon (odd or even). With
--input both (default) each of the K samples is generated twice (once conditioned on odd, once on
even) and the two are averaged into that sample's pred_img.nii.gz.

Run twice per NFE — `pred` generates the K stochastic samples, `avg` writes the K-averages:
    python CT_experiments/predict_2D_imf.py --epoch 200 --mode pred --num_steps 5 --iteration_num 20
    python CT_experiments/predict_2D_imf.py --epoch 200 --mode avg  --num_steps 5 --cleanup
(or just use CT_experiments/run_mayo_nfe_sweep.sh, which loops both over a list of NFEs)

Notes on this build (vs the plain main-branch script):
  * v-head       : the U-Net is built with auxiliary_v_head=True to match how the Mayo v2 model was
                   trained (train_2D_imf_mayo.py). Without it, model-200.pt strict-load-FAILS with
                   `Unexpected key(s): ...final_conv_v...`.
  * slice batch  : adaptive (see --auto_batch). The old fixed `slice_batch = 2` was tuned for a 12GB
                   card and leaves an 80GB A800 ~95% idle.
  * ground truth : OPTIONAL. The xlsx points gt at .../low_dose_CT/nii_imgs/<PID>/img.nii.gz; if that
                   is not uploaded, prediction still runs and simply skips writing gt_img.nii.gz
                   (download the preds and score them against a local gt). Pass --require_gt to make
                   a missing gt a hard error instead.
  * disk         : `avg` writes ONLY the --k_save averages (default K=10,20) instead of all K=1..20,
                   and --cleanup then drops the per-sample volumes.
  * output dir   : the run config is encoded in the folder name, so different NFE / solver /
                   schedule / slice_range never share a folder (the per-iteration 'already done'
                   resume check would otherwise silently reuse a wrong-config result).
"""
import os
import sys

# --- make `import IMF_denoising...` work regardless of where the repo is mounted ---
_HERE = os.path.dirname(os.path.abspath(__file__))            # .../IMF_denoising/CT_experiments
_REPO_PARENT = os.path.dirname(os.path.dirname(_HERE))        # dir that CONTAINS the IMF_denoising folder
for _p in (_REPO_PARENT, '/gpfs/work/aac/xingyiyao23/Code', '/host/c/Users/ROG/Documents/GitHub'):
    if _p not in sys.path:
        sys.path.append(_p)

import argparse
import glob
import numpy as np
import nibabel as nb
import torch

import IMF_denoising.improved_mean_flow as imf
import IMF_denoising.functions_collection as ff
import IMF_denoising.Build_lists.Build_list as Build_list
import IMF_denoising.Generator as Generator
from IMF_denoising.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion import Unet


def _detect_base():
    """Where the Mayo xlsx + recon data + model outputs live (HPC cluster root first, then the
    docker D: mounts). Mirrors train_2D_imf_mayo.py so both resolve to the same tree."""
    for b in ('/gpfs/work/aac/xingyiyao23', '/host/d/research', '/host/d', 'D:/research', '/d/research'):
        if os.path.isdir(os.path.join(b, '新mayo_data')) or os.path.isdir(os.path.join(b, 'projects/denoising')):
            return b
    return '/gpfs/work/aac/xingyiyao23'


_BASE = _detect_base()
_MAYO_DATA = os.path.join(_BASE, '新mayo_data')


def _remap(p):
    """Resolve a stored recon/gt path onto the current base. The xlsx stores E:-drive docker paths
    like /host/e/D/Data/low_dose_CT/...; on the cluster that whole tree lives under <base>/新mayo_data/
    (so .../low_dose_CT/X -> <base>/新mayo_data/X). Same logic as train_2D_imf_mayo.py."""
    if p is None:
        return p
    p = str(p)
    if os.path.exists(p):
        return p
    if '/low_dose_CT/' in p:
        cand = os.path.join(_MAYO_DATA, p.split('/low_dose_CT/', 1)[1])
        if os.path.exists(cand):
            return cand
    for src, dsts in (('/host/e', ('/host/e', 'E:', '/e')),
                      ('/host/d', ('/host/d/research', '/host/d', 'D:', '/d'))):
        if p.startswith(src + '/'):
            tail = p[len(src):]
            for d in dsts:
                cand = d + tail
                if os.path.exists(cand):
                    return cand
    return p


def _save_nifti(arr, affine, path):
    """Atomic NIfTI save: write a temp file then rename, so an interrupted run can never leave a
    half-written .nii.gz that the resume check would treat as 'done'."""
    tmp = path + '.tmp.nii.gz'
    nb.save(nb.Nifti1Image(arr, affine), tmp)
    os.replace(tmp, path)


def get_args_parser():
    p = argparse.ArgumentParser('Mayo iMF inference (HPC)')
    p.add_argument('--trial_name', type=str, default='imf_v2_unsupervised_gaussian_mayo',
                   help='must match train_2D_imf_mayo.py (the old non-v-head default strict-load-fails)')
    p.add_argument('--epoch', type=int, required=True)
    p.add_argument('--mode', type=str, required=True, choices=['pred', 'avg'])
    p.add_argument('--input', type=str, default='both', choices=['both', 'odd', 'even'])
    p.add_argument('--slice_range', type=str, default='150-200', help='Mayo: 150-200 to match the prior paper')
    p.add_argument('--iteration_num', type=int, default=20, help='K: number of stochastic samples')
    p.add_argument('--num_steps', type=int, default=1, help='NFE per sample: 1 for one-step, 2+ for multistep')
    p.add_argument('--solver', type=str, default='euler', choices=['euler', 'midpoint', 'heun'])
    p.add_argument('--schedule', type=str, default='uniform', choices=['uniform', 'optimal'])
    p.add_argument('--slice_batch', type=int, default=8, help='initial slices per GPU forward (auto-tuned when --auto_batch)')
    p.add_argument('--auto_batch', dest='auto_batch', action='store_true', default=True,
                   help='adaptively grow slice_batch (x2) until CUDA OOM / --max_slice_batch, halving on OOM (default ON)')
    p.add_argument('--no_auto_batch', dest='auto_batch', action='store_false',
                   help='disable adaptive batching: keep a fixed --slice_batch (still shrinks on OOM)')
    p.add_argument('--max_slice_batch', type=int, default=64,
                   help='hard cap for auto-grown slice_batch (0 = unlimited; the slice count itself is the real ceiling)')
    p.add_argument('--k_save', type=int, nargs='+', default=[10, 20], help='which avg-of-K volumes to write in avg mode')
    p.add_argument('--cleanup', action='store_true',
                   help='avg mode: after averaging, delete the per-sample volumes + non-kept scans to free disk')
    p.add_argument('--require_gt', action='store_true',
                   help='fail if a ground-truth volume is missing (default: skip gt, still predict)')
    p.add_argument('--study_folder', type=str, default=os.path.join(_BASE, 'projects/denoising/models'))
    p.add_argument('--patient_list_file', type=str,
                   default=os.path.join(_MAYO_DATA, 'mayo_low_dose_CT_gaussian_simulation_highnoise_v2.xlsx'))
    p.add_argument('--batch', type=str, nargs='+', default=['test'], help="xlsx 'batch' value(s) to run")
    return p


def run(args):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    supervision = 'unsupervised'
    epoch = args.epoch
    input_condition = args.input

    trained_model_filename = os.path.join(args.study_folder, args.trial_name, 'models', f'model-{epoch}.pt')
    # Encode the run config in the output path so different NFE / solver / schedule / slice_range never
    # share a folder -- otherwise the per-iteration "already done" resume check would silently reuse a
    # wrong-config result (e.g. an nfe1 run reused for an nfe5 request).
    cfg = f'nfe{args.num_steps}'
    if args.solver != 'euler':        cfg += '_' + args.solver
    if args.schedule != 'uniform':    cfg += '_' + args.schedule
    if args.slice_range != '150-200': cfg += '_slice' + args.slice_range
    save_folder = os.path.join(args.study_folder, args.trial_name, f'pred_images_input_{input_condition}_{cfg}')
    os.makedirs(save_folder, exist_ok=True)

    print('data base  :', _BASE)
    print('checkpoint :', trained_model_filename)
    print('save_folder:', save_folder, '| mode:', args.mode, '| NFE:', args.num_steps)
    if not os.path.isfile(trained_model_filename):
        raise FileNotFoundError(trained_model_filename)
    if not os.path.isfile(args.patient_list_file):
        raise FileNotFoundError(args.patient_list_file)

    image_size = [512, 512]
    histogram_equalization = False        # abdomen: plain HU window, no hist-eq
    background_cutoff = -200
    maximum_cutoff = 250
    normalize_factor = 'equation'

    # ========== Patient list ==========
    build_sheet = Build_list.Build(args.patient_list_file)
    _, patient_id_list, random_num_list, _, noise_file_odd_list, noise_file_even_list, ground_truth_file_list, _ = \
        build_sheet.__build__(batch_list=args.batch)
    n = np.arange(patient_id_list.shape[0])
    print(f'batch {args.batch}: {n.shape[0]} cases')

    # ========== Model (built once) ==========
    base_model = Unet(
        problem_dimension='2D',
        init_dim=64,
        out_dim=1,
        channels=1,
        conditional_diffusion=True,
        condition_channels=1,          # Mayo: a single recon as condition
        downsample_list=(True, True, True, False),
        upsample_list=(True, True, True, False),
        full_attn=(None, None, False, True),
        auxiliary_v_head=True,         # matches train_2D_imf_mayo.py; without it model-200.pt won't load
    )
    diffusion_model = imf.ImprovedMeanFlow(
        base_model,
        image_size=image_size,
        ratio_r_neq_t=0.5,
        clip_or_not=False,
        auto_normalize=False,
    )

    sampler = imf.Sampler(diffusion_model, generator=None, batch_size=1)
    sampler.background_cutoff = background_cutoff
    sampler.maximum_cutoff = maximum_cutoff
    sampler.normalize_factor = normalize_factor
    sampler.histogram_equalization = histogram_equalization
    sampler.load_model(trained_model_filename)
    print('model + EMA loaded.')

    # Adaptive slice-batching -- set ONCE here (not per case) so the auto-tuned batch and the
    # discovered OOM ceiling persist across every case + sample (the warm-up is paid once).
    # NOTE: the real ceiling is the slice count itself (150-200 -> 50), so the batch settles at one
    # forward per volume; a slice_batch beyond that simply never fills.
    sampler.slice_batch = args.slice_batch
    sampler.auto_batch = args.auto_batch
    sampler.max_slice_batch = args.max_slice_batch
    print(f'slice-batching: start={args.slice_batch} auto={args.auto_batch} max={args.max_slice_batch}')

    G = Generator.Dataset_2D
    missing_gt = []

    for i in range(n.shape[0]):
        patient_id = str(patient_id_list[n[i]])
        random_num = random_num_list[n[i]]
        noise_file_odd = _remap(noise_file_odd_list[n[i]])
        noise_file_even = _remap(noise_file_even_list[n[i]])
        gt_file = _remap(ground_truth_file_list[n[i]])

        if input_condition == 'both':
            condition_files, condition_names = [noise_file_odd, noise_file_even], ['odd', 'even']
        elif input_condition == 'odd':
            condition_files, condition_names = [noise_file_odd], ['odd']
        else:
            condition_files, condition_names = [noise_file_even], ['even']

        print(i, patient_id, random_num)
        if not all(os.path.isfile(f) for f in condition_files):
            print('  [skip] missing condition volume(s):', condition_files); continue

        if args.slice_range != 'all':
            slice_start, slice_end = (int(v) for v in args.slice_range.split('-'))
        else:
            slice_start, slice_end = 0, nb.load(condition_files[0]).shape[2]

        affine = nb.load(condition_files[0]).affine

        # Ground truth is OPTIONAL: it is only needed to score the preds, never to make them. If the
        # nii_imgs/ tree was not uploaded, keep predicting and just don't write gt_img.nii.gz.
        gt_img = None
        if os.path.isfile(gt_file):
            gt_img = nb.load(gt_file).get_fdata()[:, :, slice_start:slice_end]
        else:
            missing_gt.append(patient_id)
            msg = f'  ground truth not found: {gt_file}'
            if args.require_gt:
                raise FileNotFoundError(msg.strip())
            print(msg + '  -> predicting anyway, no gt_img.nii.gz will be written', flush=True)

        case_root = os.path.join(save_folder, patient_id, f'random_{random_num}')

        if args.mode == 'pred':
            # The condition volume + its Generator are identical across all K samples (the samples
            # differ only by the sampler's fresh init noise), so build each once and reuse. Lazy, so
            # a fully 'already done' case never touches the ~90MB volume.
            cond_imgs = [None] * len(condition_files)
            generators = [None] * len(condition_files)

            for iteration in range(1, args.iteration_num + 1):
                save_folder_case = os.path.join(case_root, f'epoch{epoch}_{iteration}')
                ff.make_folder([os.path.join(save_folder, patient_id), case_root, save_folder_case])
                if os.path.isfile(os.path.join(save_folder_case, 'pred_img.nii.gz')):
                    print('  iter', iteration, 'already done'); continue

                pred_per_condition = []
                for ci_i, condition_file in enumerate(condition_files):
                    if generators[ci_i] is None:
                        ci = nb.load(condition_file).get_fdata()[:, :, slice_start:slice_end]
                        cond_imgs[ci_i] = ci
                        generators[ci_i] = G(
                            supervision=supervision,
                            img_list=np.array([condition_file]),
                            condition_list=np.array([condition_file]),
                            image_size=image_size,
                            num_slices_per_image=ci.shape[2],
                            random_pick_slice=False,
                            slice_range=None if args.slice_range == 'all' else [slice_start, slice_end],
                            histogram_equalization=histogram_equalization,
                            bins=None, bins_mapped=None,
                            background_cutoff=background_cutoff, maximum_cutoff=maximum_cutoff,
                            normalize_factor=normalize_factor,
                            shuffle=False, augment=False,
                        )
                    sampler.generator = generators[ci_i]
                    original_model_ref = sampler.model
                    try:
                        sampler.model = sampler.ema.ema_model     # sample from EMA weights
                        pred_img = sampler.sample_2D(
                            trained_model_filename, cond_imgs[ci_i], direct_use_of_model=True,
                            num_steps=args.num_steps, solver=args.solver, schedule=args.schedule)
                    finally:
                        sampler.model = original_model_ref
                    pred_per_condition.append(pred_img)

                    if len(condition_files) > 1:
                        _save_nifti(pred_img, affine,
                                    os.path.join(save_folder_case, f'pred_img_{condition_names[ci_i]}.nii.gz'))

                # pred_img.nii.gz is what `avg` reads: the odd/even mean for --input both, else the
                # single-condition prediction. Averaged in-memory (no reloading what we just wrote).
                final = (np.mean(np.stack(pred_per_condition, axis=0), axis=0)
                         if len(condition_files) > 1 else pred_per_condition[0])
                _save_nifti(final, affine, os.path.join(save_folder_case, 'pred_img.nii.gz'))

                if iteration == 1:
                    if gt_img is not None:
                        _save_nifti(gt_img, affine, os.path.join(save_folder_case, 'gt_img.nii.gz'))
                    _save_nifti(cond_imgs[0], affine, os.path.join(save_folder_case, 'condition_img.nii.gz'))

        else:  # mode == 'avg' -> average the K generated samples
            save_folder_avg = os.path.join(case_root, f'epoch{epoch}avg')
            ff.make_folder([os.path.join(save_folder, patient_id), case_root, save_folder_avg])
            made = ff.sort_timeframe(ff.find_all_target_files([f'epoch{epoch}_*'], case_root), 0, '_', '/')
            completed = [p for p in made if os.path.isfile(os.path.join(p, 'pred_img.nii.gz'))]
            if not completed:
                print('  no completed predicts, skip'); continue

            arrays = []
            for p in completed:
                fp = os.path.join(p, 'pred_img.nii.gz')
                try:
                    arrays.append(np.asarray(nb.load(fp).get_fdata(), dtype=np.float32))
                except Exception as e:
                    print(f'  [corrupt] deleting {fp} ({str(e)[:50]}) — re-run pred to regenerate', flush=True)
                    try:
                        os.remove(fp)
                    except OSError:
                        pass
            if not arrays:
                print('  no valid predicts, skip'); continue
            n_ok = len(arrays)
            print(f'  averaging from {n_ok} valid samples')

            stack = np.stack(arrays, axis=-1)
            written = []
            for k in sorted(set(args.k_save)):
                if k > n_ok:
                    print(f'  [warn] requested K={k} but only {n_ok} valid samples; skipping')
                    continue
                _save_nifti(stack[..., :k].mean(axis=-1), affine,
                            os.path.join(save_folder_avg, f'pred_img_scans{k}.nii.gz'))
                written.append(k)
            np.save(os.path.join(save_folder_avg, 'sample_std.npy'), stack.std(axis=-1).astype(np.float32))
            if gt_img is not None:
                _save_nifti(gt_img, affine, os.path.join(save_folder_avg, 'gt_img.nii.gz'))

            if args.cleanup:
                # Only drop the raw samples once EVERY requested K was actually written -- they are
                # the sole source the averages can ever be rebuilt from.
                if set(written) != set(args.k_save):
                    print(f'  [cleanup] SKIPPED: wrote K={written}, wanted {args.k_save} — keeping raw samples')
                else:
                    removed = 0
                    for f in glob.glob(os.path.join(save_folder_avg, 'pred_img_scans*.nii.gz')):
                        try:
                            kk = int(os.path.basename(f).split('scans')[1].split('.nii')[0])
                        except Exception:
                            continue
                        if kk not in set(args.k_save):
                            os.remove(f); removed += 1
                    for p in completed:
                        for name in ('pred_img.nii.gz', 'pred_img_odd.nii.gz', 'pred_img_even.nii.gz'):
                            fp = os.path.join(p, name)
                            if os.path.isfile(fp):
                                os.remove(fp); removed += 1
                    print(f'  [cleanup] removed {removed} files (per-sample volumes + non-kept scans)')

    if args.mode == 'pred':
        print(f'[auto_batch] final tuned slice_batch = {getattr(sampler, "slice_batch", None)} '
              f'(largest per-forward batch that fit; ceiling={getattr(sampler, "_sb_ceiling", None)})')
    if missing_gt:
        print(f'[gt] NO ground truth for {len(missing_gt)} case(s): {missing_gt} — '
              f'preds written without gt_img.nii.gz; score them against a local gt.')


if __name__ == '__main__':
    run(get_args_parser().parse_args())
