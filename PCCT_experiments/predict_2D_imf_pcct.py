"""
predict_2D_imf_pcct.py — real-world PCCT inference for the iMF (v-head) model, with or without GAN.

Run twice per NFE -- `pred` generates the K stochastic samples, `avg` writes the K-averages:
    python PCCT_experiments/predict_2D_imf_pcct.py --trial_name imf_v2_unsupervised_PCCT --epoch 200 --mode pred --num_steps 5
    python PCCT_experiments/predict_2D_imf_pcct.py --trial_name imf_v2_unsupervised_PCCT --epoch 200 --mode avg  --num_steps 5 --cleanup
(or use PCCT_experiments/run_pcct_nfe_sweep.sh, which loops both over a list of NFEs)

PCCT-specific behaviour, all forced by the data rather than by preference:
  * NO ground truth. This is real clinical data -- the patient list has no ground_truth_file column
    at all, so nothing is written or compared. Quality is measured by CNR against hand-drawn GM/WM
    ROIs instead (PCCT_experiments/eval_pcct_cnr.py). condition_img.nii.gz IS written, because the
    noisy input is the CNR baseline the denoised result must beat.
  * NO random_num column either -> Build_thinsliceCT returns None for it; we default to 0 rather
    than indexing None (which is what the brain-CT predict would do, and crash).
  * ALL 50 slices are predicted (--slice_range all). The GM and WM ROIs sit on DIFFERENT slices and
    together span the whole volume (e.g. case 29: GM on z=30..44, WM on z=3..10), so restricting to
    a sub-range would silently drop one tissue class and make CNR uncomputable.
  * window [-1000, 1000] HU, no histogram equalisation -- matching how the model was trained.

Output: <study>/<trial>/pred_images_nfe{N}/<case>/epoch{E}avg/pred_img_scans{10,20}.nii.gz
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_PARENT = os.path.dirname(os.path.dirname(_HERE))
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
import IMF_denoising.Generator_thinslice as Generator
from IMF_denoising.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion import Unet


def _detect_base():
    for b in ('/gpfs/work/aac/xingyiyao23', '/host/d/research', '/host/d'):
        if os.path.isdir(os.path.join(b, 'Data')):
            return b
    return '/gpfs/work/aac/xingyiyao23'


_BASE = _detect_base()
_PCCT = os.path.join(_BASE, 'Data', 'PCCT')
_SLICED = 'soft_thins_0_noblank_sliced.nii.gz'


def _remap(p, case=None):
    """Resolve a patient-list path onto the uploaded PCCT data (directory AND filename may be stale)."""
    if p is None:
        return p
    p = str(p)
    if os.path.exists(p):
        return p
    if case is not None:
        for name in (_SLICED, 'soft_thins_0_noblank.nii.gz'):
            cand = os.path.join(_PCCT, 'soft_thins_xy', str(case), name)
            if os.path.isfile(cand):
                return cand
    for marker in ('/soft_thins/', '/soft_thins_xy/'):
        if marker in p:
            cand = os.path.join(_PCCT, 'soft_thins_xy', p.split(marker, 1)[1])
            if os.path.isfile(cand):
                return cand
            cand = os.path.join(os.path.dirname(cand), _SLICED)
            if os.path.isfile(cand):
                return cand
    return p


def _save_nifti(arr, affine, path):
    """Atomic NIfTI save: write a temp file then rename, so an interrupted run can never leave a
    half-written .nii.gz that the resume check would treat as 'done'."""
    tmp = path + '.tmp.nii.gz'
    nb.save(nb.Nifti1Image(arr, affine), tmp)
    os.replace(tmp, path)


def _edge_padded_copy(condition_file, out_path):
    """Write an edge-replicated copy of a volume: [s0, s0, s1, ... s_{n-1}, s_{n-1}], n -> n+2.

    Adjacent-slice conditioning reads s-1 and s+1 with NO bounds handling (Generator_thinslice), so
    on the raw volume the first slice silently wraps to the last (numpy index -1) and the last slice
    raises IndexError. Restricting to the interior would avoid that but drop slices 0 and n-1 from
    the output -- and 5 of the 8 PCCT test cases carry ROI voxels on exactly those slices (WM on
    z=0 for cases 30/32/33, ROI on z=49 for 31/34), so their CNR would be computed over a different
    voxel population than Chen's.

    Padding instead gives every real slice a genuine neighbour and keeps the output at the full n
    slices, aligned 1:1 with the ROI masks. The two duplicated end slices are conditioning input
    only; they are never themselves predicted. Edge replication is the usual convention for
    neighbour operations at a boundary.
    """
    img = nb.load(condition_file)
    vol = img.get_fdata()
    padded = np.concatenate([vol[:, :, :1], vol, vol[:, :, -1:]], axis=2)
    nb.save(nb.Nifti1Image(padded, img.affine), out_path)
    return out_path, vol.shape[2]


def get_args_parser():
    p = argparse.ArgumentParser('PCCT iMF inference')
    p.add_argument('--trial_name', type=str, default='imf_v2_unsupervised_PCCT',
                   help='imf_v2_unsupervised_PCCT (no GAN) or imf_gan_unsupervised_PCCT')
    p.add_argument('--epoch', type=int, required=True)
    p.add_argument('--mode', type=str, required=True, choices=['pred', 'avg'])
    p.add_argument('--slice_range', type=str, default='all',
                   help="'all' (default, = the full 50 slices). The ROIs span the whole volume, so "
                        "do NOT restrict this if you intend to compute CNR.")
    p.add_argument('--iteration_num', type=int, default=20, help='K: number of stochastic samples')
    p.add_argument('--num_steps', type=int, default=5, help='NFE per sample')
    p.add_argument('--solver', type=str, default='euler', choices=['euler', 'midpoint', 'heun'])
    p.add_argument('--schedule', type=str, default='uniform', choices=['uniform', 'optimal'])
    p.add_argument('--slice_batch', type=int, default=8, help='initial slices per GPU forward (auto-tuned)')
    p.add_argument('--auto_batch', dest='auto_batch', action='store_true', default=True,
                   help='adaptively grow slice_batch (x2) until CUDA OOM / --max_slice_batch (default ON)')
    p.add_argument('--no_auto_batch', dest='auto_batch', action='store_false')
    p.add_argument('--max_slice_batch', type=int, default=64,
                   help='cap for the auto-grown batch; the 50-slice volume is the real ceiling')
    p.add_argument('--weights', type=str, default='ema', choices=['ema', 'raw'],
                   help="Which copy of the trained model to sample from. 'ema' (default) reproduces "
                        "every number reported so far. 'raw' uses the online weights -- needed here "
                        "because the FLOW trainer calls ema.update() once per EPOCH rather than per "
                        "optimizer step (improved_mean_flow.py), so with update_every=10 and "
                        "update_after_step=100 its 'EMA' is ~96%% the epoch-100 weights. The GAN "
                        "trainer updates per step, so 'ema' there is a genuine average. Comparing a "
                        "no-GAN EMA against a GAN EMA therefore compares two very different points "
                        "in training; --weights raw removes that confound.")
    p.add_argument('--k_save', type=int, nargs='+', default=[10, 20], help='which avg-of-K volumes to write')
    p.add_argument('--cleanup', action='store_true',
                   help='avg mode: after averaging, delete the per-sample volumes to free disk')
    p.add_argument('--study_folder', type=str, default=os.path.join(_BASE, 'projects/denoising/models'))
    p.add_argument('--patient_list_file', type=str,
                   default=os.path.join(_PCCT, 'Patient_lists', 'PCCT_split_hpc.xlsx'))
    p.add_argument('--batch', type=int, nargs='+', default=[2], help="xlsx 'batch' to run (2 = test, the ROI cases)")
    return p


def run(args):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    supervision = 'unsupervised'
    condition_channel = 2
    epoch = args.epoch
    image_size = [512, 512]
    histogram_equalization = False
    background_cutoff, maximum_cutoff = -1000, 1000
    normalize_factor = 'equation'

    trained_model_filename = os.path.join(args.study_folder, args.trial_name, 'models', f'model-{epoch}.pt')
    cfg = f'nfe{args.num_steps}'
    if args.solver != 'euler':     cfg += '_' + args.solver
    if args.schedule != 'uniform': cfg += '_' + args.schedule
    if args.slice_range != 'all':  cfg += '_slice' + args.slice_range
    if args.weights != 'ema':      cfg += '_' + args.weights   # keep raw-weight runs in their own folder
    save_folder = os.path.join(args.study_folder, args.trial_name, f'pred_images_{cfg}')
    os.makedirs(save_folder, exist_ok=True)

    print('data base  :', _BASE)
    print('checkpoint :', trained_model_filename)
    print('save_folder:', save_folder, '| mode:', args.mode, '| NFE:', args.num_steps)
    if not os.path.isfile(trained_model_filename):
        raise FileNotFoundError(trained_model_filename)
    if not os.path.isfile(args.patient_list_file):
        raise FileNotFoundError(args.patient_list_file)

    bs = Build_list.Build_thinsliceCT(args.patient_list_file)
    _, pid_list, _, random_num_list, cond_list, _ = bs.__build__(batch_list=args.batch)
    n = np.arange(pid_list.shape[0])
    print(f'batch {args.batch}: {n.shape[0]} cases')

    base_model = Unet(
        problem_dimension='2D', init_dim=64, out_dim=1, channels=1,
        conditional_diffusion=True, condition_channels=condition_channel,
        downsample_list=(True, True, True, False), upsample_list=(True, True, True, False),
        full_attn=(None, None, False, True), auxiliary_v_head=True,
    )
    diffusion_model = imf.ImprovedMeanFlow(
        base_model, image_size=image_size, ratio_r_neq_t=0.5, clip_or_not=False, auto_normalize=False)

    sampler = imf.Sampler(diffusion_model, generator=None, batch_size=1)
    sampler.background_cutoff = background_cutoff
    sampler.maximum_cutoff = maximum_cutoff
    sampler.normalize_factor = normalize_factor
    sampler.histogram_equalization = histogram_equalization
    sampler.bins = None
    sampler.bins_mapped = None
    sampler.load_model(trained_model_filename)
    print('model + EMA loaded.')

    # Adaptive slice-batching -- set ONCE so the tuned batch and the discovered OOM ceiling persist
    # across every case and sample (the warm-up is paid once, not per case).
    sampler.slice_batch = args.slice_batch
    sampler.auto_batch = args.auto_batch
    sampler.max_slice_batch = args.max_slice_batch
    print(f'slice-batching: start={args.slice_batch} auto={args.auto_batch} max={args.max_slice_batch}')

    G = Generator.Dataset_2D
    for i in range(n.shape[0]):
        case = str(pid_list[n[i]])
        # PCCT_split has no random_num column -> Build_thinsliceCT returns None; default to 0.
        random_num = 0 if random_num_list is None else random_num_list[n[i]]
        condition_file = _remap(cond_list[n[i]], case)
        print(i, 'case', case)
        if not os.path.isfile(condition_file):
            print('  [skip] missing volume:', condition_file); continue

        vol_slices = nb.load(condition_file).shape[2]
        if args.slice_range != 'all':
            slice_start, slice_end = (int(v) for v in args.slice_range.split('-'))
        else:
            slice_start, slice_end = 0, vol_slices
        slice_num = slice_end - slice_start

        condition_img = nb.load(condition_file).get_fdata()[:, :, slice_start:slice_end]
        affine = nb.load(condition_file).affine
        case_root = os.path.join(save_folder, case, f'random_{random_num}')

        if args.mode == 'pred':
            # Feed the generator an EDGE-PADDED copy so all `slice_num` real slices keep a valid
            # neighbour pair; slices 1..slice_num of the padded volume are exactly 0..slice_num-1 of
            # the original, so the output stays aligned with the ROI masks. See _edge_padded_copy.
            pad_path = os.path.join(case_root, f'.cond_padded_{case}.nii.gz')
            ff.make_folder([os.path.join(save_folder, case), case_root])
            _edge_padded_copy(condition_file, pad_path)

            # Build the generator ONCE per case: the condition volume is identical across the K
            # samples (diversity comes from the sampler's init noise, not from the data).
            generator = G(
                supervision=supervision,
                img_list=np.array([pad_path]),               # N2N: same volume is input and target
                condition_list=np.array([pad_path]),
                image_size=image_size,
                num_slices_per_image=slice_num,
                random_pick_slice=False,
                slice_range=[1, 1 + slice_num],              # the real slices inside the padded volume
                histogram_equalization=histogram_equalization, bins=None, bins_mapped=None,
                background_cutoff=background_cutoff, maximum_cutoff=maximum_cutoff,
                normalize_factor=normalize_factor, shuffle=False, augment=False,
            )
            sampler.generator = generator
            if args.weights == 'ema':
                sampler.model = sampler.ema.ema_model        # the averaged copy (see --weights)
            # else: keep sampler.model as loaded = the online weights

            for iteration in range(1, args.iteration_num + 1):
                save_folder_case = os.path.join(case_root, f'epoch{epoch}_{iteration}')
                ff.make_folder([os.path.join(save_folder, case), case_root, save_folder_case])
                if os.path.isfile(os.path.join(save_folder_case, 'pred_img.nii.gz')):
                    print('  iter', iteration, 'already done'); continue

                pred_img = sampler.sample_2D(
                    trained_model_filename, condition_img, direct_use_of_model=True,
                    num_steps=args.num_steps, solver=args.solver, schedule=args.schedule)
                _save_nifti(pred_img, affine, os.path.join(save_folder_case, 'pred_img.nii.gz'))
                if iteration == 1:
                    # the noisy input = the CNR baseline. No gt_img: real PCCT has no ground truth.
                    _save_nifti(condition_img, affine, os.path.join(save_folder_case, 'condition_img.nii.gz'))

            if os.path.isfile(pad_path):        # scratch conditioning copy, not a result
                os.remove(pad_path)

        else:  # mode == 'avg'
            save_folder_avg = os.path.join(case_root, f'epoch{epoch}avg')
            ff.make_folder([os.path.join(save_folder, case), case_root, save_folder_avg])
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
                    print(f'  [corrupt] deleting {fp} ({str(e)[:50]}) — re-run pred', flush=True)
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
            # keep the noisy input alongside the averages so the CNR baseline is self-contained
            _save_nifti(condition_img, affine, os.path.join(save_folder_avg, 'condition_img.nii.gz'))

            if args.cleanup:
                # Drop the raw samples ONLY once every requested K was written -- they are the sole
                # source the averages could ever be rebuilt from.
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
                        fp = os.path.join(p, 'pred_img.nii.gz')
                        if os.path.isfile(fp):
                            os.remove(fp); removed += 1
                    print(f'  [cleanup] removed {removed} files (per-sample volumes + non-kept scans)')

    if args.mode == 'pred':
        print(f'[auto_batch] final tuned slice_batch = {getattr(sampler, "slice_batch", None)} '
              f'(ceiling={getattr(sampler, "_sb_ceiling", None)})')


if __name__ == '__main__':
    run(get_args_parser().parse_args())
