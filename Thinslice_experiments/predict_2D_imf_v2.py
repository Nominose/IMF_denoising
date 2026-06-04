"""
predict_2D_imf_v2.py — Brain CT thin-slice inference for the iMF **v2** model (with aux v-head).

Why a separate script from predict_2D_imf.py:
  * model-200 (trial `imf_v2_unsupervised_gaussian_brainCT`) was trained WITH the auxiliary
    v-head (`auxiliary_v_head=True`). The old predict_2D_imf.py builds the U-Net WITHOUT it,
    so loading this checkpoint there fails with `Unexpected key(s): ...final_conv_v...`.
  * This script builds the U-Net WITH the v-head (verified to strict-load model-200), and
    writes predictions to a per-NFE folder so NFE=3 and NFE=5 runs don't collide.

Paths assume the docker mount  D:\research -> /host/d  and code under /host/.../IMF_denoising.

Usage (run twice per NFE: first `pred`, then `avg`):
    python Thinslice_experiments/predict_2D_imf_v2.py --epoch 200 --mode pred --iteration_num 20 --num_steps 3
    python Thinslice_experiments/predict_2D_imf_v2.py --epoch 200 --mode avg                      --num_steps 3
    python Thinslice_experiments/predict_2D_imf_v2.py --epoch 200 --mode pred --iteration_num 20 --num_steps 5
    python Thinslice_experiments/predict_2D_imf_v2.py --epoch 200 --mode avg                      --num_steps 5
"""
import os
import sys

# --- make `import IMF_denoising...` work regardless of where the repo is mounted ---
_HERE = os.path.dirname(os.path.abspath(__file__))            # .../IMF_denoising/Thinslice_experiments
_REPO_PARENT = os.path.dirname(os.path.dirname(_HERE))        # dir that CONTAINS the IMF_denoising folder
for _p in (_REPO_PARENT, '/host/c/Users/ROG/Documents/GitHub'):
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


# --- auto-detect the real data root. Docker mounts the whole D: drive at /host/d, and the actual
#     data lives under D:\research, i.e. /host/d/research. Older scripts/xlsx assume /host/d. ---
def _detect_base():
    for b in ('/host/d/research', '/host/d'):
        if os.path.isdir(os.path.join(b, 'Data')):
            return b
    return '/host/d/research'


_BASE = _detect_base()


def _remap(p):
    """Remap a stored '/host/d/...' path onto the detected data root if needed."""
    if p is None:
        return p
    p = str(p)
    if os.path.exists(p):
        return p
    if p.startswith('/host/d/') and not p.startswith(_BASE + '/'):
        cand = _BASE + p[len('/host/d'):]
        if os.path.exists(cand):
            return cand
    return p


def _save_nifti(arr, affine, path):
    """Atomic NIfTI save: write a temp file then rename. An interrupted run can then never
    leave a half-written (corrupt) .nii.gz that the resume check would treat as 'done'."""
    tmp = path + '.tmp.nii.gz'
    nb.save(nb.Nifti1Image(arr, affine), tmp)
    os.replace(tmp, path)


def get_args_parser():
    p = argparse.ArgumentParser('Brain CT iMF v2 (v-head) inference')
    p.add_argument('--trial_name', type=str, default='imf_v2_unsupervised_gaussian_brainCT')
    p.add_argument('--epoch', type=int, default=200)
    p.add_argument('--mode', type=str, required=True, choices=['pred', 'avg'])
    p.add_argument('--slice_range', type=str, default='30-80')
    p.add_argument('--iteration_num', type=int, default=20, help='K: number of stochastic samples')
    p.add_argument('--num_steps', type=int, default=3, help='NFE per sample (euler). Use 3 and 5.')
    p.add_argument('--solver', type=str, default='euler', choices=['euler', 'midpoint', 'heun'])
    p.add_argument('--schedule', type=str, default='uniform', choices=['uniform', 'optimal'])
    p.add_argument('--slice_batch', type=int, default=8, help='slices per GPU forward (auto-halves on CUDA OOM)')
    p.add_argument('--amp', action='store_true', help='fp16 autocast during sampling (faster on RTX, ~same result)')
    p.add_argument('--k_save', type=int, nargs='+', default=[10, 20], help='which avg-of-K volumes to write in avg mode')
    p.add_argument('--cleanup', action='store_true', help='avg mode: after averaging, delete per-sample volumes + non-kept scans to free disk')
    p.add_argument('--batch', type=int, nargs='+', default=[5], help='patient-list batch(es) to load (test=5; training=0-4)')
    p.add_argument('--patient', type=str, default=None, help='only process patient IDs containing this string, e.g. 214841')
    # data / model roots — defaults derived from the auto-detected data base (e.g. /host/d/research)
    p.add_argument('--study_folder', type=str, default=os.path.join(_BASE, 'projects/denoising/models'))
    p.add_argument('--patient_list_file', type=str,
                   default=os.path.join(_BASE, 'Data/brain_CT/Patient_lists/fixedCT_static_simulation_train_test_gaussian_xjtlu.xlsx'))
    p.add_argument('--bins', type=str, default=os.path.join(_BASE, 'file/histogram_equalization/bins.npy'))
    p.add_argument('--bins_mapped', type=str, default=os.path.join(_BASE, 'file/histogram_equalization/bins_mapped.npy'))
    return p


def build_model(condition_channel):
    """U-Net config that EXACTLY matches the imf_v2 checkpoint (strict-load verified)."""
    base_model = Unet(
        problem_dimension='2D',
        init_dim=64,
        out_dim=1,
        channels=1,
        conditional_diffusion=True,
        condition_channels=condition_channel,
        downsample_list=(True, True, True, False),
        upsample_list=(True, True, True, False),
        full_attn=(None, None, False, True),
        auxiliary_v_head=True,          # <-- the key difference vs the old predict script
    )
    diffusion_model = imf.ImprovedMeanFlow(
        base_model,
        image_size=[512, 512],
        ratio_r_neq_t=0.5,
        clip_or_not=False,
        auto_normalize=False,
    )
    return diffusion_model


def run(args):
    # GPU throughput knobs (safe; help conv/matmul on RTX/Ada)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    supervision = 'unsupervised'
    condition_channel = 2
    epoch = args.epoch

    trained_model_filename = os.path.join(args.study_folder, args.trial_name, 'models', f'model-{epoch}.pt')
    # per-NFE output folder so NFE=3 and NFE=5 never overwrite each other
    save_folder = os.path.join(args.study_folder, args.trial_name, f'pred_images_nfe{args.num_steps}')
    os.makedirs(save_folder, exist_ok=True)
    print('data base  :', _BASE)
    print('checkpoint :', trained_model_filename)
    print('save_folder:', save_folder, '| mode:', args.mode, '| NFE:', args.num_steps)
    if not os.path.isfile(trained_model_filename):
        raise FileNotFoundError(trained_model_filename)

    image_size = [512, 512]
    histogram_equalization = True
    background_cutoff = -1000
    maximum_cutoff = 2000
    normalize_factor = 'equation'

    bins = np.load(args.bins)
    bins_mapped = np.load(args.bins_mapped)

    # load requested batch(es); default test set = batch 5
    build_sheet = Build_list.Build_thinsliceCT(args.patient_list_file)
    _, patient_id_list, patient_subid_list, random_num_list, condition_list, x0_list = \
        build_sheet.__build__(batch_list=args.batch)
    if args.patient is not None:
        matches = [i for i in range(patient_id_list.shape[0]) if args.patient in str(patient_id_list[i])]
        if not matches:
            raise SystemExit(f'patient "{args.patient}" not found in batch(es) {args.batch}')
        n = np.array(matches[:1])  # one case: first matching row (random_num 0)
        print(f'patient filter "{args.patient}" -> {patient_id_list[n[0]]}/{patient_subid_list[n[0]]} random {random_num_list[n[0]]}')
    else:
        n = ff.get_X_numbers_in_interval(total_number=patient_id_list.shape[0], start_number=0, end_number=1, interval=2)
    print('total cases to process:', n.shape[0])

    # build model + sampler once
    diffusion_model = build_model(condition_channel)
    sampler = imf.Sampler(diffusion_model, generator=None, batch_size=1)
    sampler.background_cutoff = background_cutoff
    sampler.maximum_cutoff = maximum_cutoff
    sampler.normalize_factor = normalize_factor
    sampler.histogram_equalization = histogram_equalization
    sampler.bins = bins
    sampler.bins_mapped = bins_mapped
    sampler.load_model(trained_model_filename)   # strict load of model + EMA (verified OK for v2)
    print('model + EMA loaded.')

    G = Generator.Dataset_2D

    for i in range(n.shape[0]):
        patient_id = str(patient_id_list[n[i]])
        patient_subid = str(patient_subid_list[n[i]])
        random_num = random_num_list[n[i]]
        x0_file = _remap(x0_list[n[i]])
        condition_file = _remap(condition_list[n[i]])
        print(i, patient_id, patient_subid, random_num)
        if not (os.path.isfile(x0_file) and os.path.isfile(condition_file)):
            print('  [skip] missing data:', x0_file, '|', condition_file); continue

        if args.slice_range != 'all':
            slice_start, slice_end = (int(v) for v in args.slice_range.split('-'))
        else:
            slice_start, slice_end = 0, nb.load(condition_file).get_fdata().shape[2]
        slice_num = slice_end - slice_start

        gt_img = nb.load(x0_file).get_fdata()[:, :, slice_start:slice_end]
        condition_img = nb.load(condition_file).get_fdata()[:, :, slice_start:slice_end]
        affine = nb.load(condition_file).affine

        # case folders
        case_root = os.path.join(save_folder, patient_id, patient_subid, f'random_{random_num}')

        if args.mode == 'pred':
            # Build the data generator ONCE per case. The condition volume is identical across the K
            # samples (diversity comes from random init noise, not the data), so there is no need to
            # reload + histogram-equalize the ~90MB volume on every sample.
            generator = G(
                supervision=supervision,
                img_list=np.array([x0_file]),
                condition_list=np.array([condition_file]),
                image_size=image_size,
                num_slices_per_image=slice_num,
                random_pick_slice=False,
                slice_range=[slice_start, slice_end],
                histogram_equalization=histogram_equalization,
                bins=bins, bins_mapped=bins_mapped,
                background_cutoff=background_cutoff, maximum_cutoff=maximum_cutoff,
                normalize_factor=normalize_factor,
                shuffle=False, augment=False,
            )
            sampler.generator = generator
            sampler.slice_batch = args.slice_batch
            sampler.model = sampler.ema.ema_model            # sample from EMA weights

            for iteration in range(1, args.iteration_num + 1):
                save_folder_case = os.path.join(case_root, f'epoch{epoch}_{iteration}')
                ff.make_folder([os.path.join(save_folder, patient_id),
                                os.path.join(save_folder, patient_id, patient_subid),
                                case_root, save_folder_case])
                if os.path.isfile(os.path.join(save_folder_case, 'pred_img.nii.gz')):
                    print('  iter', iteration, 'already done'); continue

                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
                    pred_img = sampler.sample_2D(
                        trained_model_filename, condition_img, direct_use_of_model=True,
                        num_steps=args.num_steps, solver=args.solver, schedule=args.schedule)

                _save_nifti(pred_img, affine, os.path.join(save_folder_case, 'pred_img.nii.gz'))
                if iteration == 1:
                    _save_nifti(gt_img, affine, os.path.join(save_folder_case, 'gt_img.nii.gz'))
                    _save_nifti(condition_img, affine, os.path.join(save_folder_case, 'condition_img.nii.gz'))

        else:  # mode == 'avg'  -> cumulative average of the K generated samples
            save_folder_avg = os.path.join(case_root, f'epoch{epoch}avg')
            ff.make_folder([os.path.join(save_folder, patient_id),
                            os.path.join(save_folder, patient_id, patient_subid),
                            case_root, save_folder_avg])
            made = ff.sort_timeframe(ff.find_all_target_files([f'epoch{epoch}_*'], case_root), 0, '_', '/')
            completed = [p for p in made if os.path.isfile(os.path.join(p, 'pred_img.nii.gz'))]
            if len(completed) == 0:
                print('  no completed predicts, skip'); continue

            # load all samples, tolerating (and deleting) any corrupt / half-written file
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
            for k in sorted(set(args.k_save)):
                if k > n_ok:
                    print(f'  [warn] requested K={k} but only {n_ok} valid samples; skipping')
                    continue
                avg_k = stack[..., :k].mean(axis=-1)
                _save_nifti(avg_k, affine, os.path.join(save_folder_avg, f'pred_img_scans{k}.nii.gz'))
            # per-pixel std of all samples (free; useful for variance-weighted fusion later)
            np.save(os.path.join(save_folder_avg, 'sample_std.npy'), stack.std(axis=-1).astype(np.float32))
            _save_nifti(gt_img, affine, os.path.join(save_folder_avg, 'gt_img.nii.gz'))

            if args.cleanup:
                keep = set(args.k_save)
                removed = 0
                # drop cumulative averages we don't keep (e.g. scans1-9, 11-19)
                for f in glob.glob(os.path.join(save_folder_avg, 'pred_img_scans*.nii.gz')):
                    try:
                        kk = int(os.path.basename(f).split('scans')[1].split('.nii')[0])
                    except Exception:
                        continue
                    if kk not in keep:
                        os.remove(f); removed += 1
                # drop per-sample volumes (no longer needed once averaged; std map is kept)
                for p in completed:
                    fp = os.path.join(p, 'pred_img.nii.gz')
                    if os.path.isfile(fp):
                        os.remove(fp); removed += 1
                print(f'  [cleanup] removed {removed} files (per-sample volumes + non-kept scans)')


if __name__ == '__main__':
    run(get_args_parser().parse_args())
