"""
predict_2D_imf.py — Mayo Low-Dose CT inference with improved MeanFlow (iMF)

1-NFE sampling: 1 forward pass per slice per sample
iMF K=8 = 8 forward passes per slice vs FM K=8 = 400 per slice
"""
import sys
sys.path.append('/host/c/Users/ROG/Documents/GitHub')
import argparse
import os
import torch
import numpy as np
import nibabel as nb

import IMF_denoising.improved_mean_flow as imf
import IMF_denoising.functions_collection as ff
import IMF_denoising.Build_lists.Build_list as Build_list
import IMF_denoising.Generator as Generator
from IMF_denoising.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion import Unet


def _detect_base():
    for b in ('/host/d/research', '/host/d', 'D:/research', '/d/research'):
        if os.path.isdir(os.path.join(b, 'Data', '新mayo_data')):
            return b
    return '/host/d/research'


_BASE = _detect_base()
_MAYO_DATA = os.path.join(_BASE, 'Data', '新mayo_data')
# stored recon/gt paths are /host/e/D/Data/low_dose_CT/<tail>; on D: the data is split across a few
# folders (recons under Data/新mayo_data/simulation_highnoise_v2, clean GT under file/新建文件夹/mayo/nii_imgs).
_REMAP_BASES = [
    _MAYO_DATA,
    os.path.join(_BASE, 'file', '新建文件夹', 'mayo'),
    os.path.join(_BASE, 'file', 'new', 'mayo'),
]


def _remap(p):
    """Resolve a stored /host/e path onto whichever D: folder actually holds it."""
    if p is None:
        return p
    p = str(p)
    if os.path.exists(p):
        return p
    old = '/host/e/D/Data/low_dose_CT'
    if p.startswith(old + '/'):
        tail = p[len(old):]
        for base in _REMAP_BASES:
            cand = base + tail
            if os.path.exists(cand):
                return cand
    return p


def get_args_parser():
    parser = argparse.ArgumentParser('iMF Inference Script')
    parser.add_argument('--trial_name', type=str, default='imf_v2_unsupervised_gaussian_mayo')  # match train_2D_imf_mayo.py (old non-v-head default would strict-load-fail against the v-head model)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--mode', type=str, required=True, choices=['pred', 'avg'])
    parser.add_argument('--input', type=str, default='both', choices=['both', 'odd', 'even'])
    parser.add_argument('--slice_range', type=str, default="150-200")  # Mayo: slices 150-200 to match the prior paper
    parser.add_argument('--iteration_num', type=int, default=20)
    parser.add_argument('--num_steps', type=int, default=1, help='NFE per sample: 1 for one-step, 2+ for multistep')
    parser.add_argument('--solver', type=str, default='euler', choices=['euler', 'midpoint', 'heun'], help='ODE solver type')
    parser.add_argument('--schedule', type=str, default='uniform', choices=['uniform', 'optimal'], help='Time step schedule')
    return parser


def run(args):
    trial_name = args.trial_name
    epoch = args.epoch
    do_pred_or_avg = args.mode
    input_condition = args.input

    supervision = 'unsupervised'
    print('supervision:', supervision)

    study_folder = os.path.join(_BASE, 'projects/denoising/models')
    trained_model_filename = os.path.join(study_folder, trial_name, 'models/model-' + str(epoch) + '.pt')
    # Encode the run config in the output path so different NFE / solver / schedule / slice_range never
    # share a folder — otherwise the per-iteration "already done" resume check would silently reuse a
    # wrong-config result (e.g. an nfe1 run reused for an nfe3 request). (audit P1)
    cfg = 'nfe' + str(args.num_steps)
    if args.solver != 'euler':        cfg += '_' + args.solver
    if args.schedule != 'uniform':    cfg += '_' + args.schedule
    if args.slice_range != '150-200': cfg += '_slice' + args.slice_range
    save_folder = os.path.join(study_folder, trial_name, 'pred_images_input_' + input_condition + '_' + cfg)
    os.makedirs(save_folder, exist_ok=True)

    image_size = [512, 512]
    histogram_equalization = False
    background_cutoff = -200
    maximum_cutoff = 250
    normalize_factor = 'equation'

    # ========== Patient list ==========
    build_sheet = Build_list.Build(os.path.join(_MAYO_DATA, 'mayo_low_dose_CT_gaussian_simulation_highnoise_v2.xlsx'))
    _, patient_id_list, random_num_list, noise_file_all_list, noise_file_odd_list, noise_file_even_list, ground_truth_file_list, _ = \
        build_sheet.__build__(batch_list=['test'])

    print('total cases:', patient_id_list.shape[0])
    n = np.arange(patient_id_list.shape[0])
    print('total number:', n.shape[0])

    # ========== Model (built once) ==========
    base_model = Unet(
        problem_dimension='2D',
        init_dim=64,
        out_dim=1,
        channels=1,
        conditional_diffusion=True,
        condition_channels=1,
        downsample_list=(True, True, True, False),
        upsample_list=(True, True, True, False),
        full_attn=(None, None, False, True),
        auxiliary_v_head=True,
    )

    diffusion_model = imf.ImprovedMeanFlow(
        base_model,
        image_size=image_size,
        ratio_r_neq_t=0.5,
        clip_or_not=False,
        auto_normalize=False,
    )

    # Create sampler ONCE, load model+EMA ONCE
    sampler = imf.Sampler(diffusion_model, generator=None, batch_size=1)
    # Set denorm params manually (same for all generators)
    sampler.background_cutoff = background_cutoff
    sampler.maximum_cutoff = maximum_cutoff
    sampler.normalize_factor = normalize_factor
    sampler.histogram_equalization = histogram_equalization
    sampler.load_model(trained_model_filename)
    # Batch 2 slices per forward. Tuned on a 12GB card: batch 8 thrashes the VRAM allocator
    # (~11GB, 3.0 s/slice) while batch 2 stays at ~2GB and runs ~0.5 s/slice. Batching is
    # numerically a no-op (each slice keeps its own init noise) — purely a throughput knob.
    sampler.slice_batch = 2
    print("Model and EMA loaded from:", trained_model_filename)

    G = Generator.Dataset_2D
    for i in range(n.shape[0]):
        patient_id = patient_id_list[n[i]]
        random_num = random_num_list[n[i]]
        noise_file_odd = _remap(noise_file_odd_list[n[i]])
        noise_file_even = _remap(noise_file_even_list[n[i]])
        gt_file = _remap(ground_truth_file_list[n[i]])

        if input_condition == 'both':
            condition_files = [noise_file_odd, noise_file_even]
            condition_names = ['odd', 'even']
        elif input_condition == 'odd':
            condition_files = [noise_file_odd]
        elif input_condition == 'even':
            condition_files = [noise_file_even]

        print(i, patient_id, random_num)

        # Parse slice range
        if args.slice_range != "all":
            slice_start, slice_end = args.slice_range.split('-')
            slice_start, slice_end = int(slice_start), int(slice_end)
        else:
            tmp = nb.load(condition_files[0]).get_fdata()
            slice_start, slice_end = 0, tmp.shape[2]

        # Load ground truth
        gt_img = nb.load(gt_file).get_fdata()[:, :, slice_start:slice_end]
        # Load first condition for affine and shape reference
        affine = nb.load(condition_files[0]).affine

        if do_pred_or_avg == 'pred':
            iteration_num = args.iteration_num

            # PERF: the condition volume + its Generator are identical across all K iterations
            # (the K samples differ only by the sampler's fresh init noise, not the condition).
            # Build each once, lazily, and reuse — avoids reloading the ~90MB condition volume and
            # rebuilding the Generator (histogram-eq etc.) every iteration. Lazy build keeps fully
            # 'already done' cases (which `continue` before use) from ever touching the volume.
            cond_imgs = [None] * len(condition_files)
            generators = [None] * len(condition_files)

            for iteration in range(1, iteration_num + 1):
                print('iteration:', iteration)

                save_folder_case = os.path.join(
                    save_folder, patient_id, 'random_' + str(random_num),
                    'epoch' + str(epoch) + '_' + str(iteration))
                ff.make_folder([
                    os.path.join(save_folder, patient_id),
                    os.path.join(save_folder, patient_id, 'random_' + str(random_num)),
                    save_folder_case])

                if os.path.isfile(os.path.join(save_folder_case, 'pred_img.nii.gz')):
                    print('already done')
                    continue

                pred_per_condition = []
                for condition_i in range(len(condition_files)):
                    condition_file = condition_files[condition_i]
                    print('condition file:', condition_file)

                    # Build condition_img + Generator once (lazily), then reuse across iterations.
                    # sample_2D reads the actual conditioning from sampler.generator[...] and only
                    # reads condition_img for shape/min (never mutates it), so reuse is safe.
                    if generators[condition_i] is None:
                        ci = nb.load(condition_file).get_fdata()[:, :, slice_start:slice_end]
                        cond_imgs[condition_i] = ci
                        generators[condition_i] = G(
                            supervision=supervision,
                            img_list=np.array([condition_file]),
                            condition_list=np.array([condition_file]),
                            image_size=image_size,
                            num_slices_per_image=ci.shape[2],
                            random_pick_slice=False,
                            slice_range=None if args.slice_range == "all" else [slice_start, slice_end],
                            histogram_equalization=histogram_equalization,
                            bins=None,
                            bins_mapped=None,
                            background_cutoff=background_cutoff,
                            maximum_cutoff=maximum_cutoff,
                            normalize_factor=normalize_factor,
                            shuffle=False,
                            augment=False,
                        )
                    condition_img = cond_imgs[condition_i]

                    sampler.generator = generators[condition_i]
                    # Use EMA model without re-loading (already loaded once above)
                    original_model_ref = sampler.model
                    try:
                        sampler.model = sampler.ema.ema_model
                        pred_img = sampler.sample_2D(trained_model_filename, condition_img, direct_use_of_model=True, num_steps=args.num_steps, solver=args.solver, schedule=args.schedule)
                    finally:
                        sampler.model = original_model_ref

                    print(pred_img.shape)
                    pred_per_condition.append(pred_img)

                    if len(condition_files) == 1:
                        nb.save(nb.Nifti1Image(pred_img, affine),
                                os.path.join(save_folder_case, 'pred_img.nii.gz'))
                    else:
                        nb.save(nb.Nifti1Image(pred_img, affine),
                                os.path.join(save_folder_case, 'pred_img_' + condition_names[condition_i] + '.nii.gz'))

                if len(condition_files) == 2:
                    # Average the two conditions in-memory (avoid reloading the odd/even .nii.gz we
                    # just wrote — the avg step downstream only reads pred_img.nii.gz).
                    pred_img_final = np.mean(np.stack(pred_per_condition, axis=0), axis=0)
                    nb.save(nb.Nifti1Image(pred_img_final, affine),
                            os.path.join(save_folder_case, 'pred_img.nii.gz'))

                if iteration == 1:
                    nb.save(nb.Nifti1Image(gt_img, affine),
                            os.path.join(save_folder_case, 'gt_img.nii.gz'))
                    # Save first condition image for reference (reuse already-loaded volume)
                    cond_ref = cond_imgs[0] if cond_imgs[0] is not None else nb.load(condition_files[0]).get_fdata()[:, :, slice_start:slice_end]
                    nb.save(nb.Nifti1Image(cond_ref, affine),
                            os.path.join(save_folder_case, 'condition_img.nii.gz'))

        if do_pred_or_avg == 'avg':
            # Load condition_img for shape reference
            condition_img = nb.load(condition_files[0]).get_fdata()[:, :, slice_start:slice_end]

            save_folder_avg = os.path.join(
                save_folder, patient_id, 'random_' + str(random_num),
                'epoch' + str(epoch) + 'avg')
            ff.make_folder([
                os.path.join(save_folder, patient_id),
                os.path.join(save_folder, patient_id, 'random_' + str(random_num)),
                save_folder_avg])

            made_predicts = ff.sort_timeframe(
                ff.find_all_target_files(
                    ['epoch' + str(epoch) + '_*'],
                    os.path.join(save_folder, patient_id, 'random_' + str(random_num))),
                0, '_', '/')

            if len(made_predicts) == 0:
                print('skip, no made predicts')
                continue

            # FIX #3: Filter to only completed directories
            completed_predicts = [
                p for p in made_predicts
                if os.path.isfile(os.path.join(p, 'pred_img.nii.gz'))
            ]
            total_predicts = len(completed_predicts)

            if total_predicts < 1:
                print('skip, no completed predicts')
                continue

            print(f'found {total_predicts} completed predictions')

            loaded_data = np.zeros((condition_img.shape[0], condition_img.shape[1],
                                    condition_img.shape[2], total_predicts))
            for j, p in enumerate(completed_predicts):
                loaded_data[:, :, :, j] = nb.load(os.path.join(p, 'pred_img.nii.gz')).get_fdata()

            for avg_num in range(1, total_predicts + 1):
                print('avg_num:', avg_num)
                predicts_avg = loaded_data[:, :, :, :avg_num].mean(axis=-1)
                nb.save(nb.Nifti1Image(predicts_avg, affine),
                        os.path.join(save_folder_avg, 'pred_img_scans' + str(avg_num) + '.nii.gz'))

            # Save gt in avg folder for easy metric computation
            nb.save(nb.Nifti1Image(gt_img, affine),
                    os.path.join(save_folder_avg, 'gt_img.nii.gz'))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    run(args)
