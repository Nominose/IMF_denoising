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


def get_args_parser():
    parser = argparse.ArgumentParser('iMF Inference Script')
    parser.add_argument('--trial_name', type=str, default='imf_unsupervised_gaussian_mayo')
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--mode', type=str, required=True, choices=['pred', 'avg'])
    parser.add_argument('--input', type=str, default='both', choices=['both', 'odd', 'even'])
    parser.add_argument('--slice_range', type=str, default="100-200")
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

    study_folder = '/host/d/projects/denoising/models'
    trained_model_filename = os.path.join(study_folder, trial_name, 'models/model-' + str(epoch) + '.pt')
    save_folder = os.path.join(study_folder, trial_name, 'pred_images_input_' + input_condition)
    os.makedirs(save_folder, exist_ok=True)

    image_size = [512, 512]
    histogram_equalization = False
    background_cutoff = -200
    maximum_cutoff = 250
    normalize_factor = 'equation'

    # ========== Patient list ==========
    build_sheet = Build_list.Build(os.path.join('/host/d/file/新建文件夹/mayo/mayo_flow_matching.xlsx'))
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
    print("Model and EMA loaded from:", trained_model_filename)

    G = Generator.Dataset_2D
    for i in range(n.shape[0]):
        patient_id = patient_id_list[n[i]]
        random_num = random_num_list[n[i]]
        noise_file_odd = noise_file_odd_list[n[i]]
        noise_file_even = noise_file_even_list[n[i]]
        gt_file = ground_truth_file_list[n[i]]

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

                for condition_i in range(len(condition_files)):
                    condition_file = condition_files[condition_i]
                    print('condition file:', condition_file)

                    # FIX #2: Load condition_img per branch
                    condition_img = nb.load(condition_file).get_fdata()[:, :, slice_start:slice_end]
                    slice_num = condition_img.shape[2]

                    generator = G(
                        supervision=supervision,
                        img_list=np.array([condition_file]),
                        condition_list=np.array([condition_file]),
                        image_size=image_size,
                        num_slices_per_image=slice_num,
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

                    sampler.generator = generator
                    # Use EMA model without re-loading (already loaded once above)
                    original_model_ref = sampler.model
                    try:
                        sampler.model = sampler.ema.ema_model
                        pred_img = sampler.sample_2D(trained_model_filename, condition_img, direct_use_of_model=True, num_steps=args.num_steps, solver=args.solver, schedule=args.schedule)
                    finally:
                        sampler.model = original_model_ref

                    print(pred_img.shape)

                    if len(condition_files) == 1:
                        nb.save(nb.Nifti1Image(pred_img, affine),
                                os.path.join(save_folder_case, 'pred_img.nii.gz'))
                    else:
                        nb.save(nb.Nifti1Image(pred_img, affine),
                                os.path.join(save_folder_case, 'pred_img_' + condition_names[condition_i] + '.nii.gz'))

                if len(condition_files) == 2:
                    pred_img_final = np.zeros([len(condition_files), pred_img.shape[0], pred_img.shape[1], pred_img.shape[2]])
                    for condition_i in range(len(condition_files)):
                        pred_img_final[condition_i] = nb.load(
                            os.path.join(save_folder_case, 'pred_img_' + condition_names[condition_i] + '.nii.gz')).get_fdata()
                    pred_img_final = np.mean(pred_img_final, axis=0)
                    nb.save(nb.Nifti1Image(pred_img_final, affine),
                            os.path.join(save_folder_case, 'pred_img.nii.gz'))

                if iteration == 1:
                    nb.save(nb.Nifti1Image(gt_img, affine),
                            os.path.join(save_folder_case, 'gt_img.nii.gz'))
                    # Save first condition image for reference
                    cond_ref = nb.load(condition_files[0]).get_fdata()[:, :, slice_start:slice_end]
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
