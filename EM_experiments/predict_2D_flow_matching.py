"""
Example: predict_2D_flow_matching.py

Minimal modification of CT_experiments/predict_2D.py for flow matching.
Only 3 lines change (marked with # <-- CHANGED).
"""
import sys
sys.path.append('/gpfs/work/aac/xingyiyao23/Code')
import argparse
import os
import torch
import numpy as np
import nibabel as nb

# ======================================================================
# CHANGE 1: import flow matching instead of ddpm
# ======================================================================
import Diffusion_denoising_thin_slice.conditional_flow_matching as fm           # <-- CHANGED

import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.Generator as Generator


def get_args_parser():
    parser = argparse.ArgumentParser('Flow Matching Inference Script')
    parser.add_argument('--trial_name', type=str, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--mode', type=str, required=True, help='pred or avg')
    parser.add_argument('--input', type=str, default='both', choices=['both', 'odd', 'even', 'all'])
    parser.add_argument('--slice_range', type=str, default="all")
    return parser


def run(args):
    trial_name = args.trial_name
    epoch = args.epoch
    do_pred_or_avg = args.mode
    input_condition = args.input

    supervision = 'supervised' if trial_name[:2] == 'su' else 'unsupervised'
    print('supervision:', supervision)

    study_folder = '/gpfs/work/aac/xingyiyao23/projects'
    trained_model_filename = os.path.join(study_folder, trial_name, 'models', f'model-{epoch}.pt')
    save_folder = os.path.join(study_folder, trial_name, f'pred_images_input_{input_condition}')
    os.makedirs(save_folder, exist_ok=True)

    image_size = [512, 512]
    sampling_timesteps = 50              # FM typically needs only 50 steps

    histogram_equalization = False
    background_cutoff = -200
    maximum_cutoff = 250
    normalize_factor = 'equation'

    # patient list
    build_sheet = Build_list.Build(
        os.path.join('/gpfs/work/aac/xingyiyao23/Data/low_dose_CT/Patient_lists/mayo_low_dose_CT_gaussian_simulation_v2.xlsx')
    )
    _, patient_id_list, random_num_list, noise_file_all_list, noise_file_odd_list, noise_file_even_list, \
        ground_truth_file_list, _ = build_sheet.__build__(batch_list=['test'])
    n = ff.get_X_numbers_in_interval(total_number=patient_id_list.shape[0], start_number=0, end_number=1, interval=1)
    print('total number:', n.shape[0])

    # ======================================================================
    # CHANGE 2: same Unet, but wrap in ConditionalFlowMatching
    # ======================================================================
    model = fm.Unet(
        problem_dimension='2D',
        init_dim=64, out_dim=1, channels=1,
        conditional_diffusion=True, condition_channels=1,
        downsample_list=(True, True, True, False),
        upsample_list=(True, True, True, False),
        full_attn=(None, None, False, True),
    )

    diffusion_model = fm.ConditionalFlowMatching(                               # <-- CHANGED
        model,
        image_size=image_size,
        sampling_timesteps=sampling_timesteps,
        clip_or_not=True,
        clip_range=[-1, 1],
    )

    G = Generator.Dataset_2D
    for i in range(n.shape[0]):
        patient_id = patient_id_list[n[i]]
        random_num = random_num_list[n[i]]
        noise_file_all = noise_file_all_list[n[i]]
        noise_file_odd = noise_file_odd_list[n[i]]
        noise_file_even = noise_file_even_list[n[i]]
        gt_file = ground_truth_file_list[n[i]]

        if supervision == 'supervised':
            assert input_condition in ['all']
        if input_condition == 'both':
            condition_files = [noise_file_odd, noise_file_even]
        elif input_condition == 'odd':
            condition_files = [noise_file_odd]
        elif input_condition == 'even':
            condition_files = [noise_file_even]
        elif input_condition == 'all':
            condition_files = [noise_file_all]

        condition_names = ['odd', 'even'] if len(condition_files) == 2 else []
        print(i, patient_id, random_num)

        affine = nb.load(condition_files[0]).affine
        condition_img = nb.load(condition_files[0]).get_fdata()
        if args.slice_range != "all":
            s, e = map(int, args.slice_range.split('-'))
        else:
            s, e = 0, condition_img.shape[2]
        condition_img = condition_img[:, :, s:e]
        slice_num = condition_img.shape[2]
        print('slice num:', slice_num)

        gt_img = nb.load(gt_file).get_fdata()[:, :, s:e]

        if do_pred_or_avg == 'pred':
            iteration_num = 20 if supervision == 'unsupervised' else 1

            for iteration in range(1, iteration_num + 1):
                print('iteration:', iteration)
                save_folder_case = os.path.join(
                    save_folder, patient_id, f'random_{random_num}', f'epoch{epoch}_{iteration}'
                )
                ff.make_folder([
                    os.path.join(save_folder, patient_id),
                    os.path.join(save_folder, patient_id, f'random_{random_num}'),
                    save_folder_case,
                ])

                if os.path.isfile(os.path.join(save_folder_case, 'pred_img.nii.gz')):
                    print('already done'); continue

                for ci in range(len(condition_files)):
                    cond_file = condition_files[ci]
                    print('condition file:', cond_file)

                    generator = G(
                        supervision=supervision,
                        img_list=np.array([cond_file]),
                        condition_list=np.array([cond_file]),
                        image_size=image_size,
                        num_slices_per_image=slice_num,
                        random_pick_slice=False,
                        slice_range=None if args.slice_range == "all" else [s, e],
                        histogram_equalization=histogram_equalization,
                        bins=None, bins_mapped=None,
                        background_cutoff=background_cutoff,
                        maximum_cutoff=maximum_cutoff,
                        normalize_factor=normalize_factor,
                    )

                    # ==============================================================
                    # CHANGE 3: use fm.Sampler instead of ddpm.Sampler
                    # ==============================================================
                    sampler = fm.Sampler(diffusion_model, generator, batch_size=1)  # <-- CHANGED
                    pred_img = sampler.sample_2D(trained_model_filename, condition_img)
                    print(pred_img.shape)

                    if len(condition_files) == 1:
                        nb.save(nb.Nifti1Image(pred_img, affine),
                                os.path.join(save_folder_case, 'pred_img.nii.gz'))
                    else:
                        nb.save(nb.Nifti1Image(pred_img, affine),
                                os.path.join(save_folder_case, f'pred_img_{condition_names[ci]}.nii.gz'))

                if len(condition_files) == 2:
                    final = np.zeros([2, *pred_img.shape])
                    for ci in range(2):
                        final[ci] = nb.load(
                            os.path.join(save_folder_case, f'pred_img_{condition_names[ci]}.nii.gz')
                        ).get_fdata()
                    final = np.mean(final, axis=0)
                    nb.save(nb.Nifti1Image(final, affine),
                            os.path.join(save_folder_case, 'pred_img.nii.gz'))

                if iteration == 1:
                    nb.save(nb.Nifti1Image(gt_img, affine),
                            os.path.join(save_folder_case, 'gt_img.nii.gz'))
                    nb.save(nb.Nifti1Image(condition_img, affine),
                            os.path.join(save_folder_case, 'condition_img.nii.gz'))

        if do_pred_or_avg == 'avg':
            save_avg = os.path.join(save_folder, patient_id, f'random_{random_num}', f'epoch{epoch}avg')
            ff.make_folder([os.path.join(save_folder, patient_id),
                            os.path.join(save_folder, patient_id, f'random_{random_num}'),
                            save_avg])

            made = ff.sort_timeframe(
                ff.find_all_target_files([f'epoch{epoch}_*'],
                                         os.path.join(save_folder, patient_id, f'random_{random_num}')),
                0, '_', '/')
            if len(made) == 0:
                print('skip'); continue

            total = sum(os.path.isfile(os.path.join(m, 'pred_img.nii.gz')) for m in made)
            if total < 2:
                print('not enough'); continue

            loaded = np.zeros((*condition_img.shape, total))
            for j in range(total):
                loaded[:, :, :, j] = nb.load(os.path.join(made[j], 'pred_img.nii.gz')).get_fdata()

            for avg_num in [1, 2, 5, 10, 20]:
                if avg_num > total:
                    continue
                avg = np.mean(loaded[:, :, :, :avg_num], axis=-1)
                nb.save(nb.Nifti1Image(avg, affine),
                        os.path.join(save_avg, f'pred_img_scans{avg_num}.nii.gz'))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    run(args)
