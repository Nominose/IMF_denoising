"""
predict_2D_imf.py — Brain CT thin-slice denoising inference with improved MeanFlow (iMF)

Usage:
  python predict_2D_imf.py --epoch 30 --mode pred --iteration_num 20 --num_steps 3
  python predict_2D_imf.py --epoch 30 --mode avg
"""
import sys
sys.path.append('/gpfs/work/aac/xingyiyao23/Code')
import argparse
import os
import torch
import numpy as np
import nibabel as nb

import Diffusion_denoising_thin_slice.improved_mean_flow as imf
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.Generator_thinslice as Generator
from Diffusion_denoising_thin_slice.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion import Unet


def get_args_parser():
    parser = argparse.ArgumentParser('Brain CT iMF Inference Script')
    parser.add_argument('--trial_name', type=str, default='imf_unsupervised_gaussian_brainCT')
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--mode', type=str, required=True, choices=['pred', 'avg'])
    parser.add_argument('--slice_range', type=str, default="30-80")
    parser.add_argument('--iteration_num', type=int, default=20)
    parser.add_argument('--num_steps', type=int, default=3, help='NFE per sample: 1 for one-step, 3 recommended for N2N')
    return parser


def run(args):
    trial_name = args.trial_name
    epoch = args.epoch
    do_pred_or_avg = args.mode

    supervision = 'unsupervised'
    condition_channel = 2
    print('supervision:', supervision)

    study_folder = '/gpfs/work/aac/xingyiyao23/projects'
    trained_model_filename = os.path.join(study_folder, trial_name, 'models/model-' + str(epoch) + '.pt')
    save_folder = os.path.join(study_folder, trial_name, 'pred_images_nfe' + str(args.num_steps))
    os.makedirs(save_folder, exist_ok=True)

    image_size = [512, 512]
    histogram_equalization = True
    background_cutoff = -1000
    maximum_cutoff = 2000
    normalize_factor = 'equation'

    # ========== Histogram equalization bins ==========
    bins = np.load('/gpfs/work/aac/xingyiyao23/Data/histogram_equalization/bins.npy')
    bins_mapped = np.load('/gpfs/work/aac/xingyiyao23/Data/histogram_equalization/bins_mapped.npy')

    # ========== Patient list (test set = batch 5) ==========
    patient_list_file = '/gpfs/work/aac/xingyiyao23/Data/brain_CT/Patient_lists/fixedCT_static_simulation_train_test_gaussian_xjtlu.xlsx'
    build_sheet = Build_list.Build_thinsliceCT(patient_list_file)
    _, patient_id_list, patient_subid_list, random_num_list, condition_list, x0_list = \
        build_sheet.__build__(batch_list=[5])

    print('total cases:', patient_id_list.shape[0])
    n = ff.get_X_numbers_in_interval(total_number=patient_id_list.shape[0], start_number=0, end_number=1, interval=2)
    print('total number:', n.shape[0])

    # ========== Model ==========
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
    )

    diffusion_model = imf.ImprovedMeanFlow(
        base_model,
        image_size=image_size,
        ratio_r_neq_t=0.5,
        clip_or_not=False,
        auto_normalize=False,
    )

    # Create sampler, load model once
    sampler = imf.Sampler(diffusion_model, generator=None, batch_size=1)
    sampler.background_cutoff = background_cutoff
    sampler.maximum_cutoff = maximum_cutoff
    sampler.normalize_factor = normalize_factor
    sampler.histogram_equalization = histogram_equalization
    sampler.bins = bins
    sampler.bins_mapped = bins_mapped
    sampler.load_model(trained_model_filename)
    print("Model and EMA loaded from:", trained_model_filename)

    G = Generator.Dataset_2D

    for i in range(n.shape[0]):
        patient_id = patient_id_list[n[i]]
        patient_subid = patient_subid_list[n[i]]
        random_num = random_num_list[n[i]]
        x0_file = x0_list[n[i]]
        condition_file = condition_list[n[i]]

        print(i, patient_id, patient_subid, random_num)

        # Parse slice range
        if args.slice_range != "all":
            slice_start, slice_end = args.slice_range.split('-')
            slice_start, slice_end = int(slice_start), int(slice_end)
        else:
            tmp = nb.load(condition_file).get_fdata()
            slice_start, slice_end = 0, tmp.shape[2]

        slice_num = slice_end - slice_start

        # Load ground truth and condition
        gt_img = nb.load(x0_file).get_fdata()[:, :, slice_start:slice_end]
        condition_img = nb.load(condition_file).get_fdata()[:, :, slice_start:slice_end]
        affine = nb.load(condition_file).affine

        if do_pred_or_avg == 'pred':
            iteration_num = args.iteration_num

            for iteration in range(1, iteration_num + 1):
                print('iteration:', iteration)

                # Make folders
                save_folder_case = os.path.join(
                    save_folder, patient_id, patient_subid,
                    'random_' + str(random_num),
                    'epoch' + str(epoch) + '_' + str(iteration))
                ff.make_folder([
                    os.path.join(save_folder, patient_id),
                    os.path.join(save_folder, patient_id, patient_subid),
                    os.path.join(save_folder, patient_id, patient_subid, 'random_' + str(random_num)),
                    save_folder_case])

                if os.path.isfile(os.path.join(save_folder_case, 'pred_img.nii.gz')):
                    print('already done')
                    continue

                # Generator for this case
                generator = G(
                    supervision=supervision,
                    img_list=np.array([x0_file]),
                    condition_list=np.array([condition_file]),
                    image_size=image_size,
                    num_slices_per_image=slice_num,
                    random_pick_slice=False,
                    slice_range=[slice_start, slice_end],
                    histogram_equalization=histogram_equalization,
                    bins=bins,
                    bins_mapped=bins_mapped,
                    background_cutoff=background_cutoff,
                    maximum_cutoff=maximum_cutoff,
                    normalize_factor=normalize_factor,
                    shuffle=False,
                    augment=False,
                )

                sampler.generator = generator
                original_model_ref = sampler.model
                try:
                    sampler.model = sampler.ema.ema_model
                    pred_img = sampler.sample_2D(trained_model_filename, condition_img, direct_use_of_model=True, num_steps=args.num_steps)
                finally:
                    sampler.model = original_model_ref

                print(pred_img.shape)

                # Save prediction
                nb.save(nb.Nifti1Image(pred_img, affine),
                        os.path.join(save_folder_case, 'pred_img.nii.gz'))

                # Save gt and condition on first iteration
                if iteration == 1:
                    nb.save(nb.Nifti1Image(gt_img, affine),
                            os.path.join(save_folder_case, 'gt_img.nii.gz'))
                    nb.save(nb.Nifti1Image(condition_img, affine),
                            os.path.join(save_folder_case, 'condition_img.nii.gz'))

        if do_pred_or_avg == 'avg':
            save_folder_avg = os.path.join(
                save_folder, patient_id, patient_subid,
                'random_' + str(random_num),
                'epoch' + str(epoch) + 'avg')
            ff.make_folder([
                os.path.join(save_folder, patient_id),
                os.path.join(save_folder, patient_id, patient_subid),
                os.path.join(save_folder, patient_id, patient_subid, 'random_' + str(random_num)),
                save_folder_avg])

            made_predicts = ff.sort_timeframe(
                ff.find_all_target_files(
                    ['epoch' + str(epoch) + '_*'],
                    os.path.join(save_folder, patient_id, patient_subid, 'random_' + str(random_num))),
                0, '_', '/')

            if len(made_predicts) == 0:
                print('skip, no made predicts')
                continue

            completed_predicts = [
                p for p in made_predicts
                if os.path.isfile(os.path.join(p, 'pred_img.nii.gz'))
            ]
            total_predicts = len(completed_predicts)

            if total_predicts < 1:
                print('skip, no completed predicts')
                continue

            print(f'found {total_predicts} completed predictions')

            # Skip if avg already done
            if os.path.isfile(os.path.join(save_folder_avg, 'pred_img_scans' + str(total_predicts) + '.nii.gz')):
                print('already done')
                continue

            first_pred = nb.load(os.path.join(completed_predicts[0], 'pred_img.nii.gz')).get_fdata()
            loaded_data = np.zeros((first_pred.shape[0], first_pred.shape[1],
                                    first_pred.shape[2], total_predicts))
            loaded_data[:, :, :, 0] = first_pred
            for j, p in enumerate(completed_predicts[1:], 1):
                loaded_data[:, :, :, j] = nb.load(os.path.join(p, 'pred_img.nii.gz')).get_fdata()

            for avg_num in range(1, total_predicts + 1):
                print('avg_num:', avg_num)
                predicts_avg = loaded_data[:, :, :, :avg_num].mean(axis=-1)
                nb.save(nb.Nifti1Image(predicts_avg, affine),
                        os.path.join(save_folder_avg, 'pred_img_scans' + str(avg_num) + '.nii.gz'))

            # Save gt in avg folder
            nb.save(nb.Nifti1Image(gt_img, affine),
                    os.path.join(save_folder_avg, 'gt_img.nii.gz'))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    run(args)
