"""
predict_2D_flow_matching.py — Mayo Low-Dose CT with Flow Matching

Fixes applied from review:
  1. Generator uses preload=True with preloaded data (real IO optimization)
  2. avg mode filters completed folders before reading
  3. get_X_numbers_in_interval uses correct end_number for all patients
  4. Test generator explicitly sets augment=False, shuffle=False
  5. clip_or_not=False to match training config
  6. Preloaded data stored as float32
  7. Slice number printed during inference

Usage:
  python3 predict_2D_flow_matching.py --trial_name flow_matching_unsupervised_gaussian_mayo --epoch 190 --mode pred --input both --slice_range 100-200 --iteration_num 10
  python3 predict_2D_flow_matching.py --trial_name flow_matching_unsupervised_gaussian_mayo --epoch 190 --mode avg --input both --slice_range 100-200
"""
import sys
sys.path.append('/gpfs/work/aac/xingyiyao23/Code
import argparse
import os
import torch
import numpy as np
import nibabel as nb
from ema_pytorch import EMA

import Diffusion_denoising_thin_slice.conditional_flow_matching as fm
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.Generator as Generator
import Diffusion_denoising_thin_slice.Data_processing as Data_processing


def get_args_parser():
    parser = argparse.ArgumentParser('Flow Matching Inference Script')
    parser.add_argument('--trial_name', type=str, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--mode', type=str, required=True, choices=['pred', 'avg'], help='pred or avg')
    parser.add_argument('--input', type=str, default='both', choices=['both', 'odd', 'even', 'all'])
    parser.add_argument('--slice_range', type=str, default="all")
    parser.add_argument('--iteration_num', type=int, default=10, help='number of K samples to generate')
    return parser


def run(args):
    trial_name = args.trial_name
    epoch = args.epoch
    do_pred_or_avg = args.mode
    input_condition = args.input

    supervision = 'supervised' if trial_name[:2] == 'su' else 'unsupervised'
    print('supervision:', supervision)

    study_folder = '/gpfs/work/aac/xingyiyao23/results'
    trained_model_filename = os.path.join(study_folder, trial_name, 'models', f'model-{epoch}.pt')
    save_folder = os.path.join(study_folder, trial_name, f'pred_images_input_{input_condition}')
    os.makedirs(save_folder, exist_ok=True)

    image_size = [512, 512]
    sampling_timesteps = 50

    histogram_equalization = False
    background_cutoff = -200
    maximum_cutoff = 250
    normalize_factor = 'equation'

    # patient list
    patient_list_file = '/gpfs/work/aac/xingyiyao23/Data/新建文件夹/mayo/mayo_flow_matching.xlsx'
    build_sheet = Build_list.Build(patient_list_file)

    _, patient_id_list, random_num_list, noise_file_all_list, noise_file_odd_list, noise_file_even_list, \
        ground_truth_file_list, _ = build_sheet.__build__(batch_list=['test'])

    # FIX #3: end_number should cover all patients, not just 1
    total_patients = patient_id_list.shape[0]
    n = np.arange(total_patients)
    print('total test patients:', n.shape[0])

    # =================================================================
    #  BUILD MODEL ONCE
    # =================================================================
    model = fm.Unet(
        problem_dimension='2D',
        init_dim=64, out_dim=1, channels=1,
        conditional_diffusion=True, condition_channels=1,
        downsample_list=(True, True, True, False),
        upsample_list=(True, True, True, False),
        full_attn=(None, None, False, True),
    )

    # FIX #5: clip_or_not=False to match training config
    diffusion_model = fm.ConditionalFlowMatching(
        model,
        image_size=image_size,
        sampling_timesteps=sampling_timesteps,
        clip_or_not=False,
        auto_normalize=False,
    )

    # LOAD WEIGHTS ONCE
    print('loading model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(trained_model_filename, map_location=device)
    diffusion_model.load_state_dict(checkpoint['model'])
    diffusion_model.to(device)
    diffusion_model.eval()
    print('model loaded to', device)

    ema = EMA(diffusion_model)
    ema.load_state_dict(checkpoint['ema'])
    ema.to(device)
    ema.ema_model.eval()
    print('EMA loaded')

    del checkpoint
    torch.cuda.empty_cache()

    # =================================================================
    #  PRELOAD ALL TEST NII.GZ FILES INTO RAM (float32)
    # =================================================================
    print('\npreloading test data...')
    preloaded_data = {}   # fpath -> np.ndarray (float32)
    preloaded_affine = {}  # fpath -> affine

    for i in range(n.shape[0]):
        for fpath in [noise_file_all_list[n[i]], noise_file_odd_list[n[i]],
                      noise_file_even_list[n[i]], ground_truth_file_list[n[i]]]:
            if fpath not in preloaded_data:
                print(f'  loading {os.path.basename(os.path.dirname(fpath))}/{os.path.basename(fpath)}')
                nii = nb.load(fpath)
                # FIX #6: store as float32 to save RAM
                preloaded_data[fpath] = np.asarray(nii.dataobj, dtype=np.float32)
                preloaded_affine[fpath] = nii.affine
    print(f'preloaded {len(preloaded_data)} files\n')

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
        print(f"\n===== Patient {i+1}/{n.shape[0]}: {patient_id}, random={random_num} =====")

        affine = preloaded_affine[condition_files[0]]
        condition_img_full = preloaded_data[condition_files[0]]
        if args.slice_range != "all":
            s, e = map(int, args.slice_range.split('-'))
        else:
            s, e = 0, condition_img_full.shape[2]
        condition_img = condition_img_full[:, :, s:e]
        slice_num = condition_img.shape[2]
        print('slice num:', slice_num)

        gt_img = preloaded_data[gt_file][:, :, s:e]

        # =================================================================
        #  PRED mode
        # =================================================================
        if do_pred_or_avg == 'pred':
            iteration_num = args.iteration_num if supervision == 'unsupervised' else 1

            # FIX #1: Build generators with preload=True, passing preloaded data
            # FIX #4: explicitly set augment=False, shuffle=False for test
            generators = {}
            for ci, cond_file in enumerate(condition_files):
                cond_data_preloaded = preloaded_data[cond_file]
                # preload_data expects (list_of_x0_volumes, list_of_cond_volumes)
                # matching the shape of img_list / condition_list
                generators[ci] = G(
                    supervision=supervision,
                    preload=True,
                    preload_data=([cond_data_preloaded], [cond_data_preloaded]),
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
                    shuffle=False,
                    augment=False,
                )

            for iteration in range(1, iteration_num + 1):
                print(f'\n--- iteration {iteration}/{iteration_num} ---')
                save_folder_case = os.path.join(
                    save_folder, patient_id, f'random_{random_num}', f'epoch{epoch}_{iteration}'
                )
                ff.make_folder([
                    os.path.join(save_folder, patient_id),
                    os.path.join(save_folder, patient_id, f'random_{random_num}'),
                    save_folder_case,
                ])

                if os.path.isfile(os.path.join(save_folder_case, 'pred_img.nii.gz')):
                    print('already done, skip')
                    continue

                for ci in range(len(condition_files)):
                    print('condition:', condition_names[ci] if condition_names else 'all')

                    gen = generators[ci]
                    pred = np.zeros((image_size[0], image_size[1], slice_num), dtype=np.float32)

                    with torch.inference_mode():
                        for z in range(slice_num):
                            # FIX #7: print slice number
                            print(f'  slice {z+1}/{slice_num}')
                            datas = gen[z]
                            cond = datas[1]
                            if isinstance(cond, np.ndarray):
                                cond = torch.from_numpy(cond).float()
                            if cond.dim() == 2:
                                cond = cond.unsqueeze(0).unsqueeze(0)
                            elif cond.dim() == 3:
                                cond = cond.unsqueeze(0)
                            data_cond = cond.to(device)

                            out = ema.ema_model.sample(condition=data_cond, batch_size=1)
                            pred[:, :, z] = out[0, 0].detach().cpu().numpy()

                    # denormalize
                    pred = Data_processing.crop_or_pad(
                        pred,
                        [condition_img.shape[0], condition_img.shape[1], condition_img.shape[2]],
                        value=np.min(condition_img),
                    )
                    pred = Data_processing.normalize_image(
                        pred, normalize_factor=normalize_factor,
                        image_max=maximum_cutoff, image_min=background_cutoff, invert=True
                    )
                    pred = Data_processing.correct_shift_caused_in_pad_crop_loop(pred)

                    if len(condition_files) == 1:
                        nb.save(nb.Nifti1Image(pred, affine),
                                os.path.join(save_folder_case, 'pred_img.nii.gz'))
                    else:
                        nb.save(nb.Nifti1Image(pred, affine),
                                os.path.join(save_folder_case, f'pred_img_{condition_names[ci]}.nii.gz'))

                    print(f'  done, shape={pred.shape}')

                # average odd/even if both conditions
                if len(condition_files) == 2:
                    final = np.zeros([2, *pred.shape], dtype=np.float32)
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

        # =================================================================
        #  AVG mode
        # =================================================================
        if do_pred_or_avg == 'avg':
            save_avg = os.path.join(save_folder, patient_id, f'random_{random_num}', f'epoch{epoch}avg')
            ff.make_folder([
                os.path.join(save_folder, patient_id),
                os.path.join(save_folder, patient_id, f'random_{random_num}'),
                save_avg,
            ])

            made = ff.sort_timeframe(
                ff.find_all_target_files([f'epoch{epoch}_*'],
                                         os.path.join(save_folder, patient_id, f'random_{random_num}')),
                0, '_', '/')

            # FIX #2: filter completed folders before reading
            completed = [m for m in made if os.path.isfile(os.path.join(m, 'pred_img.nii.gz'))]
            total = len(completed)
            print(f'found {total} completed iterations')

            if total < 1:
                print('no completed iterations, skip')
                continue

            loaded = np.zeros((*condition_img.shape, total), dtype=np.float32)
            for j, m in enumerate(completed):
                loaded[:, :, :, j] = nb.load(os.path.join(m, 'pred_img.nii.gz')).get_fdata()

            for avg_num in [1, 3, 5, 8, 10]:
                if avg_num > total:
                    continue
                avg = np.mean(loaded[:, :, :, :avg_num], axis=-1)
                nb.save(nb.Nifti1Image(avg, affine),
                        os.path.join(save_avg, f'pred_img_scans{avg_num}.nii.gz'))
                print(f'  saved K={avg_num} average')

            nb.save(nb.Nifti1Image(gt_img, affine),
                    os.path.join(save_avg, 'gt_img.nii.gz'))
            nb.save(nb.Nifti1Image(condition_img, affine),
                    os.path.join(save_avg, 'condition_img.nii.gz'))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    run(args)