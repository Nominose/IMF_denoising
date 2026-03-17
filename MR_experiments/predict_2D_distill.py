import sys
sys.path.append('/host/d/Github')
import argparse
import os
import torch
import numpy as np 
import nibabel as nb
import Diffusion_denoising_thin_slice.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion as ddpm
import Diffusion_denoising_thin_slice.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_EDM as edm
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.Generator as Generator


def get_args_parser():
    parser = argparse.ArgumentParser('Diffusion Inference Script')

    parser.add_argument('--trial_name', type=str, required=True,
                        help='trial name such as unsupervised_gaussian')
    parser.add_argument('--epoch', type=int, required=True,
                        help='epoch number of the model')
    parser.add_argument('--mode', type=str, required = True, 
                        help='predict mode: avg or pred')
    
    parser.add_argument('--input', type=str, default='both', choices=['both', 'odd', 'even', 'all'],
                        help='input condition: both, odd, even, all')
    
    parser.add_argument('--slice_range', type=str, default="all",
                        help='slice range such as 100-200 or None for all slices')
        

    return parser

###########

def run(args):
    trial_name = args.trial_name
    epoch = args.epoch
    do_pred_or_avg = args.mode  #'avg' #'pred'
    input_condition = args.input  #'both', 'odd', 'even', 'all'

    supervision = 'supervised' 

    study_folder = '/host/d/projects/denoising/models'
    trained_model_filename = os.path.join(study_folder,trial_name, 'models/model-' + str(epoch)+ '.pt')
    save_folder = os.path.join(study_folder, trial_name, 'pred_images_input_'+ input_condition); os.makedirs(save_folder, exist_ok=True)

    image_size = [512,512] 
    objective = 'pred_x0' 
    sampling_timesteps = 50 # 100

    histogram_equalization = False
    background_cutoff = -200
    maximum_cutoff = 250
    normalize_factor = 'equation'

    ###########
    build_sheet =  Build_list.Build(os.path.join('/host/d/Data/low_dose_CT/Patient_lists/mayo_low_dose_CT_distill_v2.xlsx'))
    _, patient_id_list, random_num_list, noise_file_all_list, noise_file_odd_list, noise_file_even_list, ground_truth_file_list, _,_,_ = build_sheet.__build__(batch_list = ['test'], distill=True)
    print('total cases:', patient_id_list.shape[0])
    n = ff.get_X_numbers_in_interval(total_number = patient_id_list.shape[0],start_number = 0,end_number = 1, interval = 1)
    print('total number:', n.shape[0])

    model = ddpm.Unet(
        problem_dimension = '2D',
        init_dim = 64,
        out_dim = 1,
        channels = 1, 
        conditional_diffusion = True,
        condition_channels = 1,
        downsample_list = (True, True, True, False),
        upsample_list = (True, True, True, False),
        full_attn = (None, None, False, True),)


    diffusion_model = ddpm.GaussianDiffusion(
        model,
        image_size = image_size,
        timesteps = 1000,           # number of steps
        sampling_timesteps = sampling_timesteps,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        ddim_sampling_eta = 1.,
        force_ddim = False,
        auto_normalize=False,
        objective = objective,
        clip_or_not = True, 
        clip_range = [-1,1], )


    G = Generator.Dataset_2D
    for i in range(0,n.shape[0]):
        patient_id, random_num,noise_file_all, noise_file_odd, noise_file_even, gt_file = patient_id_list[n[i]], random_num_list[n[i]], noise_file_all_list[n[i]], noise_file_odd_list[n[i]], noise_file_even_list[n[i]], ground_truth_file_list[n[i]]
        
        if input_condition == 'both':
            condition_files = [noise_file_odd, noise_file_even]
        elif input_condition == 'odd':
            condition_files = [noise_file_odd]
        elif input_condition == 'even':
            condition_files = [noise_file_even]
        elif input_condition == 'all':
            condition_files = [noise_file_all]

        if len(condition_files) == 2:
            condition_names = ['odd','even']

        print(i,patient_id, random_num)

        # get the condition image (noise odd)
        affine = nb.load(condition_files[0]).affine
        condition_img = nb.load(condition_files[0]).get_fdata()
        if args.slice_range != "all":
            slice_start, slice_end = args.slice_range.split('-')
            slice_start, slice_end = int(slice_start), int(slice_end)
        else:
            slice_start, slice_end = 0, condition_img.shape[2]
        condition_img = condition_img[:,:,slice_start:slice_end]
            
        slice_num = condition_img.shape[2]; print('slice num:', slice_num)

        # get ground truth image
        gt_img = nb.load(gt_file).get_fdata()[:,:,slice_start:slice_end]

        if do_pred_or_avg == 'pred':
            iteration_num = 20 if supervision == 'unsupervised' else 1

            for iteration in range(1, iteration_num + 1):
                print('iteration:', iteration)

                # make folders
                save_folder_case = os.path.join(save_folder, patient_id, 'random_' + str(random_num), 'epoch' + str(epoch)+'_' + str(iteration))
                ff.make_folder([os.path.join(save_folder, patient_id), os.path.join(save_folder, patient_id, 'random_' + str(random_num)), save_folder_case])

                if os.path.isfile(os.path.join(save_folder_case, 'pred_img.nii.gz')):
                    print('already done')
                    continue
                
                for condition_i in range(0,len(condition_files)):

                    condition_file = condition_files[condition_i]
                    print('condition file:', condition_file)

                    # generator
                    generator = G(
                        supervision = supervision,

                        img_list = np.array([condition_file]), # this is a dummy, we do not use it
                        condition_list = np.array([condition_file]),
                        image_size = image_size,

                        num_slices_per_image = slice_num,
                        random_pick_slice = False,
                        slice_range = None if args.slice_range is None else [slice_start, slice_end],

                        histogram_equalization = histogram_equalization,
                        bins = None if histogram_equalization == False else np.load('/host/d/Github/Diffusion_denoising_thin_slice/help_data/histogram_equalization/bins_lowdoseCT.npy'),
                        bins_mapped = None if histogram_equalization == False else np.load('/host/d/Github/Diffusion_denoising_thin_slice/help_data/histogram_equalization/bins_mapped_lowdoseCT.npy'),
                        background_cutoff = background_cutoff,
                        maximum_cutoff = maximum_cutoff,
                        normalize_factor = normalize_factor,)

                    # sample:
                    sampler = ddpm.Sampler(diffusion_model,generator,batch_size = 1)

                    pred_img = sampler.sample_2D(trained_model_filename, condition_img)
                    print(pred_img.shape)
        
                    # save
                    if len(condition_files) == 1:
                        nb.save(nb.Nifti1Image(pred_img, affine), os.path.join(save_folder_case, 'pred_img.nii.gz'))
                    else:
                        nb.save(nb.Nifti1Image(pred_img, affine), os.path.join(save_folder_case, 'pred_img_' + condition_names[condition_i] + '.nii.gz'))

                if len(condition_files) == 2:
                    pred_img_final = np.zeros([len(condition_files), pred_img.shape[0], pred_img.shape[1], pred_img.shape[2]])
                    for condition_i in range(0,len(condition_files)):
                        pred_img_final[condition_i,:,:,:] = nb.load(os.path.join(save_folder_case, 'pred_img_' + condition_names[condition_i] + '.nii.gz')).get_fdata()
                    # average the two conditions
                    pred_img_final = np.mean(pred_img_final, axis = 0)
                    assert pred_img_final.shape == pred_img.shape
                    nb.save(nb.Nifti1Image(pred_img_final, affine), os.path.join(save_folder_case, 'pred_img.nii.gz'))


                if iteration == 1:
                    nb.save(nb.Nifti1Image(gt_img, affine), os.path.join(save_folder_case, 'gt_img.nii.gz'))
                    nb.save(nb.Nifti1Image(condition_img, affine), os.path.join(save_folder_case, 'condition_img.nii.gz'))
        

        if do_pred_or_avg == 'avg':

            save_folder_avg = os.path.join(save_folder, patient_id, 'random_' + str(random_num), 'epoch' + str(epoch)+'avg')
            ff.make_folder([os.path.join(save_folder, patient_id), os.path.join(save_folder, patient_id, 'random_' + str(random_num)), save_folder_avg])

            if os.path.isfile(os.path.join(save_folder_avg, 'pred_img_scans20.nii.gz')):
                print('already done')
                continue
            
            made_predicts = ff.sort_timeframe(ff.find_all_target_files(['epoch' + str(epoch)+'_*'], os.path.join(save_folder, patient_id,  'random_' + str(random_num))),0,'_','/')
            if len(made_predicts) == 0:
                print('skip, no made predicts')
                continue

            total_predicts = 0
            for jj in range(len(made_predicts)):
                total_predicts += os.path.isfile(os.path.join(made_predicts[jj],'pred_img.nii.gz'))
            if total_predicts != 20:
                print('skip, not enough predicts')
                continue

            loaded_data = np.zeros((condition_img.shape[0], condition_img.shape[1], condition_img.shape[2], total_predicts))
            for j in range(total_predicts):
                loaded_data[:,:,:,j] = nb.load(os.path.join(made_predicts[j],'pred_img.nii.gz')).get_fdata()

            for avg_num in [6,10,20]:#range(1,total_predicts+1):
                print('avg_num:', avg_num)
                predicts_avg = np.zeros((condition_img.shape[0], condition_img.shape[1], condition_img.shape[2], avg_num))
                print('predict_num:', avg_num)
                for j in range(avg_num):
                    print('file:', made_predicts[j])
                    predicts_avg[:,:,:,j] = loaded_data[:,:,:,j]
                # average across last axis
                predicts_avg = np.mean(predicts_avg, axis = -1)
                nb.save(nb.Nifti1Image(predicts_avg, affine), os.path.join(save_folder_avg, 'pred_img_scans' + str(avg_num) + '.nii.gz'))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    run(args)