import sys
sys.path.append('/host/d/Github')
import os
import torch
import argparse
import numpy as np
import nibabel as nb
import Diffusion_denoising_thin_slice.noise2noise.model as noise2noise
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.Generator as Generator


def get_args_parser():
    parser = argparse.ArgumentParser('Diffusion Inference Script')

    parser.add_argument('--trial_name', type=str, required=True,
                        help='trial name such as noise2noise')
    
    parser.add_argument('--epoch', type=int, required=True,
                        help='epoch number of the model')

    
    parser.add_argument('--input', type=str, default='both', choices=['both', 'odd', 'even', 'all'],
                        help='input condition: both, odd, even, all')
    
    parser.add_argument('--slice_range', type=str, default="all",
                        help='slice range such as 100-200 or None for all slices')

    parser.add_argument('--noise_type', type=str, default='gaussian', choices=['gaussian', 'poisson'],
                        help='type of noise added in the simulation')
        

    return parser

#######################

def run(args):
    trial_name = args.trial_name
    epoch = args.epoch
    input_condition = args.input  #'both', 'odd', 'even', 'all'
    print('noise type:', args.noise_type)

    study_folder = '/host/d/projects/denoising/models'

    trained_model_filename = os.path.join(study_folder,trial_name, 'models/model-' + str(epoch)+ '.pt')
    save_folder = os.path.join(study_folder, trial_name, 'pred_images_input_'+ input_condition) 
    
    os.makedirs(save_folder, exist_ok=True)

    image_size = [512,512]

    histogram_equalization = False
    background_cutoff = -200
    maximum_cutoff = 250
    normalize_factor = 'equation'
    print('background cutoff:', background_cutoff, '; maximum cutoff:', maximum_cutoff, '; normalize factor:', normalize_factor)
    #######################
    if args.noise_type == 'gaussian':
        build_sheet_v2 =  Build_list.Build(os.path.join('/host/d/Data/low_dose_CT/Patient_lists/mayo_low_dose_CT_gaussian_simulation_v2.xlsx'))
        build_sheet_v3 = Build_list.Build(os.path.join('/host/d/Data/low_dose_CT/Patient_lists/mayo_low_dose_CT_gaussian_simulation_v3.xlsx'))
        batch_list, patient_id_list, random_num_list,noise_file_all_list, noise_file_odd_list, _, ground_truth_file_list, slice_num_list = build_sheet_v2.__build__(batch_list = ['train','val','test'])
        _, _, _, _, _,noise_file_even_list, _, _ = build_sheet_v3.__build__(batch_list = ['train','val','test'])

    elif args.noise_type == 'poisson':
        build_sheet =  Build_list.Build(os.path.join('/host/d/Data/low_dose_CT/Patient_lists/mayo_low_dose_CT_poisson_simulation_v2.xlsx'))
        batch_list, patient_id_list, random_num_list, noise_file_all_list, noise_file_odd_list, noise_file_even_list, ground_truth_file_list, slice_num_list = build_sheet.__build__(batch_list = ['train','val','test'])

    print('example of noise file all:', noise_file_all_list[0])
    n = ff.get_X_numbers_in_interval(total_number = patient_id_list.shape[0],start_number = 0,end_number = 1, interval = 1)
    print('total number:', n.shape[0])


    # build model
    model = noise2noise.Unet(
        problem_dimension = '2D',  # '2D' or '3D'
        input_channels = 1,
        out_channels = 1,  
        initial_dim = 16,  # initial feature dimension after first conv layer
        dim_mults = (2,4,8,16),
        full_attn_paths = (None, None, False,True), # these are for downsampling and upsampling paths
        full_attn_bottleneck = True, # this is for the middle bottleneck layer
        act = 'ReLU',)

    # main
    G = Generator.Dataset_2D
    for i in range(0,n.shape[0]):
        patient_id = patient_id_list[n[i]]
        random_num = random_num_list[n[i]]
        noise_file_all = noise_file_all_list[n[i]]
        noise_file_odd = noise_file_odd_list[n[i]]
        noise_file_even = noise_file_even_list[n[i]]
        gt_file = ground_truth_file_list[n[i]]

        # here we only use noise odd as condition
        if input_condition == 'all':
            condition_files =[noise_file_all]# [noise_file_odd, noise_file_even]  # can be one or two condition files
        elif input_condition =='both':
            condition_files = [noise_file_odd, noise_file_even]
        elif input_condition == 'odd':
            condition_files = [noise_file_odd]
        elif input_condition == 'even':
            condition_files = [noise_file_even]
        print('condition files:', condition_files)

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
        slice_num = condition_img.shape[2]
        print('slice num:', slice_num)

        # get ground truth image
        gt_img = nb.load(gt_file).get_fdata()[:,:,slice_start:slice_end]

        # make folders
        save_folder_case = os.path.join(save_folder, patient_id, 'random_' + str(random_num), 'epoch' + str(epoch))
        ff.make_folder([os.path.join(save_folder, patient_id), os.path.join(save_folder, patient_id, 'random_' + str(random_num)), save_folder_case])

        if os.path.isfile(os.path.join(save_folder_case, 'pred_img.nii.gz')):
            print('already done')
            continue
                
        for condition_i in range(0,len(condition_files)):

            condition_file = condition_files[condition_i]
            print('condition file:', condition_file)

            # generator
            generator = G(
                supervision = 'unsupervised',

                img_list = np.array([condition_file]), # this is a dummy, we do not use it
                condition_list = np.array([condition_file]),
                image_size = image_size,

                num_slices_per_image = 100,#slice_num, 
                random_pick_slice = False,
                slice_range = [100,200],# None,
                
                histogram_equalization = histogram_equalization,
                bins = None if histogram_equalization == False else np.load('/host/d/Github/Diffusion_denoising_thin_slice/help_data/histogram_equalization/bins_lowdoseCT.npy'),
                bins_mapped = None if histogram_equalization == False else np.load('/host/d/Github/Diffusion_denoising_thin_slice/help_data/histogram_equalization/bins_mapped_lowdoseCT.npy'),
                background_cutoff = background_cutoff,
                maximum_cutoff = maximum_cutoff,
                normalize_factor = normalize_factor,)

            # sample:
            sampler = noise2noise.Sampler(model,generator,batch_size = 1, image_size = image_size)

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

        # save condition
        # nb.save(nb.Nifti1Image(condition_img, affine),  os.path.join(save_folder_case, 'condition_img.nii.gz'))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    run(args)