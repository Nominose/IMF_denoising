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
import Diffusion_denoising_thin_slice.Generator_EM as Generator_EM
import Diffusion_denoising_thin_slice.Data_processing as Data_processing

def get_args_parser():
    parser = argparse.ArgumentParser('Diffusion Inference Script')

    parser.add_argument('--trial_name', type=str, required=True,
                        help='trial name such as noise2noise')
    
    parser.add_argument('--epoch', type=int, required=True,
                        help='epoch number of the model')
    
    parser.add_argument('--slice_range', type=str, default="all",
                        help='slice range such as 100-200 or None for all slices')

    parser.add_argument('--noise_type', type=str, default='gaussian', choices=['gaussian', 'poisson'],
                        help='type of noise added in the simulation')
        

    return parser

#######################

def run(args):
    trial_name = args.trial_name
    epoch = args.epoch

    study_folder = '/host/d/projects/denoising/models'

    trained_model_filename = os.path.join(study_folder,trial_name, 'models/model-' + str(epoch)+ '.pt')
    save_folder = os.path.join(study_folder, trial_name, 'pred_images')
    
    os.makedirs(save_folder, exist_ok=True)

    histogram_equalization = False
    background_cutoff = 0
    maximum_cutoff = 1
    normalize_factor = 'equation'
    final_max = 1
    final_min = -1

    #######################
    build_sheet =  Build_list.Build_EM(os.path.join('/host/d/Data/minnie_EM/Patient_lists/minnie_EM_split_gaussian_simulation_v1.xlsx'))

    # define train patient list
    batch_list, patient_id_list, _,_,simulation_file_1_list, simulation_file_2_list, ground_truth_file_list, slice_num_list = build_sheet.__build__(batch_list = ['test'])
    n = ff.get_X_numbers_in_interval(total_number = patient_id_list.shape[0],start_number = 0,end_number = 1, interval = 1)
    print('total number:', n.shape[0])


    # build model
    model = noise2noise.Unet(
        problem_dimension = '2D',  # '2D' or '3D'
        input_channels = 1,
        out_channels = 1,  
        initial_dim = 16,  # initial feature dimension after first conv layer
        dim_mults = (2,4,8,16),
        full_attn_paths = (None, None, False, True), # these are for downsampling and upsampling paths
        full_attn_bottleneck = True, # this is for the middle bottleneck layer
        act = 'ReLU',)

    # main
    G =  Generator_EM.Dataset_2D
    for i in range(0,2):#n.shape[0]):
        patient_id = patient_id_list[n[i]]
        simulation_file_1 = simulation_file_1_list[n[i]]
        simulation_file_2 = simulation_file_2_list[n[i]]
        gt_file = ground_truth_file_list[n[i]]

        condition_file = simulation_file_2
    
        print(i,patient_id)

        # get the condition image 
        affine = nb.load(condition_file).affine
        condition_img = nb.load(condition_file).get_fdata()
        if args.slice_range != "all":
            slice_start, slice_end = args.slice_range.split('-')
            slice_start, slice_end = int(slice_start), int(slice_end)
        else:
            slice_start, slice_end = 0, condition_img.shape[2]
        condition_img = condition_img[:,:,slice_start:slice_end]
            
        slice_num = condition_img.shape[2]

        # this image shape (x and y should be divisible by 16)
        x_shape, y_shape = condition_img.shape[0], condition_img.shape[1]
        target_x_shape = (x_shape //16 ) *16
        target_y_shape = (y_shape //16 ) *16
        print('original image size:', x_shape, y_shape, '; target image size:', target_x_shape, target_y_shape)

        # get ground truth image
        gt_img = nb.load(gt_file).get_fdata()
        gt_img = gt_img[:,:,slice_start:slice_end]

        # define model for each case
        image_size = [target_x_shape, target_y_shape] # this image size is different for each case

        # make folders
        save_folder_case = os.path.join(save_folder, patient_id, 'epoch' + str(epoch))
        ff.make_folder([os.path.join(save_folder, patient_id), save_folder_case])

        # if os.path.isfile(os.path.join(save_folder_case, 'pred_img.nii.gz')):
        #     print('already done')
        #     continue
                

        # generator
        generator = G(
            gt_file_list = np.asarray([simulation_file_2]), # this is dummy, not used in inference
            simulation_1_file_list = np.asarray([simulation_file_2]), # this is dummy, not used in inference
            simulation_2_file_list = np.asarray([simulation_file_2]),

            image_size = [target_x_shape, target_y_shape],

            num_slices_per_image = slice_num,
            random_pick_slice = False,
            slice_range = None if args.slice_range is None else [slice_start, slice_end],

            background_cutoff = background_cutoff, 
            maximum_cutoff = maximum_cutoff,
            normalize_factor = normalize_factor,
            final_max = final_max,
            final_min = final_min,)


        # sample:
        sampler = noise2noise.Sampler(model,generator,batch_size = 1, image_size = image_size)
        need_denormalize = False if final_min == 0 else True
        pred_img = sampler.sample_2D(trained_model_filename, condition_img, need_denormalize = need_denormalize)

        print(pred_img.shape)

        nb.save(nb.Nifti1Image(pred_img, affine), os.path.join(save_folder_case, 'pred_img.nii.gz'))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    run(args)