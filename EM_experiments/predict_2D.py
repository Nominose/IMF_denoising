import sys 
sys.path.append('/host/d/Github')
import argparse
import os
import torch
import numpy as np 
import nibabel as nb
import Diffusion_denoising_thin_slice.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion as ddpm
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.Generator_EM as Generator_EM
import Diffusion_denoising_thin_slice.Data_processing as Data_processing

def get_args_parser():
    parser = argparse.ArgumentParser('Diffusion Inference Script')

    parser.add_argument('--trial_name', type=str, required=True,
                        help='trial name such as unsupervised_MR')

    parser.add_argument('--epoch', type=int, required=True,
                        help='epoch number of the model')

    parser.add_argument('--mode', type=str, required = True, 
                        help='predict mode: avg or pred')
    
    parser.add_argument('--finalmax_finalmin', type=str, default="1_-1",
                        help='final max and min values for clipping, such as 1_-1')
    
    parser.add_argument('--slice_range', type=str, default="all",
                        help='slice range such as 100-200 or None for all slices')
        

    return parser

###########

def run(args):
    trial_name = args.trial_name
    epoch = args.epoch
    do_pred_or_avg = args.mode  #'avg' #'pred'

    supervision = 'supervised' if trial_name[0:2] == 'su' else 'unsupervised'; print('supervision:', supervision)

    study_folder = '/host/d/projects/denoising/models'
    trained_model_filename = os.path.join(study_folder,trial_name, 'models/model-' + str(epoch)+ '.pt')
    save_folder = os.path.join(study_folder, trial_name, 'pred_images'); os.makedirs(save_folder, exist_ok=True)

    objective = 'pred_x0' 
    sampling_timesteps = 50 # 100

    histogram_equalization = False
    background_cutoff =  0
    maximum_cutoff = 1
    final_max = args.finalmax_finalmin.split('_')[0]
    final_max = int(final_max)
    final_min = args.finalmax_finalmin.split('_')[1]
    final_min = int(final_min)
    normalize_factor = 'equation'

    ###########
    build_sheet =  Build_list.Build_EM(os.path.join('/host/d/Data/minnie_EM/Patient_lists/minnie_EM_split_gaussian_simulation_v1.xlsx'))

    # define train patient list
    batch_list, patient_id_list, _,_,simulation_file_1_list, simulation_file_2_list, ground_truth_file_list, slice_num_list = build_sheet.__build__(batch_list = ['test'])
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
            clip_range = [final_min, final_max],)

        if do_pred_or_avg == 'pred':
            iteration_num = 40 if supervision == 'unsupervised' else 1

            for iteration in range(1,iteration_num + 1):
                print('iteration:', iteration)

                # make folders
                save_folder_case = os.path.join(save_folder, patient_id, 'epoch' + str(epoch)+'_' + str(iteration))
                ff.make_folder([os.path.join(save_folder, patient_id), save_folder_case])

                if os.path.isfile(os.path.join(save_folder_case, 'pred_img.nii.gz')):
                    print('already done')
                    continue
    
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

                #sample:
                sampler = ddpm.Sampler(diffusion_model,generator,batch_size = 1)
                need_denormalize = False if final_min == 0 else True
                pred_img = sampler.sample_2D(trained_model_filename, condition_img, need_change_dim = True, need_denormalize = need_denormalize)
        
                # save
                nb.save(nb.Nifti1Image(pred_img, affine), os.path.join(save_folder_case, 'pred_img.nii.gz'))
                if iteration == 1:
                    nb.save(nb.Nifti1Image(gt_img, affine), os.path.join(save_folder_case, 'gt_img.nii.gz'))
                    nb.save(nb.Nifti1Image(condition_img, affine), os.path.join(save_folder_case, 'condition_img.nii.gz'))
        

        if do_pred_or_avg == 'avg':

            save_folder_avg = os.path.join(save_folder, patient_id, 'epoch' + str(epoch)+'avg')
            ff.make_folder([os.path.join(save_folder, patient_id), os.path.join(save_folder, patient_id), save_folder_avg])

            # if os.path.isfile(os.path.join(save_folder_avg, 'pred_img_scans20.nii.gz')):
            #     print('already done')
            #     continue
            
            made_predicts = ff.sort_timeframe(ff.find_all_target_files(['epoch' + str(epoch)+'_*'], os.path.join(save_folder, patient_id)),0,'_','/')
            if len(made_predicts) == 0:
                print('skip, no made predicts')
                continue

            total_predicts = 0
            for jj in range(len(made_predicts)):
                total_predicts += os.path.isfile(os.path.join(made_predicts[jj],'pred_img.nii.gz'))
            if total_predicts < 10:
                print('skip, not enough predicts')
                continue

            loaded_data = np.zeros((condition_img.shape[0], condition_img.shape[1], condition_img.shape[2], total_predicts))
            for j in range(total_predicts):
                loaded_data[:,:,:,j] = nb.load(os.path.join(made_predicts[j],'pred_img.nii.gz')).get_fdata()

            for avg_num in [40]:#range(1,total_predicts+1):
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