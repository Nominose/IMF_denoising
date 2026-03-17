import sys
sys.path.append('/host/d/Github')
import os
import torch
import argparse
import numpy as np 
import nibabel as nb
import Diffusion_denoising_thin_slice.Thinslice_experiments.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion as ddpm
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.Generator_thinslice as Generator 



def get_args_parser():
    parser = argparse.ArgumentParser('Diffusion Inference Script')

    parser.add_argument('--trial_name', type=str, required=True,
                        help='trial name such as unsupervised_gaussian')
    parser.add_argument('--epoch', type=int, required=True,
                        help='epoch number of the model')
    
    parser.add_argument('--mode', type=str, required = True, 
                        help='predict mode: avg or pred')

    
    parser.add_argument('--slice_range', type=str, default="all",
                        help='slice range such as 100-200 or None for all slices')
        

    return parser


###########
def run(args):
    trial_name = args.trial_name
    epoch = args.epoch
    do_pred_or_avg = args.mode  #'avg' #'pred'
    

    problem_dimension = '2D'
    supervision = 'supervised' if trial_name[0:2] == 'su' else 'unsupervised'; print('supervision:', supervision)

    trained_model_filename = os.path.join('/host/d/projects/denoising/models', trial_name, 'models/model-' + str(epoch)+ '.pt')
    save_folder = os.path.join('/host/d/projects/denoising/models', trial_name, 'pred_images'); os.makedirs(save_folder, exist_ok=True)

    # bias 
    beta = 0

    # model condition 
    condition_channel = 2
    # target = 'mean' if 'mean' in trial_name else 'current'

    image_size = [512,512] 
    objective = 'pred_x0'
    sampling_timesteps = 100

    histogram_equalization = False
    background_cutoff = -1000
    maximum_cutoff = 1000
    normalize_factor = 'equation'
    clip_range = [-1,1]

    ###########
    build_sheet =  Build_list.Build_thinsliceCT(os.path.join('/host/d/Data/PCCT/Patient_lists/PCCT_split.xlsx'))
    _,patient_id_list,_,_, condition_list, x0_list = build_sheet.__build__(batch_list = [2])
    x0_list = condition_list 
    print('total cases:', patient_id_list.shape[0])
  

    model = ddpm.Unet(
        problem_dimension = problem_dimension,
        init_dim = 64,
        out_dim = 1,
        channels = 1, 
        conditional_diffusion = True,
        condition_channels = condition_channel,

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
        clip_range = clip_range, )

    for i in range(0,patient_id_list.shape[0]):
        patient_id = patient_id_list[i]
        x0_file = x0_list[i]
        condition_file = condition_list[i]

        print(i,patient_id )

        # load condition image
        condition_file_load = nb.load(condition_file)
        affine = condition_file_load.affine
        condition_img = condition_file_load.get_fdata()

        print("slice_range: ", args.slice_range)
        if args.slice_range != "all":
            slice_start, slice_end = args.slice_range.split('-')
            slice_start, slice_end = int(slice_start), int(slice_end)
        else:
            slice_start, slice_end = 0, condition_img.shape[2]
        condition_img = condition_img[:,:,slice_start:slice_end]
        gt_img = condition_img # dummy
            
        slice_num = condition_img.shape[2]; print('slice num:', slice_num)

        if do_pred_or_avg == 'pred':

            # get the condition image
            for iteration in range(5,9):
                print('iteration:', iteration)

                # make folders
                ff.make_folder([os.path.join(save_folder, patient_id)])
                save_folder_case = os.path.join(save_folder, patient_id, 'epoch' + str(epoch)+'_'+str(iteration)); os.makedirs(save_folder_case, exist_ok=True)


                if os.path.isfile(os.path.join(save_folder_case, 'pred_img.nii.gz')):
                    print('already done')
                    continue

                # generator
                generator = Generator.Dataset_2D(
                    supervision = supervision,

                    img_list = np.array([x0_file]),
                    condition_list = np.array([condition_file]),
                    image_size = image_size,

                    num_slices_per_image = slice_num,
                    random_pick_slice = False,
                    slice_range = [slice_start, slice_end],

                    histogram_equalization = histogram_equalization,
                    bins = np.load('/host/d/Github/Diffusion_denoising_thin_slice/help_data/histogram_equalization/bins.npy'),
                    bins_mapped = np.load('/host/d/Github/Diffusion_denoising_thin_slice/help_data/histogram_equalization/bins_mapped.npy'),
                    background_cutoff = background_cutoff,
                    maximum_cutoff = maximum_cutoff,
                    normalize_factor = normalize_factor,)

                # sample:
                sampler = ddpm.Sampler(diffusion_model,generator,batch_size = 1)

                pred_img = sampler.sample_2D(trained_model_filename, gt_img)
                print(pred_img.shape)

                pred_img_final = pred_img
        
                # save
                nb.save(nb.Nifti1Image(pred_img_final, affine), os.path.join(save_folder_case, 'pred_img.nii.gz'))

                if iteration == 1:
                    nb.save(nb.Nifti1Image(condition_img, affine), os.path.join(save_folder_case, 'condition_img.nii.gz'))
        

        if do_pred_or_avg == 'avg':

            save_folder_avg = os.path.join(save_folder, patient_id, 'epoch' + str(epoch)+'avg'); os.makedirs(save_folder_avg, exist_ok=True)

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
            print('total made predicts:', total_predicts)
            # if total_predicts != 20:
            #     print('skip, not enough predicts')
            #     continue

            loaded_data = np.zeros((gt_img.shape[0], gt_img.shape[1], gt_img.shape[2], total_predicts))
            for j in range(total_predicts):
                loaded_data[:,:,:,j] = nb.load(os.path.join(made_predicts[j],'pred_img.nii.gz')).get_fdata()

            for avg_num in [8]:#[2,4,6,8,10,12,14,16,18,20]:#range(1,total_predicts+1):
                print('avg_num:', avg_num)
                if os.path.isfile(os.path.join(save_folder_avg, 'pred_img_scans' + str(avg_num) + '.nii.gz')):
                    print('already done')
                    continue
                predicts_avg = np.zeros((gt_img.shape[0], gt_img.shape[1], gt_img.shape[2], avg_num))
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