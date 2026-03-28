import sys
sys.path.append('/gpfs/work/aac/xingyiyao23/Code')
import os
import torch
import numpy as np 
import nibabel as nb
import Diffusion_denoising_thin_slice.Thinslice_experiments.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion as ddpm
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.Generator_thinslice as Generator 

###########
trial_name = 'supervised_gaussian_brainCT'
problem_dimension = '2D'
supervision = 'supervised' if trial_name[0:2] == 'su' else 'unsupervised'; print('supervision:', supervision)

epoch = 75
trained_model_filename = os.path.join('/gpfs/work/aac/xingyiyao23/results', trial_name, 'models/model-' + str(epoch)+ '.pt')
save_folder = os.path.join('/gpfs/work/aac/xingyiyao23/results', trial_name, 'pred_images'); os.makedirs(save_folder, exist_ok=True)

# bias 
beta = 0

# model condition 
# if 'mean' in trial_name: condition on current slice, target the mean of neighboring slices
# else: condition on neighboring slices, target the current slice
condition_channel = 1 if (supervision == 'supervised') or ('mean' in trial_name) else 2
# target = 'mean' if 'mean' in trial_name else 'current'

image_size = [512,512] 
objective = 'pred_x0'
sampling_timesteps = 100

histogram_equalization = True
background_cutoff = -1000
maximum_cutoff = 2000
normalize_factor = 'equation'
clip_range = [-1,1]

do_pred_or_avg = 'pred'

###########
build_sheet =  Build_list.Build_thinsliceCT(os.path.join('/gpfs/work/aac/xingyiyao23/Data/brain_CT/Patient_lists/fixedCT_static_simulation_train_test_gaussian_xjtlu.xlsx'))
_,patient_id_list,patient_subid_list,random_num_list, condition_list, x0_list = build_sheet.__build__(batch_list = [5]) 
print('total cases:', patient_id_list.shape[0])
n = ff.get_X_numbers_in_interval(total_number = patient_id_list.shape[0],start_number = 0,end_number = 1, interval = 2)
print('total number:', n.shape[0])
# x0_list = x0_list[0:1]; condition_list = condition_list[0:1]

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

for i in range(0,n.shape[0]):
    patient_id = patient_id_list[n[i]]
    patient_subid = patient_subid_list[n[i]]
    random_num = random_num_list[n[i]]
    x0_file = x0_list[n[i]]
    condition_file = condition_list[n[i]]

    print(i,patient_id, patient_subid, random_num)

    if do_pred_or_avg == 'pred':
        # get the ground truth image
        gt_img = nb.load(x0_file)
        print('x0_file:', x0_file, 'shape:', gt_img.get_fdata().shape)
        affine = gt_img.affine; gt_img = gt_img.get_fdata()[:,:,30:80]

        # get the condition image
        print('condition_file:', condition_file, 'shape: ', nb.load(condition_file).get_fdata().shape)
        condition_img = nb.load(condition_file).get_fdata()[:,:,30:80]

        for iteration in range(1,2):#21):
            print('iteration:', iteration)

            # make folders
            ff.make_folder([os.path.join(save_folder, patient_id), os.path.join(save_folder, patient_id, patient_subid), os.path.join(save_folder, patient_id, patient_subid, 'random_' + str(random_num))])
            save_folder_case = os.path.join(save_folder, patient_id, patient_subid, 'random_' + str(random_num), 'epoch' + str(epoch)+'_'+str(iteration)); os.makedirs(save_folder_case, exist_ok=True)


            if os.path.isfile(os.path.join(save_folder_case, 'pred_img.nii.gz')):
                print('already done')
                continue

            # generator
            generator = Generator.Dataset_2D(
                supervision = supervision,

                img_list = np.array([x0_file]),
                condition_list = np.array([condition_file]),
                image_size = image_size,

                num_slices_per_image = 50, 
                random_pick_slice = False,
                slice_range = [30,80],

                histogram_equalization = histogram_equalization,
                bins = np.load('/gpfs/work/aac/xingyiyao23/Code/Diffusion_denoising_thin_slice/help_data/histogram_equalization/bins.npy'),
                bins_mapped = np.load('/gpfs/work/aac/xingyiyao23/Code/Diffusion_denoising_thin_slice/help_data/histogram_equalization/bins_mapped.npy'),
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
                nb.save(nb.Nifti1Image(gt_img, affine), os.path.join(save_folder_case, 'gt_img.nii.gz'))
                nb.save(nb.Nifti1Image(condition_img, affine), os.path.join(save_folder_case, 'condition_img.nii.gz'))
       

    if do_pred_or_avg == 'avg':

        save_folder_avg = os.path.join(save_folder, patient_id, patient_subid, 'random_' + str(random_num), 'epoch' + str(epoch)+'avg'); os.makedirs(save_folder_avg, exist_ok=True)

        # if os.path.isfile(os.path.join(save_folder_avg, 'pred_img_scans20.nii.gz')):
        #     print('already done')
        #     continue

        # get the ground truth image
        gt_img = nb.load(x0_file)
        print('x0_file:', x0_file, 'shape:', gt_img.get_fdata().shape)
        affine = gt_img.affine; gt_img = gt_img.get_fdata()[:,:,30:80]

        
        made_predicts = ff.sort_timeframe(ff.find_all_target_files(['epoch' + str(epoch)+'_*'], os.path.join(save_folder, patient_id, patient_subid, 'random_' + str(random_num))),0,'_','/')
        if len(made_predicts) == 0:
            print('skip, no made predicts')
            continue
        total_predicts = 0
        for jj in range(len(made_predicts)):
            total_predicts += os.path.isfile(os.path.join(made_predicts[jj],'pred_img.nii.gz'))
        print('total made predicts:', total_predicts)
        if total_predicts != 20:
            print('skip, not enough predicts')
            continue

        loaded_data = np.zeros((gt_img.shape[0], gt_img.shape[1], gt_img.shape[2], total_predicts))
        for j in range(total_predicts):
            loaded_data[:,:,:,j] = nb.load(os.path.join(made_predicts[j],'pred_img.nii.gz')).get_fdata()

        for avg_num in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
            if os.path.isfile(os.path.join(save_folder_avg, 'pred_img_scans' + str(avg_num) + '.nii.gz')):
                print('already done avg num:', avg_num)
                continue
            print('avg_num:', avg_num)
            predicts_avg = np.zeros((gt_img.shape[0], gt_img.shape[1], gt_img.shape[2], avg_num))
            print('predict_num:', avg_num)
            for j in range(avg_num):
                print('file:', made_predicts[j])
                predicts_avg[:,:,:,j] = loaded_data[:,:,:,j]
            # average across last axis
            predicts_avg = np.mean(predicts_avg, axis = -1)
            nb.save(nb.Nifti1Image(predicts_avg, affine), os.path.join(save_folder_avg, 'pred_img_scans' + str(avg_num) + '.nii.gz'))