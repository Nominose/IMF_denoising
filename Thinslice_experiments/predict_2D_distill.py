import sys
sys.path.append('/host/d/Github')
import os
import torch
import numpy as np 
import nibabel as nb
import Diffusion_denoising_thin_slice.Thinslice_experiments.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion as ddpm
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.Generator_thinslice as Generator

###########
trial_name = 'distill_brainCT'
problem_dimension = '2D'
supervision = 'supervised'

epoch = 164
trained_model_filename = os.path.join('/host/d/projects/denoising/models', trial_name, 'models/model-' + str(epoch)+ '.pt')
if os.path.isfile(trained_model_filename) ==0:
    print('no model')
save_folder = os.path.join('/host/d/projects/denoising/models', trial_name, 'pred_images'); os.makedirs(save_folder, exist_ok=True)

# bias 
beta = 0

# model condition 
condition_channel = 2

image_size = [512,512] 
objective = 'pred_x0'
sampling_timesteps = 100

histogram_equalization = True
background_cutoff = -1000
maximum_cutoff = 2000
normalize_factor = 'equation'
clip_range = [-1,1]


###########
build_sheet =  Build_list.Build_thinsliceCT(os.path.join('/host/d/Data/brain_CT/Patient_lists/fixedCT_static_simulation_train_test_gaussian_xjtlu.xlsx'))
_,patient_id_list,patient_subid_list,random_num_list, condition_list, x0_list = build_sheet.__build__(batch_list = [5]) 
n = ff.get_X_numbers_in_interval(total_number = patient_id_list.shape[0],start_number = 0,end_number = 1, interval = 2)
print('total number:', n.shape[0])

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

   
    # get the ground truth image
    gt_img = nb.load(x0_file)
    print('x0_file:', x0_file, 'shape:', gt_img.get_fdata().shape)
    affine = gt_img.affine; gt_img = gt_img.get_fdata()[:,:,30:80]

    # get the condition image
    print('condition_file:', condition_file, 'shape: ', nb.load(condition_file).get_fdata().shape)
    condition_img = nb.load(condition_file).get_fdata()[:,:,30:80]
    for iteration in range(1,2):
        print('iteration:', iteration)

        # make folders
        ff.make_folder([os.path.join(save_folder, patient_id), os.path.join(save_folder, patient_id, patient_subid), os.path.join(save_folder, patient_id, patient_subid, 'random_' + str(random_num))])
        save_folder_case = os.path.join(save_folder, patient_id, patient_subid, 'random_' + str(random_num), 'epoch' + str(epoch)+'_'+str(iteration)); os.makedirs(save_folder_case, exist_ok=True)


        if os.path.isfile(os.path.join(save_folder_case, 'pred_img.nii.gz')):
            print('already done')
            continue

        # generator
        generator = Generator.Dataset_2D_distilled(

            img_list = np.array([x0_file]),
            condition_list = np.array([condition_file]),
            image_size = image_size,

            num_slices_per_image = 50, 
            random_pick_slice = False,
            slice_range = [30,80],

            histogram_equalization = histogram_equalization,
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
        # nb.save(nb.Nifti1Image(condition_img, affine), os.path.join(save_folder_case, 'condition_img.nii.gz'))
        # nb.save(nb.Nifti1Image(gt_img, affine), os.path.join(save_folder_case, 'gt_img.nii.gz'))

