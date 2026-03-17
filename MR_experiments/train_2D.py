import sys 
sys.path.append('/host/d/Github')
import os
import torch
import numpy as np 
import nibabel as nb
import Diffusion_denoising_thin_slice.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion as ddpm
import Diffusion_denoising_thin_slice.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_EDM as edm
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.Generator_MR as Generator_MR

trial_name = 'unsupervised_MR_try2'
problem_dimension = '2D'
supervision = 'supervised' if trial_name[0:2] == 'su' else 'unsupervised'

preload = True

# bias  
beta = 0
lpips_weight = 0#0.2
edge_weight = 0#0.05

# model condition 
condition_channel = 1 
train_batch_size = 3
objective = 'pred_x0' #if 'noise' not in trial_name else 'pred_noise'

pre_trained_model =  None#os.path.join('/host/d/projects/denoising/models', trial_name, 'models/model-225.pt') #None
start_step = 0#225

# image condition
image_size = [640,320]
num_patches_per_slice = 2
patch_size = [320,320]#[128,128]

histogram_equalization = False
background_cutoff =  2.5e-06
maximum_cutoff = 5.5e-05# 0.00015
normalize_factor = 'equation'

######Patient list
# define train
if supervision == 'supervised':
    build_sheet =  Build_list.Build(os.path.join('/host/d/Data/NYU_MR/Patient_lists/NYU_MR_simulation_undersample4_equispaced.xlsx'))
else:
    build_sheet =  Build_list.Build(os.path.join('/host/d/Data/NYU_MR/Patient_lists/NYU_MR_simulation.xlsx'))

# define train patient list
_, _, _, noise_file_all_list_train, noise_file_odd_list_train, noise_file_even_list_train, gt_file_list_train, slice_num_list_train = build_sheet.__build__(batch_list = ['train']) 
# noise_file_all_list_train = noise_file_all_list_train[25:26]
# noise_file_odd_list_train = noise_file_odd_list_train[25:26]
# noise_file_even_list_train = noise_file_even_list_train[25:26]
# gt_file_list_train = gt_file_list_train[25:26]
# slice_num_list_train = slice_num_list_train[25:26]

# define val patient list
_, _, _,  noise_file_all_list_val, noise_file_odd_list_val, noise_file_even_list_val,  gt_file_list_val, slice_num_list_val = build_sheet.__build__(batch_list = ['val'])

print('number of training cases:', gt_file_list_train.shape[0], '; number of validation cases:', gt_file_list_val.shape[0], ' example of noise_odd_list_train:', noise_file_odd_list_train[0])

######Define models
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
    image_size = image_size if num_patches_per_slice == None else patch_size,
    timesteps = 1000,
    sampling_timesteps = 250,
    objective = objective,
    clip_or_not =False,
    auto_normalize = False,)

# ######Define data generator
# # first we define the x0 and condition list 
if supervision == 'supervised':
    x0_list_train,condition_list_train = gt_file_list_train,noise_file_odd_list_train
    x0_list_val, condition_list_val = gt_file_list_val, noise_file_odd_list_val
elif supervision == 'unsupervised':
    x0_list_train, condition_list_train = noise_file_even_list_train, noise_file_odd_list_train
    x0_list_val, condition_list_val = noise_file_even_list_val, noise_file_odd_list_val
print('example x0 train file:', x0_list_train[0], '; example condition file:', condition_list_train[0])
print('example x0 val file:', x0_list_val[0], '; example condition file:', condition_list_val[0])

# preload_data if needed
if preload  == True:
    print('loading!!')
    x0_data_train, condition_data_train = ff.preload_data(x0_list_train, transpose = True), ff.preload_data(condition_list_train, transpose = True)
    x0_data_val, condition_data_val = ff.preload_data(x0_list_val,transpose = True), ff.preload_data(condition_list_val, transpose = True)

# data generator
G =  Generator_MR.Dataset_2D
generator_train = G(
        supervision = supervision,

        preload = preload,
        preload_data = (x0_data_train, condition_data_train) if preload == True else None,

        img_list = x0_list_train,
        condition_list = condition_list_train,
        image_size = image_size,

        num_slices_per_image = 30, # no matter how many slices, we only randomly pick 50 slices each time
        random_pick_slice = True,
        slice_range = None,

        background_cutoff = background_cutoff,
        maximum_cutoff = maximum_cutoff,
        normalize_factor = normalize_factor,

        num_patches_per_slice = num_patches_per_slice,
        patch_size = patch_size,

        shuffle = True,
        augment = True,
        augment_frequency = 0.5,

        switch_odd_and_even_frequency = -1 if (supervision == 'supervised' ) else 0.5, 
        )

generator_val = G(
        supervision = supervision,

        preload = preload,
        preload_data = (x0_data_val, condition_data_val) if preload == True else None,

        img_list = x0_list_val,
        condition_list = condition_list_val,
        image_size = image_size,

        background_cutoff = background_cutoff,
        maximum_cutoff = maximum_cutoff,
        normalize_factor = normalize_factor,

        num_slices_per_image = 20,
        random_pick_slice = False,
        slice_range = [10,30],

        num_patches_per_slice = 1,
        patch_size = image_size,)

# #######Start to train
# define a saved model folder
save_models_folder = os.path.join('/host/d/projects/denoising/models', trial_name, 'models');ff.make_folder([os.path.dirname(save_models_folder), save_models_folder])
trainer = ddpm.Trainer(
    diffusion_model= diffusion_model,
    generator_train = generator_train,
    generator_val = generator_val,
    train_batch_size = train_batch_size,
    
    accum_iter = 1,
    train_num_steps = 1000, # total training epochs
    results_folder = save_models_folder,
   
    train_lr = 1e-4,
    train_lr_decay_every = 200, 
    save_models_every = 10,
    validation_every = 10,)


trainer.train(pre_trained_model=pre_trained_model, start_step= start_step, beta = beta, lpips_weight = lpips_weight, edge_weight = edge_weight)