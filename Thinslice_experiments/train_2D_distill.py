import sys 
sys.path.append('/host/d/Github')
import os
import torch
import numpy as np 
import Diffusion_denoising_thin_slice.Thinslice_experiments.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion as ddpm
# import Diffusion_denoising_thin_slice.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_EDM as edm
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.Generator_thinslice as Generator

trial_name = 'distill_brainCT'
problem_dimension = '2D'
supervision = 'supervised'

# bias  
beta = 0
lpips_weight = 0#0.2
edge_weight = 0#0.05

# model condition 
# if 'mean' in trial_name: condition on current slice, target the mean of neighboring slices
# else: condition on neighboring slices, target the current slice
condition_channel = 2

pre_trained_model = None#os.path.join('/host/d/projects/denoising/models', trial_name, 'models/model-2640.pt') #None
start_step = 0
image_size = [512,512]
num_patches_per_slice = 2
patch_size = [128,128]

objective = 'pred_x0'

histogram_equalization = True
background_cutoff = -1000
maximum_cutoff = 2000
normalize_factor = 'equation'

###########################
# define train
build_sheet =  Build_list.Build_thinsliceCT(os.path.join('/host/d/Data/brain_CT/Patient_lists/fixedCT_static_distilled_model_train_test_xjtlu.xlsx'))
_,_,_,_, condition_list_train, x0_list_train = build_sheet.__build__(batch_list = [0,1,2,3,4,5]) 
# find out who has avg20 
index_list = []
for i in range(x0_list_train.shape[0]):
    if os.path.isfile(x0_list_train[i]):
        index_list.append(i)
condition_list_train = condition_list_train[index_list]; x0_list_train = x0_list_train[index_list]

# define val
_,_,_,_, condition_list_val, x0_list_val = build_sheet.__build__(batch_list = [5])
# x0_list_val = x0_list_val[0:1]; condition_list_val = condition_list_val[0:1]

print('train:', x0_list_train.shape, condition_list_train.shape, 'val:', x0_list_val.shape, condition_list_val.shape)
print(x0_list_train[0:5], condition_list_train[0:5], x0_list_val[0:5], condition_list_val[0:5])

# define u-net and diffusion model
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

# generator definition
generator_train = Generator.Dataset_2D_distilled(

        img_list = x0_list_train,
        condition_list = condition_list_train,
        image_size = image_size,

        num_slices_per_image = 20,
        random_pick_slice = True,
        slice_range = None,

        num_patches_per_slice = num_patches_per_slice,
        patch_size = patch_size,

        histogram_equalization = histogram_equalization,
        background_cutoff = background_cutoff,
        maximum_cutoff = maximum_cutoff,
        normalize_factor = normalize_factor,

        shuffle = True,
        augment = True,
        augment_frequency = 0.5,)

generator_val = Generator.Dataset_2D_distilled(

        img_list = x0_list_val,
        condition_list = condition_list_val,
        image_size = image_size,

        num_slices_per_image = 20,
        random_pick_slice = False,
        slice_range = [20,40], #[50,70],

        num_patches_per_slice = 1,
        patch_size = [512,512],

        histogram_equalization = histogram_equalization,
        background_cutoff = background_cutoff,
        maximum_cutoff = maximum_cutoff,
        normalize_factor = normalize_factor,)

# start to train
trainer = ddpm.Trainer(
    diffusion_model= diffusion_model,
    generator_train = generator_train,
    generator_val = generator_val,
    train_batch_size = 25,
    
    accum_iter = 1,
    train_num_steps = 200, # total training epochs
    results_folder = os.path.join('/host/d/projects/denoising/models', trial_name, 'models'),
   
    train_lr = 1e-4,
    train_lr_decay_every = 200, 
    save_models_every = 1,
    validation_every = 1,)



trainer.train(pre_trained_model=pre_trained_model, start_step= start_step, beta = beta, lpips_weight = lpips_weight, edge_weight = edge_weight)