import sys 
sys.path.append('/gpfs/work/aac/xingyiyao23/Code')
import os
import torch
import numpy as np 
import IMF_denoising.Thinslice_experiments.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion as ddpm
import IMF_denoising.functions_collection as ff
import IMF_denoising.Build_lists.Build_list as Build_list
import IMF_denoising.Generator_thinslice as Generator

trial_name = 'unsupervised_gaussian_PCCT_nohist_beta1_lpips0.025_cutoff-1000to1000'
problem_dimension = '2D'
supervision = 'supervised' if trial_name[0:2] == 'su' else 'unsupervised'; print('supervision:', supervision)

train_batch_size = 10
preload = False

# bias  
beta = 1
lpips_weight = 0.025#0.2
edge_weight = 0#0.05

# model condition 
# if 'mean' in trial_name: condition on current slice, target the mean of neighboring slices
# else: condition on neighboring slices, target the current slice
condition_channel = 1 if (supervision == 'supervised') or ('mean' in trial_name) else 2

pre_trained_model = os.path.join('/gpfs/work/aac/xingyiyao23/projects', trial_name, 'models/model-150.pt') #None
start_step = 150
image_size = [512,512]
num_patches_per_slice = 2
patch_size = [128,128]
 
objective = 'pred_x0'

histogram_equalization = False
background_cutoff = -1000
maximum_cutoff = 1000
normalize_factor = 'equation'

###########################
# define train
build_sheet =  Build_list.Build_thinsliceCT(os.path.join('/gpfs/work/aac/xingyiyao23/Data/PCCT/Patient_lists/PCCT_split.xlsx'))
_,_,_,_, condition_list_train, _ = build_sheet.__build__(batch_list = [0]) 
x0_list_train = condition_list_train
# x0_list_train = x0_list_train[0:2]; condition_list_train = condition_list_train[0:2]

# define val
_,_,_,_, condition_list_val, _ = build_sheet.__build__(batch_list = [1])
x0_list_val = condition_list_val
# x0_list_val = x0_list_val[0:2]; condition_list_val = condition_list_val[0:2]

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

## preload data
if preload  == True:
    print('preloading data ...')
    condition_data_train =  ff.preload_data(condition_list_train)
    condition_data_val =  ff.preload_data(condition_list_val)

# generator definition
generator_train = Generator.Dataset_2D(
        supervision = supervision,

        preload = preload,
        preload_data = (condition_data_train,condition_data_train) if preload == True else None,

        img_list = x0_list_train,
        condition_list = condition_list_train,
        image_size = image_size,

        num_slices_per_image = 50,
        random_pick_slice = True,
        slice_range = None,

        num_patches_per_slice = num_patches_per_slice,
        patch_size = patch_size,

        histogram_equalization = histogram_equalization,
        bins = None if histogram_equalization == False else np.load('/gpfs/work/aac/xingyiyao23/Code/IMF_denoising/help_data/histogram_equalization/bins.npy'),
        bins_mapped = None if histogram_equalization == False else np.load('/gpfs/work/aac/xingyiyao23/Code/IMF_denoising/help_data/histogram_equalization/bins_mapped.npy'),
        background_cutoff = background_cutoff,
        maximum_cutoff = maximum_cutoff,
        normalize_factor = normalize_factor,

        shuffle = True,
        augment = True,
        augment_frequency = 0.5,)

generator_val = Generator.Dataset_2D(
        supervision = supervision,

        preload = preload,
        preload_data = (condition_data_val,condition_data_val) if preload == True else None,

        img_list = x0_list_val,
        condition_list = condition_list_val,
        image_size = image_size,

        num_slices_per_image = 20,
        random_pick_slice = False,
        slice_range = [20,40], #[50,70],

        num_patches_per_slice = 1,
        patch_size = [512,512],

        histogram_equalization = histogram_equalization,
        bins = None if histogram_equalization == False else np.load('/gpfs/work/aac/xingyiyao23/Code/IMF_denoising/help_data/histogram_equalization/bins.npy'),
        bins_mapped = None if histogram_equalization == False else np.load('/gpfs/work/aac/xingyiyao23/Code/IMF_denoising/help_data/histogram_equalization/bins_mapped.npy'),

        background_cutoff = background_cutoff,
        maximum_cutoff = maximum_cutoff,
        normalize_factor = normalize_factor,)

# start to train
trainer = ddpm.Trainer(
    diffusion_model= diffusion_model,
    generator_train = generator_train,
    generator_val = generator_val,
    train_batch_size = train_batch_size,
    
    accum_iter = 1,
    train_num_steps = 250, # total training epochs
    results_folder = os.path.join('/gpfs/work/aac/xingyiyao23/projects', trial_name, 'models'),
   
    train_lr = 1e-4,
    train_lr_decay_every = 100,#200, 
    save_models_every = 5,
    validation_every = 10,)


trainer.train(pre_trained_model=pre_trained_model, start_step= start_step, beta = beta, lpips_weight = lpips_weight, edge_weight = edge_weight)