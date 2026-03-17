import sys 
sys.path.append('/host/d/Github')
import os
import torch
import numpy as np 
import nibabel as nb
import Diffusion_denoising_thin_slice.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion as ddpm
import Diffusion_denoising_thin_slice.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_EDM as edm
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.prepare_sliced_data as prepare_sliced_data
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.Generator as Generator

trial_name = 'distill_2'
problem_dimension = '2D'
supervision = 'supervised' 
adjacent_condition = True if 'adjacent' in trial_name else False
print('supervision type:', supervision, '; adjacent condition:', adjacent_condition)

preload = True

# bias  
beta = 0
lpips_weight = 0#0.2
edge_weight = 0#0.05

# model condition 
condition_channel = 1 if not adjacent_condition else 2
train_batch_size = 5
objective = 'pred_x0' #if 'noise' not in trial_name else 'pred_noise'

pre_trained_model =  None#os.path.join('/host/d/projects/denoising/models', trial_name, 'models/model-400.pt') #None
start_step = 0

# image condition
image_size = [512,512]
num_patches_per_slice = 2
patch_size = [256,256]#[128,128]

histogram_equalization = False# if 'hist' in trial_name else False
print('histogram equalization:', histogram_equalization)
background_cutoff = -200
maximum_cutoff = 250
normalize_factor = 'equation'

###### prepare data pair if not ready
prepare_sliced_data.prepare_sliced_data()

######Patient list
# define train
build_sheet =  Build_list.Build(os.path.join('/host/d/Data/low_dose_CT/Patient_lists/mayo_low_dose_CT_distill_v2.xlsx'))

# define train patient list
_, _, _, noise_file_all_list_train, noise_file_odd_list_train, noise_file_even_list_train, gt_file_list_train, slice_num_list_train,generated_20_list_train,generated_10_list_train = build_sheet.__build__(batch_list = ['train'], distill=True)
# noise_file_all_list_train = noise_file_all_list_train[0:1]
# noise_file_odd_list_train = noise_file_odd_list_train[0:1]
# noise_file_even_list_train = noise_file_even_list_train[0:1]
# gt_file_list_train = gt_file_list_train[0:1]
# slice_num_list_train = slice_num_list_train[0:1]

# define val patient list
_, _, _,  noise_file_all_list_val, noise_file_odd_list_val, noise_file_even_list_val,  gt_file_list_val, slice_num_list_val,generated_20_list_val,generated_10_list_val = build_sheet.__build__(batch_list = ['val'], distill=True)

print('number of training cases:', gt_file_list_train.shape[0], '; number of validation cases:', gt_file_list_val.shape[0])

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

######Define data generator
# first we define the x0 and condition list 
x0_list_train, condition_list_train = generated_20_list_train, noise_file_all_list_train
x0_list_val, condition_list_val = generated_20_list_val, noise_file_all_list_val
print('example x0 train file:', x0_list_train[0], '; example condition file:', condition_list_train[0])
print('example x0 val file:', x0_list_val[0], '; example condition file:', condition_list_val[0])

    
# preload_data if needed
if preload  == True:
    x0_data_train, condition_data_train = ff.preload_data(x0_list_train), ff.preload_data(condition_list_train)
    x0_data_val, condition_data_val = ff.preload_data(x0_list_val), ff.preload_data(condition_list_val)

# data generator
# if 'adjacent' in trial_name, we use Generator.Dataset_2D_adjacent_slices, else use Generator.Dataset_2D
# can you define generator first?
G = Generator.Dataset_2D_adjacent_slices  if adjacent_condition else Generator.Dataset_2D
generator_train = G(
        supervision = supervision,

        preload = preload,
        preload_data = (x0_data_train, condition_data_train) if preload == True else None,

        img_list = x0_list_train,
        condition_list = condition_list_train,
        image_size = image_size,

        num_slices_per_image = 50, # no matter how many slices, we only randomly pick 50 slices each time
        random_pick_slice = True,
        slice_range = None,

        num_patches_per_slice = num_patches_per_slice,
        patch_size = patch_size,

        histogram_equalization = histogram_equalization,
        bins = None if histogram_equalization == False else np.load('/host/d/Github/Diffusion_denoising_thin_slice/help_data/histogram_equalization/bins_lowdoseCT.npy'),
        bins_mapped = None if histogram_equalization == False else np.load('/host/d/Github/Diffusion_denoising_thin_slice/help_data/histogram_equalization/bins_mapped_lowdoseCT.npy'),
        background_cutoff = background_cutoff,
        maximum_cutoff = maximum_cutoff,
        normalize_factor = normalize_factor,

        shuffle = True,
        augment = True,
        augment_frequency = 0.5,

        switch_odd_and_even_frequency = -1,
        )

generator_val = G(
        supervision = supervision,

        preload = preload,
        preload_data = (x0_data_val, condition_data_val) if preload == True else None,

        img_list = x0_list_val,
        condition_list = condition_list_val,
        image_size = image_size,

        num_slices_per_image = 50,
        random_pick_slice = False,
        slice_range = [30,80],# [100,190],

        num_patches_per_slice = 1,
        patch_size = [512,512],

        histogram_equalization = histogram_equalization,
        bins = None if histogram_equalization == False else np.load('/host/d/Github/Diffusion_denoising_thin_slice/help_data/histogram_equalization/bins_lowdoseCT.npy'),
        bins_mapped = None if histogram_equalization == False else np.load('/host/d/Github/Diffusion_denoising_thin_slice/help_data/histogram_equalization/bins_mapped_lowdoseCT.npy'),
        background_cutoff = background_cutoff,
        maximum_cutoff = maximum_cutoff,
        normalize_factor = normalize_factor,)

#######Start to train
# define a saved model folder
save_models_folder = os.path.join('/host/d/projects/denoising/models', trial_name, 'models');ff.make_folder([os.path.dirname(save_models_folder), save_models_folder])
trainer = ddpm.Trainer(
    diffusion_model= diffusion_model,
    generator_train = generator_train,
    generator_val = generator_val,
    train_batch_size = train_batch_size,
    
    accum_iter = 1,
    train_num_steps = 500, # total training epochs
    results_folder = save_models_folder,
   
    train_lr = 1e-4,
    train_lr_decay_every = 200, 
    save_models_every = 5,
    validation_every = 5,)


trainer.train(pre_trained_model=pre_trained_model, start_step= start_step, beta = beta, lpips_weight = lpips_weight, edge_weight = edge_weight)