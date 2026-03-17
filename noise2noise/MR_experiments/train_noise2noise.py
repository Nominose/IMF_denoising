import sys 
sys.path.append('/host/d/Github')
import os
import torch 
import numpy as np
import Diffusion_denoising_thin_slice.noise2noise.model as noise2noise
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.Generator_MR as Generator_MR

#######################
trial_name = 'noise2noise_simple_MR'
preload = True
supervision = 'unsupervised' 

pre_trained_model = None#os.path.join('/host/d/projects/denoising/models', trial_name, 'models/model-20.pt')
start_step = 0
train_batch_size = 1

image_size = [640,320]
num_patches_per_slice = 2
patch_size = [320,320]

background_cutoff = 2.5e-06
maximum_cutoff = 0.00015
normalize_factor = 'equation'
#######################
# define train
build_sheet =  Build_list.Build(os.path.join('/host/d/Data/NYU_MR/Patient_lists/NYU_MR_simulation.xlsx'))

# define train patient list
_, _, _, noise_file_all_list_train, noise_file_odd_list_train, noise_file_even_list_train, gt_file_list_train, slice_num_list_train = build_sheet.__build__(batch_list = ['train']) 

# define val patient list
_, _, _,  noise_file_all_list_val, noise_file_odd_list_val, noise_file_even_list_val,  gt_file_list_val, slice_num_list_val = build_sheet.__build__(batch_list = ['val'])

print('number of training cases:', gt_file_list_train.shape[0], '; number of validation cases:', gt_file_list_val.shape[0], ' example of noise_odd_list_train:', noise_file_odd_list_train[0])


# build model
model = noise2noise.Unet(
    problem_dimension = '2D',  # '2D' or '3D'
  
    input_channels = 1,
    out_channels = 1,  
    initial_dim = 16,  # initial feature dimension after first conv layer
    dim_mults = (2,4,8,16),
    groups = 8,
      
    attn_dim_head = 32,
    attn_heads = 4,
    full_attn_paths = (None, None, None, None), # these are for downsampling and upsampling paths
    full_attn_bottleneck = None, # this is for the middle bottleneck layer
    act = 'ReLU',
)

# build generator
# first we define the image list 
x0_list_train, condition_list_train = noise_file_even_list_train, noise_file_odd_list_train
x0_list_val, condition_list_val = noise_file_even_list_val, noise_file_odd_list_val
print('example x0 train file:', x0_list_train[0], '; example condition file:', condition_list_train[0])
print('example x0 val file:', x0_list_val[0], '; example condition file:', condition_list_val[0])
# preload_data if needed
if preload  == True:
    x0_data_train, condition_data_train = ff.preload_data(x0_list_train), ff.preload_data(condition_list_train)
    x0_data_val, condition_data_val = ff.preload_data(x0_list_val), ff.preload_data(condition_list_val)

G = Generator_MR.Dataset_2D
generator_train = G(
        supervision = supervision,

        preload = preload,
        preload_data = (x0_data_train, condition_data_train) if preload == True else None,

        img_list = x0_list_train,
        condition_list = condition_list_train,
        image_size = image_size,

        num_slices_per_image = 30,
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

        maximum_cutoff = maximum_cutoff,
        background_cutoff = background_cutoff,
        normalize_factor = normalize_factor,

        num_slices_per_image = 20,
        random_pick_slice = False,
        slice_range = [10,30],

        num_patches_per_slice = 1,
        patch_size = image_size,)


# train
save_models_folder = os.path.join('/host/d/projects/denoising/models', trial_name, 'models');ff.make_folder([os.path.dirname(save_models_folder), save_models_folder])
trainer = noise2noise.Trainer(
    model= model,
    generator_train = generator_train,
    generator_val = generator_val,
    train_batch_size = train_batch_size,

    train_num_steps = 150, # total training epochs
    results_folder = save_models_folder,
   
    train_lr = 1e-4,
    train_lr_decay_every = 200, 
    save_models_every = 5,
    validation_every = 5,
)

trainer.train(pre_trained_model=pre_trained_model, start_step= start_step)