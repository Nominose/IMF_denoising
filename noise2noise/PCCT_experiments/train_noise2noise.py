import sys 
sys.path.append('/host/d/Github')
import os
import torch 
import numpy as np
import Diffusion_denoising_thin_slice.noise2noise.model as noise2noise
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.Generator_thinslice as Generator

#######################
trial_name = 'noise2noise_PCCT'
preload = True
supervision = 'unsupervised' 

pre_trained_model = None#os.path.join('/host/d/projects/denoising/models', trial_name, 'models/model-10.pt')
start_step = 0
train_batch_size = 5

image_size = [512,512]
num_patches_per_slice = 1
patch_size = [512,512] # if GPU memory is limited, you can reduce the patch size

histogram_equalization = True
background_cutoff = -1000
maximum_cutoff = 2000
normalize_factor = 'equation'
#######################
# define train
build_sheet =  Build_list.Build_thinsliceCT(os.path.join('/host/d/Data/PCCT/Patient_lists/PCCT_split.xlsx'))
_,_,_,_, condition_list_train, _ = build_sheet.__build__(batch_list = [0]) 
x0_list_train = condition_list_train
# x0_list_train = x0_list_train[0:2]; condition_list_train = condition_list_train[0:2]

# define val
_,_,_,_, condition_list_val, _ = build_sheet.__build__(batch_list = [1])
x0_list_val = condition_list_val
# x0_list_val = x0_list_val[0:2]; condition_list_val = condition_list_val[0:2]

print('train:', x0_list_train.shape, condition_list_train.shape, 'val:', x0_list_val.shape, condition_list_val.shape)
print(x0_list_train[0:5], condition_list_train[0:5], x0_list_val[0:5], condition_list_val[0:5])


# build model
model = noise2noise.Unet(
    problem_dimension = '2D',  # '2D' or '3D'
    input_channels = 1,
    out_channels = 1,  
    initial_dim = 16,  # initial feature dimension after first conv layer
    dim_mults = (2,4,8,16),
    full_attn_paths = (None, None, False, True), # these are for downsampling and upsampling paths
    full_attn_bottleneck = True, # this is for the middle bottleneck layer
    act = 'ReLU',
)

# build generator
# first we define the image list 
if preload  == True:
    print('preloading data ...')
    condition_data_train =  ff.preload_data(condition_list_train)
    condition_data_val =  ff.preload_data(condition_list_val)

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
        bins = None if histogram_equalization == False else np.load('/host/d/Github/Diffusion_denoising_thin_slice/help_data/histogram_equalization/bins.npy'),
        bins_mapped = None if histogram_equalization == False else np.load('/host/d/Github/Diffusion_denoising_thin_slice/help_data/histogram_equalization/bins_mapped.npy'),
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
        bins = None if histogram_equalization == False else np.load('/host/d/Github/Diffusion_denoising_thin_slice/help_data/histogram_equalization/bins.npy'),
        bins_mapped = None if histogram_equalization == False else np.load('/host/d/Github/Diffusion_denoising_thin_slice/help_data/histogram_equalization/bins_mapped.npy'),

        background_cutoff = background_cutoff,
        maximum_cutoff = maximum_cutoff,
        normalize_factor = normalize_factor,)



# train
save_models_folder = os.path.join('/host/d/projects/denoising/models', trial_name, 'models');ff.make_folder([os.path.dirname(save_models_folder), save_models_folder])
trainer = noise2noise.Trainer(
    model= model,
    generator_train = generator_train,
    generator_val = generator_val,
    train_batch_size = train_batch_size,

    train_num_steps = 100, # total training epochs
    results_folder = save_models_folder,
   
    train_lr = 1e-4,
    train_lr_decay_every = 200, 
    save_models_every = 10,
    validation_every = 10,
)

trainer.train(pre_trained_model=pre_trained_model, start_step= start_step)