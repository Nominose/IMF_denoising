import sys 
sys.path.append('/host/d/Github')
import os
import torch 
import numpy as np
import Diffusion_denoising_thin_slice.noise2noise.model as noise2noise
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.Generator_EM as Generator_EM

#######################
trial_name = 'noise2noise_EM_range-1+1'
preload = False
supervision = 'unsupervised' 

pre_trained_model = None#os.path.join('/host/d/projects/denoising/models', trial_name, 'models/model-10.pt')
start_step = 0
train_batch_size = 1

image_size = None
num_patches_per_slice = 2
patch_size = [320,320]

histogram_equalization = False
background_cutoff = 0
maximum_cutoff = 1
final_max = 1
final_min = -1
normalize_factor = 'equation'
#######################
build_sheet =  Build_list.Build_EM(os.path.join('/host/d/Data/minnie_EM/Patient_lists/minnie_EM_split_gaussian_simulation_v1.xlsx'))

# define train patient list
_, patient_id_list_train, _, _, simulation_file_1_list_train, simulation_file_2_list_train, ground_truth_file_list_train, _ = build_sheet.__build__(batch_list = ['train'])
# patient_id_list_train = patient_id_list_train[0:1]
# simulation_file_1_list_train = simulation_file_1_list_train[0:1]
# simulation_file_2_list_train = simulation_file_2_list_train[0:1]
# ground_truth_file_list_train = ground_truth_file_list_train[0:1]

# define val patient list
_, patient_id_list_val, _, _, simulation_file_1_list_val, simulation_file_2_list_val, ground_truth_file_list_val, _ = build_sheet.__build__(batch_list = ['val'])
# patient_id_list_val = patient_id_list_val[0:1]
# simulation_file_1_list_val = simulation_file_1_list_val[0:1]
# simulation_file_2_list_val = simulation_file_2_list_val[0:1]
# ground_truth_file_list_val = ground_truth_file_list_val[0:1]

print('number of training cases:', ground_truth_file_list_train.shape[0], '; number of validation cases:', ground_truth_file_list_val.shape[0], ' example of simulation_file_1_list_train:', simulation_file_1_list_train[0])


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
# preload_data if needed
if preload  == True:
    print('loading!!')
    simulation_1_data_train, simulation_2_data_train, gt_data_train = ff.preload_data(simulation_file_1_list_train), ff.preload_data(simulation_file_2_list_train), ff.preload_data(ground_truth_file_list_train)
    simulation_1_data_val, simulation_2_data_val, gt_data_val = ff.preload_data(simulation_file_1_list_val), ff.preload_data(simulation_file_2_list_val), ff.preload_data(ground_truth_file_list_val)

G =  Generator_EM.Dataset_2D
generator_train = G(
        gt_file_list = ground_truth_file_list_train,
        simulation_1_file_list = simulation_file_1_list_train,
        simulation_2_file_list = simulation_file_2_list_train,

        num_slices_per_image = 30,  
        random_pick_slice = True,
        slice_range = None, # None or [a,b]

        background_cutoff = background_cutoff, 
        maximum_cutoff = maximum_cutoff,
        normalize_factor = normalize_factor,
        final_max = final_max,
        final_min = final_min,

        num_patches_per_slice = num_patches_per_slice,
        patch_size = patch_size,

        shuffle = True,
        augment = True,
        augment_frequency = 0.5,

        preload = preload,
        preload_data = (simulation_1_data_train, simulation_2_data_train, gt_data_train) if preload else None,
        )

generator_val = G(
        gt_file_list = ground_truth_file_list_val,
        simulation_1_file_list = simulation_file_1_list_val,
        simulation_2_file_list = simulation_file_2_list_val,

        num_slices_per_image = 30, 
        random_pick_slice = False,
        slice_range = [0,30], # None or [a,b]

        background_cutoff = background_cutoff, 
        maximum_cutoff = maximum_cutoff,
        normalize_factor = normalize_factor,
        final_max = final_max,
        final_min = final_min,

        num_patches_per_slice = 1,
        patch_size = [320,320],
        preset_patch_origin = [0,0],

        shuffle = False,

        preload = preload,
        preload_data = (simulation_1_data_val, simulation_2_data_val, gt_data_val) if preload else None,)


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
    train_lr_decay_every = 150, 
    save_models_every = 10,
    validation_every = 10,
)

trainer.train(pre_trained_model=pre_trained_model, start_step= start_step)