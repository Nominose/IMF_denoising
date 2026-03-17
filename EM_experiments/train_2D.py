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
import Diffusion_denoising_thin_slice.Generator_EM as Generator_EM

trial_name = 'unsupervised_gaussian_EM_range01_try'
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

pre_trained_model =  os.path.join('/host/d/projects/denoising/models', trial_name, 'models/model-100.pt') #None
start_step = 100

# image condition
image_size = None
num_patches_per_slice = 2
patch_size = [320,320]#[128,128]

histogram_equalization = False
background_cutoff =  0
maximum_cutoff = 1
final_max = 1
final_min = 0
normalize_factor = 'equation'

######Patient list
# define train
if supervision == 'supervised':
    print('no supervised yet. only unsupervised')
    # build_sheet =  Build_list.Build_EM(os.path.join('/host/d/Data/NYU_MR/Patient_lists/NYU_MR_simulation_undersample4_equispaced.xlsx'))
else:
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
# preload_data if needed
if preload  == True:
    print('loading!!')
    simulation_1_data_train, simulation_2_data_train, gt_data_train = ff.preload_data(simulation_file_1_list_train), ff.preload_data(simulation_file_2_list_train), ff.preload_data(ground_truth_file_list_train)
    simulation_1_data_val, simulation_2_data_val, gt_data_val = ff.preload_data(simulation_file_1_list_val), ff.preload_data(simulation_file_2_list_val), ff.preload_data(ground_truth_file_list_val)

# data generator
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

        preload = True,
        preload_data = (simulation_1_data_train, simulation_2_data_train, gt_data_train),
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

        preload = True,
        preload_data = (simulation_1_data_val, simulation_2_data_val, gt_data_val),)

# #######Start to train
# define a saved model folder
save_models_folder = os.path.join('/host/d/projects/denoising/models', trial_name, 'models');ff.make_folder([os.path.dirname(save_models_folder), save_models_folder])
trainer = ddpm.Trainer(
    diffusion_model= diffusion_model,
    generator_train = generator_train,
    generator_val = generator_val,
    train_batch_size = train_batch_size,
    
    accum_iter = 1,
    train_num_steps = 180, # total training epochs
    results_folder = save_models_folder,
   
    train_lr = 1e-4,
    train_lr_decay_every = 200, 
    save_models_every = 5,
    validation_every = 5,)
    


trainer.train(pre_trained_model=pre_trained_model, start_step= start_step, beta = beta, lpips_weight = lpips_weight, edge_weight = edge_weight)