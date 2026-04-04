"""
train_2D_flow_matching.py — Brain CT thin-slice denoising with standard Flow Matching

For comparing FM NFE=1 with cDDPM DDIM=1 and iMF NFE=1.
"""
import sys
sys.path.append('/host/c/Users/ROG/Documents/GitHub')
import os
import torch
import numpy as np

import IMF_denoising.conditional_flow_matching as fm
import IMF_denoising.functions_collection as ff
import IMF_denoising.Build_lists.Build_list as Build_list
import IMF_denoising.Generator_thinslice as Generator
from IMF_denoising.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion import Unet

# ========== Parameters ==========
trial_name = 'fm_unsupervised_gaussian_brainCT'
problem_dimension = '2D'
supervision = 'unsupervised'

condition_channel = 2
train_batch_size = 3
pre_trained_model = None
start_step = 0

image_size = [512, 512]
num_patches_per_slice = 2
patch_size = [128, 128]

histogram_equalization = True
background_cutoff = -1000
maximum_cutoff = 2000
normalize_factor = 'equation'

# ========== Patient list ==========
patient_list_file = '/host/d/Data/brain_CT/Patient_lists/fixedCT_static_simulation_train_test_gaussian_xjtlu.xlsx'
build_sheet = Build_list.Build_thinsliceCT(patient_list_file)

_, _, _, _, condition_list_train, x0_list_train = build_sheet.__build__(batch_list=[0, 1, 2, 3])
n = ff.get_X_numbers_in_interval(total_number=x0_list_train.shape[0], start_number=0, end_number=1, interval=2)
x0_list_train = x0_list_train[n]; condition_list_train = condition_list_train[n]

_, _, _, _, condition_list_val, x0_list_val = build_sheet.__build__(batch_list=[4])
n = ff.get_X_numbers_in_interval(total_number=x0_list_val.shape[0], start_number=0, end_number=1, interval=2)
x0_list_val = x0_list_val[n]; condition_list_val = condition_list_val[n]

print('supervision:', supervision)
print('train:', x0_list_train.shape, condition_list_train.shape, 'val:', x0_list_val.shape, condition_list_val.shape)

# ========== Histogram equalization bins ==========
bins = np.load('/host/d/file/histogram_equalization/bins.npy') if histogram_equalization else None
bins_mapped = np.load('/host/d/file/histogram_equalization/bins_mapped.npy') if histogram_equalization else None

# ========== Base U-Net (same as iMF and cDDPM) ==========
base_model = Unet(
    problem_dimension=problem_dimension,
    init_dim=64,
    out_dim=1,
    channels=1,
    conditional_diffusion=True,
    condition_channels=condition_channel,
    downsample_list=(True, True, True, False),
    upsample_list=(True, True, True, False),
    full_attn=(None, None, False, True),
)

# ========== Flow Matching ==========
diffusion_model = fm.ConditionalFlowMatching(
    base_model,
    image_size=patch_size,
    sampling_timesteps=50,  # default for multi-step, can override at inference
    clip_or_not=False,
    auto_normalize=False,
)

# ========== Data generators ==========
generator_train = Generator.Dataset_2D(
    supervision=supervision,
    img_list=x0_list_train,
    condition_list=condition_list_train,
    image_size=image_size,
    num_slices_per_image=50,
    random_pick_slice=True,
    slice_range=None,
    num_patches_per_slice=num_patches_per_slice,
    patch_size=patch_size,
    histogram_equalization=histogram_equalization,
    bins=bins,
    bins_mapped=bins_mapped,
    background_cutoff=background_cutoff,
    maximum_cutoff=maximum_cutoff,
    normalize_factor=normalize_factor,
    shuffle=True,
    augment=True,
    augment_frequency=0.5,
)

generator_val = Generator.Dataset_2D(
    supervision=supervision,
    img_list=x0_list_val,
    condition_list=condition_list_val,
    image_size=image_size,
    num_slices_per_image=20,
    random_pick_slice=False,
    slice_range=[20, 40],
    num_patches_per_slice=1,
    patch_size=[256, 256],
    histogram_equalization=histogram_equalization,
    bins=bins,
    bins_mapped=bins_mapped,
    background_cutoff=background_cutoff,
    maximum_cutoff=maximum_cutoff,
    normalize_factor=normalize_factor,
)

# ========== Training ==========
save_models_folder = os.path.join('/host/d/projects/denoising/models', trial_name, 'models')
ff.make_folder([os.path.dirname(save_models_folder), save_models_folder])

trainer = fm.Trainer(
    diffusion_model=diffusion_model,
    generator_train=generator_train,
    generator_val=generator_val,
    train_batch_size=train_batch_size,
    accum_iter=1,
    train_num_steps=100,
    results_folder=save_models_folder,
    train_lr=1e-4,
    train_lr_decay_every=200,
    save_models_every=5,
    validation_every=5,
)

trainer.train(
    pre_trained_model=pre_trained_model,
    start_step=start_step,
    beta=0,
    lpips_weight=0,
    edge_weight=0,
)
