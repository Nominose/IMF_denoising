"""
train_2D_imf.py — Mayo Low-Dose CT with improved MeanFlow (iMF)

Key difference from FM: 1-NFE sampling instead of 50-step ODE.
iMF K=8 = 8 total steps vs FM K=8 = 400 steps.
"""
import sys
sys.path.append('/gpfs/work/aac/xingyiyao23/Code')
import os
import torch
import numpy as np

import Diffusion_denoising_thin_slice.improved_mean_flow as imf
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.Generator as Generator
from Diffusion_denoising_thin_slice.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion import Unet

# ========== Parameters ==========
trial_name = 'imf_unsupervised_gaussian_mayo'
problem_dimension = '2D'
supervision = 'unsupervised'

preload = True
condition_channel = 1
train_batch_size = 1
pre_trained_model = None
start_step = 0

image_size = [512, 512]
num_patches_per_slice = 2
patch_size = [256, 256]

histogram_equalization = False
background_cutoff = -200
maximum_cutoff = 250
normalize_factor = 'equation'

# ========== Patient list ==========
patient_list_file = '/gpfs/work/aac/xingyiyao23/Data/新建文件夹/mayo/mayo_flow_matching.xlsx'
build_sheet = Build_list.Build(patient_list_file)

_, _, _, noise_file_all_list_train, noise_file_odd_list_train, noise_file_even_list_train, gt_file_list_train, slice_num_list_train = \
    build_sheet.__build__(batch_list=['train'])

_, _, _, noise_file_all_list_val, noise_file_odd_list_val, noise_file_even_list_val, gt_file_list_val, slice_num_list_val = \
    build_sheet.__build__(batch_list=['val'])

print('train:', gt_file_list_train.shape[0], '; val:', gt_file_list_val.shape[0])

# ========== Base U-Net (same architecture as DDPM/FM) ==========
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

# ========== improved MeanFlow ==========
diffusion_model = imf.ImprovedMeanFlow(
    base_model,
    image_size=patch_size,
    ratio_r_neq_t=0.50,        # 50% of samples use MeanFlow, 50% use FM
    clip_or_not=False,
    auto_normalize=False,
    adaptive_weight_power=1.0,  # p=1.0 as recommended by iMF paper
)

# ========== Data generators ==========
x0_list_train, condition_list_train = noise_file_even_list_train, noise_file_odd_list_train
x0_list_val, condition_list_val = noise_file_even_list_val, noise_file_odd_list_val

print('x0 train:', x0_list_train[0])
print('cond train:', condition_list_train[0])

if preload:
    x0_data_train = ff.preload_data(x0_list_train)
    condition_data_train = ff.preload_data(condition_list_train)
    x0_data_val = ff.preload_data(x0_list_val)
    condition_data_val = ff.preload_data(condition_list_val)

G = Generator.Dataset_2D
generator_train = G(
    supervision=supervision,
    preload=preload,
    preload_data=(x0_data_train, condition_data_train) if preload else None,
    img_list=x0_list_train,
    condition_list=condition_list_train,
    image_size=image_size,
    num_slices_per_image=50,
    random_pick_slice=True,
    slice_range=None,
    num_patches_per_slice=num_patches_per_slice,
    patch_size=patch_size,
    histogram_equalization=histogram_equalization,
    bins=None,
    bins_mapped=None,
    background_cutoff=background_cutoff,
    maximum_cutoff=maximum_cutoff,
    normalize_factor=normalize_factor,
    shuffle=True,
    augment=True,
    augment_frequency=0.5,
    switch_odd_and_even_frequency=0.5,
)

generator_val = G(
    supervision=supervision,
    preload=preload,
    preload_data=(x0_data_val, condition_data_val) if preload else None,
    img_list=x0_list_val,
    condition_list=condition_list_val,
    image_size=image_size,
    num_slices_per_image=90,
    random_pick_slice=False,
    slice_range=[100, 190],
    num_patches_per_slice=1,
    patch_size=[512, 512],
    histogram_equalization=histogram_equalization,
    bins=None,
    bins_mapped=None,
    background_cutoff=background_cutoff,
    maximum_cutoff=maximum_cutoff,
    normalize_factor=normalize_factor,
)

# ========== Training ==========
save_models_folder = os.path.join('/gpfs/work/aac/xingyiyao23/results', trial_name, 'models')
ff.make_folder([os.path.dirname(save_models_folder), save_models_folder])

trainer = imf.Trainer(
    diffusion_model=diffusion_model,
    generator_train=generator_train,
    generator_val=generator_val,
    train_batch_size=train_batch_size,
    accum_iter=1,
    train_num_steps=200,
    results_folder=save_models_folder,
    train_lr=1e-4,
    train_lr_decay_every=200,
    save_models_every=5,
    validation_every=5,
)

trainer.train(
    pre_trained_model=pre_trained_model,
    start_step=start_step,
)
