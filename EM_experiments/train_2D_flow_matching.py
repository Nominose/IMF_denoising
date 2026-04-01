"""
Example: train_2D_flow_matching.py

This is a minimal modification of CT_experiments/train_2D.py.
Lines marked with  # <-- CHANGED  are the only differences from the DDPM version.
"""
import sys
sys.path.append('/host/c/Users/ROG/Documents/GitHub')
import os
import torch
import numpy as np
import nibabel as nb

# ======================================================================
# CHANGE 1:  import flow matching instead of DDPM
# ======================================================================
# BEFORE (DDPM):
#   import IMF_denoising.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion as ddpm
# AFTER (Flow Matching):
import IMF_denoising.conditional_flow_matching as fm          # <-- CHANGED

import IMF_denoising.functions_collection as ff
import IMF_denoising.Build_lists.Build_list as Build_list
import IMF_denoising.Generator as Generator

# ---------- hyper-parameters (same as before) ----------
trial_name = 'flow_matching_gaussian_mayo'
problem_dimension = '2D'
supervision = 'supervised' if trial_name[0:2] == 'su' else 'unsupervised'
adjacent_condition = 'adjacent' in trial_name
print('supervision:', supervision, '; adjacent:', adjacent_condition)

preload = True
beta = 0
lpips_weight = 0
edge_weight = 0

condition_channel = 1 if not adjacent_condition else 2
train_batch_size = 5
pre_trained_model = None
start_step = 0

image_size = [512, 512]
num_patches_per_slice = 2
patch_size = [256, 256]

histogram_equalization = False
background_cutoff = -200
maximum_cutoff = 250
normalize_factor = 'equation'

# ---------- patient list (unchanged) ----------
build_sheet = Build_list.Build(
    os.path.join('/host/d/Data/low_dose_CT/Patient_lists/mayo_low_dose_CT_gaussian_simulation_v2.xlsx')
)
_, _, _, noise_file_all_list_train, noise_file_odd_list_train, noise_file_even_list_train, gt_file_list_train, slice_num_list_train = \
    build_sheet.__build__(batch_list=['train'])
_, _, _, noise_file_all_list_val, noise_file_odd_list_val, noise_file_even_list_val, gt_file_list_val, slice_num_list_val = \
    build_sheet.__build__(batch_list=['val'])
print('train:', gt_file_list_train.shape[0], '; val:', gt_file_list_val.shape[0])


# ======================================================================
# CHANGE 2:  Use the SAME Unet — import it from flow matching module
#            (it re-exports the original Unet unchanged)
# ======================================================================
# The Unet class is identical; we just import it from the FM module
# which re-exports it from conditional_diffusion.
from IMF_denoising.conditional_flow_matching import Unet     # <-- CHANGED (import path only)

model = Unet(
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


# ======================================================================
# CHANGE 3:  Replace GaussianDiffusion with ConditionalFlowMatching
# ======================================================================
# BEFORE (DDPM):
#   diffusion_model = ddpm.GaussianDiffusion(
#       model,
#       image_size=patch_size,
#       timesteps=1000,
#       sampling_timesteps=250,
#       objective='pred_x0',
#       clip_or_not=False,
#       auto_normalize=False,
#   )
#
# AFTER (Flow Matching):
diffusion_model = fm.ConditionalFlowMatching(                                 # <-- CHANGED
    model,
    image_size=patch_size,               # same as before
    sampling_timesteps=50,               # FM needs far fewer steps (50 is usually enough)
    clip_or_not=False,
    auto_normalize=False,
)


# ---------- data generators (completely unchanged) ----------
if supervision == 'supervised':
    x0_list_train, condition_list_train = gt_file_list_train, noise_file_all_list_train
    x0_list_val, condition_list_val = gt_file_list_val, noise_file_all_list_val
else:
    x0_list_train, condition_list_train = noise_file_even_list_train, noise_file_odd_list_train
    x0_list_val, condition_list_val = noise_file_even_list_val, noise_file_odd_list_val

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
    switch_odd_and_even_frequency=-1 if supervision == 'supervised' else 0.5,
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


# ======================================================================
# CHANGE 4:  Use fm.Trainer instead of ddpm.Trainer
# ======================================================================
save_models_folder = os.path.join('/host/d/projects/denoising/models', trial_name, 'models')
ff.make_folder([os.path.dirname(save_models_folder), save_models_folder])

# BEFORE:  trainer = ddpm.Trainer(...)
# AFTER:
trainer = fm.Trainer(                                                          # <-- CHANGED
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
    beta=beta,
    lpips_weight=lpips_weight,
    edge_weight=edge_weight,
)
