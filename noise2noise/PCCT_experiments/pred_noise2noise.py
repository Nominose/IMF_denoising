# make sure you have Github copilot installed, search it in the VSCode extension marketplace, it will make your coding much easier
import sys 
sys.path.append('/host/d/Github/')
import os
import torch
import numpy as np
import nibabel as nb
import Diffusion_denoising_thin_slice.noise2noise.model as noise2noise
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.Generator_thinslice as Generator

trial_name = 'noise2noise_PCCT' 
epoch = 80
# define your own saved model path and prediction save path
trained_model_filename = os.path.join('/host/d/projects/denoising/models', trial_name, 'models/model-' + str(epoch)+ '.pt')
save_folder = os.path.join('/host/d/projects/denoising/models', trial_name, 'pred_images'); os.makedirs(save_folder, exist_ok=True)

### parameters no need to change
image_size = [512,512]

histogram_equalization = True
background_cutoff = -1000
maximum_cutoff = 2000
normalize_factor = 'equation' 

# define patient list
build_sheet =  Build_list.Build_thinsliceCT(os.path.join('/host/d/Data/PCCT/Patient_lists/PCCT_split.xlsx'))
_,patient_id_list,patient_subid_list,random_num_list, condition_list, x0_list = build_sheet.__build__(batch_list = [2]) 
x0_list = condition_list
print('total cases:', patient_id_list.shape[0])


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

#main
for i in range(0, patient_id_list.shape[0]):
    patient_id = patient_id_list[i]
    x0_file = x0_list[i]
    condition_file = condition_list[i]

    print(i,patient_id)

    # make folders
    ff.make_folder([os.path.join(save_folder, patient_id)])
    save_folder_case = os.path.join(save_folder, patient_id, 'epoch_' + str(epoch))
    ff.make_folder([save_folder_case])

    # get the condition image
    condition_img = nb.load(condition_file)
    affine = condition_img.affine
    condition_img = condition_img.get_fdata()[:,:,30:80]

    # save condition image
    nb.save(nb.Nifti1Image(condition_img, affine), os.path.join(save_folder_case,'condition_img.nii.gz'))

    if os.path.isfile(os.path.join(save_folder_case, 'pred_img.nii.gz')):
        print('prediction already exists, skip to next case')
        continue
    
    # # generator
    generator = Generator.Dataset_2D(
        supervision = 'unsupervised',

        img_list = np.array([x0_file]),
        condition_list = np.array([condition_file]),
        image_size = image_size,

        num_slices_per_image = 50,
        random_pick_slice = False,
        slice_range = [30,80],

        histogram_equalization = histogram_equalization,
        bins = np.load('/host/d/Github/Diffusion_denoising_thin_slice/help_data/histogram_equalization/bins.npy'),
        bins_mapped = np.load('/host/d/Github/Diffusion_denoising_thin_slice/help_data/histogram_equalization/bins_mapped.npy'),
        background_cutoff = background_cutoff,
        maximum_cutoff = maximum_cutoff,
        normalize_factor = normalize_factor,)

    # # sample:
    sampler = noise2noise.Sampler(model,generator,batch_size = 1, image_size = image_size)

    pred_img = sampler.sample_2D(trained_model_filename, condition_img)
    pred_img_final = pred_img
    
    # save
    nb.save(nb.Nifti1Image(pred_img_final, affine), os.path.join(save_folder_case, 'pred_img.nii.gz'))
    

