# make sure you have Github copilot installed, search it in the VSCode extension marketplace, it will make your coding much easier
import sys 
sys.path.append('/host/d/Github/')
import os
import torch
import numpy as np
import nibabel as nb
import Diffusion_denoising_thin_slice.noise2noise.Thinslice_experiments.model_previous_version as noise2noise
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.noise2noise.Thinslice_experiments.Generator as Generator 

trial_name = 'noise2noise_brainCT' 
epoch = 78
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
build_sheet =  Build_list.Build_thinsliceCT(os.path.join('/host/d/Data/brain_CT/Patient_lists/fixedCT_static_simulation_train_test_gaussian_xjtlu.xlsx'))
_,patient_id_list,patient_subid_list,random_num_list, condition_list, x0_list = build_sheet.__build__(batch_list = [5]) 
print('total cases:', patient_id_list.shape[0])
n = ff.get_X_numbers_in_interval(total_number = patient_id_list.shape[0],start_number = 0,end_number = 1, interval = 2)
print('total number:', n.shape[0])

# build model
model = noise2noise.Unet2D(
    init_dim = 16,
    channels = 2, 
    out_dim = 1,
    dim_mults = (2,4,8,16),
    full_attn = (None,None, False, True),
    act = 'ReLU',
)

#main
for i in range(0, n.shape[0]):
    patient_id = patient_id_list[n[i]]
    patient_subid = patient_subid_list[n[i]]
    random_num = random_num_list[n[i]]
    x0_file = x0_list[n[i]]
    condition_file = condition_list[n[i]]

    print(i,patient_id, patient_subid, random_num)

    # make folders
    ff.make_folder([os.path.join(save_folder, patient_id), os.path.join(save_folder, patient_id, patient_subid), os.path.join(save_folder, patient_id, patient_subid, 'random_' + str(random_num))])
    save_folder_case = os.path.join(save_folder, patient_id, patient_subid, 'random_' + str(random_num), 'epoch' + str(epoch)); os.makedirs(save_folder_case, exist_ok=True)

    # get the ground truth image
    gt_img = nb.load(x0_file)
    affine = gt_img.affine
    gt_img = gt_img.get_fdata()[:,:,30:80]

    # get the condition image
    condition_img = nb.load(condition_file).get_fdata()[:,:,30:80]


    # save condition image
    nb.save(nb.Nifti1Image(condition_img, affine), os.path.join(save_folder_case,'condition_img.nii.gz'))
    # nb.save(nb.Nifti1Image(gt_img, affine), os.path.join(save_folder_case,'gt_img.nii.gz'))

    if os.path.isfile(os.path.join(save_folder_case, 'pred_img.nii.gz')):
        print('prediction already exists, skip to next case')
        continue
    

    # # generator
    bins = np.load('/host/d/Github/CTDenoising_Diffusion_N2N/example_data/histogram_equalization/bins.npy') 
    bins_mapped = np.load('/host/d/Github/CTDenoising_Diffusion_N2N/example_data/histogram_equalization/bins_mapped.npy')
    generator = Generator.Dataset_2D(
        img_list = np.array([condition_file]),
        image_size = image_size,

        num_slices_per_image = 50,
        random_pick_slice = False,
        slice_range = [30,80],

        bins = bins,
        bins_mapped = bins_mapped,
        histogram_equalization = histogram_equalization,
        background_cutoff = background_cutoff,
        maximum_cutoff = maximum_cutoff,
        normalize_factor = normalize_factor,)

    # # sample:
    sampler = noise2noise.Sampler(model,generator,batch_size = 1, image_size = image_size)

    pred_img = sampler.sample_2D(trained_model_filename, condition_img)
    pred_img_final = pred_img
    
    # save
    nb.save(nb.Nifti1Image(pred_img_final, affine), os.path.join(save_folder_case, 'pred_img.nii.gz'))
    

