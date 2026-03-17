import sys 
sys.path.append('/host/d/Github')
import os
import torch
import numpy as np 
import nibabel as nb
import Diffusion_denoising_thin_slice.functions_collection as ff


def prepare_sliced_data():
    cases = ff.find_all_target_files(['*'], '/host/d/Data/low_dose_CT/simulation_v2')
    for c in cases:
        patient_id = os.path.basename(c)
     
        if os.path.exists(os.path.join('/host/d/Data/low_dose_CT/nii_imgs', patient_id, 'img_sliced.nii.gz')):
            continue
        # simulated data (pick slice 100-200, except L109 28-128)
        file = os.path.join('/host/d/Data/low_dose_CT/simulation_v2', patient_id, 'gaussian_random_0', 'recon_all.nii.gz')
        img_file = nb.load(file)
        img = img_file.get_fdata()[:,:, 28:128] if patient_id == 'L109' else img_file.get_fdata()[:,:, 100:200]
        assert img.shape[2] == 100, f"Error: patient {patient_id} has unexpected number of slices: {img.shape[2]}"
        new_img = nb.Nifti1Image(img, img_file.affine, img_file.header)
        nb.save(new_img, os.path.join('/host/d/Data/low_dose_CT/simulation_v2', patient_id, 'gaussian_random_0', 'recon_all_sliced.nii.gz'))

        file = os.path.join('/host/d/Data/low_dose_CT/simulation_v2', patient_id, 'gaussian_random_0', 'recon_odd.nii.gz')
        img_file = nb.load(file)
        img = img_file.get_fdata()[:,:, 28:128] if patient_id == 'L109' else img_file.get_fdata()[:,:, 100:200]
        new_img = nb.Nifti1Image(img, img_file.affine, img_file.header)
        nb.save(new_img, os.path.join('/host/d/Data/low_dose_CT/simulation_v2', patient_id, 'gaussian_random_0', 'recon_odd_sliced.nii.gz'))    

        file = os.path.join('/host/d/Data/low_dose_CT/simulation_v2', patient_id, 'gaussian_random_0', 'recon_even.nii.gz')
        img_file = nb.load(file)
        img = img_file.get_fdata()[:,:, 28:128] if patient_id == 'L109' else img_file.get_fdata()[:,:, 100:200]
        new_img = nb.Nifti1Image(img, img_file.affine, img_file.header)
        nb.save(new_img, os.path.join('/host/d/Data/low_dose_CT/simulation_v2', patient_id, 'gaussian_random_0', 'recon_even_sliced.nii.gz'))

        # original data
        file = os.path.join('/host/d/Data/low_dose_CT/nii_imgs', patient_id, 'img.nii.gz')
        img_file = nb.load(file)
        img = img_file.get_fdata()[:,:, 28:128] if patient_id == 'L109' else img_file.get_fdata()[:,:, 100:200]
        new_img = nb.Nifti1Image(img, img_file.affine, img_file.header)
        nb.save(new_img, os.path.join('/host/d/Data/low_dose_CT/nii_imgs', patient_id, 'img_sliced.nii.gz'))
