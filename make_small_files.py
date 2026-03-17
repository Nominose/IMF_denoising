import sys
sys.path.append('/workspace/Documents')
import os
import torch
import numpy as np 
import nibabel as nb
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list


main_path = '/mnt/camca_NAS/denoising/models'

model_name = 'unsupervised_gaussian_beta0'
epoch_num = 'epoch58_1'

data_path = os.path.join(main_path, model_name, 'pred_images')
save_path = os.path.join(main_path, model_name, 'pred_images_compressed'); os.makedirs(save_path, exist_ok=True)

build_sheet =  Build_list.Build(os.path.join('/mnt/camca_NAS/denoising/Patient_lists/fixedCT_static_simulation_train_test_gaussian_NAS.xlsx'))
_,patient_id_list,patient_subid_list,random_num_list, condition_list, x0_list = build_sheet.__build__(batch_list = [5]) 
print('total cases:', patient_id_list.shape[0])
n = ff.get_X_numbers_in_interval(total_number = patient_id_list.shape[0],start_number = 0,end_number = 1, interval = 2)

for i in range(0, n.shape[0]):
    patient_id = patient_id_list[n[i]]
    patient_subid = patient_subid_list[n[i]]
    random_num = random_num_list[n[i]]
    
    print(i,patient_id, patient_subid, random_num)

    original_image = ff.find_all_target_files([epoch_num+'/pred_img*'],os.path.join(data_path, patient_id, patient_subid, 'random_' + str(random_num)))
    print('original image: ', original_image[0])
    if len(original_image) == 0:
        print('no image found')
        continue


    epoch_base = os.path.basename(os.path.dirname(original_image[0]))
    save_path_data = os.path.join(save_path, patient_id, patient_subid, 'random_' + str(random_num), epoch_base)
    if os.path.exists(save_path_data):
        print('already done')
        continue
    
    # make it into integer
    data = original_image[0]
    a = nb.load(data)
    affine = a.affine
    img = a.get_fdata()
    img_integer = np.round(img).astype(np.int16)

    # save
    ff.make_folder([os.path.join(save_path, patient_id), os.path.join(save_path, patient_id, patient_subid), os.path.join(save_path, patient_id, patient_subid, 'random_' + str(random_num)), save_path_data])
    nb.save(nb.Nifti1Image(img_integer, affine), os.path.join(save_path_data, 'pred_img_int.nii.gz'))
 
    
    





