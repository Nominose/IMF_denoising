import sys 
sys.path.append('/workspace/Documents')
import os
import pandas as pd
import numpy as np
import shutil
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list

## change file name
# build_sheet =  Build_list.Build(os.path.join('/mnt/camca_NAS/denoising/Patient_lists/fixedCT_static_simulation_train_test_gaussian.xlsx'))
# _,patient_id_list,patient_subid_list,random_num_list, condition_list, x0_list = build_sheet.__build__(batch_list = [0,1,2,3,4,5])
# n = ff.get_X_numbers_in_interval(total_number = patient_id_list.shape[0],start_number = 0,end_number = 1, interval = 3)
# print('total number:', n.shape[0])

# for i in range(0, n.shape[0]):
#     patient_id = patient_id_list[n[i]]
#     patient_subid = patient_subid_list[n[i]]
#     print(f"Processing patient {patient_id} {patient_subid}...")

#     folder = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_DDPM_gaussian_2D/pred_images',patient_id, patient_subid)
#     for index in range(1,21):
#         correct_name = os.path.join(folder,'random_0', 'epoch70_' + str(index))
#         if os.path.isdir(correct_name):
#             if os.path.isdir(os.path.join(folder,'random_0', 'epoch70_' + str(index) + '_200')):
#                 # delete the folder
#                 shutil.rmtree(os.path.join(folder,'random_0', 'epoch70_' + str(index) + '_200'))
#                 print('deleted:', os.path.join(folder,'random_0', 'epoch70_' + str(index) + '_200'))
#             else:
#                 continue
#         else:
#             if os.path.isdir(os.path.join(folder,'random_0', 'epoch70_' + str(index) + '_200')):
#                 # rename the folder
#                 os.rename(os.path.join(folder,'random_0', 'epoch70_' + str(index) + '_200'),correct_name)
#                 print('renamed:', os.path.join(folder,'random_0', 'epoch70_' + str(index) + '_200'), 'to', correct_name)
            

### transfer data from NAS to local
nas_path = '/mnt/camca_NAS/denoising/Data'
local_path = '/workspace/Documents/Data/denoising'

# patient_sheet = pd.read_excel(os.path.join('/mnt/camca_NAS/denoising/Patient_lists/fixedCT_static_shuffled_batched.xlsx'),dtype={'Patient_ID': str, 'Patient_subID': str})
# noise_types = ['gaussian','possion']

# for noise_type in noise_types:
#     results = []
#     for i in range(0, len(patient_sheet)):
#         patient_id = patient_sheet['Patient_ID'][i]
#         patient_subid = patient_sheet['Patient_subID'][i]
#         batch = patient_sheet['batch'][i]
#         print(f"Processing patient {patient_id} {patient_subid}...")

#         ground_truth_file = os.path.join(nas_path,'fixedCT',patient_id,patient_subid,'img_thinslice_partial.nii.gz')
#         ff.make_folder([os.path.join(local_path, 'fixedCT'), os.path.join(local_path, 'fixedCT',patient_id),os.path.join(local_path, 'fixedCT',patient_id,patient_subid)])

#         # os.remove(os.path.join(local_path,'fixedCT',patient_id,patient_subid,'img_thinslice_partial.nii.gz'))

#         if not os.path.exists(os.path.join(local_path,'fixedCT',patient_id,patient_subid,'img_thinslice_partial.nii.gz')):
#             print('copy')
#             shutil.copy(ground_truth_file,os.path.join(local_path,'fixedCT',patient_id,patient_subid,'img_thinslice_partial.nii.gz'))

#         for simulation_n in range(0,2):
#             simulation_files = os.path.join(nas_path,'simulation',patient_id,patient_subid,noise_type + '_random_' + str(simulation_n), 'recon.nii.gz')
#             ff.make_folder([os.path.join(local_path, 'simulation'), os.path.join(local_path, 'simulation',patient_id),os.path.join(local_path, 'simulation',patient_id,patient_subid),os.path.join(local_path, 'simulation',patient_id,patient_subid,noise_type + '_random_' + str(simulation_n))])
#             if not os.path.exists(os.path.join(local_path,'simulation',patient_id,patient_subid,noise_type + '_random_' + str(simulation_n),'recon.nii.gz')):
#                 shutil.copy(simulation_files,os.path.join(local_path,'simulation',patient_id,patient_subid,noise_type + '_random_' + str(simulation_n),'recon.nii.gz'))

# transfer generated images
# build_sheet =  Build_list.Build(os.path.join('/mnt/camca_NAS/denoising/Patient_lists/fixedCT_static_simulation_train_test_gaussian_NAS.xlsx'))
# _,patient_id_list,patient_subid_list,random_num_list, condition_list, x0_list = build_sheet.__build__(batch_list = [0,1,2,3,4,5]) 
# n = ff.get_X_numbers_in_interval(total_number = patient_id_list.shape[0],start_number = 0,end_number = 1, interval = 2)

# for i in range(0,n.shape[0]):
#     patient_id = patient_id_list[n[i]]
#     patient_subid = patient_subid_list[n[i]]
#     random_num = random_num_list[n[i]]
#     print(i, patient_id, patient_subid, random_num)

#     data = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_gaussian_current_beta0/pred_images', patient_id, patient_subid, 'random_' + str(random_num), 'epoch61avg/pred_img_scans20.nii.gz')

#     if os.path.isfile(data) == 0:
#         print('not generated yet')
#         continue
#     des_folder = os.path.join(local_path, 'pred_images', patient_id, patient_subid, 'random_' + str(random_num))
#     ff.make_folder([os.path.join(local_path, 'pred_images'), os.path.join(local_path, 'pred_images', patient_id), os.path.join(local_path, 'pred_images', patient_id, patient_subid), os.path.join(local_path, 'pred_images', patient_id, patient_subid, 'random_' + str(random_num))])
#     if not os.path.exists(os.path.join(des_folder, 'pred_img_scans20.nii.gz')):
#         shutil.copy(data, os.path.join(des_folder, 'pred_img_scans20.nii.gz'))
#         # print('copied:', data, 'to', os.path.join(des_folder, 'pred_img_scans20.nii.gz'))

#     condition = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_gaussian_current_beta0/pred_images', patient_id, patient_subid, 'random_' + str(random_num), 'epoch61_1/condition_img.nii.gz')
#     des_folder = os.path.join(local_path, 'pred_images', patient_id, patient_subid, 'random_' + str(random_num))
#     ff.make_folder([os.path.join(local_path, 'pred_images'), os.path.join(local_path, 'pred_images', patient_id), os.path.join(local_path, 'pred_images', patient_id, patient_subid), os.path.join(local_path, 'pred_images', patient_id, patient_subid, 'random_' + str(random_num))])
#     if not os.path.exists(os.path.join(des_folder, 'condition_img.nii.gz')):
#         shutil.copy(condition, os.path.join(des_folder, 'condition_img.nii.gz'))
#         print('copied:', condition, 'to', os.path.join(des_folder, 'condition_img.nii.gz'))

    # gt = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_DDPM_gaussian_2D/pred_images', patient_id, patient_subid, 'random_' + str(random_num), 'epoch73_1/gt_img.nii.gz')
    # des_folder = os.path.join(local_path, 'pred_images', patient_id, patient_subid, 'random_' + str(random_num))
    # ff.make_folder([os.path.join(local_path, 'pred_images'), os.path.join(local_path, 'pred_images', patient_id), os.path.join(local_path, 'pred_images', patient_id, patient_subid), os.path.join(local_path, 'pred_images', patient_id, patient_subid, 'random_' + str(random_num))])
    # if not os.path.exists(os.path.join(des_folder, 'gt_img.nii.gz')):
    #     shutil.copy(gt, os.path.join(des_folder, 'gt_img.nii.gz'))
    #     print('copied:', gt, 'to', os.path.join(des_folder, 'gt_img.nii.gz'))



### delete
folders = ff.find_all_target_files(['pred*'],os.path.join('/mnt/camca_NAS/denoising/models/supervised_gaussian_current_beta0_distilled_EDM'))
for f in folders:
    if os.path.isdir(f):
        shutil.rmtree(f)
        print('deleted:', f)
    else:
        print('not a folder:', f)
   