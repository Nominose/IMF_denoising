#!/usr/bin/env bash

# run in terminal of your own laptop

##############
## Settings ##
##############

set -o nounset
set -o errexit
set -o pipefail

#shopt -s globstar nullglob

###########
## Logic ##
###########

# define the folder where dcm2niix function is saved, save it in your local laptop
dcm2niix_fld="/mnt/d/Github/Diffusion_denoising_thin_slice/Data_preparation/dcm2niix_11-Apr-2019/"

main_path="/mnt/d/Data/low_dose_CT/full_3mm"
# # define patient lists (the directory where you save all the patient data)
PATIENTS=(${main_path}/*)

echo ${#PATIENTS[@]}


# for p in ${PATIENTS[*]};
# do

#   echo ${p}
  
#   if [ -d ${p} ];
#   then

#   patient_id=$(basename $(dirname ${p}))
#   patient_subid=$(basename ${p})

#   output_folder=${main_path}/nii_imgs_202404_new
#   mkdir -p ${output_folder}/${patient_id}/
#   mkdir -p ${output_folder}/${patient_id}/${patient_subid}/
#   nii_folder=${output_folder}/${patient_id}/${patient_subid}/

#   IMGS=(${p}/*)  # Find all the images under this patient ID
  

#   for i in $(seq 0 $(( ${#IMGS[*]} - 1 )));
#       do

#       echo ${IMGS[${i}]}
      
#       if [ "$(ls -A ${IMGS[${i}]})" ]; then  # check whether the image folder is empty
        
#         filename='img'
#         o_file=${nii_folder}${filename}_${i}.nii.gz # define the name of output nii files, the name will be "timeframe.nii.gz"
#         echo ${o_file}

#         if [ -f ${o_file} ];then
#           echo "already done this file"
#           continue

#         else
#         # if dcm2niix doesn't work (error = ignore image), remove -i y
#         ${dcm2niix_fld}dcm2niix -i y -m y -b n -o "${nii_folder}" -f "${filename}" -9 -z y "${IMGS[${i}]}"
#         fi

#       else
#         echo "${IMGS[${i}]} is emtpy; Skipping"
#         continue
#       fi
      
#     done

#   else
#     echo "${p} missing dicom image folder"
#     continue
    
#   fi

# done
