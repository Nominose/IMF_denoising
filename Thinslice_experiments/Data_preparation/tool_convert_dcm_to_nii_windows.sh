#!/usr/bin/env bash

# run in terminal of your own laptop
# In Windows, make sure you do these things first:
# 1. Install WSL, we need to run bash script in WSL
# 2. In the terminal of WSL, Install dcm2niix in WSL, run "sudo apt update" and then "sudo apt install dcm2niix"
# 3. In the terminal of WSL, install dos2unix, run "sudo apt update" and then "sudo apt install dos2unix"
# 4. In the terminal of WSL, run "dos2unix tool_convert_dcm_to_nii_windows.sh" to convert the format of this script from Windows to Linux
# 5. to run this script, in the terminal of WSL, navigate to the folder where this script is saved, then run "bash tool_convert_dcm_to_nii_windows.sh"

set -o nounset
set -o errexit
set -o pipefail

# define the folder where dcm2niix function is saved, save it in your local laptop
# dcm2niix_fld="/mnt/d/Github/Diffusion_denoising_thin_slice/Data_preparation/dcm2niix_11-Apr-2019/"

main_path="/mnt/d/Data/low_dose_CT/"
# # define patient lists (the directory where you save all the patient data)
PATIENTS=(${main_path}/dcm_quarter/*)

echo ${#PATIENTS[@]}


for p in ${PATIENTS[*]};
do

  echo ${p}
  
  if [ -d ${p} ];
  then

 
  patient_id=$(basename ${p})

  output_folder=${main_path}/nii_imgs_quarter
  mkdir -p ${output_folder}/${patient_id}/
  nii_folder=${output_folder}/${patient_id}/

  IMGS=(${p}/*)  # Find all the images under this patient ID
  

  for i in $(seq 0 $(( ${#IMGS[*]} - 1 )));
      do

      echo ${IMGS[${i}]}
      
      if [ "$(ls -A ${IMGS[${i}]})" ]; then  # check whether the image folder is empty
        
        filename='img'
        # o_file=${nii_folder}${filename}_${i}.nii.gz 
        o_file=${nii_folder}${filename}.nii.gz
        echo ${o_file}

        if [ -f ${o_file} ];then
          echo "already done this file"
          continue

        else
        # if dcm2niix doesn't work (error = ignore image), remove -i y
        # in Window
        dcm2niix -i y -m y -b n -o "${nii_folder}" -f "${filename}" -9 -z y "${IMGS[${i}]}"
        fi

      else
        echo "${IMGS[${i}]} is emtpy; Skipping"
        continue
      fi
      
    done

  else
    echo "${p} missing dicom image folder"
    continue
    
  fi

done
