import numpy as np
import os
import pandas as pd


class Build():  # this build class for lowdose CT (mayo dataset), and MR dataset (where we have odd and even reconstructions)
    def __init__(self,file_list):
        self.a = 1
        self.file_list = file_list
        # self.data = pd.read_excel(file_list, dtype = {'Patient_ID': str, 'Patient_subID': str})
        self.data = pd.read_excel(file_list, dtype = {'Patient_ID': str})

    def __build__(self,batch_list, distill=False):
        for b in range(len(batch_list)):
            cases = self.data.loc[self.data['batch'] == batch_list[b]]
            if b == 0:
                c = cases.copy()
            else:
                c = pd.concat([c,cases])

        batch_list = np.asarray(c['batch']) if 'batch' in c.columns else None
        patient_id_list = np.asarray(c['Patient_ID']) if 'Patient_ID' in c.columns else None
        random_num_list = np.asarray(c['random_num']) if 'random_num' in c.columns else None
        noise_file_all_list = np.asarray(c['simulation_file_all']) if 'simulation_file_all' in c.columns else None
        noise_file_odd_list = np.asarray(c['simulation_file_odd']) if 'simulation_file_odd' in c.columns else None
        noise_file_even_list = np.asarray(c['simulation_file_even']) if 'simulation_file_even' in c.columns else None
        ground_truth_file_list = np.asarray(c['ground_truth_file']) if 'ground_truth_file' in c.columns else None
        slice_num_list = np.asarray(c['slice_num']) if 'slice_num' in c.columns else None if 'slice_num' in c.columns else None
        if distill:
            generated_20_file_list = np.asarray(c['generated_20_file']) if 'generated_20_file' in c.columns else None
            generated_10_file_list = np.asarray(c['generated_10_file']) if 'generated_10_file' in c.columns else None

        if distill:
            return batch_list, patient_id_list, random_num_list, noise_file_all_list, noise_file_odd_list, noise_file_even_list, ground_truth_file_list, slice_num_list, generated_20_file_list, generated_10_file_list
        else:
            return batch_list, patient_id_list, random_num_list, noise_file_all_list, noise_file_odd_list, noise_file_even_list, ground_truth_file_list, slice_num_list

class Build_thinsliceCT(): # this is the build class for thinslice CT (brain CT) where we use adjacent slices as condition
    def __init__(self,file_list):
        self.a = 1
        self.file_list = file_list
        self.data = pd.read_excel(file_list, dtype = {'Patient_ID': str, 'Patient_subID': str})
        # self.data = pd.read_excel(file_list, dtype = {'Patient_ID': str})

    def __build__(self,batch_list):
        for b in range(len(batch_list)):
            cases = self.data.loc[self.data['batch'] == batch_list[b]]
            if b == 0:
                c = cases.copy()
            else:
                c = pd.concat([c,cases])

        batch_list = np.asarray(c['batch']) if 'batch' in c.columns else None
        patient_id_list = np.asarray(c['Patient_ID']) if 'Patient_ID' in c.columns else None
        patient_subid_list = np.asarray(c['Patient_subID']) if 'Patient_subID' in c.columns else None
        random_num_list = np.asarray(c['random_num']) if 'random_num' in c.columns else None
        noise_file_list = np.asarray(c['noise_file']) if 'noise_file' in c.columns else None
        ground_truth_file_list = np.asarray(c['ground_truth_file']) if 'ground_truth_file' in c.columns else None
        # slice_num_list = np.asarray(c['slice_num']) if 'slice_num' in c.columns else None

        
        return batch_list, patient_id_list, patient_subid_list, random_num_list,noise_file_list, ground_truth_file_list


class Build_EM(): # this is the build class for electron microscopy where we have two independent noisy simulations
    def __init__(self,file_list):
        self.a = 1
        self.file_list = file_list
        self.data = pd.read_excel(file_list, dtype = {'Patient_ID': str, 'Patient_subID': str})
        # self.data = pd.read_excel(file_list, dtype = {'Patient_ID': str})

    def __build__(self,batch_list):
        for b in range(len(batch_list)):
            cases = self.data.loc[self.data['batch'] == batch_list[b]]
            if b == 0:
                c = cases.copy()
            else:
                c = pd.concat([c,cases])

        batch_list = np.asarray(c['batch']) if 'batch' in c.columns else None
        patient_id_list = np.asarray(c['Patient_ID']) if 'Patient_ID' in c.columns else None
        patient_subid_list = np.asarray(c['Patient_subID']) if 'Patient_subID' in c.columns else None
        random_num_list = np.asarray(c['random_num']) if 'random_num' in c.columns else None
        simulation_file_1_list = np.asarray(c['simulation_file_1']) if 'simulation_file_1' in c.columns else None
        simulation_file_2_list = np.asarray(c['simulation_file_2']) if 'simulation_file_2' in c.columns else None
        ground_truth_file_list = np.asarray(c['ground_truth_file']) if 'ground_truth_file' in c.columns else None
        slice_num_list = np.asarray(c['slice_num']) if 'slice_num' in c.columns else None

        
        return batch_list, patient_id_list, patient_subid_list, random_num_list,simulation_file_1_list, simulation_file_2_list, ground_truth_file_list, slice_num_list