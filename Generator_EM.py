# dataset classes

import os
import numpy as np
import nibabel as nb
import random
from scipy import ndimage
from skimage.measure import block_reduce

import torch
from torch.utils.data import Dataset
import Diffusion_denoising_thin_slice.Data_processing as Data_processing
import Diffusion_denoising_thin_slice.functions_collection as ff

# random function
def random_rotate(i, z_rotate_degree = None, z_rotate_range = [-10,10], fill_val = None, order = 1):
    # only do rotate according to z (in-plane rotation)
    if z_rotate_degree is None:
        z_rotate_degree = random.uniform(z_rotate_range[0], z_rotate_range[1])

    if fill_val is None:
        fill_val = np.min(i)
    
    if z_rotate_degree == 0:
        return i, z_rotate_degree
    else:
        if len(i.shape) == 2:
            return Data_processing.rotate_image(np.copy(i), z_rotate_degree, order = order, fill_val = fill_val, ), z_rotate_degree
        else:
            return Data_processing.rotate_image(np.copy(i), [0,0,z_rotate_degree], order = order, fill_val = fill_val, ), z_rotate_degree

def random_translate(i, x_translate = None,  y_translate = None, translate_range = [-10,10]):
    # only do translate according to x and y
    if x_translate is None or y_translate is None:
        x_translate = int(random.uniform(translate_range[0], translate_range[1]))
        y_translate = int(random.uniform(translate_range[0], translate_range[1]))
    
    if len(i.shape) == 2:
        return Data_processing.translate_image(np.copy(i), [x_translate,y_translate]), x_translate,y_translate
    else:
        return Data_processing.translate_image(np.copy(i), [x_translate,y_translate,0]), x_translate,y_translate


class Dataset_2D(Dataset):
    def __init__(
        self,
        gt_file_list,
        simulation_1_file_list,
        simulation_2_file_list,

        num_slices_per_image,
        random_pick_slice,
        slice_range, # None or [a,b]

        background_cutoff, 
        maximum_cutoff,
        normalize_factor,
        final_max = 1,
        final_min = -1,

        image_size = None,

        num_patches_per_slice = None,
        patch_size = None,
        preset_patch_origin = None,

        shuffle = False,
        augment = False,
        augment_frequency = 0,

        preload = False,
        preload_data = None,):

        super().__init__()
        self.gt_file_list = gt_file_list
        self.simulation_1_file_list = simulation_1_file_list
        self.simulation_2_file_list = simulation_2_file_list

        self.image_size = image_size

        self.num_slices_per_image = num_slices_per_image
        self.random_pick_slice = random_pick_slice
        self.slice_range = slice_range

        self.num_patches_per_slice = num_patches_per_slice
        self.patch_size = patch_size
        self.preset_patch_origin = preset_patch_origin

        self.preload = preload
        self.preload_gt_data = preload_data[-1] if self.preload == True else None
        self.preload_simulation_1_data = preload_data[0] if self.preload == True else None
        self.preload_simulation_2_data = preload_data[1] if self.preload == True else None

        self.background_cutoff = background_cutoff
        self.maximum_cutoff = maximum_cutoff
        self.normalize_factor = normalize_factor
        self.final_max = final_max
        self.final_min = final_min
        self.shuffle = shuffle
        self.augment = augment
        self.augment_frequency = augment_frequency

        self.num_files = len(gt_file_list)

        self.index_array = self.generate_index_array()
        self.current_gt_file = None
        self.current_gt_data = None
        self.current_simulation_1_file = None
        self.current_simulation_1_data = None
        self.current_simulation_2_file = None
        self.current_simulation_2_data = None
       

    def generate_index_array(self): 
        np.random.seed()
        index_array = []; index_array_patches = []
        
        if self.shuffle == True:
            f_list = np.random.permutation(self.num_files)
        else:
            f_list = np.arange(self.num_files)

        for f in f_list:
            s_list = np.arange(self.num_slices_per_image)
            for s in s_list:
                index_array.append([f, s])
                if self.num_patches_per_slice != None:
                    patch_list = np.arange(self.num_patches_per_slice)
                    for p in patch_list:
                        index_array_patches.append([f, s,p])
        if self.num_patches_per_slice != None:
            return index_array_patches
        else:
            return index_array

    def __len__(self):
        if self.num_patches_per_slice != None:
            return self.num_files * self.num_slices_per_image * self.num_patches_per_slice
        else:
            return self.num_files * self.num_slices_per_image
    

    def load_file(self, filename = None, preload_data = None):
        if self.preload == False:
            ii = nb.load(filename).get_fdata()
        else:
            ii = preload_data
    
        # # do histogram equalization first
        # if self.histogram_equalization == True: 
        #     ii = Data_processing.apply_transfer_to_img(ii, self.bins, self.bins_mapped)

        # cutoff and normalization
        ii = Data_processing.cutoff_intensity(ii,cutoff_low = self.background_cutoff, cutoff_high = self.maximum_cutoff)

        if self.final_max != 1 or self.final_min != 0:
            ii = Data_processing.normalize_image(ii, normalize_factor = self.normalize_factor, image_max = self.maximum_cutoff, image_min = self.background_cutoff ,final_max = self.final_max, final_min = self.final_min, invert = False)
        if self.image_size is not None: 
            if ii.shape[0] != self.image_size[0] or ii.shape[1] != self.image_size[1]:
                ii = Data_processing.crop_or_pad(ii, [self.image_size[0], self.image_size[1], ii.shape[2]], value= np.min(ii))
        # print('max and min after normalization:', np.max(ii), np.min(ii))
        return ii
        
    def __getitem__(self, index):
        # print('in this geiitem, self.index_array is: ', self.index_array)
        if self.num_patches_per_slice != None:
            f,s,p = self.index_array[index]
        else:
            f,s = self.index_array[index]

        gt_filename = self.gt_file_list[f]
        simulation_1_filename = self.simulation_1_file_list[f]
        simulation_2_filename = self.simulation_2_file_list[f]
        # print('simulation_1_filename: ', simulation_1_filename, '; simulation_2_filename: ', simulation_2_filename)

        # load data
        if simulation_1_filename != self.current_simulation_1_file:
            if self.preload == False:
                simulation_1_img = self.load_file(filename = simulation_1_filename)
            else:
                simulation_1_img = self.load_file(preload_data = self.preload_simulation_1_data[f])
            self.current_simulation_1_file = simulation_1_filename
            self.current_simulation_1_data = np.copy(simulation_1_img)

            # define slice range
            if self.slice_range == None:
                total_slice_range = [0,self.current_simulation_1_data.shape[2]] 
                # print('total slice range is: ', total_slice_range)
            else:
                total_slice_range = self.slice_range
      
            if self.random_pick_slice == False:
                self.slice_index_list = np.arange(total_slice_range[0], total_slice_range[1])
                self.slice_index_list = self.slice_index_list[:self.num_slices_per_image]
            else:
                self.slice_index_list = np.random.permutation(np.arange(total_slice_range[0], total_slice_range[1]))[:self.num_slices_per_image]

        if simulation_2_filename != self.current_simulation_2_file:
            if self.preload == False:
                simulation_2_img = self.load_file(filename = simulation_2_filename)
            else:
                simulation_2_img = self.load_file(preload_data = self.preload_simulation_2_data[f])
            self.current_simulation_2_file = simulation_2_filename
            self.current_simulation_2_data = np.copy(simulation_2_img)
      
        # print('slice index list: ', self.slice_index_list)

        # pick the slice
        # print('s is: ', s)
        slice_index = self.slice_index_list[s]

        # set x0 and condition data (important)
        self.current_x0_data = np.copy(self.current_simulation_1_data)
        self.current_condition_data = np.copy(self.current_simulation_2_data)

        # pick the patch:
        if self.num_patches_per_slice != None:
            x_shape, y_shape = self.current_condition_data.shape[0], self.current_condition_data.shape[1]
            random_origin_x, random_origin_y = random.randint(0, x_shape - self.patch_size[0]), random.randint(0, y_shape - self.patch_size[1])
            if self.preset_patch_origin is not None:
                random_origin_x, random_origin_y = self.preset_patch_origin[0], self.preset_patch_origin[1]

        # target image
        x0_image_data = np.copy(self.current_x0_data)[:,:,slice_index]
        # crop the patch
        if self.num_patches_per_slice != None:
            x0_image_data = x0_image_data[random_origin_x:random_origin_x + self.patch_size[0], random_origin_y:random_origin_y + self.patch_size[1]]
        
        # condition image
        condition_image_data = np.copy(self.current_condition_data)[:,:,slice_index]
        if self.num_patches_per_slice != None:
            condition_image_data = condition_image_data[random_origin_x:random_origin_x + self.patch_size[0], random_origin_y:random_origin_y + self.patch_size[1]]
          
        # augmentation
        if self.augment == True:
            if random.uniform(0,1) < self.augment_frequency:
                x0_image_data, z_rotate_degree = random_rotate(x0_image_data,  order = 1)
                x0_image_data, x_translate, y_translate = random_translate(x0_image_data)
                condition_image_data, _ = random_rotate(condition_image_data, z_rotate_degree = z_rotate_degree, order = 1)
                condition_image_data, _, _ = random_translate(condition_image_data, x_translate = x_translate, y_translate = y_translate)
                # print('augment : z_rotate_degree, x_translate, y_translate: ', z_rotate_degree, x_translate, y_translate)
        
            
        x0_image_data = torch.from_numpy(x0_image_data).unsqueeze(0).float()
        condition_image_data = torch.from_numpy(condition_image_data).unsqueeze(0).float()

        # print('shape of x0 image data: ', x0_image_data.shape, ' and condition image data: ', condition_image_data.shape)
        # print('max and min of x0 image data: ', torch.max(x0_image_data), torch.min(x0_image_data), ' and condition image data: ', torch.max(condition_image_data), torch.min(condition_image_data))
        return x0_image_data, condition_image_data
    
    def on_epoch_end(self):
        print('now run on_epoch_end function')
        self.index_array = self.generate_index_array()
        self.current_gt_file = None
        self.current_gt_data = None
        self.current_simulation_1_file = None
        self.current_simulation_1_data = None
        self.current_simulation_2_file = None
        self.current_simulation_2_data = None
    

