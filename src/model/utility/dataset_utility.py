import os
import glob
import numpy as np
from scipy import misc
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

figure_configuration_names = ['center_single', 'distribute_four', 'distribute_nine', 'in_center_single_out_center_single', 'in_distribute_four_out_center_single', 'left_center_single_right_center_single', 'up_center_single_down_center_single']
        
class ToTensor(object):
    def __call__(self, sample):
        #return torch.tensor(sample, dtype=torch.float32)
        return torch.from_numpy(sample)

class dataset(Dataset):
    def __init__(self, root_dir, dataset_type, img_size, transform=None, shuffle=False):
        self.root_dir = root_dir
        self.transform = transform
        self.file_names = []
        for idx in [0,1,2,3,4,5,6]: #figure configurations
            tmp = [f for f in glob.glob(os.path.join(root_dir, figure_configuration_names[idx], "*.npz")) if dataset_type in os.path.basename(f)]
            self.file_names += tmp
        self.img_size = img_size
        self.embeddings = np.load(os.path.join(root_dir, 'embedding.npy'), allow_pickle=True, encoding='ASCII')
        self.shuffle = shuffle
        self.switch = [3,4,5,0,1,2,6,7]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        data_path = self.file_names[idx]
        data = np.load(data_path)
        image = data["image"].reshape(16, 160, 160)
        target = data["target"]
        structure = data["structure"]
        meta_target = data["meta_target"]
        meta_structure = data["meta_structure"]

        if self.shuffle:
            context = image[:8, :, :]
            choices = image[8:, :, :]
            indices = range(8)
            np.random.shuffle(indices)
            new_target = indices.index(target)
            new_choices = choices[indices, :, :]
            switch_2_rows = np.random.rand()            
            if switch_2_rows < 0.5:                
                context = context[self.switch, :, :]
            image = np.concatenate((context, new_choices))
            target = new_target
        
        resize_image_arr = []
        for idx in range(0, 16):
            # resize_image.append(misc.imresize(image[idx,:,:], (self.img_size, self.img_size)))
            # resize_image.append(np.array(Image.fromarray(image[idx,:,:])).resize((self.img_size, self.img_size)))
            img_to_resize = Image.fromarray(image[idx,:,:])
            resized_image = img_to_resize.resize((self.img_size, self.img_size))
            resized_arr = np.array(resized_image)
            resize_image_arr.append(resized_arr)
        
        resize_image_arr = np.stack(resize_image_arr)

        embedding = torch.zeros((6, 300), dtype=torch.float)
        indicator = torch.zeros(1, dtype=torch.float)
        element_idx = 0
        for element in structure:
            if element != '/':
                embedding[element_idx, :] = torch.tensor(self.embeddings.item().get(element), dtype=torch.float)
                element_idx += 1
        if element_idx == 6:
            indicator[0] = 1.
        # if meta_target.dtype == np.int8:
        #     meta_target = meta_target.astype(np.uint8)
        # if meta_structure.dtype == np.int8:
        #     meta_structure = meta_structure.astype(np.uint8)
    
        del data
        if self.transform:
            resize_image_arr = self.transform(resize_image_arr)
            target = torch.tensor(target, dtype=torch.long)
            meta_target = self.transform(meta_target) 
            meta_structure = self.transform(meta_structure)

        return resize_image_arr, target, meta_target, meta_structure, embedding, indicator
        
