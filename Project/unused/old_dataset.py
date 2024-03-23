# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 00:06:08 2024

@author: 19714
"""

import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class MPISintelDataset(Dataset):
    def __init__(self, root_dir, split='training', pass_type='final', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.pass_type = pass_type
        self.transform = transform

        # List all the sequence directories in the specified split
        self.sequence_dirs = sorted(os.listdir(os.path.join(root_dir, split, pass_type)))

        # Create a list of all image file paths and corresponding flow file paths
        self.image_paths = []
        self.flow_paths = []
        for seq_dir in self.sequence_dirs:
            image_dir = os.path.join(root_dir, split, pass_type, seq_dir)
            flow_dir = os.path.join(root_dir, split, 'flow', seq_dir)
            num_frames = len(os.listdir(image_dir))
            #print(f"Sequence {seq_dir}: {num_frames} frames")  # Print the number of frames in each sequence
            
            for frame_idx in range(1, num_frames):
                img1_path = os.path.join(image_dir, f'frame_{frame_idx:04d}.png')
                img2_path = os.path.join(image_dir, f'frame_{frame_idx+1:04d}.png')
                flow_path = os.path.join(flow_dir, f'frame_{frame_idx:04d}.flo')

                self.image_paths.append((img1_path, img2_path))
                self.flow_paths.append(flow_path)
        #print(f"Total image pairs loaded: {len(self.image_paths)}")  # Print the total number of image pairs loaded
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img1_path, img2_path = self.image_paths[idx]
        flow_path = self.flow_paths[idx]

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        flow = self.read_flo_file(flow_path)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, flow

    def read_flo_file(self, file_path):
        with open(file_path, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if magic != 202021.25:
                raise Exception('Invalid .flo file format')
            width, height = np.fromfile(f, np.int32, count=2)
            flow_data = np.fromfile(f, np.float32, count=2 * width * height)
        return flow_data.reshape((height, width, 2)).transpose(2, 0, 1)

# # Example usage:
# if __name__ == "__main__":
#     dataset_root = 'C:/Users/19714/OneDrive - Oregon State University/Assignments/Winter 2024/Deep learning/Final Project/MPI-Sintel-complete'
#     train_dataset = MPISintelDataset(root_dir=dataset_root, split='training', pass_type='final')
#     train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

#     print(f"Total sequences in dataset: {len(train_dataset.sequence_dirs)}")
#     print(f"Total image pairs in dataset: {len(train_dataset)}")

