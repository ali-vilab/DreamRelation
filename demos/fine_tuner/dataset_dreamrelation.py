from pathlib import Path

import click
import torch
from torch.utils.data import DataLoader, Dataset

import os
from PIL import Image
from torchvision import transforms
import numpy as np
import torch.nn.functional as F


def resize_with_mask_preservation(image, h, w):
    if isinstance(image, Image.Image):
        image = transforms.ToTensor()(image)

    batch_size, original_h, original_w = image.shape
    scale_x = w / original_w
    scale_y = h / original_h

    # Resize the image using nearest neighbor interpolation for each batch
    resized_image = F.interpolate(image.unsqueeze(1), size=(h, w), mode='nearest').squeeze(1)

    # Find positions with value 1 in the original image
    pos = torch.nonzero(image == 1, as_tuple=False)

    # Calculate new positions after scaling
    new_x = (pos[:, 2].float() * scale_x).floor().long().clamp(0, w - 1)
    new_y = (pos[:, 1].float() * scale_y).floor().long().clamp(0, h - 1)
    batch_indices = pos[:, 0]

    # Preserve the mask in the resized image
    resized_image[batch_indices, new_y, new_x] = 1.0

    return resized_image


def load_to_cpu(x):
    return torch.load(x, map_location=torch.device("cpu"), weights_only=True)


class LatentEmbedDatasetMask(Dataset):
    def __init__(self, file_paths, mask_paths, repeat=1, frame_offset=0):
        self.items = []
        self.full_mask_paths = os.path.join(mask_paths)
        self.temporal_downsample_ratio = 6
        self.frame_offset = frame_offset
        self.subject_mask_color_dict = {}
        
        for p in file_paths:
            latent_path = Path(p).with_suffix(".latent.pt")
            embed_path = Path(p).with_suffix(".embed.pt")

            video_name = os.path.splitext(os.path.basename(p))[0]
            full_mask_path = Path(os.path.join(self.full_mask_paths, video_name))
            
            if latent_path.is_file() and embed_path.is_file() and full_mask_path.is_dir():
                self.items.append((latent_path, embed_path, full_mask_path))
        
        self.items = self.items * repeat
        print(f"Loaded {len(self.items)}/{len(file_paths)} valid file triplets.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        latent_path, embed_path, full_mask_path = self.items[idx]
        latent, embed = load_to_cpu(latent_path), load_to_cpu(embed_path)
        latent_shape = latent["mean"].shape
        video_name = os.path.basename(latent_path).split('.')[0]
        full_latent_mask = self.load_full_mask_folder_to_tensors(full_mask_path, self.temporal_downsample_ratio, latent_shape, video_name, self.frame_offset)
        if full_latent_mask.shape[1] == 1:
            zeros = torch.zeros(full_latent_mask.shape[0], 1, full_latent_mask.shape[2], full_latent_mask.shape[3])
            full_latent_mask = torch.cat((full_latent_mask, zeros), dim=1)

        return (latent, embed, full_latent_mask)
    
    def load_full_mask_folder_to_tensors(self, folder_path, temporal_downsample_ratio, latent_shape, video_name, frame_offset):
        b, c, f, h, w = latent_shape
        
        if not isinstance(folder_path, Path):
            folder_path = Path(folder_path)
        
        tensors_list = []

        files = sorted(folder_path.glob('*.png'))
        if not files:
            raise Exception("no files find!")
        
        file_count = len(files)
        
        ### Directly add the first frame
        if (0+frame_offset) >= file_count:
            print(f"Error!!--------Index {0+frame_offset} as First Frame of {folder_path} exceed video frame length!--------") 
            tensors = torch.zeros(f, c, h, w)
            return tensors
        first_image = Image.open(files[0+frame_offset])

        first_tensor_image = torch.from_numpy(np.array(first_image)).permute(2, 0, 1)
        # Get unique colors
        if video_name in self.subject_mask_color_dict.keys():
            unique_colors = self.subject_mask_color_dict[video_name]
        else:
            # Flatten the image to a 2D tensor where each row is a color
            flatten_image = first_tensor_image.flatten(1).permute(1, 0)  # Now it's (N, C)
            # Get all unique colors
            unique_colors = torch.unique(flatten_image, dim=0)
            self.subject_mask_color_dict[video_name] = unique_colors

        # Generate masks
        subject_masks_first = []
        for color in unique_colors:
            if torch.any(color != 0):  # Skip the black background
                # Create a mask by comparing colors
                mask = (first_tensor_image.permute(1, 2, 0) == color).all(dim=2)  # Shape: HxW
                subject_masks_first.append(mask.float())
        
        first_tensor_subjects = torch.stack(subject_masks_first)
        resized_first_tensor_subjects = resize_with_mask_preservation(first_tensor_subjects, h, w)
        tensors_list.append(resized_first_tensor_subjects)
        
        # Process each frame to extract masks for subjects
        for i in range(0, f-1):
            start_index = i * temporal_downsample_ratio + 1 + frame_offset
            
            if start_index >= file_count:
                break
            
            end_index = min(start_index + temporal_downsample_ratio, file_count)
            
            masks = []
            for idx in range(start_index, end_index):
                # Load and convert image to tensor
                image = Image.open(files[idx])
                tensor_image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
                # Generate masks
                subject_masks = []
                for color in unique_colors:
                    if torch.any(color != 0):  # Skip the black background
                        mask = (tensor_image.permute(1, 2, 0) == color).all(dim=2)  # Shape: HxW
                        subject_masks.append(mask.float())

                # Stack the subject masks into a tensor
                tensor_subjects = torch.stack(subject_masks)
                resized_tensor_subjects = resize_with_mask_preservation(tensor_subjects, h, w)
                masks.append(resized_tensor_subjects)
            
            if masks:
                average_mask = torch.mean(torch.stack(masks), dim=0)
                tensors_list.append(average_mask)
        
        if len(tensors_list) > 0:
            tensors = torch.stack(tensors_list)
        else:
            raise Exception("tensors_list is empty!")
        
        if tensors.shape[0] != f:
            print(f"Error!!--------tensor shape is {tensors.shape} in --{video_name}--------") 
            tensors = torch.zeros(f, c, h, w)
        
        return tensors
