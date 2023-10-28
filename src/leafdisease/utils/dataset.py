import os
import torch
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset
from leafdisease.utils.image import mean_smoothing, normalise, denormalise

class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform, norm=True, threshold=0.05):
        self.data_dir = data_dir
        self.transform = transform
        self.threshold = threshold
        self.norm = norm
        self.filenames = [filename for filename in os.listdir(data_dir)]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.data_dir, filename)
        image = Image.open(image_path)

        image = self.transform(image)
            
        mask = torch.mean(image, dim=0, keepdim=True)
        mask = (mask > self.threshold).float()
        
        if self.norm:
            image = normalise(image)

        return {"filename": filename, "image": image, "mask": mask}