import os
from PIL import Image
from torch.utils.data import Dataset
from leafdisease.utils.image import mean_smoothing

class CustomImageDataset(Dataset):
    def __init__(self, data_dir, threshold=0.5, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.threshold = threshold
        self.filenames = [filename for filename in os.listdir(data_dir)]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.data_dir, filename)
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
            noisy_mask = ((image+1)/2 != 0).all(dim=0).float()
            noisy_mask = noisy_mask.unsqueeze(0)
            mask = mean_smoothing(noisy_mask)
            mask = (mask > self.threshold).float()

        return {"filename": filename, "image": image, "mask": mask}