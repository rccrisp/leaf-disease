import os
from PIL import Image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.filenames = [filename for filename in os.listdir(data_dir)]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.data_dir, filename)
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)
        return {"filename": filename, "image": image}
