import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd

transform = transforms.Compose([
    transforms.ToTensor()
])

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): files
            root_dir (string): Directory where the image is located
            transform (callable, optional): 
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name)
        image_class = int(self.data_frame.iloc[idx, -1])  # Case where class is the last line

        if self.transform:
            image = self.transform(image)

        return image, image_class

# Creating datasets and data loaders
dataset = CustomDataset(csv_file='E:/sample_output.csv', root_dir='E:/', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

# use dataloader
for images, labels in dataloader:
    print(images.shape, labels)
