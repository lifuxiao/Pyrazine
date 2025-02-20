import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor()  # Convert PIL Image to Tensor
])

# Custom Dataset class to handle image loading and transformations
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')  # Ensure 3 channels
        content_label = int(self.data_frame.iloc[idx]['content'])

        if self.transform:
            image = self.transform(image)

        return image, content_label

# Create dataset instance
dataset = CustomDataset(csv_file='E:/分类任务/224_224/含量分类+验证集测试集分类-四甲基吡嗪.csv', root_dir='E:/分类任务/224_224', transform=transform)

# Create DataLoader
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

# Helper function to convert DataLoader batches to NumPy arrays
def dataloader_to_numpy(dataloader):
    images, content_labels = [], []
    for imgs, content_lbls in dataloader:
        images.append(imgs.view(imgs.size(0), -1).numpy())  # Flatten images
        content_labels.append(content_lbls.numpy())
    return np.vstack(images), np.concatenate(content_labels)

# Convert DataLoader data to NumPy arrays
X, y = dataloader_to_numpy(data_loader)

# Standardize the data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_standardized)

# Map numerical labels to descriptive labels
label_mapping = {4: 'Low concentration', 5: 'Medium concentration', 6: 'High concentration'}
descriptive_labels = np.vectorize(label_mapping.get)(y)

# Create color mapping
colors = {'Low concentration': 'blue', 'Medium concentration': 'green', 'High concentration': 'red'}
color_labels = np.vectorize(colors.get)(descriptive_labels)

# Visualize the t-SNE results
plt.figure(dpi=200, figsize=(10, 5))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=color_labels, label=descriptive_labels, s=10)  # Adjust s to change the size
plt.title('t-SNE Visualization')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

# Create a legend manually
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
           for label, color in colors.items()]
plt.legend(handles=handles, title="Content Label")

# Save the figure as an SVG file to E drive
plt.savefig('E:/分类任务/tSNE_Visualization.svg', format='svg')

plt.show()
