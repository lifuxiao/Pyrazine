import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Define image transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
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
        image_class = int(self.data_frame.iloc[idx]['class'])  # Specify 'class' column
        content_label = int(self.data_frame.iloc[idx]['content'])  # Specify 'content' column

        if self.transform:
            image = self.transform(image)

        return img_name, image, image_class, content_label

# Create dataset instance
dataset = CustomDataset(csv_file='E:/分类任务/224_224/含量分类+验证集测试集分类-四甲基吡嗪.csv', root_dir='E:/分类任务/224_224', transform=transform)

# Split dataset based on 'class' column to create validation set
validation_indices = dataset.data_frame[dataset.data_frame['class'] == 3].index.tolist()
validation_dataset = torch.utils.data.Subset(dataset, validation_indices)

# Create DataLoader for validation set
validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=False, num_workers=0)

# Convert validation DataLoader to NumPy arrays
def dataloader_to_numpy_with_filenames(dataloader):
    filenames, images, labels = [], [], []
    for fnames, imgs, _, lbls in dataloader:
        filenames.extend([os.path.basename(fname) for fname in fnames])  # Extract base filenames
        images.append(imgs.view(imgs.size(0), -1).numpy())  # Flatten images
        labels.append(lbls.numpy())
    return filenames, np.vstack(images), np.concatenate(labels)

filenames, X_validation, y_validation = dataloader_to_numpy_with_filenames(validation_loader)

# Load the saved model
model_path = 'E:/分类任务/模型保存/random_best_forest_model.pkl'
model = joblib.load(model_path)

# Predict the results for the validation set
y_validation_pred = model.predict(X_validation)

# Calculate accuracy for validation set predictions
validation_accuracy = accuracy_score(y_validation, y_validation_pred)
print(f'Validation Accuracy: {validation_accuracy:.2f}')

# Save the predictions for the validation set to a CSV file
validation_results = pd.DataFrame({
    'Image': filenames,
    'True Label': y_validation,
    'Predicted Label': y_validation_pred
})
validation_results.to_csv('E:/分类任务/模型保存/random_forest_BEST_2.csv', index=False)
print('Validation results saved to validation_predictions.csv')

