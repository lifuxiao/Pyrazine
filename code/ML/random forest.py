import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import joblib
import matplotlib.pyplot as plt

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
        image = Image.open(img_name).convert("RGB")
        image_class = int(self.data_frame.iloc[idx]['class'])
        content_label = int(self.data_frame.iloc[idx]['content'])

        if self.transform:
            image = self.transform(image)

        return image, image_class, content_label

# Create dataset instance
dataset = CustomDataset(csv_file='E:/X.csv', root_dir='E:/', transform=transform)

# Split dataset based on 'class' column
train_indices = dataset.data_frame[dataset.data_frame['class'] == 1].index.tolist()
test_indices = dataset.data_frame[dataset.data_frame['class'] == 2].index.tolist()

train_dataset = torch.utils.data.Subset(dataset, train_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)

# Helper function to convert DataLoader to NumPy arrays
def dataloader_to_numpy(dataloader):
    images, labels = [], []
    for imgs, _, lbls in dataloader:
        images.append(imgs.view(imgs.size(0), -1).numpy())
        labels.append(lbls.numpy())
    return np.vstack(images), np.concatenate(labels)

# Convert data
X_train, y_train = dataloader_to_numpy(train_loader)
X_test, y_test = dataloader_to_numpy(test_loader)

# Train RandomForest model
clf = RandomForestClassifier(n_estimators=40, max_depth=7, min_samples_leaf=1, min_samples_split=3, random_state=42)  #The default parameters are (n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model
model_path = 'E:/random_best_forest_model.pkl'
joblib.dump(clf, model_path)
print(f'Model saved to {model_path}')

# Evaluate model on test set
y_test_pred = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy:.2f}')

# Function to predict image using the model
def predict_image(model, image_path, transform):
    image = Image.open(image_path).convert("RGB")
    if transform:
        image = transform(image)
    image = image.view(1, -1).numpy()

    prediction = model.predict(image)
    return prediction[0]

# Function to batch predict images and save results to CSV
def batch_predict(model_path, image_folder, output_csv, transform):
    model = joblib.load(model_path)
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

    results = [(image_file, predict_image(model, os.path.join(image_folder, image_file), transform)) for image_file in image_files]

    df = pd.DataFrame(results, columns=['Image', 'Prediction'])
    df.to_csv(output_csv, index=False)
    print(f'Results saved to {output_csv}')

# Example: batch predict images and save results to CSV
image_folder = 'E:/'
output_csv = 'predictions.csv'
batch_predict(model_path, image_folder, output_csv, transform)

# Function to evaluate and plot different parameters
def evaluate_parameter(param_name, param_values, x_train, y_train):
    train_set_scores = []
    cross_val_scores = []
    for value in param_values:
        rf = RandomForestClassifier(**{param_name: value}, random_state=42)
        rf.fit(x_train, y_train)
        scores = cross_val_score(rf, x_train, y_train, cv=5)
        cross_val_scores.append(scores.mean())
        train_set_scores.append(rf.score(x_train, y_train))
    return train_set_scores, cross_val_scores

# Parameters to evaluate
param_dict = {
    'n_estimators': range(1, 51),
    'max_depth': range(1, 21),
    'min_samples_leaf': range(1, 21),
    'min_samples_split': range(2, 22)
}

# Plotting the results
fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=200)

for ax, (param_name, param_values) in zip(axes.flatten(), param_dict.items()):
    train_set_scores, cross_val_scores = evaluate_parameter(param_name, param_values, X_train, y_train)
    ax.plot(param_values, train_set_scores, label='Train set scores')
    ax.plot(param_values, cross_val_scores, label='Cross-validation scores')
    ax.set_title(f'Scores for {param_name}')
    ax.set_xlabel(param_name)
    ax.set_ylabel('Score')
    ax.legend()

plt.tight_layout()
plt.savefig('Parameter_adjustment.svg', format='svg')
plt.show()
