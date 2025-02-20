import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib

# Define image transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name)
        image_class = int(self.data_frame.iloc[idx]['class'])
        content_label = int(self.data_frame.iloc[idx]['content'])

        if self.transform:
            image = self.transform(image)

        return image, image_class, content_label

# Helper function to convert DataLoader to NumPy arrays
def dataloader_to_numpy(dataloader):
    images, labels = [], []
    for imgs, _, lbls in dataloader:
        images.append(imgs.view(imgs.size(0), -1).numpy())  # Flatten the image
        labels.append(lbls.numpy())
    return np.vstack(images), np.concatenate(labels)

# Define function to predict a single image
def predict_image(model, image_path, transform):
    image = Image.open(image_path)
    if transform:
        image = transform(image)
    image = image.view(1, -1).numpy()  # Flatten the image
    prediction = model.predict(image)
    return prediction[0]

# Define function to batch predict images in a folder
def batch_predict(model_path, image_folder, output_csv, transform):
    model = joblib.load(model_path)
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('jpg', 'jpeg', 'png'))]
    results = []
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        prediction = predict_image(model, image_path, transform)
        results.append((image_file, prediction))
    df = pd.DataFrame(results, columns=['Image', 'Prediction'])
    df.to_csv(output_csv, index=False)
    print(f'Results saved to {output_csv}')

if __name__ == "__main__":
    # Create dataset instance
    dataset = CustomDataset(csv_file='E:/x.csv',
                            root_dir='E:/', transform=transform)

    # Split dataset based on 'class' column
    train_indices = dataset.data_frame[dataset.data_frame['class'] == 1].index.tolist()
    test_indices = dataset.data_frame[dataset.data_frame['class'] == 2].index.tolist()

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)

    # Convert DataLoader to NumPy arrays
    X_train, y_train = dataloader_to_numpy(train_loader)
    X_test, y_test = dataloader_to_numpy(test_loader)

    # Define parameter grid for GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': [0.001, 0.01, 0.1, 1]  # Only used for 'rbf' kernel
    }

    # Initialize SVM model
    svc = SVC(random_state=42)

    # Perform GridSearchCV
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Print best parameters and best score
    print(f'Best parameters found: {grid_search.best_params_}')
    print(f'Best cross-validation accuracy: {grid_search.best_score_:.2f}')

    # Train final model with best parameters
    best_model = grid_search.best_estimator_

    # Save the best model
    model_path = 'E:/svm_best_model.pkl'
    joblib.dump(best_model, model_path)
    print(f'Best model saved to {model_path}')

    # Evaluate the best model
    y_test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f'Test Accuracy of best model: {test_accuracy:.2f}')

    # Example: batch predict images in a folder and save results to CSV
    image_folder = 'E:/'
    output_csv = 'predictions.csv'
    batch_predict(model_path, image_folder, output_csv, transform)
