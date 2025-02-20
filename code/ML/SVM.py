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

# image transformation
transform = transforms.Compose([
    transforms.ToTensor()
])


# Custom Dataset Classes
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
        image_class = int(self.data_frame.iloc[idx]['class'])  # Specify the ‘class’ column directly
        content_label = int(self.data_frame.iloc[idx]['content'])  # Specify the ‘content’ column directly

        if self.transform:
            image = self.transform(image)

        return image, image_class, content_label


# Creating DataSet
dataset = CustomDataset(csv_file='E:/X.csv', root_dir='E:/',
                        transform=transform)

# Split the dataset according to the ‘class’ columns
train_indices = dataset.data_frame[dataset.data_frame['class'] == 1].index.tolist()
test_indices = dataset.data_frame[dataset.data_frame['class'] == 2].index.tolist()

train_dataset = torch.utils.data.Subset(dataset, train_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

# Creating DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)


# Define function to convert DataLoader to NumPy array
def dataloader_to_numpy(dataloader):
    images, labels = [], []
    for imgs, _, lbls in dataloader:
        images.append(imgs.view(imgs.size(0), -1).numpy())  
        labels.append(lbls.numpy())
    return np.vstack(images), np.concatenate(labels)



X_train, y_train = dataloader_to_numpy(train_loader)
X_test, y_test = dataloader_to_numpy(test_loader)

# # 训练SVM模型
# clf = SVC(C=1,kernel='rbf', gamma=0.001)
# clf.fit(X_train, y_train)
#
# # 保存模型
# model_path = 'E:/分类任务/模型保存/svm_model_best2.pkl'
# joblib.dump(clf, model_path)
# print(f'Model saved to {model_path}')
#
# # 在测试集上评估模型
# y_test_pred = clf.predict(X_test)
# test_accuracy = accuracy_score(y_test, y_test_pred)
# print(f'Test Accuracy: {test_accuracy:.2f}')


# # 定义一个函数，用于加载模型并对新图像进行预测
# def predict_image(model, image_path, transform):
#     # 读取图像并进行预处理
#     image = Image.open(image_path)
#     if transform:
#         image = transform(image)
#     image = image.view(1, -1).numpy()  # 展平图像并转换为NumPy数组
#
#     # 进行预测
#     prediction = model.predict(image)
#     return prediction[0]
#
#
# # 定义一个函数，用于批量预测文件夹中的图像，并保存结果到CSV文件
# def batch_predict(model_path, image_folder, output_csv, transform):
#     # 加载模型
#     model = joblib.load(model_path)
#
#     # 获取文件夹中的所有图像文件
#     image_files = [f for f in os.listdir(image_folder) if f.endswith(('jpg', 'jpeg', 'png'))]
#
#     results = []
#     for image_file in image_files:
#         image_path = os.path.join(image_folder, image_file)
#         prediction = predict_image(model, image_path, transform)
#         results.append((image_file, prediction))
#
#     # 将结果保存到CSV文件
#     df = pd.DataFrame(results, columns=['Image', 'Prediction'])
#     df.to_csv(output_csv, index=False)
#     print(f'Results saved to {output_csv}')
#
#
# # 示例：批量预测文件夹中的图像，并保存结果到CSV文件
# image_folder = 'E:/分类任务/224_224/新建文件夹'
# output_csv = 'predictions1.csv'
# batch_predict(model_path, image_folder, output_csv, transform)


#使用class标签3进行预测
#Split dataset based on 'class' column to create validation set
validation_indices = dataset.data_frame[dataset.data_frame['class'] == 3].index.tolist()

validation_dataset = torch.utils.data.Subset(dataset, validation_indices)
#
# Create DataLoader for validation set
validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=True, num_workers=0)

# Convert validation DataLoader to NumPy arrays
X_validation, y_validation = dataloader_to_numpy(validation_loader)

# Load the saved model
model_path = 'E:/分类任务/模型保存/svm_model_best2.pkl'
model = joblib.load(model_path)

# Predict the results for the validation set
y_validation_pred = model.predict(X_validation)

# Calculate accuracy for validation set predictions
validation_accuracy = accuracy_score(y_validation, y_validation_pred)
print(f'Validation Accuracy: {validation_accuracy:.2f}')


# Optional: Save the predictions for the validation set to a CSV file
validation_results = pd.DataFrame({'True Label': y_validation, 'Predicted Label': y_validation_pred})
validation_results.to_csv('E:/分类任务/模型保存/validation_predictions.csv', index=False)
print('Validation results saved to validation_predictions.csv')
