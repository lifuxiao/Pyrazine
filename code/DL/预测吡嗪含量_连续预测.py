import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch.nn as nn
import pandas as pd

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def load_model(model_path):
    model_name = os.path.basename(model_path).split('_')[0]  # Assuming the model name is a prefix in the filename

    if model_name == 'densenet201':
        model = models.densenet201(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 3)
    elif model_name == 'alexnet':
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 3)
    elif model_name == 'googlenet':
        model = models.googlenet(weights=True)
        model.fc = nn.Linear(model.fc.in_features, 3)
    elif model_name == 'resnet101':
        model = models.resnet101(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 3)
    elif model_name == 'resnet152':
        model = models.resnet152(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 3)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 3)
    elif model_name == 'vgg16':
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 3)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    state_dict = torch.load(model_path)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # Adjust state_dict keys if there's a mismatch
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)

    model.eval()
    model = model.to(device)
    return model


def predict_image(model, image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)  # Use 'transform' here
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)

    return outputs.cpu().numpy()


def predict_filtered_images(model, data_frame, root_dir, output_excel):
    results = []
    validation_indices = data_frame[data_frame['class'] == 3].index.tolist()

    for idx in validation_indices:
        image_name = data_frame.iloc[idx]['image_name']  # Assuming 'image_name' column has image names
        image_path = os.path.join(root_dir, image_name)

        if os.path.exists(image_path):
            predictions = predict_image(model, image_path)
            results.append([image_name, *predictions.flatten()])
        else:
            print(f"Image {image_name} not found in {root_dir}")

    # Convert results to DataFrame and save to Excel
    df = pd.DataFrame(results, columns=['Image', 'Four_Pyrazine', 'Three_Pyrazine', 'Other_Pyrazine'])
    df.to_excel(output_excel, index=False)


# New paths for the CSV file and root directory
csv_file = 'E:/新柱子-2/原始图像/RGB/224X224/CNN训练预测-新柱子-纯原始数据-点png - 标记-删除XXXXX.csv'  # Path to your CSV file
root_dir = 'E:/新柱子-2/原始图像/RGB/224X224/新柱子-原始图像全'  # Root directory for images

# Load the dataset
data_frame = pd.read_csv(csv_file)

# Directory containing the models
model_dir = 'F:/python-deepL/Pyrazine'
# Directory to save the predictions
output_dir = 'D:/新柱子-全'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate through each model file in the model directory
for model_filename in os.listdir(model_dir):
    if model_filename.endswith('.pth'):
        model_path = os.path.join(model_dir, model_filename)
        try:
            model = load_model(model_path)
        except ValueError as e:
            print(e)
            continue

        # Create output file name
        model_name = os.path.splitext(model_filename)[0]
        output_excel = os.path.join(output_dir, f'{model_name}_predictions.xlsx')

        # Make predictions and save to Excel
        predict_filtered_images(model, data_frame, root_dir, output_excel)
        print(f"Filtered predictions saved to {output_excel}")

# # Directory containing the models
# model_dir = 'F:/python-deepL/Pyrazine'
# # Directory to save the predictions
# output_dir = 'D:/1'