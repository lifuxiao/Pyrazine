import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from Dataloader import ImageDataset, build_parser
import pandas as pd

# Load the pretrained alexnet model
alexnet = models.alexnet(weights='DEFAULT')

# Modify the classifier to output three values
alexnet.classifier[6] = nn.Linear(alexnet.classifier[6].in_features, 3)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
alexnet = alexnet.to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(alexnet.parameters(), lr=0.0001)

# Load dataset and dataloader
parser = build_parser()
args = parser.parse_args()

dataset = ImageDataset(csv_file=args.train_file, image_dir=args.image_dir)
train_indices = (dataset.classes != 2).nonzero(as_tuple=True)[0].tolist()
val_indices = (dataset.classes == 3).nonzero(as_tuple=True)[0].tolist()

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

# Training loop
num_epochs = 200

# Lists to store loss values
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    alexnet.train()
    running_loss = 0.0
    for i, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        labels = torch.stack([batch['four_pyrazine'], batch['three_pyrazine'], batch['other_pyrazine']], dim=1).to(device)

        # Forward pass
        outputs = alexnet(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Print details about the predictions and true values
        if i % 20 == 0:  # Adjust the frequency of printing as needed
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            print(f"Predicted values: {outputs.cpu().detach().numpy()}")
            print(f"True values: {labels.cpu().detach().numpy()}")

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}')

    # Validation
    alexnet.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = torch.stack([batch['four_pyrazine'], batch['three_pyrazine'], batch['other_pyrazine']], dim=1).to(device)

            outputs = alexnet(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    #torch.save(alexnet.state_dict(), 'alexnet_trained_200_256X1024.pth')
    if (epoch + 1) % 50 == 0:
        torch.save(alexnet.state_dict(), f'alexnet_epoch_{epoch+1}.pth')
        print(f"Model saved at epoch {epoch+1}")
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    print(f'Validation Loss: {val_loss:.4f}')

print("Training complete.")

# Save the loss values to an Excel file
loss_data = {
    'Epoch': list(range(1, num_epochs + 1)),
    'Training Loss': train_losses,
    'Validation Loss': val_losses
}

df = pd.DataFrame(loss_data)
df.to_excel('F:/python-deepL/1.训练好的模型/Alexnet_50-pretrain/alexnet_training_validation_loss_200_224X224.xlsx', index=False)
print("Loss values saved to training_validation_loss.xlsx.")
