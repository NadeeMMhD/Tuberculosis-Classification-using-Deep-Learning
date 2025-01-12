import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import time

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data augmentation and normalization for training
transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Paths to datasets
train_dir = r'C:\tuber\TB_Chest_Radiography_Database\Train'
val_dir = r'C:\tuber\TB_Chest_Radiography_Database\Val'
test_dir = r'C:\tuber\TB_Chest_Radiography_Database\Test'

# Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform['train'])
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform['val'])
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform['test'])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the EfficientNet model
class EfficientNetModel(nn.Module):
    def __init__(self, model_name='efficientnet_b0'):
        super(EfficientNetModel, self).__init__()
        # Load the pretrained EfficientNet model
        if model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        elif model_name == 'efficientnet_b1':
            self.model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        elif model_name == 'efficientnet_b2':
            self.model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        
        # Get the number of features in the last layer
        num_features = self.model.classifier[1].in_features
        
        # Replace the classifier with a new one for binary classification
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, 2)
        )

    def forward(self, x):
        return self.model(x)

# Initialize model (you can choose between b0, b1, or b2)
model = EfficientNetModel(model_name='efficientnet_b0').to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping criteria
early_stopping_patience = 5
best_val_loss = np.inf
patience_counter = 0

# Training the model
num_epochs = 50
# Fix: Initialize lists correctly
train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

for epoch in range(num_epochs):
    model.train()
    train_loss, correct = 0, 0
    total = 0
    start_time = time.time()

    # Training loop
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    train_accuracy = 100 * correct / total
    train_loss /= len(train_loader)
    train_loss_history.append(train_loss)
    train_acc_history.append(train_accuracy)

    # Validation loop
    model.eval()
    val_loss, correct = 0, 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    val_accuracy = 100 * correct / total
    val_loss /= len(val_loader)
    val_loss_history.append(val_loss)
    val_acc_history.append(val_accuracy)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_accuracy:.2f}, Val Acc: {val_accuracy:.2f}, Time: {time.time() - start_time:.2f}s')

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_efficientnet_model.pth')
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        print("Early stopping triggered.")
        break

# Load the best model for testing
model.load_state_dict(torch.load('best_efficientnet_model.pth'))

# Testing the model
model.eval()
test_loss, correct = 0, 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / total
test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

# Plotting loss and accuracy curves
epochs_range = range(1, len(train_loss_history) + 1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_loss_history, label='Train Loss')
plt.plot(epochs_range, val_loss_history, label='Val Loss')
plt.legend(loc='upper right')
plt.title('Loss History')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_acc_history, label='Train Accuracy')
plt.plot(epochs_range, val_acc_history, label='Val Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy History')

plt.tight_layout()
plt.show()