import os
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm  # 用於導入 Vision Transformer
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

# 自定義資料集
class ImageDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 資料處理與讀取
data_path = './train'
train_data = []
val_data = []

for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)
    if os.path.isdir(folder_path):  # 確保只處理資料夾
        files = os.listdir(folder_path)
        num_train = int(0.8 * len(files))
        files_train = random.sample(files, num_train)
        files_val = list(set(files) - set(files_train))
        
        for file in files_train:
            file_path = os.path.join(folder_path, file)
            img = cv2.imread(file_path)
            img = cv2.resize(img, (224, 224))
            train_data.append((img, folder))
            
        for file in files_val:
            file_path = os.path.join(folder_path, file)
            img = cv2.imread(file_path)
            img = cv2.resize(img, (224, 224))
            val_data.append((img, folder))

# 資料預處理
le = LabelEncoder()
y_train = le.fit_transform([label for _, label in train_data])
y_val = le.transform([label for _, label in val_data])

train_images = [img for img, _ in train_data]
val_images = [img for img, _ in val_data]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = ImageDataset(train_images, y_train, transform=transform)
val_dataset = ImageDataset(val_images, y_val, transform=transform)

# 減小批次大小以節省 GPU 記憶體
BATCH_SIZE = 16

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 模型定義 - 使用 Vision Transformer (ViT)
class CustomViT(nn.Module):
    def __init__(self, num_classes):
        super(CustomViT, self).__init__()
        # 使用預訓練的 ViT 模型，來自 timm
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)
    
    def forward(self, x):
        return self.vit(x)

# 創建模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(le.classes_)
model = CustomViT(num_classes).to(device)

# 損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練與驗證
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).long()
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    return history

# 訓練模型
history = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20)

# 保存模型
model_save_path = './my_vit_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"模型已保存至 {model_save_path}")

# 繪製訓練和驗證的損失與準確率
train_loss = history['train_loss']
val_loss = history['val_loss']
train_acc = history['train_acc']
val_acc = history['val_acc']

epochs = range(1, len(train_loss) + 1)

plt.figure()
plt.plot(epochs, train_loss, label='Training loss', marker='o')
plt.plot(epochs, val_loss, label='Validation loss', marker='o')
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, train_acc, label='Training accuracy', marker='o')
plt.plot(epochs, val_acc, label='Validation accuracy', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 混淆矩陣
model.eval()
test_path = './test'
real_label = []
predicted_class = []

for folder in os.listdir(test_path):
    folder_path = os.path.join(test_path, folder)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        img = cv2.imread(file_path)
        img = cv2.resize(img, (224, 224))
        img = transform(img).unsqueeze(0).to(device)

        prediction = model(img)
        real_label.append(folder)
        predicted_class_index = prediction.argmax(dim=1).item()
        predicted_class.append(le.classes_[predicted_class_index])

conf_matrix = confusion_matrix(real_label, predicted_class)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
