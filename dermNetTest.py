import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import models, transforms
import matplotlib.pyplot as plt
import os

# 定義模型結構
class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()
        self.resnet = models.resnet101(weights=None)  # 不需要預訓練權重
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

# 載入資料夾名稱對應的類別
data_path = './train'
folders = sorted([folder for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, folder))])
print("資料夾對應類別：", folders)

# 載入已保存的模型
model_path = './my_resnet101_model.pth'  # 保存模型的路徑
num_classes = len(folders)  # 使用資料夾數作為類別數量
model = CustomResNet(num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

# 定義圖像預處理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加載並處理圖片
img_path = './test/Acne and Rosacea Photos/07RosaceaMilia0120.jpg'  # 輸入圖像的路徑
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 轉換顏色順序
input_img = transform(img).unsqueeze(0)  # 增加批次維度

# 將圖像傳入模型
with torch.no_grad():
    output = model(input_img)
    predicted_class_index = torch.argmax(output, dim=1).item()

# 將預測的類別編號轉換為對應的資料夾名稱
predicted_class_name = folders[predicted_class_index]
print(f"Predicted class: {predicted_class_name}")

# 顯示輸入的圖像
plt.imshow(img)
plt.title(f"Predicted Class: {predicted_class_name}")
plt.show()
