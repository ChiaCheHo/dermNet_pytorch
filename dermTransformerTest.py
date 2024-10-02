import torch
import torch.nn as nn  # 確保 nn 已正確導入
import cv2
import numpy as np
from torchvision import transforms
import timm
import matplotlib.pyplot as plt
import os

# 定義模型結構
class CustomViT(nn.Module):
    def __init__(self, num_classes):
        super(CustomViT, self).__init__()
        # 使用預訓練的 ViT 模型
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)
    
    def forward(self, x):
        return self.vit(x)

# 載入資料夾名稱對應的類別
data_path = './train'
folders = sorted([folder for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, folder))])
print("資料夾對應類別：", folders)

# 載入已保存的模型
model_path = './my_vit_model.pth'  # 保存模型的路徑
num_classes = len(folders)  # 使用資料夾數作為類別數量
model = CustomViT(num_classes)

# 強制模型載入到 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 定義圖像預處理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加載並處理輸入圖像
img_path = './test/Acne and Rosacea Photos/07RosaceaMilia0120.jpg'  # 輸入圖像的路徑
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 轉換顏色順序
input_img = transform(img).unsqueeze(0).to(device)  # 增加批次維度並傳入 GPU/CPU

# 將圖像傳入模型並進行預測
with torch.no_grad():
    output = model(input_img)
    predicted_class_index = torch.argmax(output, dim=1).item()

# 將預測的類別編號轉換為對應的資料夾名稱
predicted_class_name = folders[predicted_class_index]
print(f"Predicted class: {predicted_class_name}")

# 顯示輸入的圖像和預測結果
plt.imshow(img)
plt.title(f"Predicted Class: {predicted_class_name}")
plt.show()
