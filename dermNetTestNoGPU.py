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
folders = ['Acne and Rosacea Photos', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 'Atopic Dermatitis Photos', 'Bullous Disease Photos', 'Cellulitis Impetigo and other Bacterial Infections', 'Eczema Photos', 'Exanthems and Drug Eruptions', 'Hair Loss Photos Alopecia and other Hair Diseases', 'Herpes HPV and other STDs Photos', 'Light Diseases and Disorders of Pigmentation', 'Lupus and other Connective Tissue diseases', 'Melanoma Skin Cancer Nevi and Moles', 'Nail Fungus and other Nail Disease', 'Poison Ivy Photos and other Contact Dermatitis', 'Psoriasis pictures Lichen Planus and related diseases', 'Scabies Lyme Disease and other Infestations and Bites', 'Seborrheic Keratoses and other Benign Tumors', 'Systemic Disease', 'Tinea Ringworm Candidiasis and other Fungal Infections', 'Urticaria Hives', 'Vascular Tumors', 'Vasculitis Photos', 'Warts Molluscum and other Viral Infections']
zh_tw_folders = ['痤瘡和紅斑痤瘡', '光化性角化病基底細胞癌和其他惡性病變', '異位性皮膚炎', '大皰性疾病', '蜂窩性組織炎膿皰病和其他細菌感染', '濕疹', '皮疹和藥物皮疹', '脫髮、脫髮和其他頭髮疾病', '皰疹HPV 和其他性傳染病', '輕度疾病和色素沉著疾病', '狼瘡和其他結締組織疾病', '黑色素瘤、皮膚癌、痣和痣', '灰指甲和其他指甲疾病', '毒藤和其他接觸性皮膚炎', '牛皮癬圖片、扁平苔癬和相關疾病', '疥瘡萊姆病和其他感染和叮咬', '脂漏性角化症和其他良性腫瘤', '全身性疾病', '癬癬念珠菌病和其他真菌感染', '蕁麻疹', '血管腫瘤', '血管炎', '疣軟疣和其他病毒感染']

# 載入已保存的模型
model_path = './my_resnet101_model.pth'  # 保存模型的路徑
num_classes = len(folders)  # 使用資料夾數作為類別數量
model = CustomResNet(num_classes)

# 強制將模型設置在 CPU 上
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# 定義圖像預處理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加載並處理圖片
# img_path = './test/Cellulitis Impetigo and other Bacterial Infections/impetigo-58.jpg'  # 輸入圖像的路徑
img_path = './train/Cellulitis Impetigo and other Bacterial Infections/atypical-mycobacterium-14.jpg'  # 輸入圖像的路徑
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

predicted_class_name = zh_tw_folders[predicted_class_index]
print(f"Predicted class: {predicted_class_name}")

# 顯示輸入的圖像
# plt.imshow(img)
# plt.title(f"Predicted Class: {predicted_class_name}")
# plt.show()
