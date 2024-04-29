import argparse
import os
import random
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def setup_args():
    parser = argparse.ArgumentParser(description='PyTorch SEM Training')
    parser.add_argument('--data_dir', type=str, default="/home/data/Project/hynix",
                        help='SEM 데이터 경로.')
    parser.add_argument('--weight_dir', type=str, default="/home/workspace/weights",
                        help='모델 가중치를 저장할 경로.')
    parser.add_argument('--manualSeed', default=2024, type=int, help='재현성.')
    parser.add_argument('--epochs', default=150, type=int, help='훈련 에포크 수.')
    parser.add_argument('--batch', default=128, type=int, help='훈련 배치 크기.')
    parser.add_argument('--crop_size', default=384, type=int, help='입력 이미지를 자르는 크기.')
    parser.add_argument('--lr', type=float, default=1e-4, help='초기 학습률.')
    
    return parser.parse_args()

def set_seed(seed: int = 2024) -> None:
    """
    재현성을 위해 시드 설정.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

# 이미지 중앙 crop 함수, 512 size로 crop

def center_crop(img, new_width=512, new_height=512):
    width, height = img.size   

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    img = img.crop((left, top, right, bottom))
    return img

# 주어진 디렉토리에서 이미지 파일을 처리하는 함수

def process_folder(dir_path,defocus=False):
    
    if defocus:
        label_mapping = {'40': 0, '48': 1, '49': 2, '50': 3, '51': 4, '52': 5, '53': 13, '54': 6, '55': 7, '56': 8, '57': 9, '58': 10, '59': 11}   
    else:
        # ast,tilt
        label_mapping = {'13': 13, '21': 12, '22': 14, '23': 15, '24': 16, '25': 17, '26': 18, '27': 19, '28': 20, '29': 21, '30': 22, '31': 23, '32': 24,
                        '33': 25, '34': 26, '35': 27, '36': 28, '37': 29, '38': 30, '39': 31, '40': 32,
                        '48': 33, '49': 34, '50': 35, '51': 36, '52': 37, '53': 38, '54': 39, '55': 40, '56': 41, '57': 42, '58': 43, '59': 44}   
        
    contents = os.listdir(dir_path)
    print(dir_path,"폴더 처리")
    # 이미지 폴더만 필터링
    
    filtered_contents = [item for item in contents if os.path.isdir(os.path.join(dir_path, item)) and item.endswith('_')]
    

    Y = []  
    X = []  

    # 이미지 폴더를 정렬된 순서로 처리
    
    for index, category in enumerate(sorted(filtered_contents, key=lambda x: (x.split('_')[3], x.split('_')[4]))):
         
        file_paths = sorted(os.listdir(os.path.join(dir_path, category)))
       
        ms_file_paths = [file_path for file_path in file_paths if file_path.endswith('MS.jpeg')]
      
        # 학습 폴더 처리
        
        for file_path in tqdm(ms_file_paths[:30], desc=f"{category} 처리 중"):
       
            img_path = os.path.join(dir_path, category, file_path)
            if "CBL" in img_path:
                continue
            
            img = Image.open(img_path)
            if "ISO" in img_path: 
               img = img.rotate(-20) # 시계 방향으로 20도 회전 
            elif "F16Y" in img_path:
               img = img.rotate(90)  # 반시계 방향으로 90도 회전
            img = center_crop(img)
            img_array = np.array(img)
            
            label = category[-3:-1]
            mapped_label = label_mapping[label]

    
            X.append(img_array)
            Y.append(mapped_label)

    X = np.array(X)
    Y = np.array(Y)



    return X, Y 

def to_categorical(y, num_classes):
    """
    원-핫 인코딩
    """
    return np.eye(num_classes, dtype='uint8')[y]


class CustomDataset(Dataset):
    """
    데이터셋 클래스 정의
    """
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = Image.fromarray(self.X[idx])
        y = self.Y[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

    
def create_dataset_and_loader(X, Y, transform, batch_size, shuffle=False):
    """
    데이터셋 및 로더 생성 함수
    """
    dataset = CustomDataset(X, Y, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,num_workers=4)
    return loader


def plot_confusion_matrix(true_labels, pred_labels, labels_description, filename='confusion_matrix.png'):
    """
    confusion matrix 시각화 함수
    """

    cm = confusion_matrix(true_labels, pred_labels)


    plt.figure(figsize=(15, 10))
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, square=True, cmap='Blues', cbar=False,
                xticklabels=labels_description, yticklabels=labels_description)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=90)  
    plt.yticks(rotation=0)
    plt.savefig(filename)
    plt.close()
    

    
def plot_grouped_confusion_matrix(total_labels, total_preds, labels_description, group_indices, filename_prefix='confusion_matrix'):
    for group_name, indices in group_indices.items():
        # 해당 그룹의 라벨만 선택
        filtered_labels = [labels_description[i] for i in indices]

        # 실제 라벨과 예측 라벨을 필터링
        mask = np.isin(total_labels, indices)
        filtered_true_labels = np.array([label for label in total_labels if label in indices])
        filtered_pred_labels = np.array([pred for pred, m in zip(total_preds, mask) if m])

        # 컨퓨전 매트릭스 계산
        cm = confusion_matrix(filtered_true_labels, filtered_pred_labels, labels=indices)

        # Simplify the matrix values if any element is greater than 100
        cm_simplified = cm.astype(float)

        # Create annotation array with formatting
        annot_array = np.array([[f'{val:.0f}' if val < 100 else f'{val/100}x' for val in row] for row in cm_simplified])


        # 시각화
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_simplified, annot=annot_array, fmt='', linewidths=.5, square=True, cmap='Blues', cbar=False,
                    xticklabels=filtered_labels, yticklabels=filtered_labels)
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Confusion Matrix for {group_name}')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.savefig(f'{filename_prefix}_{group_name}.png')
        plt.close()
        
# resnet50 
# resnet18 95.17%
