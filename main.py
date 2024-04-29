import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import torchvision.models as models
from utill import *


def main():

    args = setup_args()
    
    # 데이터 및 모델 저장 디렉토리 설정
    
    defocus_path = os.path.join(args.data_dir, 'defocus_add')
    ast_path = os.path.join(args.data_dir, 'astigmatism')
    ast_add_path = os.path.join(args.data_dir, 'ast_add')
    tilt_path = os.path.join(args.data_dir, 'tilt')
    
    model_save_dir = args.weight_dir
    
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        
    start_time = time.time()

    # 데이터 처리
    
    X_defocus, Y_defocus = process_folder(defocus_path,defocus=True)
    X_ast, Y_ast = process_folder(ast_path,defocus=False)
    X_ast_add, Y_ast_add = process_folder(ast_add_path,defocus=False)
    X_tilt, Y_tilt = process_folder(tilt_path,defocus=False)

    X1 = X_defocus
    Y1 = Y_defocus
    X2 = X_ast
    Y2 = Y_ast
    X3 = X_ast_add
    Y3 = Y_ast_add
    X4 = X_tilt
    Y4 = Y_tilt

    elapsed_time = time.time() - start_time
    print(f"process_image took {elapsed_time:.2f} seconds.")
    
    # 데이터 병합
    
    X = np.concatenate((X1,X2,X3,X4), axis=0)
    Y = np.concatenate((Y1,Y2,Y3,Y4), axis=0)

    num_classes = len(np.unique(Y))
        
    print("num_classes",num_classes)
    
    # 원핫 인코딩
    
    Y_one_hot =  to_categorical(Y, num_classes=45)

    # 훈련, 테스트 분리
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_one_hot, test_size=0.2, stratify=Y_one_hot, random_state=args.manualSeed)
    # 훈련, 검증 분리
    
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, stratify=Y_train, test_size=0.2, random_state=args.manualSeed)


    print("X_train:", X_train.shape[0], "training images,", "height:", X_train.shape[1], ", width:", X_train.shape[2])
    print("X_val:", X_val.shape[0], "validation images,", "height:", X_val.shape[1], ", width:", X_val.shape[2])
    print("X_test:", X_test.shape[0], "test images,", "height:", X_test.shape[1], ", width:", X_test.shape[2])
  
    # X_train: 60444 training images, height: 512 , width: 512
    # X_val: 15112 validation images, height: 512 , width: 512
    # X_test: 18890 test images, height: 512 , width: 512

    
    # 데이터 처리 
    
    train_transform = transforms.Compose([
        transforms.RandomCrop((args.crop_size,args.crop_size)),
        # transforms.RandomHorizontalFlip()
       #  transforms.RandomVerticalFlip()
      #  transforms.RandomRotation(180)
        # transforms.ColorJitter()
        # transforms.GaussianBlur(kernel_size=(5, 5))
        transforms.ToTensor(),
     #   transforms.Normalize((0.5),(0.5))
    ])

    test_transform = transforms.Compose([
        transforms.CenterCrop((args.crop_size,args.crop_size)),
        transforms.ToTensor(),
     #   transforms.Normalize((0.5),(0.5))
    ])
    

    # 데이터셋과 로더 생성
    
    batch_size = args.batch
    
    train_loader = create_dataset_and_loader(X_train, Y_train, train_transform, batch_size, True)
    valid_loader = create_dataset_and_loader(X_val, Y_val, test_transform, batch_size)
    test_loader = create_dataset_and_loader(X_test, Y_test, test_transform, batch_size)


    #resnet 18 
    
    # model = models.resnet18() 
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 45)
    # model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    model = models.efficientnet_v2_s(weights=None)
    # 첫 번째 컨볼루션 레이어의 입력 채널을 1로 변경
    first_conv_layer = model.features[0][0]
    model.features[0][0] = torch.nn.Conv2d(1, first_conv_layer.out_channels, kernel_size=first_conv_layer.kernel_size, stride=first_conv_layer.stride, padding=first_conv_layer.padding, bias=False)

    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, 45)
    
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    model_filename = 'best.pth'
    model_filepath = os.path.join(model_save_dir, model_filename)
    num_epochs = args.epochs
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()

        train_loss, val_loss = 0.0, 0.0
        train_correct, val_correct = 0, 0
        train_total, val_total = 0, 0

        
        # 훈련 데이터셋으로 모델 훈련
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.type(torch.FloatTensor).cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == torch.argmax(labels,axis=1)).sum().item()
            
        # 검증 데이터셋으로 모델 평가
        
        model.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.cuda(), labels.type(torch.FloatTensor).cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == torch.argmax(labels,axis=1)).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total    
        val_loss /= len(valid_loader)
        val_accuracy = 100 * val_correct / val_total
        print(f'Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        
        # best_val_loss가 갱신되면 모델 저장
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_filepath)
        

        # lr 스케줄러 업데이트
        
        scheduler.step(val_loss)#
    
   # 테스트 데이터 성능 확인
    
    print("start test")                 
    model.load_state_dict(torch.load(model_filepath))
    model.eval()
    correct = 0
    total = 0

    total_labels = []
    total_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(), labels.type(torch.FloatTensor).cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == torch.argmax(labels,axis=1)).sum().item()

            total_labels.extend(list(torch.argmax(labels,axis=1).detach().cpu().numpy()))
            total_preds.extend(predicted.detach().cpu().numpy())

    test_accuracy = 100 * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    
    
    label_descriptions = [
        "defocus -18", "defocus -15", "defocus -12", "defocus -9", "defocus -6",   # 0-4
        "defocus -3", "defocus +3", "defocus +6", "defocus +9", "defocus +12",     # 5-9
        "defocus +15", "defocus +18", "ast 21",                                    # 10-12
        "normal",                                                                  # 13
        "ast 22", "ast 23", "ast 24", "ast 25", "ast 26", "ast 27",                # 14-19
        "ast 28", "ast 29", "ast 30", "ast 31", "ast 32",                          # 20-24
        "tilt 33", "tilt 34", "tilt 35", "tilt 36", "tilt 37",                     # 25-29
        "tilt 38", "tilt 39", "tilt 40",                                           # 30-32
        "ast 48", "ast 49", "ast 50", "ast 51", "ast 52", "ast 53",                # 33-38
        "ast 54", "ast 55", "ast 56", "ast 57", "ast 58", "ast 59"                 # 39-44
    ]

    group_indices = {
    'defocus': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13],
    'ast': [12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,33,34,35,36,37,38,39,40,41,42,43,44,13],
    'tilt': [25, 26, 27, 28, 29, 30, 31, 32, 13]
     }       
    # Confusion matrix 
    
    plot_confusion_matrix(total_labels, total_preds,label_descriptions)
    plot_grouped_confusion_matrix(total_labels, total_preds, label_descriptions, group_indices)

if __name__ == '__main__':
    main()
    
    
    