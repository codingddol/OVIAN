# -*- coding: utf-8 -*-
"""
train_attrimil.py

AttriMIL 기반 난소암 조직 분석 모델을 학습하는 메인 실행 파일입니다.

📌 주요 기능:
- MIL(HDF5 기반) 데이터셋 로딩 및 5-Fold Cross Validation 분할
- Spatial Constraint + Rank Constraint 포함한 AttriMIL 모델 학습
- 클래스 불균형 보정 (가중치 적용)
- 검증 기반 Early Stopping 및 Best Model 저장
- 테스트 결과 AUC 및 클래스별 Accuracy 출력

입력: 학습용 CSV + HDF5 feature 파일 경로 + Fold split CSV 경로
출력: fold별 최적 모델 (.pt) 및 평가 로그
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from models.AttriMIL import AttriMIL  # AttriMIL 모델 정의
from constraints import spatial_constraint, rank_constraint  # 정규화 제약 함수
from utils import get_split_loader, calculate_error  # 데이터 로더 및 유틸 함수
from dataloader import Generic_MIL_Dataset  # MIL 데이터셋 클래스

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc

import numpy as np
import queue

# 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 사용 여부

label_dict = {'HGSC': 0, 'LGSC': 1, 'CC': 2, 'EC': 3, 'MC': 4}  # 클래스 라벨 매핑
n_classes = 5  # 클래스 수
feature_dim = 512  # feature dimension (ResNet18 기반)

# 정확도 기록용 클래스
class Accuracy_Logger:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]
        return None if count == 0 else correct / count, correct, count

# 전체 학습 루프 (5-Fold)
def train_attrimil(csv_path, h5_dir, split_path, save_path, max_epoch=200):
    dataset = Generic_MIL_Dataset(
        csv_path=csv_path,
        data_dir=h5_dir,
        shuffle=False,
        seed=7,
        print_info=True,
        label_dict=label_dict,
        patient_strat=False
    )

    for fold in range(5):  # 5개 Fold 학습
        split_file = os.path.join(split_path, f"splits_{fold}.csv")
        train_set, val_set, test_set = dataset.return_splits(from_id=False, csv_path=split_file)

        model = AttriMIL(dim=feature_dim, n_classes=n_classes).to(device)  # 모델 초기화
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, momentum=0.9, weight_decay=1e-5)
        loss_fn = nn.CrossEntropyLoss(weight=calculate_class_weights(train_set, n_classes).to(device))  # 클래스 가중치 적용

        label_positive_list = [queue.Queue(maxsize=4) for _ in range(n_classes)]  # 순위제약용 큐
        label_negative_list = [queue.Queue(maxsize=4) for _ in range(n_classes)]

        train_loader = get_split_loader(train_set, training=True, testing=False, weighted=True)
        val_loader = get_split_loader(val_set, testing=False)
        test_loader = get_split_loader(test_set, testing=False)

        best_loss = float('inf')
        patience = 0  # Early stopping 용

        for epoch in range(max_epoch):
            model.train()
            total_loss = 0
            for data, label, coords, nearest in train_loader:
                data, label = data.to(device), label.to(device)

                # 모델 추론 및 손실 계산
                logits, _, _, attribute_score, _ = model(data)
                loss_bag = loss_fn(logits, label)
                loss_spa = spatial_constraint(attribute_score, n_classes, nearest, ks=3)
                loss_rank, label_positive_list, label_negative_list = rank_constraint(data, label, model, attribute_score, n_classes, label_positive_list, label_negative_list)

                loss = loss_bag + 1.0 * loss_spa + 5.0 * loss_rank  # 총 손실
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()

            val_loss = validate(model, val_loader, loss_fn)  # 검증 손실 평가
            if val_loss < best_loss:
                best_loss = val_loss
                patience = 0
                torch.save(model.state_dict(), os.path.join(save_path, f"fold{fold}_best.pt"))  # 모델 저장
            else:
                patience += 1

            if patience > 20 and epoch > 50:
                print(f"Early stopping at epoch {epoch}")
                break

        model.load_state_dict(torch.load(os.path.join(save_path, f"fold{fold}_best.pt")))  # 최적 모델 불러오기
        test_result(model, test_loader)  # 테스트 평가

# 검증 손실 계산
def validate(model, loader, loss_fn):
    model.eval()
    loss_total = 0
    with torch.no_grad():
        for data, label, coords, nearest in loader:
            data, label = data.to(device), label.to(device)
            logits, _, _, _, _ = model(data)
            loss = loss_fn(logits, label)
            loss_total += loss.item()
    return loss_total / len(loader)

# 테스트 평가 및 출력
def test_result(model, loader):
    model.eval()
    logger = Accuracy_Logger(n_classes)
    all_probs = np.zeros((len(loader), n_classes))  # 예측 확률 저장
    all_labels = np.zeros(len(loader))  # 실제 라벨 저장

    for i, (data, label, coords, nearest) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        with torch.no_grad():
            _, Y_prob, Y_hat, _, _ = model(data)

        logger.log(Y_hat, label)
        all_probs[i] = Y_prob.cpu().numpy()
        all_labels[i] = label.item()

    # AUC 및 클래스별 정확도 출력
    auc = roc_auc_score(label_binarize(all_labels, classes=list(range(n_classes))), all_probs, multi_class='ovr')
    print(f"Test AUC: {auc:.4f}")
    for c in range(n_classes):
        acc, correct, count = logger.get_summary(c)
        print(f"Class {c}: Acc={acc:.4f}, Correct={correct}/{count}")

# 클래스 불균형 보정용 가중치 계산
def calculate_class_weights(dataset, n_classes):
    labels = [label for label in dataset.labels]
    class_counts = np.bincount(labels, minlength=n_classes)
    class_weights = 1. / (class_counts + 1e-6)
    class_weights = class_weights * (len(labels) / np.sum(class_weights * class_counts))
    return torch.FloatTensor(class_weights)
