# -*- coding: utf-8 -*-
"""
utils.py

MIL(Multiple Instance Learning) 기반 병리 이미지 분류 모델 학습에 필요한
데이터 로딩, 샘플링, 최적화, 평가 등 유틸리티 함수들을 정의한 모듈입니다.

📌 주요 기능:
- Custom Sampler 및 MIL 전용 collate 함수 정의
- 학습/검증/테스트용 DataLoader 생성 (weighted/순차/무작위 샘플링)
- 모델 파라미터 초기화 및 구조 출력
- 클래스 불균형 대응을 위한 가중치 생성
- MIL 학습을 위한 split 생성기 제공 (train/val/test 분리)
- 정확도 및 오류율 계산 함수

이 모듈은 AttriMIL 모델 학습 코드(train_attrimil.py)와 함께 사용됩니다.
"""

import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections

# GPU 사용 여부
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 커스텀 샘플러: 주어진 인덱스 리스트에서 순차적으로 샘플링
class SubsetSequentialSampler(Sampler):
    """
    주어진 인덱스 리스트를 순차적으로 샘플링하는 커스텀 Sampler.
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

# MIL 학습을 위한 collate 함수 (좌표 정보 없이 이미지 + 라벨만 반환)
def collate_MIL(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]

# MIL 학습을 위한 collate 함수 (좌표와 최근접 패치 정보 포함)
def collate_MIL_coords(batch):
    imgs = torch.cat([item[0] for item in batch], dim=0)
    labels = torch.LongTensor([item[1] for item in batch])
    coords = []
    nearest = []

    for item in batch:
        coord = item[2]
        near = item[3]

        if isinstance(coord, np.ndarray):
            coords.append(coord)
        else:
            coords.append(coord.cpu().numpy())

        if isinstance(near, np.ndarray):
            nearest.append(near)
        else:
            nearest.append(near.cpu().numpy())

    return imgs, labels, coords, nearest

# 패치 추론용 collate 함수 (이미지 + 좌표만 반환)
def collate_features(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]

# 일반적인 순차 DataLoader (Validation/Test 용도)
def get_simple_loader(dataset, batch_size=1, num_workers=1):
    kwargs = {'num_workers': 4, 'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler.SequentialSampler(dataset),
        collate_fn=collate_MIL,
        **kwargs
    )
    return loader 

# split_dataset에 대해 학습/검증/테스트용 DataLoader 생성
def get_split_loader(split_dataset, training=False, testing=False, weighted=False, batch_size=1):
    """
    split_dataset을 입력으로 받아 상황에 따라 적절한 DataLoader 반환
    """
    kwargs = {'num_workers': 4} if device.type == "cuda" else {}

    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(
                    split_dataset,
                    batch_size=batch_size,
                    sampler=WeightedRandomSampler(weights, len(weights)),
                    collate_fn=collate_MIL_coords,
                    **kwargs
                )
            else:
                loader = DataLoader(
                    split_dataset,
                    batch_size=batch_size,
                    sampler=RandomSampler(split_dataset),
                    collate_fn=collate_MIL_coords,
                    **kwargs
                )
        else:
            loader = DataLoader(
                split_dataset,
                batch_size=batch_size,
                sampler=SequentialSampler(split_dataset),
                collate_fn=collate_MIL_coords,
                **kwargs
            )
    else:
        ids = np.random.choice(np.arange(len(split_dataset)), int(len(split_dataset) * 0.1), replace=False)
        loader = DataLoader(
            split_dataset,
            batch_size=batch_size,
            sampler=SubsetSequentialSampler(ids),
            collate_fn=collate_MIL,
            **kwargs
        )

    return loader

# 학습 optimizer 생성
def get_optim(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer

# 모델 구조 및 파라미터 수 출력
def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)
    
    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
    
    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)

# MIL에서 클래스별 split 생성기 (train/val/test 분리)
def generate_split(cls_ids, val_num, test_num, samples, n_splits=5,
                   seed=7, label_frac=1.0, custom_test_ids=None):
    indices = np.arange(samples).astype(int)
    
    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)

    np.random.seed(seed)
    for i in range(n_splits):
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []
        
        if custom_test_ids is not None:
            all_test_ids.extend(custom_test_ids)

        for c in range(len(val_num)):
            possible_indices = np.intersect1d(cls_ids[c], indices)
            val_ids = np.random.choice(possible_indices, val_num[c], replace=False)
            remaining_ids = np.setdiff1d(possible_indices, val_ids)
            all_val_ids.extend(val_ids)

            if custom_test_ids is None:
                test_ids = np.random.choice(remaining_ids, test_num[c], replace=False)
                remaining_ids = np.setdiff1d(remaining_ids, test_ids)
                all_test_ids.extend(test_ids)

            if label_frac == 1:
                sampled_train_ids.extend(remaining_ids)
            else:
                sample_num = math.ceil(len(remaining_ids) * label_frac)
                slice_ids = np.arange(sample_num)
                sampled_train_ids.extend(remaining_ids[slice_ids])

        yield sampled_train_ids, all_val_ids, all_test_ids

# iterable에서 n번째 요소 가져오기
def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator, n, None), default)

# 예측 오류율 계산 (1 - accuracy)
def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()
    return error

# 클래스 불균형 보정용 샘플 가중치 계산 (train용)
def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))                                           
    weight_per_class = [N / len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        y = dataset.getlabel(idx)                        
        weight[idx] = weight_per_class[y]                                  
    return torch.DoubleTensor(weight)

# 모델 파라미터 초기화 (Linear, BatchNorm 계층 대상)
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
