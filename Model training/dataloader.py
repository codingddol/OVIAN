# -*- coding: utf-8 -*-
"""
dataloader.py
- MIL(Multiple Instance Learning) 방식 학습/추론을 위한 데이터 로더 정의.
- 주요 구성:
  1) SimpleMILDataset : CSV 기반 전체 슬라이드 단위 Dataset
  2) SimpleMILSplit   : slide_data를 학습/검증/테스트 split별로 관리하는 Dataset
  3) mil_collate_fn   : DataLoader에서 batch 단위로 묶을 때 coords, nearest를 list 형태로 유지하는 Collate 함수
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import h5py


class SimpleMILDataset:
    """
    [목적]
        - MIL 학습을 위해 슬라이드 단위 데이터를 관리하는 Dataset 클래스.
        - CSV 파일에는 각 슬라이드의 image_id, label 등이 기록되어 있으며,
          label_dict를 이용해 문자열 라벨을 정수 인덱스로 매핑.

    Args:
        csv_path (str): 슬라이드 메타데이터(csv) 경로
        data_dir (str): HDF5 파일들이 저장된 디렉토리
        label_dict (dict): {라벨명: 정수} 매핑 딕셔너리

    Attributes:
        slide_data (pd.DataFrame): csv 로드한 전체 데이터프레임
        df (pd.DataFrame): 동일한 참조, 접근 편의용
        data_dir (str): 데이터 디렉토리 경로
        label_dict (dict): 라벨 매핑 딕셔너리
        num_classes (int): 클래스 개수
        slide_cls_ids (List[List[int]]): 각 클래스별 sample 인덱스 리스트
        labels (List[int]): 각 샘플 라벨 리스트
    """
    def __init__(self, csv_path, data_dir, label_dict):
        self.slide_data = pd.read_csv(csv_path)
        # 문자열 라벨 → 정수 라벨 변환
        self.slide_data['label'] = self.slide_data['label'].map(label_dict)

        self.df = self.slide_data
        self.data_dir = data_dir
        self.label_dict = label_dict
        self.num_classes = len(label_dict)

        # 클래스별 인덱스 리스트 구성
        self.slide_cls_ids = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0].tolist()

        self.labels = self.slide_data['label'].tolist()

    def __len__(self):
        """데이터셋 크기 반환"""
        return len(self.df)

    def getlabel(self, idx):
        """특정 인덱스 샘플의 라벨 반환"""
        return self.df['label'].iloc[idx]

    @classmethod
    def from_df(cls, df, data_dir, label_dict):
        """
        DataFrame으로부터 SimpleMILSplit 객체를 생성하는 편의 메서드
        """
        instance = cls.__new__(cls)
        instance.df = df
        instance.data_dir = data_dir
        instance.label_dict = label_dict
        instance.image_ids = df['image_id'].tolist()
        instance.labels = df['label'].tolist()
        instance.num_classes = len(label_dict)

        instance.slide_cls_ids = [[] for _ in range(instance.num_classes)]
        for i in range(instance.num_classes):
            instance.slide_cls_ids[i] = np.where(df['label'] == i)[0].tolist()

        return SimpleMILSplit(df, data_dir, instance.num_classes, label_dict)

    def return_splits(self, from_id=True, csv_path=None):
        """
        미리 정의된 split(csv)에 따라 train/val/test 데이터셋을 나누어 반환
        Args:
            from_id (bool): image_id 기준 split 여부 (현재 False로만 사용)
            csv_path (str): split csv 경로 (train/val/test 컬럼 포함)
        Returns:
            Tuple(SimpleMILSplit, SimpleMILSplit, SimpleMILSplit)
        """
        assert not from_id and csv_path is not None
        splits = pd.read_csv(csv_path)
        return (
            SimpleMILSplit(
                self.slide_data[self.slide_data['image_id'].isin(splits['train'])].reset_index(drop=True),
                self.data_dir,
                self.num_classes,
                self.label_dict
            ),
            SimpleMILSplit(
                self.slide_data[self.slide_data['image_id'].isin(splits['val'])].reset_index(drop=True),
                self.data_dir,
                self.num_classes,
                self.label_dict
            ),
            SimpleMILSplit(
                self.slide_data[self.slide_data['image_id'].isin(splits['test'])].reset_index(drop=True),
                self.data_dir,
                self.num_classes,
                self.label_dict
            )
        )


class SimpleMILSplit(Dataset):
    """
    [목적]
        - 특정 split(train/val/test)에 해당하는 슬라이드 데이터셋 관리
        - HDF5 파일에서 feature, coords, nearest 정보를 로드

    Args:
        slide_data (pd.DataFrame): split 대상 슬라이드 데이터
        data_dir (str): 데이터 디렉토리
        num_classes (int): 클래스 개수
        label_dict (dict): 라벨 매핑 딕셔너리
    """
    def __init__(self, slide_data, data_dir, num_classes, label_dict):
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_dict = label_dict

        # 클래스별 인덱스 정리
        self.slide_cls_ids = [[] for _ in range(num_classes)]
        for i in range(num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0].tolist()

    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): 샘플 인덱스
        Returns:
            features (torch.Tensor): 패치 feature 텐서
            label (int): 정수 라벨
            coords (np.ndarray): 패치 좌표
            nearest (np.ndarray): 최근접 패치 인덱스
        """
        row = self.slide_data.iloc[idx]
        image_id = row['image_id']
        label = row['label']

        # 정수라벨 → 문자열 라벨 역매핑
        inv_label_dict = {v: k for k, v in self.label_dict.items()}
        label_str = inv_label_dict[label]

        # HDF5 파일명 생성 (예: "CC_12345.h5")
        h5_filename = f"{label_str}_{image_id}.h5"
        h5_path = os.path.join(os.path.dirname(__file__), self.data_dir, h5_filename)

        if not os.path.isfile(h5_path):
            raise FileNotFoundError(f"❌ HDF5 file not found: {h5_path}")

        with h5py.File(h5_path, 'r') as f:
            features = torch.from_numpy(f['features'][:])  # (num_instances, feature_dim)
            coords = f['coords'][:]                       # (num_instances, 2)
            nearest = f['nearest'][:]                     # (num_instances, k)

        return features, label, coords, nearest

    def getlabel(self, idx):
        """특정 인덱스 라벨 반환"""
        return self.slide_data['label'].iloc[idx]


# --- Collate 함수 ---
def mil_collate_fn(batch):
    """
    DataLoader에서 batch 단위로 묶을 때 사용하는 collate 함수.
    coords/nearest는 numpy array이므로 stack 대신 list 형태를 유지한다.
    """
    data_list, label_list, coords_list, nearest_list = zip(*batch)

    data = torch.stack(data_list, dim=0)   # (batch_size, num_instances, feature_dim)
    labels = torch.tensor(label_list)      # (batch_size,)
    coords = list(coords_list)             # 좌표: 각 슬라이드별 numpy array
    nearest = list(nearest_list)           # 최근접 정보: 각 슬라이드별 numpy array

    return data, labels, coords, nearest
