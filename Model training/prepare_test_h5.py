# -*- coding: utf-8 -*-
"""
prepare_test_h5.py
- 슬라이드 이미지(WSI 등)를 입력받아 학습용 HDF5 파일(.h5)로 변환하는 전처리 스크립트.
- 주요 단계:
  1) 배경 필터링: 지나치게 검은/흰 패치 제거
  2) 패치 분할: 원본 이미지를 PATCH_SIZE 단위로 슬라이싱
  3) 특징 추출: ResNet18 사전학습 모델로 feature vector 생성
  4) 최근접 패치 계산: 공간적으로 인접한 패치들 간 인덱스 매핑
  5) HDF5 저장: features, coords, nearest 세 가지를 dataset으로 기록
"""

import os
import numpy as np
import h5py
from PIL import Image
from torchvision import models, transforms
import torch
from tqdm import tqdm

# ------------------------------
# 전역 설정값
# ------------------------------
PATCH_SIZE = 256                # 패치 크기(px 단위)
BLACK_THRESHOLD = 0.95           # 전체 픽셀 중 95% 이상이 검으면 배경으로 간주
WHITE_THRESHOLD = 0.99           # 전체 픽셀 중 99% 이상이 흰색이면 배경으로 간주
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Feature extractor 모델 (ResNet18)
# - ImageNet pretrained
# - 마지막 FC layer 제거 → 512-dim feature vector 출력
# ------------------------------
resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # 마지막 FC 제거
resnet.eval().to(device)

# 이미지 변환 파이프라인
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # ResNet 입력 크기
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet mean
                         [0.229, 0.224, 0.225]), # ImageNet std
])

# ===========================================================
# 1) 배경 필터링 함수
# ===========================================================
def is_black(patch):
    """
    Args:
        patch (np.ndarray): 패치 이미지 배열
    Returns:
        bool: 대부분 검은색인 경우 True
    """
    return np.mean(patch <= 10) >= BLACK_THRESHOLD

def is_white(patch):
    """
    Args:
        patch (np.ndarray): 패치 이미지 배열
    Returns:
        bool: 대부분 흰색인 경우 True
    """
    return np.mean(patch >= 245) >= WHITE_THRESHOLD


# ===========================================================
# 2) 패치 생성 (이미지 슬라이싱 후 반환)
# ===========================================================
def generate_patches(image_path):
    """
    Args:
        image_path (str): 원본 이미지 경로
    Returns:
        patches (List[PIL.Image.Image]): 전처리된 패치들
        coords  (np.ndarray): 각 패치의 좌표 (x, y)
    """
    patches, coords = [], []
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    for y in range(0, h, PATCH_SIZE):
        for x in range(0, w, PATCH_SIZE):
            patch = img_np[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            # 경계에 걸쳐 패치 크기가 모자란 경우 skip
            if patch.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
                continue

            # 배경(검정/흰색) 패치는 skip
            if is_black(patch) or is_white(patch):
                continue

            patches.append(Image.fromarray(patch))
            coords.append([x, y])

    return patches, np.array(coords)


# ===========================================================
# 3) 특징 추출
# ===========================================================
def extract_features(patches):
    """
    Args:
        patches (List[PIL.Image.Image]): 패치 이미지들
    Returns:
        np.ndarray: shape (num_patches, 512) feature 벡터
    """
    feats = []
    for p in patches:
        inp = transform(p).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = resnet(inp).squeeze().cpu().numpy()
        feats.append(feat)
    return np.array(feats)


# ===========================================================
# 4) 최근접 좌표 계산
# ===========================================================
def compute_nearest(coords):
    """
    Args:
        coords (np.ndarray): shape (N, 2) 각 패치의 (x, y) 좌표
    Returns:
        np.ndarray: shape (N, 9), 각 패치의 최근접 이웃 인덱스 리스트
                    (자신 포함 + 주변 8방향)
    """
    nearest = []
    for i, p in enumerate(coords):
        neighbors = [i]  # 자기 자신 포함
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == dy == 0:
                    continue
                neighbor = p + np.array([dx * PATCH_SIZE, dy * PATCH_SIZE])
                idx = np.where(np.all(coords == neighbor, axis=1))[0]
                neighbors.append(idx[0] if len(idx) > 0 else i)  # 없으면 자기 자신
        nearest.append(neighbors)
    return np.array(nearest)


# ===========================================================
# 5) 메인 실행 함수
# ===========================================================
def preprocess_slide(image_path, save_path, label, image_id):
    """
    Args:
        image_path (str): 원본 이미지 경로
        save_path (str): .h5 저장 경로
        label (str): 라벨명 (ex: 'CC', 'HGSC')
        image_id (str|int): 샘플 ID
    Returns:
        None (HDF5 파일 생성)
    """
    print(f"📄 처리 중: {image_id}")

    # (1) 패치 생성
    patches, coords = generate_patches(image_path)
    if len(patches) == 0:
        print("⚠️ 유효한 패치 없음")
        return

    # (2) 특징 추출
    features = extract_features(patches)

    # (3) 최근접 이웃 계산
    nearest = compute_nearest(coords)

    # (4) 저장
    os.makedirs(save_path, exist_ok=True)
    h5_name = f"{label}_{image_id}.h5"
    with h5py.File(os.path.join(save_path, h5_name), 'w') as h5:
        h5.create_dataset('features', data=features)  # (N, 512)
        h5.create_dataset('coords', data=coords)      # (N, 2)
        h5.create_dataset('nearest', data=nearest)    # (N, 9)

    print(f"✅ 저장 완료: {h5_name} | features: {features.shape}, coords: {coords.shape}, nearest: {nearest.shape}")
