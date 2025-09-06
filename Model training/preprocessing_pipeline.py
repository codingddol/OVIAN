# -*- coding: utf-8 -*-
"""
preprocessing_pipeline.py
- 원본 ZIP 데이터 → JPG 변환 → 패치 생성 → 특징 추출 → 최근접 이웃 계산 → 최종 h5 병합까지 수행하는 전처리 파이프라인
"""

import os
import zipfile
import io
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm
import imageio.v3 as iio

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# png파일이 너무 커서 로컬환경에서 학습불가 → jpg로 변환하여 저장

# ====== 파일 존재 여부 체크 =======
def image_exists(image_id, label):
    # 저장된 확장자 .jpg로 통일하여 확인
    return os.path.exists(os.path.join(output_base_dir, label, f"{image_id}.jpg"))

# ====== 이미지 리사이즈 (크면 줄임) =======
def resize_if_needed(img, image_id):
    width, height = img.size
    # MAX_DIM보다 크면 비율 유지하며 리사이즈
    if width > MAX_DIM or height > MAX_DIM:
        scale = min(MAX_DIM / width, MAX_DIM / height)
        new_size = (int(width * scale), int(height * scale))
        img = img.resize(new_size, Image.LANCZOS)
        tqdm.write(f"🔄 Resized {image_id} from ({width},{height}) to {new_size}")
    return img

# ====== ZIP에서 이미지 추출 및 JPG로 저장 =======
def extract_and_save(image_id, label, archive):
    target_filename = f"train_images/{image_id}.png"
    # zip 내 해당 이미지 찾기 (대소문자 무시)
    candidates = [f for f in archive.namelist() if f.lower() == target_filename.lower()]
    if not candidates:
        tqdm.write(f"❌ Not found in ZIP: image_id={image_id}")
        return False
    if len(candidates) > 1:
        tqdm.write(f"⚠️ Multiple candidates found for {image_id}: {candidates}, using first one")
    zip_img_path = candidates[0]
    try:
        with archive.open(zip_img_path) as file:
            img = Image.open(io.BytesIO(file.read()))
            img.load()
            img = resize_if_needed(img, image_id)
            img = img.convert("RGB")
            # 저장 경로 생성
            label_dir = os.path.join(output_base_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            dest_path = os.path.join(label_dir, f"{image_id}.jpg")
            img.save(dest_path, "JPEG", quality=80)
            tqdm.write(f"✅ Saved: {dest_path}")
            return True
    except Exception as e:
        tqdm.write(f"⚠️ Failed to process {image_id}: {e}")
        return False

# PIL 옵션 (큰 이미지 허용, 잘린 이미지 허용)
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ====== 설정값 =======
base_dir = os.path.dirname(__file__)
zip_path = os.path.join(base_dir, "data", "UBC_OCEAN1.zip")
output_base_dir = os.path.join(base_dir, "data", "Data")
csv_path = os.path.join(base_dir, "data", "train.csv")
MAX_DIM = 65000
PATCH_SIZE = 256
BLACK_THRESHOLD = 0.1  # 검정색 픽셀 비율 임계값

# zip 원본 png 데이터는 변환 후 삭제하였음

# ====== CSV 기반 데이터 확인 =======
df = pd.read_csv(csv_path)

# ====== 클래스별 파일 수 확인 =======
label_counts = {}
for label in os.listdir(output_base_dir):
    label_path = os.path.join(output_base_dir, label)
    if os.path.isdir(label_path):
        count = len([f for f in os.listdir(label_path) if f.lower().endswith(".jpg")])
        label_counts[label] = count
label_series = pd.Series(label_counts).sort_values(ascending=False)
print("폴더에 다운받아진 파일 수\n", label_series)
print("\n실제 파일 개수\n", df['label'].value_counts().rename_axis(None))

# =====================================================
# 1) 패치 생성 (배경 필터링 포함)
# =====================================================
PATCH_SIZE = 256
BLACK_THRESHOLD = 0.95
WHITE_THRESHOLD = 0.99

def is_black_patch(patch, threshold=BLACK_THRESHOLD):
    # 패치 내 검정 픽셀 비율 계산
    black_pixels = np.all(patch <= 10, axis=2)
    black_ratio = np.sum(black_pixels) / (patch.shape[0] * patch.shape[1])
    return black_ratio >= threshold

def is_white_patch(patch, threshold=WHITE_THRESHOLD):
    # 패치 내 흰 픽셀 비율 계산
    white_pixels = np.all(patch >= 245, axis=2)
    white_ratio = np.sum(white_pixels) / (patch.shape[0] * patch.shape[1])
    return white_ratio >= threshold

def patch_and_save(image_path, save_dir):
    # 원본 이미지를 패치로 분할하여 PNG로 저장
    os.makedirs(save_dir, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    h, w, _ = img_np.shape
    patch_count = 0

    for y in range(0, h, PATCH_SIZE):
        for x in range(0, w, PATCH_SIZE):
            patch = img_np[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
                continue
            if is_black_patch(patch) or is_white_patch(patch):
                continue
            patch_img = Image.fromarray(patch)
            patch_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_patch_{x}_{y}.png"
            patch_img.save(os.path.join(save_dir, patch_filename))
            patch_count += 1

    return patch_count

# =====================================================
# 2) 특징 추출 (ResNet18 backbone)
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchvision import models
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # 마지막 FC 제거
model.eval().to(device)

# 이미지 전처리 파이프라인
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def extract_features_from_patch_folder(patch_dir, h5_save_path):
    # 패치 PNG들을 불러와 feature & 좌표를 h5로 저장
    features = []
    coords = []

    for fname in tqdm(sorted(os.listdir(patch_dir))):
        if not fname.endswith(".png"):
            continue
        try:
            # 파일명에서 좌표 추출
            parts = fname.split("_patch_")[1].split(".")[0].split("_")
            x, y = int(parts[0]), int(parts[1])
        except Exception:
            print(f"⚠️ 좌표 추출 실패: {fname}")
            continue

        path = os.path.join(patch_dir, fname)
        img = Image.open(path).convert("RGB")
        img_tensor = img_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = model(img_tensor).squeeze().cpu().numpy()

        features.append(feat)
        coords.append([x, y])

    features = np.array(features).squeeze()
    coords = np.array(coords)

    import h5py
    with h5py.File(h5_save_path, 'w') as f:
        f.create_dataset('features', data=features)
        f.create_dataset('coords', data=coords)

    return features.shape, coords.shape

# =====================================================
# 3) 최근접 이웃(nearest) 계산
# =====================================================
from joblib import Parallel, delayed
import h5py

def find_nearest(input_path, output_path, patch_size=(256, 256)):
    name = os.path.basename(input_path)
    print("📂 처리 중:", name)

    h5 = h5py.File(input_path, 'r')
    coords = np.array(h5['coords'])
    h5.close()

    nearest = []

    for step, p in enumerate(coords):
        exists = [step]

        def get_neighbor(offset_x, offset_y):
            neighbor = np.array([p[0] + offset_x, p[1] + offset_y])
            loc = np.where(np.sum(coords == neighbor, axis=1) == 2)[0]
            return loc[0] if len(loc) != 0 else step

        # 8방향 이웃 탐색
        directions = [
            (0, -patch_size[1]), (0, +patch_size[1]),
            (-patch_size[0], 0), (+patch_size[0], 0),
            (-patch_size[0], -patch_size[1]), (+patch_size[0], -patch_size[1]),
            (-patch_size[0], +patch_size[1]), (+patch_size[0], +patch_size[1]),
        ]

        for dx, dy in directions:
            exists.append(get_neighbor(dx, dy))

        nearest.append(exists)

    # nearest 배열을 h5에 추가 저장
    with h5py.File(output_path, 'a') as h5:
        h5.create_dataset('nearest', data=nearest)

# =====================================================
# 4) features+coords+nearest 병합
# =====================================================
def merge_h5_files(h5_feature_path, h5_with_nearest_path, save_path):
    os.makedirs(save_path, exist_ok=True)

    for name in os.listdir(h5_feature_path):
        if not name.endswith(".h5"):
            continue

        save_file = os.path.join(save_path, name)
        if os.path.exists(save_file):
            print(f"⏩ 존재함: {name}")
            continue

        print(f"🔄 병합 중: {name}")

        # features, coords 로딩
        with h5py.File(os.path.join(h5_feature_path, name), "r") as h5:
            coords = np.array(h5["coords"])
            features = np.array(h5["features"])

        # nearest 로딩
        with h5py.File(os.path.join(h5_with_nearest_path, name), "r") as h5:
            nearest = np.array(h5["nearest"])

        # 최종 저장
        with h5py.File(save_file, "w") as h5:
            h5.create_dataset("coords", data=coords)
            h5.create_dataset("features", data=features)
            h5.create_dataset("nearest", data=nearest)

        print(f"✅ 저장 완료: {save_file} | coords: {coords.shape}, features: {features.shape}, nearest: {nearest.shape}")

# =====================================================
# 메인 실행부
# =====================================================
if __name__ == "__main__":
    base_image_dir = os.path.join(base_dir, "data", "Data")  
    patch_base_dir = os.path.join(base_dir, "data", "AttriMIL_DATA", "patches")
    h5_save_dir = os.path.join(base_dir, "data", "AttriMIL_DATA", "h5_files")
    save_path = os.path.join(base_dir, "data", "AttriMIL_DATA", "h5_coords_files")

    os.makedirs(patch_base_dir, exist_ok=True)
    os.makedirs(h5_save_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    # 라벨별 JPG 이미지 순회
    for label in os.listdir(base_image_dir):
        label_path = os.path.join(base_image_dir, label)
        if not os.path.isdir(label_path):
            continue

        print(f"\n📁 라벨 처리 중: {label}")
        for fname in os.listdir(label_path):
            if not fname.lower().endswith(".jpg"):
                continue

            slide_id = os.path.splitext(fname)[0]
            image_path = os.path.join(label_path, fname)
            patch_dir = os.path.join(patch_base_dir, label, slide_id)
            h5_path = os.path.join(h5_save_dir, f"{label}_{slide_id}.h5")

            # (a) 패치 생성
            if not os.path.exists(patch_dir) or not any(f.endswith(".png") for f in os.listdir(patch_dir)):
                print(f"🔄 {fname} → 패치 생성")
                patch_count = patch_and_save(image_path, patch_dir)
                print(f"✅ 패치 수: {patch_count}")
                if patch_count == 0:
                    print("⚠️ 유효한 패치 없음, 건너뜀")
                    continue
            else:
                print(f"⏩ 패치 있음: {fname}")

            # (b) 특징 추출
            if not os.path.exists(h5_path):
                print(f"🔄 {fname} → feature 추출")
                try:
                    feat_shape, coord_shape = extract_features_from_patch_folder(patch_dir, h5_path)
                    print(f"✅ 저장됨: {h5_path} | features: {feat_shape}, coords: {coord_shape}")
                except Exception as e:
                    print(f"❌ 오류 발생: {e}")
            else:
                print(f"⏩ h5 있음: {h5_path}")

    # (c) 최근접 이웃 계산
    h5_files = [f for f in os.listdir(h5_save_dir) if f.endswith('.h5')]
    print(f"총 {len(h5_files)}개의 h5 파일 처리 예정")
    Parallel(n_jobs=8)(delayed(find_nearest)(
        os.path.join(h5_save_dir, fname),
        os.path.join(h5_save_dir, fname),
        (256, 256)
    ) for fname in tqdm(h5_files))

    # (d) 병합 저장
    merge_h5_files(h5_save_dir, h5_save_dir, save_path)
