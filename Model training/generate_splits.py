# -*- coding: utf-8 -*-
"""
generate_splits.py
- 학습용 CSV(train.csv)를 불러와 Stratified K-Fold 기반의 검증용 split 파일을 생성.
- 현재 스크립트는 '각 폴드의 검증 세트(val)만' CSV로 저장합니다.
  (예: ./splits/split_0.csv, split_1.csv ... → 각 파일은 해당 폴드의 'val' 샘플 목록)

필드 요구:
- train.csv: 최소 다음 컬럼 필요
  - image_id: 슬라이드/샘플 식별자
  - label   : 문자열 라벨(HGSC, LGSC, CC, EC, MC)

라벨 매핑:
- label_dict = {'HGSC': 0, 'LGSC': 1, 'CC': 2, 'EC': 3, 'MC': 4}

출력:
- ./splits/split_{fold}.csv
  - 컬럼: image_id, label(정수로 매핑된 값)
"""

import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# ------------------------------
# 1) 경로/환경 로그
# ------------------------------
print("[현재 디렉토리]", os.getcwd())

# ------------------------------
# 2) 데이터 불러오기
#    - 필요한 컬럼만 선택
# ------------------------------
df = pd.read_csv("./train.csv")[["image_id", "label"]]
print("[데이터 개수]", df.shape)
print(df.head())

# ------------------------------
# 3) 라벨 매핑 (문자열 → 정수)
#    - 매핑되지 않은 라벨이 있는지 점검
# ------------------------------
label_dict = {"HGSC": 0, "LGSC": 1, "CC": 2, "EC": 3, "MC": 4}
df["label"] = df["label"].map(label_dict)

if df["label"].isna().any():
    missing = df[df["label"].isna()]
    raise ValueError(
        f"라벨 매핑 실패한 행이 있습니다. label_dict를 확인하세요.\n{missing.head(10)}"
    )

# ------------------------------
# 4) splits 디렉토리 생성
# ------------------------------
split_dir = "./splits"
os.makedirs(split_dir, exist_ok=True)

# ------------------------------
# 5) Stratified K-Fold 분할
#    - 계층적 분할로 폴드마다 라벨 분포를 유지
#    - 현재는 각 폴드의 '검증 세트(val_idx)'만 저장
# ------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(df["image_id"], df["label"])):
    # 검증 세트만 저장 (필요 시 train 세트도 별도로 저장 가능)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    csv_path = os.path.join(split_dir, f"split_{fold}.csv")
    df_val.to_csv(csv_path, index=False)

    # 간단 로그: 폴드별 샘플 수 및 라벨 분포
    label_counts = df_val["label"].value_counts().sort_index().to_dict()
    print(f"[Fold {fold}] Saved {len(df_val)} samples → {csv_path}")
    print(f"          val label distribution: {label_counts}")

# ------------------------------
# 6) 저장 확인
# ------------------------------
print("[저장된 파일 목록]", os.listdir(split_dir))

"""
[참고] train/val 모두 저장하고 싶다면:
for fold, (train_idx, val_idx) in enumerate(skf.split(df["image_id"], df["label"])):
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = df.iloc[val_idx].reset_index(drop=True)

    df_train.to_csv(os.path.join(split_dir, f"train_split_{fold}.csv"), index=False)
    df_val.to_csv(os.path.join(split_dir, f"val_split_{fold}.csv"), index=False)
"""
