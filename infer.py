# -*- coding: utf-8 -*-
"""
infer.py

기능 요약:
- 암호화 이미지 복호화 후 리사이즈/패치 분할/특징 추출
- H5 저장 및 주변 패치 정보 보강
- AttriMIL 기반 예측 및 Attention Map 시각화 (XAI 목적)
- 전체 과정은 디스크 평문 최소화, 메모리 중심 처리

이 파일은 Flask 추론 서버(flask_server.py)에서 호출되어 동작하며,
보안 흐름과 시각적 결과 출력을 모두 포함합니다.
"""

# --- 모듈 로딩 ---
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))

# 기본 라이브러리
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from io import BytesIO
import base64

# Torch 관련
import torch
import torch.nn.functional as F
from torchvision import models, transforms

# 저장용
import h5py

# 시각화
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 내부 모델
from AttriMIL import AttriMIL

# --- 설정값 ---
PATCH_SIZE = 256
MAX_DIM = 65000
JPEG_QUALITY = 80
BLACK_THRESHOLD = 0.95
WHITE_THRESHOLD = 0.99
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Image.MAX_IMAGE_PIXELS = None  # 대용량 이미지 허용

# --- AttriMIL 모델 로드 ---
model = AttriMIL(n_classes=5, dim=512).to(device)
base_dir = os.path.dirname(__file__)
ckpt_path = os.path.join(base_dir, "save_weights", "attrimil_final.pth")
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

# --- ResNet18 특징 추출기 세팅 ---
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# --- 이미지 리사이즈 및 압축 (디스크 저장 없이 메모리 처리) ---
def resize_and_save(image_input):
    """
    입력 이미지를 MAX_DIM 기준으로 리사이즈하고,
    JPEG 압축 후 다시 PIL 이미지로 로드해 반환.
    디스크 평문 저장 없음.
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise TypeError("image_input must be a file path or PIL.Image.Image")

    w, h = img.size
    if w > MAX_DIM or h > MAX_DIM:
        scale = MAX_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = BytesIO()
    img.save(buf, format='JPEG', quality=JPEG_QUALITY)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img

# --- 흑백 필터링 ---
def is_black(patch):
    return np.mean(patch < 10) > BLACK_THRESHOLD

def is_white(patch):
    return np.mean(patch > 245) > WHITE_THRESHOLD

# --- 패치 생성 ---
def create_patches(image_input, save_dir):
    """
    이미지를 PATCH_SIZE로 분할하고, 흑/백 필터링 후 유효 패치만 저장
    """
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise TypeError("image_input must be a file path or PIL.Image.Image")

    img_np = np.array(img)
    h, w, _ = img_np.shape
    count = 0
    for y in range(0, h, PATCH_SIZE):
        for x in range(0, w, PATCH_SIZE):
            patch = img_np[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            if patch.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
                continue
            if is_black(patch) or is_white(patch):
                continue
            Image.fromarray(patch).save(os.path.join(save_dir, f"patch_{x}_{y}.png"))
            count += 1
    return count

# --- 단일 패치 처리 (좌표와 특징 추출) ---
def _process_patch(fname, patch_dir, transform, device, resnet):
    if not fname.endswith(".png"):
        return None
    
    x, y = map(int, fname.replace(".png", "").split("_")[1:])
    img_path = os.path.join(patch_dir, fname)
    
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = resnet(img_tensor).squeeze().cpu().numpy()
    
    return feat, [x, y]

# --- 특징 추출 및 저장 (H5) ---
def extract_features(patch_dir, h5_path, batch_size=64):
    """
    모든 패치에 대해 ResNet 특징 추출 → H5 저장
    """
    files = sorted([f for f in os.listdir(patch_dir) if f.endswith(".png")])
    if not files:
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('features', data=np.empty((0, 512), dtype=np.float32))
            f.create_dataset('coords', data=np.empty((0, 2), dtype=np.int32))
        return

    coords = [list(map(int, os.path.splitext(f)[0].split("_")[1:])) for f in files]
    feats_chunks = []

    use_cuda = (device.type == "cuda")
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        imgs = [transform(Image.open(os.path.join(patch_dir, fname)).convert("RGB")) for fname in batch_files]
        batch = torch.stack(imgs).to(device, non_blocking=True)

        with torch.no_grad():
            if use_cuda:
                with torch.cuda.amp.autocast():
                    out = resnet(batch)
            else:
                out = resnet(batch)
            out = out.view(out.size(0), -1)

        feats_chunks.append(out.cpu().numpy())

    feats = np.vstack(feats_chunks).astype(np.float32)
    coords_np = np.array(coords, dtype=np.int32)

    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('features', data=feats)
        f.create_dataset('coords', data=coords_np)

# --- 최근접 이웃 추가 ---
def add_nearest(h5_path):
    with h5py.File(h5_path, 'r') as f:
        coords = np.array(f['coords'])

    nearest = []
    for idx, p in enumerate(coords):
        neighbors = [idx]
        for dx, dy in [(0,-PATCH_SIZE),(0,PATCH_SIZE),(-PATCH_SIZE,0),(PATCH_SIZE,0),
                       (-PATCH_SIZE,-PATCH_SIZE),(PATCH_SIZE,-PATCH_SIZE),
                       (-PATCH_SIZE,PATCH_SIZE),(PATCH_SIZE,PATCH_SIZE)]:
            neighbor = p + np.array([dx, dy])
            loc = np.where(np.all(coords == neighbor, axis=1))[0]
            neighbors.append(loc[0] if len(loc) else idx)
    with h5py.File(h5_path, 'a') as f:
        f.create_dataset('nearest', data=np.array(nearest))

# --- H5 파일 병합 ---
def merge_h5(input_h5, output_h5):
    """
    coords / features / nearest 정보 복사
    """
    with h5py.File(input_h5, 'r') as f:
        coords = np.array(f['coords'])
        features = np.array(f['features'])
        nearest = np.array(f['nearest'])
    with h5py.File(output_h5, 'w') as f:
        f.create_dataset('coords', data=coords)
        f.create_dataset('features', data=features)
        f.create_dataset('nearest', data=nearest)

# --- 최종 추론 및 attention map 시각화 ---
def infer_and_get_attention(final_h5_path, resized_img_input):
    """
    최종 H5에서 특징 → AttriMIL 추론 → Attention Map 생성 및 시각화 출력 (base64)
    """
    with h5py.File(final_h5_path, 'r') as f:
        features = torch.tensor(np.array(f['features']), dtype=torch.float32).to(device)
        coords_np = np.array(f['coords'])

    with torch.no_grad():
        logits, _, _, attribute_score, _ = model(features)

    pred_class = logits.argmax(dim=1).item()
    softmax_probs = F.softmax(logits, dim=1).cpu().numpy().flatten()

    attention_scores = attribute_score[0, pred_class].cpu().numpy().flatten()
    attention_norm = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min() + 1e-8)
    attention_norm = np.power(attention_norm, 0.5)

    # --- 이미지 로드 ---
    if isinstance(resized_img_input, Image.Image):
        wsi_img = resized_img_input.convert("RGB")
    else:
        wsi_img = Image.open(resized_img_input).convert("RGB")

    base_w, base_h = wsi_img.size
    coords_np = np.array(coords_np)

    # --- attention map → grid 재구성 ---
    def _median_step(vals, default=PATCH_SIZE):
        u = np.unique(vals)
        if u.size <= 1:
            return default
        d = np.diff(u)
        if d.size == 0:
            return default
        m = np.median(d)
        return int(m) if np.isfinite(m) and m > 0 else default

    step_x = _median_step(coords_np[:, 0])
    step_y = _median_step(coords_np[:, 1])
    step = max(1, int((step_x + step_y) / 2))

    grid_w = max(1, int((coords_np[:,0].max() - coords_np[:,0].min()) / step) + 1)
    grid_h = max(1, int((coords_np[:,1].max() - coords_np[:,1].min()) / step) + 1)

    grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    gx = ((coords_np[:,0] - coords_np[:,0].min()) / step).round().astype(int)
    gy = ((coords_np[:,1] - coords_np[:,1].min()) / step).round().astype(int)
    grid[gy, gx] = attention_norm

    grid_img = Image.fromarray((grid * 255).astype(np.uint8)).resize((base_w, base_h), Image.NEAREST)
    grid_np = np.array(grid_img, dtype=np.float32) / 255.0
    grid_img = Image.fromarray((grid_np * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=6))
    att_full = np.array(grid_img, dtype=np.float32) / 255.0

    # --- ROI 시각화 ---
    p = 85
    t = np.percentile(att_full, p)
    k = 12.0
    alpha = 1.0 / (1.0 + np.exp(-k * (att_full - t)))
    alpha = np.clip(alpha, 0, 1)

    bg_gray = ImageOps.grayscale(wsi_img).filter(ImageFilter.GaussianBlur(radius=3)).convert("RGB")
    bg_gray = ImageEnhance.Brightness(bg_gray).enhance(1.2)

    alpha_mask = Image.fromarray(((alpha > 0.5).astype(np.float32) * 255).astype(np.uint8))

    heatmap_cmap = mcolors.LinearSegmentedColormap.from_list(
        "heatmap_cmap",
        [(0.0, (1, 1, 1, 0.0)),
         (0.2, (1, 1, 1, 0.0)),
         (0.5, (1, 1, 1, 0.0)),
         (0.85, (1, 1, 1, 0.0)),
         (1.0, (1, 0, 0, 1))]
    )
    roi_color_np = (heatmap_cmap(att_full)[:, :, :3] * 255).astype(np.uint8)
    roi_colored = Image.fromarray(roi_color_np)
    composed = Image.composite(roi_colored, bg_gray, alpha_mask)

    x_min, x_max = coords_np[:,0].min(), coords_np[:,0].max()
    y_min, y_max = coords_np[:,1].min(), coords_np[:,1].max()
    scaled_x = (coords_np[:,0] - x_min) / (x_max - x_min + 1e-8) * base_w
    scaled_y = (coords_np[:,1] - y_min) / (y_max - y_min + 1e-8) * base_h

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(composed)
    ax.scatter(scaled_x, scaled_y, c=attention_norm, cmap=heatmap_cmap, s=15, alpha=1.0, edgecolors='none')
    plt.axis('off')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    attention_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)

    return pred_class, softmax_probs.tolist(), attention_base64
