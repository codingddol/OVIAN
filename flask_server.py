# -*- coding: utf-8 -*-
"""
flask_server.py

난소암 조직 이미지 분석을 위한 AI 추론 서버입니다.
Streamlit과 별도로 실행되며, 다음과 같은 기능을 제공합니다:

📌 주요 기능:
- /infer: 이미지 업로드 후 암호화 저장 → 복호화 → 전처리 → 추론 → 결과 반환
- /clear_uploads: 업로드 및 생성된 임시 파일 전체 삭제
- 자동 스케줄러를 통한 오래된 암호화 파일 정리 (기본 7일)

보안성:
- 업로드된 파일은 Fernet으로 chunk 단위 암호화 후 저장
- 분석 시 복호화하여 메모리 기반 처리 (디스크 평문 저장 없음)
"""

import sys
from flask import Flask, request, jsonify
import os
sys.path.append(os.path.dirname(__file__))  # 현재 경로를 모듈 탐색 경로에 추가

import shutil
from cryptography.fernet import Fernet
from infer import resize_and_save, create_patches, extract_features, add_nearest, merge_h5, infer_and_get_attention
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image, UnidentifiedImageError 
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import atexit

from werkzeug.utils import secure_filename     
import uuid


# 환경변수 로딩 및 암호화 키 확보
load_dotenv()
OVIAN_TOKEN = os.environ.get("OVIAN_TOKEN")
OVIAN_IMAGE_KEY = os.environ.get("OVIAN_IMAGE_KEY")

if not OVIAN_IMAGE_KEY:
    raise ValueError("환경변수 OVIAN_IMAGE_KEY를 불러오지 못했습니다.")

fernet = Fernet(OVIAN_IMAGE_KEY)

# 보관 정책 설정
KEEP_ENCRYPTED = os.getenv("KEEP_ENCRYPTED", "false").lower() == "true"
RETENTION_DAYS = int(os.getenv("RETENTION_DAYS", "7"))
RETENTION_SCAN_MINUTES = int(os.getenv("RETENTION_SCAN_MINUTES", "60"))

# 암호화 스트림 저장 함수 (메모리 기반, 디스크 평문 없음)
def encrypt_stream_to_file(infile, out_path, fernet, chunk_size=4*1024*1024):
    """
    입력 스트림(infile)을 평문으로 모두 메모리에 올리지 않고
    chunk_size씩 읽어 각 청크를 개별적으로 암호화하여 단일 파일에 순차적으로 기록.
    파일 형식: [4바이트토큰길이][토큰][4바이트토큰길이][토큰]...
    """
    with open(out_path, "wb") as out:
        while True:
            chunk = infile.read(chunk_size)
            if not chunk:
                break
            token = fernet.encrypt(chunk)
            out.write(len(token).to_bytes(4, "big"))
            out.write(token)

# 복호화 후 BytesIO로 반환 (메모리 기반 복호화 처리)
def decrypt_file_to_bytesio(enc_path, fernet):
    """
    encrypt_stream_to_file 형식으로 저장된 .enc 파일을 순차 복호화해
    BytesIO에 누적하여 반환(메모리에서 원본 바이트 스트림 사용).
    """
    bio = BytesIO()
    with open(enc_path, "rb") as f:
        while True:
            len_bytes = f.read(4)
            if not len_bytes:
                break
            tlen = int.from_bytes(len_bytes, "big")
            token = f.read(tlen)
            plain = fernet.decrypt(token)
            bio.write(plain)
    bio.seek(0)
    return bio

# 암호화 파일 정리 함수 (RETENTION_DAYS 초과 파일 제거)
def purge_old_encrypted():
    """RETENTION_DAYS보다 오래된 .enc 파일 자동 삭제"""
    now = datetime.now()
    cutoff = now - timedelta(days=RETENTION_DAYS)

    try:
        for name in os.listdir(UPLOAD_FOLDER):
            if not name.lower().endswith(".enc"):
                continue
            path = os.path.join(UPLOAD_FOLDER, name)
            try:
                st = os.stat(path)
                mtime = datetime.fromtimestamp(st.st_mtime)
                if mtime < cutoff:
                    _rm_safely(path)
            except FileNotFoundError:
                continue
    except Exception:
        pass

# 안전 삭제 함수
def _rm_safely(path: str):
    """파일/폴더가 있어도 없어도, 에러 없이 조용히 지움"""
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.isfile(path):
            os.remove(path)
    except Exception:
        pass

# Flask 앱 초기화 및 업로드 폴더 설정
app = Flask(__name__)
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 스케줄러로 주기적 정리 등록
_scheduler = BackgroundScheduler(daemon=True)
_scheduler.add_job(purge_old_encrypted, 'interval', minutes=RETENTION_SCAN_MINUTES)
_scheduler.start()
atexit.register(lambda: _scheduler.shutdown(wait=False))

# 토큰 확인 (모든 요청에 적용됨)
@app.before_request
def check_token():
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"error": "Missing token"}), 401
    token = auth_header.split(" ")[1]
    if token != OVIAN_TOKEN:
        return jsonify({"error": "Invalid token"}), 401

# 메인 추론 엔드포인트
@app.route('/infer', methods=['POST'])
def infer_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    sec = {"encrypt": False, "decrypt_probe": False, "decrypt": False}

    # 파일 이름 보안 처리 및 경로 설정
    orig_name = secure_filename(file.filename or "upload")
    ext = os.path.splitext(orig_name)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".svs", ".bmp", ".webp"]:
        pass
    safe_id = uuid.uuid4().hex
    filename = f"{safe_id}{ext}"
    slide_id = os.path.splitext(filename)[0]

    enc_path = os.path.join(UPLOAD_FOLDER, f"{filename}.enc")
    patch_dir = os.path.join(UPLOAD_FOLDER, f"patches_{slide_id}")
    h5_path = os.path.join(UPLOAD_FOLDER, f"{slide_id}.h5")
    final_h5_path = os.path.join(UPLOAD_FOLDER, f"{slide_id}_final.h5")

    tmp_paths = []
    delete_enc_after = False

    try:
        tmp_paths = [patch_dir, h5_path, final_h5_path]
        delete_enc_after = not KEEP_ENCRYPTED

        # 파일 암호화 저장
        encrypt_stream_to_file(file.stream, enc_path, fernet)
        sec["encrypt"] = True

        # 복호화 후 이미지 유효성 검사
        decrypted_stream = decrypt_file_to_bytesio(enc_path, fernet)
        try:
            img_probe = Image.open(decrypted_stream)
            img_probe.verify()
            sec["decrypt_probe"] = True
        except (UnidentifiedImageError, Exception):
            return jsonify({"error": "Unsupported or corrupted image", "security": sec}), 400

        # 실 처리용 이미지 다시 복호화
        image_stream = decrypt_file_to_bytesio(enc_path, fernet)
        image_input = Image.open(image_stream).convert("RGB")
        sec["decrypt"] = True

        # 추론 파이프라인 실행
        resized_image_input = resize_and_save(image_input)
        create_patches(resized_image_input, patch_dir)
        extract_features(patch_dir, h5_path)
        add_nearest(h5_path)
        merge_h5(h5_path, final_h5_path)

        pred_class, softmax_probs, attention_base64 = infer_and_get_attention(
            final_h5_path, resized_image_input
        )

        return jsonify({
            "pred_class": int(pred_class),
            "softmax_probs": softmax_probs,
            "attention_map_base64": attention_base64,
            "security": sec
        })

    except Exception as e:
        return jsonify({"error": str(e), "security": sec}), 500

    finally:
        # 임시 파일 정리
        for p in tmp_paths:
            _rm_safely(p)
        if delete_enc_after:
            _rm_safely(enc_path)
        try:
            image_stream.close()
        except Exception:
            pass

# 수동 업로드 정리 엔드포인트
@app.route('/clear_uploads', methods=['POST'])
def clear_uploads():
    try:
        shutil.rmtree(UPLOAD_FOLDER)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        return {"status": "success", "message": "Uploads folder cleared."}, 200
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

# Flask 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
