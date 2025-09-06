# -*- coding: utf-8 -*-
"""
constraints.py
- Attri-MIL 학습/추론 과정에서 사용하는 제약(regularization) 손실 함수 모음.
- 주요 기능:
  1) spatial_constraint: 인접 패치 간 attention 점수의 급격한 변화를 억제(공간 스무딩)
  2) rank_constraint   : 클래스별 상/하위 attention 인스턴스를 비교해 순위 일관성 유도

주의:
- 본 모듈에서는 'device' 변수를 명시적으로 정의하지 않습니다.
  rank_constraint 내부에서 torch.tensor(...).to(device) 를 호출하므로,
  이 함수를 사용하는 코드(상위 모듈)에서 전역 device를 미리 정의해야 합니다.
  (예: device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
"""

import numpy as np                  # 수치 연산
import torch                       # 텐서/자동미분
from utils import *                # 프로젝트 공용 유틸 (와일드카드 임포트 주의)
import os                          # 경로 처리 (필요 시 사용)
import queue                       # FIFO 큐(클래스별 pos/neg 특징 버퍼 관리)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc


def spatial_constraint(A, n_classes, nearest, ks=3):
    """
    [목적]
        클래스별 attention score가 공간적으로 인접한 패치들 사이에서
        급격히 변하지 않도록 L2 스무딩 제약을 부여합니다.

    Args:
        A (torch.Tensor):
            - shape: (1, n_classes, N) 또는 (n_classes, N)
            - 각 클래스 c에 대한 N개 패치의 attention 점수.
        n_classes (int):
            - 클래스 개수 C.
        nearest (np.ndarray | torch.Tensor):
            - shape: (N, ks). 각 패치 i의 k-최근접 이웃 인덱스 목록.
            - 예: nearest[i][j] = 패치 i의 j번째 최근접 이웃의 인덱스(정수).
        ks (int, 기본=3):
            - 각 패치마다 고려할 최근접 이웃 수(k).

    Returns:
        torch.Tensor 또는 float:
            - 스칼라 손실값. 평균화된 L2 스무딩 손실.

    수식(개념):
        loss = (1 / (C * N * ks)) * Σ_c Σ_i Σ_{j<ks} (A_c[i] - A_c[NN(i, j)])^2

    주의/가이드:
        - 입력 A가 (1, C, N)인 경우 squeeze(0)으로 (C, N)으로 변환합니다.
        - nearest 인덱스가 유효 범위를 벗어나면 해당 항은 건너뜁니다.
        - 계산 복잡도: O(C * N * ks)  (C: 클래스 수, N: 패치 수)
        - 메모리: A는 (C, N)로 처리되며, nearest는 (N, ks) 정수 배열이어야 합니다.
    """
    # 모델 출력 형태(1, C, N) → (C, N)로 정규화
    A = A.squeeze(0)  # -> (n_classes, N)
    if A.dim() != 2:
        raise ValueError(f"Expected attention tensor shape (n_classes, N), got {A.shape}")

    N = A.shape[1]
    loss = 0.0

    # 클래스별로 순회하며 인접 패치와의 L2 차이를 누적
    for c in range(n_classes):
        score = A[c]  # shape: (N,)
        for i in range(N):
            # 패치 i의 최근접 이웃들에 대해 차이 제곱을 합산
            for j in range(min(ks, nearest.shape[1])):
                neighbor_idx = nearest[i][j]
                if neighbor_idx >= N:
                    # 잘못된 인덱스가 들어온 경우 방어적으로 skip
                    continue
                loss += (score[i] - score[neighbor_idx]) ** 2

    # 평균화하여 스칼라 손실로 반환
    return loss / (n_classes * N * ks)


def rank_constraint(data, label, model, A, n_classes, label_positive_list, label_negative_list):
    """
    [목적]
        클래스별로 attention 상위 인스턴스(양성 후보)와 하위 인스턴스(음성 후보)를
        간이 버퍼(queue)에 유지하며, 모델이 '정답 클래스의 상위 인스턴스'에는
        높은 주의를, '비정답 클래스의 상위 인스턴스'에는 낮은 주의를 주도록
        순위 기반 제약을 가합니다.

    Args:
        data (torch.Tensor):
            - shape: (N, D) 가정. N개 인스턴스(패치) 임베딩/특징.
        label (int):
            - 현재 배치(또는 샘플)의 정답 클래스 인덱스.
        model (torch.nn.Module):
            - 호출 형태: logits, _, _, Ah, _ = model(h)
            - 여기서 Ah는 attention 맵(또는 attribute score)로 가정.
        A (torch.Tensor):
            - shape: (1, C, N) 또는 (C, N) 추정. 각 클래스-패치 attention 값.
            - topk로 클래스별 상위 인스턴스를 선택할 때 사용.
        n_classes (int):
            - 클래스 개수 C.
        label_positive_list (List[queue.Queue]):
            - 길이 C의 리스트. 각 원소는 Queue(maxsize=K) 형태로,
              "그 클래스의 양성(top attention) 인스턴스 특징"을 누적 보관.
        label_negative_list (List[queue.Queue]):
            - 길이 C의 리스트. 각 원소는 Queue(maxsize=K) 형태로,
              "그 클래스의 음성(top attention) 인스턴스 특징"을 누적 보관.

    Returns:
        Tuple[torch.Tensor, List[queue.Queue], List[queue.Queue]]:
            - loss_rank: 스칼라 텐서(순위 제약 손실)
            - label_positive_list: 업데이트된 양성 큐 리스트
            - label_negative_list: 업데이트된 음성 큐 리스트

    동작 개념(요약):
        - 각 클래스 c에 대해 A[0, c]의 top-1 인덱스를 구해 해당 특징 h를 추출.
        - 현재 샘플의 정답 라벨과 c의 일치/불일치에 따라
          - 정답인 경우: pos 큐 갱신, neg 큐에서 하나 꺼내 비교(있을 때)
          - 정답이 아닌 경우: neg 큐 갱신, pos 큐에서 하나 꺼내 비교(있을 때)
        - 비교 시 모델에 h를 통과시켜 얻은 Ah와 현재 top value 간의 차이를
          clamp(하한 0)하여 순위 일관성을 유도.
        - 모든 클래스에 대해 평균.

    주의/가이드:
        - 본 함수는 전역 변수 'device'가 정의되어 있어야 torch.tensor(...).to(device)가 안전합니다.
          (전역으로 device를 정의하거나, 이 함수 내부 코드를 device=...로 직접 지정하도록 개선해도 됩니다.)
        - topk에서 k=1만 사용(가장 높은 attention 인스턴스).
        - 큐는 maxsize를 둬 메모리 사용량을 제한하는 것을 권장합니다.
        - 모델의 반환 형태(logits, _, _, Ah, _)는 프로젝트 구현에 따라 달라질 수 있으므로
          일치하지 않으면 수정이 필요합니다.
    """
    # 순위 제약 손실 누적용 스칼라 텐서
    loss_rank = torch.tensor(0.0).to(device)

    # 각 클래스에 대해 처리
    for c in range(n_classes):
        if label == c:
            # 정답 클래스인 경우: 해당 클래스 attention 상위(=양성 후보)
            value, indice = torch.topk(A[0, c], k=1)
            h = data[indice.item(): indice.item() + 1]  # top feature (shape: (1, D))

            # pos 큐가 가득 찼다면 가장 오래된 항목 제거 후 삽입
            if label_positive_list[c].full():
                _ = label_positive_list[c].get()
            label_positive_list[c].put(h)

            # 비교할 neg가 없으면 0을 더하고 넘어감
            if label_negative_list[c].empty():
                loss_rank = loss_rank + torch.tensor(0.0).to(device)
            else:
                # neg 큐에서 하나 꺼내와 모델 통과 → 해당 클래스 attention(Ah) 비교
                h = label_negative_list[c].get()
                label_negative_list[c].put(h)  # 순환 유지
                _, _, _, Ah, _ = model(h.detach())

                # c != 0일 때와 0일 때를 구분해 clamp로 하한 0 보장
                if c != 0:
                    loss_rank = (
                        loss_rank
                        + torch.clamp(torch.mean(Ah[0, c] - value), min=0.0)
                        + torch.clamp(torch.mean(-value), min=0.0)
                        + torch.clamp(torch.mean(Ah[0, c]), min=0.0)
                    )
                else:
                    loss_rank = (
                        loss_rank
                        + torch.clamp(torch.mean(-value), min=0.0)
                        + torch.clamp(torch.mean(Ah[0, c]), min=0.0)
                    )
        else:
            # 정답 클래스가 아닌 경우: 해당 클래스의 상위는 '음성 후보'로 취급
            value, indice = torch.topk(A[0, c], k=1)
            h = data[indice.item(): indice.item() + 1]  # top feature (shape: (1, D))

            if label_negative_list[c].full():
                _ = label_negative_list[c].get()
            label_negative_list[c].put(h)

            # 비교할 pos가 없으면 0을 더하고 넘어감
            if label_positive_list[c].empty():
                loss_rank = loss_rank + torch.tensor(0.0).to(device)
            else:
                # pos 큐에서 하나 꺼내와 모델 통과 → 해당 클래스 attention(Ah) 비교
                h = label_positive_list[c].get()
                label_positive_list[c].put(h)
                _, _, _, Ah, _ = model(h.detach())

                if c != 0:
                    loss_rank = (
                        loss_rank
                        + torch.clamp(torch.mean(value - Ah[0, c]), min=0.0)
                        + torch.clamp(torch.mean(value), min=0.0)
                    )
                else:
                    loss_rank = (
                        loss_rank
                        + torch.clamp(torch.mean(value), min=0.0)
                        + torch.clamp(torch.mean(-Ah[0, c]), min=0.0)
                    )

    # 클래스 평균
    loss_rank = loss_rank / n_classes
    return loss_rank, label_positive_list, label_negative_list