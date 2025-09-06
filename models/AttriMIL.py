# -*- coding: utf-8 -*-
"""
AttriMIL.py

AttriMIL (Attribute-based Multiple Instance Learning) 모델 구현 파일입니다.
병리 이미지 분석에서 MIL 방식으로 아형 분류를 수행하며, 클래스별로 독립적인
Attention 네트워크와 Classifier를 구성하여 가중된 instance-level 정보 집계 방식으로
bag-level 예측을 수행합니다.

📌 주요 구성:
- Attn_Net_Gated: 클래스별 attention 점수를 산출하는 Gated Attention 모듈
- AttriMIL: 클래스별 attention + 분류기를 통해 해석 가능한 MIL 구조 구현

출력:
- logits: 클래스별 logit 점수
- Y_prob: softmax된 클래스별 확률
- Y_hat: 예측된 클래스 (argmax)
- attribute_score: exp(attn) x instance_score로 계산된 주목 점수
- results_dict: (현재 미사용, 확장 가능)

이 모델은 train_attrimil.py에서 학습 루프와 함께 사용됩니다.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

# 클래스별 Gated Attention 점수를 산출하는 서브 네트워크
class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        # Gated Attention 구성: tanh와 sigmoid 각각 통과
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        # Sequential로 묶기
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)  # 최종 attention score 출력

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)  # gated attention
        A = self.attention_c(A)  # shape: (N, n_classes)
        return A, x


# AttriMIL 모델 정의
class AttriMIL(nn.Module):
    def __init__(self, n_classes=2, dim=512):
        super().__init__()
        # 입력 feature 보정용 residual adaptor
        self.adaptor = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim)
        )

        # 클래스별 attention network 리스트
        self.attention_nets = nn.ModuleList([
            Attn_Net_Gated(L=dim, D=dim // 2) for _ in range(n_classes)
        ])

        # 클래스별 classifier 리스트 (instance score 생성)
        self.classifiers = nn.ModuleList([
            nn.Linear(dim, 1) for _ in range(n_classes)
        ])

        self.n_classes = n_classes
        self.bias = nn.Parameter(torch.zeros(n_classes), requires_grad=True)  # 클래스별 bias 학습 가능

    def forward(self, h):
        device = h.device
        h = h + self.adaptor(h)  # residual adaptor 적용

        # 클래스별 raw attention 저장 (shape: [C, N])
        A_raw = torch.empty(self.n_classes, h.size(0), device=device)

        # 클래스별 instance score 저장 (shape: [1, C, N])
        instance_score = torch.empty(1, self.n_classes, h.size(0), device=device)

        # 클래스별 attention과 instance score 계산
        for c in range(self.n_classes):
            A, h_out = self.attention_nets[c](h)  # (N, 1), (N, dim)
            A = A.view(-1)  # (N,) 형태로 변환
            A_raw[c, :] = A  # attention 저장
            instance_score[0, c, :] = self.classifiers[c](h_out).squeeze(-1)  # instance score 저장

        # attribute score = instance score × exp(attention)
        attribute_score = torch.empty(1, self.n_classes, h.size(0), device=device)
        for c in range(self.n_classes):
            attribute_score[0, c, :] = instance_score[0, c, :] * torch.exp(A_raw[c, :])

        # 클래스별 bag-level logit 계산
        logits = torch.empty(1, self.n_classes, device=device)
        for c in range(self.n_classes):
            logits[0, c] = (
                torch.sum(attribute_score[0, c, :], dim=-1) /
                torch.sum(torch.exp(A_raw[c, :]), dim=-1)
            ) + self.bias[c]  # bias 추가

        # 최종 출력값
        Y_hat = torch.topk(logits, 1, dim=1)[1]  # 예측 클래스
        Y_prob = F.softmax(logits, dim=1)       # softmax 확률
        results_dict = {}  # 확장 가능 (현재 미사용)

        return logits, Y_prob, Y_hat, attribute_score, results_dict
