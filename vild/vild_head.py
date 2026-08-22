# vild_head.py

import torch.nn as nn

class ViLDHead(nn.Module):
    """
    Student 모델의 region embedding을 텍스트 임베딩 공간과 동일한 차원으로 투영(projection)하는 헤드

    - CrossEntropyLoss 기반 분류용으로 cosine 정규화는 제거함
    - 구조: Linear -> LayerNorm -> ReLU
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.projection(x)
        x = self.norm(x)
        x = self.activation(x)
        return x  # 정규화 제거됨


class DualBranchStudentHead(nn.Module):
    """[변경 2026-08-22] distill_branch 의 마지막 ReLU 를 뺄 수 있게 했다(기본 뺌).

    왜: distill_branch 의 출력은 feature KD 에서 teacher 의 fused feature 와 코사인을 맞추는데,
    ReLU 로 끝나면 student 쪽 벡터가 전 성분 비음수가 된다. 반면 teacher 인코더는
    Linear(256, embedding_dim) 으로 끝나 부호 제약이 없고, unlabeled 300클립 실측에서 fused
    feature 성분의 45.8% 가 음수였다. 비음수 벡터가 그런 벡터와 낼 수 있는 코사인의 상한은
    ||f+|| / ||f|| = 0.7143 이라, feature KD 손실에 0.5 x (1 - 0.7143) = 0.1429 의 하한이
    구조적으로 깔린다. 1차 학습에서 Feat 이 0.3303 -> 0.2169 까지만 내려간 것이 이것으로 설명된다.
    즉 학습이 부족했던 것이 아니라 목표가 도달 불가능했다.

    supervised_branch 는 그대로 둔다. 그쪽 출력은 텍스트 앵커와의 코사인으로 로짓을 만드는데,
    앵커 자체가 비음수 성분이 우세해 상한이 0.69~0.72 이고 학습된 방향도 거기에 맞춰져 있다.
    두 갈래를 한꺼번에 바꾸면 1차 결과와 비교할 수 있는 축이 하나도 안 남는다.
    """

    def __init__(self, embedding_dim, distill_final_relu=False):
        super().__init__()
        self.supervised_branch = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
        )
        distill_layers = [
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        ]
        if distill_final_relu:
            distill_layers.append(nn.ReLU(inplace=True))
        self.distill_branch = nn.Sequential(*distill_layers)

    def forward(self, features):
        return self.supervised_branch(features), self.distill_branch(features)
