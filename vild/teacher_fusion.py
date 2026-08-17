from dataclasses import dataclass

import torch


@dataclass
class TeacherFusionOutput:
    logits: torch.Tensor
    features: torch.Tensor
    specialist_weights: dict
    disagreement: torch.Tensor
    uncertainty: torch.Tensor


class WeightedTeacherFusion:
    """8개 이진 전문가(mark4.1~4.8)의 출력을 9-class 로짓 한 장으로 조립한다.

    [변경 2026-08-18] others 칸을 8개 평균에서 최솟값으로 바꿨다.

    왜: 각 클래스 칸은 담당 전문가의 "내 클래스" 점수 하나를 받는데, others 칸만
    8명 전원의 "내 거 아님" 점수를 평균해서 받고 있었다. 어떤 소리가 들어오든 8명 중
    7명은 "내 거 아님"이 정답이라, others 칸에는 늘 중간 이상의 점수가 깔린다.
    그래서 담당 전문가의 확신이 어중간하면 정답을 알아봤는데도 others 에 밀렸다.

    실측(2026-08-18, val 전량 430클립 2150세그먼트를 teacher 8개에 통과):
        세그먼트 기준
          others=mean  정확도 0.6395 | others recall 0.5149(121/235) precision 0.2840 FP 305
          others=min   정확도 0.7135 | others recall 0.2298( 54/235) precision 0.8571 FP   9
          others=max   정확도 0.2367 | others recall 0.9745(229/235) precision 0.1267 FP 1579
        클립 기준
          others=mean  정확도 0.6674 | others recall 0.5319(25/47) precision 0.3086 FP 56
          others=min   정확도 0.7488 | others recall 0.1702( 8/47) precision 0.8889 FP  1

    ⚠트레이드오프를 함께 적어둔다. min 은 전체 정확도를 올리고 others 오탐을 거의 없애지만,
    others 를 제대로 others 라고 부르는 비율(recall)이 0.515 에서 0.230 으로 떨어진다.
    이 로짓이 unlabeled 2,000개의 soft label 을 만들므로, student 는 KD 에서 "배경음은 others 가
    아니다" 쪽으로 배울 재료를 받는다. eval.py 가 대표 지표로 쓰는 others FPR(=1-Recall(others))이
    바로 그 반대 방향의 값이라, 1차 학습 뒤 이 지표를 확인하고 필요하면 config 의
    fusion_others_rule 을 "mean" 으로 되돌려 재학습해 비교할 것.
    (labeled 2,000개의 hard CE·LearnableBackgroundEmbedding·apply_others_calibration 이
     others 를 따로 받쳐주므로 student 성능이 곧바로 무너지지는 않는다.)
      담당 전문가가 p>0.5 로 확신했는데 others 로 뒤집힌 세그먼트 37/400(9.2%)
      — dog_bark 21 / heavy_impact 8 / machine_noise 6 / dragging 2.
      점수 분포: 정답 칸 평균 1.422 대 others(mean) 칸 평균 0.688,
      정답 칸이 others 칸보다 낮은 경우가 16.8%.

    min 이 정의상으로도 맞다. others 는 "여덟 전문가 중 누구의 것도 아니다"라는 뜻이므로,
    가장 약하게 반대한 전문가의 점수가 그 판단의 상한이다. 한 명이라도 강하게 "내 것"이라고
    하면 others 점수가 그만큼 내려가야 한다. 평균은 나머지 7명의 무관한 찬성표를 섞는다.

    이 값은 student 가 unlabeled 데이터에서 배우는 정답(soft label)을 만든다.
    """

    def __init__(self, student_class_map, embedding_dim, device, others_rule="min"):
        self.student_class_map = student_class_map
        self.embedding_dim = embedding_dim
        self.device = device
        self.others_rule = others_rule

    def fuse(self, specialist_outputs):
        batch_size = next(iter(specialist_outputs.values()))["logits"].shape[0]
        num_classes = len(self.student_class_map)
        ensembled_logits = torch.zeros(batch_size, num_classes, device=self.device)
        others_logits_list = []
        feature_sum = torch.zeros(batch_size, self.embedding_dim, device=self.device)
        weight_sum = torch.zeros(batch_size, 1, device=self.device)
        specialist_weights = {}

        for class_name, output in specialist_outputs.items():
            logits = output["logits"]
            features = output["features"]
            probs = torch.softmax(logits, dim=1)
            positive_weight = probs[:, 0:1]
            specialist_weights[class_name] = positive_weight.detach().cpu()

            target_idx = self.student_class_map[class_name]
            ensembled_logits[:, target_idx] = logits[:, 0]
            others_logits_list.append(logits[:, 1:2])
            feature_sum += features * positive_weight
            weight_sum += positive_weight

        others_idx = self.student_class_map["others"]
        others_stack = torch.cat(others_logits_list, dim=1)          # [B, 전문가 수]
        if self.others_rule == "mean":
            others_logit = others_stack.mean(dim=1)                  # 2026-08-18 이전 방식(다수결 편향)
        elif self.others_rule == "max":
            others_logit = others_stack.max(dim=1).values
        elif self.others_rule == "min":                              # 기본값, 위 docstring 참조
            others_logit = others_stack.min(dim=1).values
        else:
            # [추가 2026-08-18] 오타가 조용히 min 으로 떨어지면 로그에는 "mean" 이라 찍히는데
            # 실제로는 min 이 도는 상황이 생긴다. 그러면 비교 실험 표에 틀린 숫자가 들어간다.
            raise ValueError(
                f"알 수 없는 others_rule: {self.others_rule!r} — 'min' | 'mean' | 'max' 중 하나여야 합니다."
            )
        ensembled_logits[:, others_idx] = others_logit
        fused_features = feature_sum / weight_sum.clamp_min(1e-6)
        stacked_weights = torch.stack(
            [specialist_weights[name].to(self.device).squeeze(1) for name in specialist_outputs.keys()],
            dim=1,
        )
        top2 = torch.topk(stacked_weights, k=min(2, stacked_weights.shape[1]), dim=1).values
        if top2.shape[1] == 1:
            disagreement = torch.zeros(batch_size, device=self.device)
        else:
            disagreement = top2[:, 0] - top2[:, 1]
        uncertainty = 1.0 - stacked_weights.max(dim=1).values
        return TeacherFusionOutput(
            logits=ensembled_logits,
            features=fused_features,
            specialist_weights=specialist_weights,
            disagreement=disagreement,
            uncertainty=uncertainty,
        )
