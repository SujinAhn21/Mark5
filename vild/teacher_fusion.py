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

    === score_mode: 담당 칸에 무엇을 넣을 것인가 (신설 2026-08-22) ===

    기존("raw")은 담당 칸에 그 전문가의 target 로짓 **원본**을 그대로 꽂았다. 그런데 그 값은
    확률이 아니라 코사인 유사도를 0.07 로 나눈 절대값이고, 8명이 각자 다른 인코더·다른 텍스트
    앵커로 계산한 것이라 서로 같은 자로 잰 숫자가 아니다. val 2150 세그먼트 실측에서 칸별
    평균이 construction -1.695 부터 machine_noise +0.370 까지 2.07 벌어지고, 표준편차는
    machine_noise 0.532 부터 construction 2.034 까지 3.8 배 차이났다. 그 상태로 한 벡터에
    나란히 놓고 argmax·softmax 를 하면 스케일이 큰 전문가가 구조적으로 유리해진다.

    "margin" 은 담당 칸에 (target - others) 를 넣는다. 같은 전문가 안에서 뺀 값이라 그 전문가의
    절대 스케일이 상쇄되고, "내 것이라고 얼마나 더 강하게 말하는가"만 남는다.

    실측(2026-08-22, val 전량 430클립 2150세그먼트):
        세그먼트 정확도  raw+min 0.7135 | raw+mean 0.6395 | margin 0.8335 | z(margin) 0.8377
        non-others 만    raw 0.7755 | margin(=positive_weight) argmax 0.8940

    이 조립 결과의 argmax 가 pseudo-label 이 되므로(train_mark5.py 의 use_pseudo_label_ce),
    여기서 버리는 11.85 포인트가 그대로 student 가 배우는 정답의 오염률이 된다.
    옛 동작으로 되돌리려면 config 의 fusion_score_mode 를 "raw" 로 두면 되고,
    그때는 others_rule(min/mean/max)이 다시 의미를 갖는다.

    === 분포 정렬 (distribution alignment, 신설 2026-08-22) ===

    문제: pseudo-label 의 클래스 분포가 균등에서 크게 벗어나 있다. unlabeled 300클립
    1500세그먼트 실측에서 construction 19.2% / others 2.4% 였다(균등이면 11.1%). labeled
    2,000개는 클래스당 222개로 균형인데, unlabeled 2,000개가 이 편향을 학습에 주입한다.

    그 결과가 test 혼동행렬에 그대로 나타난다 — construction 이 FP 21건으로 오답을 빨아들이고
    (precision 0.687), others 는 recall 0.660 에 그친다. test 에서 클래스별 가중치를 직접
    최적화하면 acc 0.9000 까지 오르는데, 그때 필요한 가중치가 construction 0.25 / others 2.2 로
    정확히 이 편향의 역수 방향이다.

    ⚠ 이 편향을 **사후 보정으로 고치려는 시도는 실패했다**(2026-08-22). val 430클립으로 클래스
    가중치를 학습시키니 val 은 0.8814 -> 0.9023 으로 올랐는데 test 는 0.8581 -> 0.8581 로
    전혀 움직이지 않았다. 파라미터 10개를 430개 샘플로 맞춰 과적합된 것이다. 그래서 사후가 아니라
    **학습 단계에서** 고친다. 여기서는 unlabeled 세그먼트 10,000개 위에서 분포를 추정하므로
    표본이 23배 크고, 추정 대상과 적용 대상이 같은 데이터다.

    수식:  aligned_p ∝ p_teacher(y|x) x (p_target(y) / p_hat(y))^tau

    p_hat 은 teacher 가 실제로 내놓는 분포이고 EMA(momentum 0.999)로 누적한다. 배치 하나로는
    분포를 못 재고, 학습 중 실시간이라 전체를 미리 알 수도 없기 때문이다. p_target 은 균등
    (labeled 가 균형이므로). tau 는 보정 강도로 0 이면 no-op, 1 이면 완전 균등화다.
    tau 값은 val 430클립에 teacher 를 통과시켜 pseudo-label 품질(정확도·클래스별 recall)이
    어떻게 변하는지 재서 정한다.
    """

    def __init__(self, student_class_map, embedding_dim, device, others_rule="min",
                 score_mode="margin", use_distribution_alignment=False,
                 dist_align_tau=0.5, dist_align_momentum=0.999):
        self.student_class_map = student_class_map
        self.embedding_dim = embedding_dim
        self.device = device
        self.others_rule = others_rule
        self.score_mode = score_mode
        # [신설 2026-08-22] 분포 정렬(distribution alignment)
        self.use_distribution_alignment = use_distribution_alignment
        self.dist_align_tau = dist_align_tau
        self.dist_align_momentum = dist_align_momentum
        num_classes = len(student_class_map)
        # 관측 분포 EMA. 균등에서 출발한다(아직 아무것도 안 봤을 때 보정이 no-op 이 되도록).
        self.observed_dist = torch.full((num_classes,), 1.0 / num_classes, device=device)
        # 목표 분포. labeled 가 클래스당 222개로 균형이므로 균등이다.
        self.target_dist = torch.full((num_classes,), 1.0 / num_classes, device=device)
        self.align_seen_batches = 0

    def _align(self, logits):
        """관측 분포를 EMA 로 누적하고, 그 역수로 확률을 보정한 로짓을 돌려준다.

        aligned_p ∝ p(y|x) x (p_target(y) / p_hat(y))^tau

        tau=0 이면 보정이 없고(현재 동작), tau=1 이면 관측 분포를 목표 분포로 완전히 끌어당긴다.
        확률 공간에서 곱한 뒤 다시 log 를 취해 로짓으로 돌려주므로, 이후 argmax·softmax·
        pseudo-label CE 가 모두 보정된 값 위에서 돈다.
        """
        probs = torch.softmax(logits, dim=1)
        # EMA 갱신은 배치 평균으로. 학습 중 매 배치 호출되므로 momentum 0.999 면
        # 사실상 최근 수천 배치의 분포에 수렴한다.
        with torch.no_grad():
            batch_dist = probs.mean(dim=0)
            m = self.dist_align_momentum
            self.observed_dist = m * self.observed_dist + (1.0 - m) * batch_dist
            self.observed_dist = self.observed_dist / self.observed_dist.sum().clamp_min(1e-12)
            self.align_seen_batches += 1

        if self.dist_align_tau == 0:
            return logits
        ratio = (self.target_dist / self.observed_dist.clamp_min(1e-8)) ** self.dist_align_tau
        aligned = probs * ratio.unsqueeze(0)
        aligned = aligned / aligned.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return torch.log(aligned.clamp_min(1e-12))

    def fuse(self, specialist_outputs):
        batch_size = next(iter(specialist_outputs.values()))["logits"].shape[0]
        num_classes = len(self.student_class_map)
        ensembled_logits = torch.zeros(batch_size, num_classes, device=self.device)
        others_logits_list = []
        margin_list = []
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
            if self.score_mode == "margin":
                # [변경 2026-08-22] 담당 칸에 target 로짓 원본이 아니라 margin(target - others)을 넣는다.
                # 근거는 위 docstring 의 "score_mode" 절.
                ensembled_logits[:, target_idx] = logits[:, 0] - logits[:, 1]
                margin_list.append((logits[:, 0] - logits[:, 1]).unsqueeze(1))
            else:
                ensembled_logits[:, target_idx] = logits[:, 0]
            others_logits_list.append(logits[:, 1:2])
            feature_sum += features * positive_weight
            weight_sum += positive_weight

        others_idx = self.student_class_map["others"]
        if self.score_mode == "margin":
            # others 는 "여덟 명 중 누구의 것도 아니다" 이므로, 가장 강하게 자기 것이라고 주장한
            # 전문가의 margin 을 뒤집은 값이 others 점수가 된다. 아무도 주장하지 않으면(모든 margin 이
            # 음수) -max 가 양수가 되어 others 가 이긴다.
            ensembled_logits[:, others_idx] = -torch.cat(margin_list, dim=1).max(dim=1).values
        else:
            others_stack = torch.cat(others_logits_list, dim=1)      # [B, 전문가 수]
            if self.others_rule == "mean":
                others_logit = others_stack.mean(dim=1)              # 2026-08-18 이전 방식(다수결 편향)
            elif self.others_rule == "max":
                others_logit = others_stack.max(dim=1).values
            elif self.others_rule == "min":                          # 위 docstring 참조
                others_logit = others_stack.min(dim=1).values
            else:
                # [추가 2026-08-18] 오타가 조용히 min 으로 떨어지면 로그에는 "mean" 이라 찍히는데
                # 실제로는 min 이 도는 상황이 생긴다. 그러면 비교 실험 표에 틀린 숫자가 들어간다.
                raise ValueError(
                    f"알 수 없는 others_rule: {self.others_rule!r} — 'min' | 'mean' | 'max' 중 하나여야 합니다."
                )
            ensembled_logits[:, others_idx] = others_logit

        # [신설 2026-08-22] 분포 정렬. 조립이 끝난 9-class 로짓에 건다.
        # 여기서 걸어야 이후의 argmax(pseudo-label)·softmax(KL)·확률 모두가 보정된 값을 본다.
        if self.use_distribution_alignment:
            ensembled_logits = self._align(ensembled_logits)

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
