# vild_config.py
# 뭐가 많아질수록 힘들구나.  
# [12:41]

import math
import torch
from sentence_transformers import SentenceTransformer
import os

SHARED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared_vild"))
# [변경] append -> insert(0): prompt_bank 등 공용 모듈이 vild/ 등 다른 경로보다 항상 우선
# 해석되도록 shared_vild를 sys.path 앞쪽에 둔다. (구버전 vild/prompt_bank.py shadow 방지)
if SHARED_DIR not in os.sys.path:
    os.sys.path.insert(0, SHARED_DIR)

from prompt_bank import get_class_synonyms, get_prompt_templates, get_prompt_texts_for_class

class AudioViLDConfig:
    def __init__(self, mark_version="mark5.0"):
        self.mark_version = mark_version

        # ==============================================================================
        # 1. 클래스 설정 (Mark Version 별)
        # ==============================================================================
        if self.mark_version == "mark4.1":
            self.classes = ["heavy_impact", "others"]
        elif self.mark_version == "mark4.2":
            self.classes = ["dragging", "others"] 
        elif self.mark_version == "mark4.3":
            self.classes = ["construction", "others"] 
        elif self.mark_version == "mark4.4":
            self.classes = ["machine_noise", "others"]
        elif self.mark_version == "mark4.5":
            self.classes = ["media_talking", "others"]
        elif self.mark_version == "mark4.6":
            self.classes = ["water_toilet", "others"]
        elif self.mark_version == "mark4.7":
            self.classes = ["water_shower", "others"]
        elif self.mark_version == "mark4.8":
            self.classes = ["dog_bark", "others"]
        elif self.mark_version == "mark5.0":
            # [변경] 'dummy_label' 제거. mark5 학습/평가 경로는 load_and_segment_with_metadata만
            # 사용하고 unlabeled는 label=-1로 처리되어 dummy_label이 실제로 쓰이지 않음(잔재).
            # student 출력/loss는 아래 9개 클래스로 고정한다.
            self.classes = [
                "heavy_impact", "dragging", "construction", "machine_noise",
                "media_talking", "water_toilet", "water_shower", "dog_bark",
                "others",
            ]
        else:
            raise ValueError(
                f"[Error] Unknown or unsupported mark_version: '{self.mark_version}'.\n"
                f"지원되는 값: ['mark4.1', 'mark4.2', 'mark4.3', 'mark4.4', 'mark4.5', 'mark4.6', 'mark4.7', 'mark4.8', 'mark5.0']"
            )

        # === 기존 속성 유지 ===
        # labeled_classes는 이제 파서가 사용할 모든 클래스를 의미하게 됨
        self.labeled_classes = self.classes
        self.unlabeled_class_identifier = "unlabeled"
        self.num_distinct_labeled_classes = len(self.labeled_classes)


        # ==============================================================================
        # 2. 오디오 및 모델 공통 파라미터
        # ==============================================================================
        # === 오디오 파라미터 ===
        self.sample_rate = 16000
        self.segment_duration = 1.0
        self.segment_samples = int(self.sample_rate * self.segment_duration)
        self.fft_size = 1024
        self.hop_length = 160
        self.n_mels = 64

        # === Segment 단위 처리 ===
        self.segment_length = 101   # Mel spectrogram time frame 수
        self.segment_hop = 50       # Segment 간 stride
        self.max_segments = 5       # Teacher가 사용할 최대 segment 수

        # === 모델 파라미터 ===
        self.embedding_dim = 384
        self.use_background_embedding = True
        # [추가] background embedding 보조 loss 가중치 (use_background_embedding=True일 때만 의미 있음)
        self.background_embedding_weight = 0.1
        # [신설 2026-08-22] 평가에서 others 로짓에 background embedding 을 max 로 덮어쓸지 여부.
        # False 가 기본이다(=덮어쓰지 않는다).
        #
        # 1차 학습(val 430) 결과에서 accuracy 0.1209 인데 ROC AUC 는 0.8871 이 나와 원인을 추적한
        # 결과 이 override 였다. prediction_details CSV 실측: 430클립 중 417개(97.0%)에서 others 가
        # 1등이었고, others 가 아닌 383개 중 316개(82.5%)는 정답이 정확히 2등이었다. others 열을
        # 빼고 argmax 만 다시 계산하면 8클래스 정확도가 0.859 였다. 즉 모델은 제대로 배웠는데
        # others 하나가 모든 클립 위에 일률적으로 얹혀 있었다(Prob_others 평균 0.81, 정답이
        # others 인 클립 0.95 와 거의 차이가 없음).
        #
        # 구조상 그럴 수밖에 없다. 학습(train_mark5.py:533)은 background 를 others 샘플 feature 의
        # 평균 방향으로 당기는 보조 loss 만 걸고 **분류 로짓에는 넣지 않는다.** 그런데 평가만
        # others_logit = max(text_logit, cos(feature, bg)/logit_temperature) 로 덮었다. max 라
        # others 를 올리기만 하고 내리지는 못하는데다, 같은 인코더가 뽑은 feature 라 다른 클래스도
        # bg 와 방향이 겹치고, logit_temperature(0.07)로 나누면서 14.3배로 증폭된다.
        # (학습 종료 시 bg_loss 0.1040 = cos(others feature, bg) 약 0.896.)
        #
        # mark4.8 이 2026-07-12 에 잡은 background embedding override 와 같은 물건이다. 다만
        # 거기서는 Hard loss 가 랜덤 근처에서 안 움직여 "학습부터 꺼야 한다"는 결론이었고,
        # mark5 는 override 가 평가에만 있어 Hard loss 가 1.396->0.495 로 정상 수렴했다.
        # 그래서 재학습 없이 평가만 다시 돌리면 된다.
        #
        # 보조 loss 자체(use_background_embedding)는 끄지 않는다. others feature 를 한 방향으로
        # 모으는 것은 해롭지 않고, 파괴적인 것은 평가 시점의 max 덮어쓰기뿐이다.
        self.use_background_override_at_eval = False
        self.use_text_aligned_student = True
        self.use_feature_kd = True
        self.feature_kd_weight = 0.3
        self.feature_kd_loss_type = "cosine_l1"

        # === DKD (Decoupled Knowledge Distillation, Zhao et al. 2022) ===
        # [신설 2026-08-09] soft loss 계산 방식을 vanilla KD <-> DKD 로 갈아끼우는 스위치.
        # False 면 기존 경로(단일 KL divergence)를 그대로 타므로 지금까지의 학습과 동일하다.
        #
        # 왜 mark5 에서야 넣는가: DKD 는 KD 손실을 TCKD(타겟 vs 비타겟 이진 분포)와
        # NCKD(비타겟 클래스들끼리의 분포)로 분해한다. mark4.x 는 2-class 라 비타겟이 1개뿐이라
        # NCKD 가 항상 0 이고 DKD 가 vanilla 의 상수배(no-op)였다(2026-07-15 확인). 9-class 인
        # mark5 에서 처음으로 의미를 갖는다.
        #
        # ⚠타겟 클래스는 **teacher 앙상블의 argmax** 를 쓴다. 원논문은 supervised 세팅이라
        # ground-truth 라벨로 타겟을 정하는데, mark5 의 KD 는 unlabeled 에 걸리므로 정답이 없다.
        # 이미 teacher 의 판정을 믿고 배우는 구조라 신뢰 구조가 새로 나빠지지는 않지만,
        # **원논문 그대로가 아닌 변형**이므로 논문에 밝혀야 한다.
        #
        # beta=1.0 인 이유: vanilla KD 를 분해하면 KD = TCKD + (1 - p_t)·NCKD 라서 teacher 가
        # 확신할수록(p_t -> 1) NCKD 가 눌린다. beta=1.0 은 "그 눌림만 푼다"는 뜻이고 그 이상
        # 증폭하지 않는다. 원논문 권장값(CIFAR-100 beta=8, ImageNet beta=2)은 클래스가 100~1000개
        # 일 때의 값이라, 9-class 인 여기서는 정보가 아니라 노이즈를 키울 위험이 크다.
        self.use_dkd = False
        self.dkd_alpha = 1.0
        self.dkd_beta = 1.0
        self.visual_view_type = "mel_delta"
        self.segment_selection_mode = "salient_topk"
        self.max_visual_segments = self.max_segments
        self.logit_temperature = 0.07
        self.segment_aggregation_mode = "confidence_saliency"
        self.segment_confidence_power = 2.0
        self.segment_saliency_power = 1.0
        self.others_confidence_threshold = 0.45
        self.others_margin_threshold = 0.05
        # [삭제 2026-07-11] others_entropy_threshold 하드코딩(0.82) 제거, 아래 property로 대체.
        # mark4_refactored에서 발견: 이 값이 클래스 수와 무관한 절대값으로 고정돼 있으면,
        # 클래스 수가 다른 mark_version 사이에서 confidence_threshold와 어긋난 엄격도로 작동함
        # (mark4.x 2-class에서 confusion matrix 완전 붕괴의 근본 원인이었음, 자세한 내용은
        # mark4_refactored/vild/vild_config.py의 동일 property 주석 참조).
        # mark5(9-class)는 기존 하드코딩값(0.82)이 우연히 confidence_threshold(0.45)와 거의
        # 같은 엄격도(top_conf≈0.47 지점)였어서 이 변경으로도 동작이 거의 그대로 유지된다.
        # [비활성 2026-08-21] class-pair 보정을 끈다(빈 dict = 아무 짝도 손대지 않음).
        # 이 보정은 두 클래스의 확률 격차가 threshold(0.03) 미만이면 둘을 평균내 동점으로
        # 만드는데, 동점은 margin=0 이라 바로 뒤 others 보정(margin<0.05 면 others 강제,
        # postprocess_utils.py:38)에 반드시 걸린다. 즉 이 보정이 예측을 실제로 바꾸는 경우
        # (그 짝이 top-2 인 경우)마다 그 결정이 others 로 덮여, 결정 기구로서는 죽어 있었다.
        # 게다가 동점을 argmax 로 풀면 클래스 목록에서 인덱스가 앞선 쪽(construction,
        # water_toilet)이 항상 이기는데, 이건 측정으로 정한 우선순위가 아니라 코드에 적힌 순서다.
        # 끄는 결정적 이유는 스윕 오염이다. 보정이 걸린 클립은 prediction_details CSV 의
        # Raw Margin 이 실제 값(0.03 미만의 어떤 값)이 아니라 인위적인 0 으로 기록되어,
        # 나중에 others margin 임계값을 0~0.03 구간에서 스윕할 때 정작 제일 궁금한 짝 클래스
        # 구간의 숫자를 믿을 수 없게 된다. 빈 dict 면 apply_class_pair_calibration 이
        # 루프를 한 번도 안 돌고 입력을 그대로 반환한다.
        self.class_pair_margin_overrides = {}
        self.enable_temporal_smoothing = True
        self.temporal_smoothing_alpha = 0.65
        self.enable_abstention = False
        self.abstention_confidence_threshold = 0.40
        # [추가] eval 시 supervised_features 로짓과 distill_features 로짓을 섞는 비중.
        # 0=supervised만, 1=distill만, 0.5=평균
        self.distill_branch_eval_weight = 0.5
        self.explain_topk_segments = 3
        self.save_visual_explanations = True
        self.encoder_type = "cnn"
        # [추가 2026-08-17] mark4.x teacher 8개는 2026-07-13 teacher 강화 이후 4블록
        # TeacherAudioEncoder(약 49만 파라미터)로 학습돼 있다. mark5 의 EnsembleTeacher 가
        # 그 .pth 를 읽을 때 같은 구조로 만들어야 하므로 True 로 둔다.
        # False 로 두면 student 용 SimpleAudioEncoder(3블록)를 만들어 state_dict 가 안 맞는다.
        # (student 자신의 인코더는 encoder_type 이 정하고 이 값의 영향을 받지 않는다.)
        self.use_large_teacher_encoder = True

        # === 학습 파라미터 ===
        self.batch_size = 16
        self.num_epochs = 80
        self.learning_rate = 1e-4
        # [추가 2026-08-17] mark4.x 와 학습 절차를 같게 맞추기 위한 값들.
        # mark4.x 는 weight_decay 가 없어 val best 를 이른 에폭에 찍고 곧장 과적합하던 것을
        # 실측하고 1e-4 를 넣었다. early stopping patience 10 도 mark4.x 와 같은 값이다.
        self.weight_decay = 1e-4
        self.early_stopping_patience = 10
        # [추가 2026-08-17] KD 하이퍼파라미터. 값은 컨퍼런스 코드(mark3.2)에서 검증한 0.7 / 4.0 그대로이고,
        # train_mark5.py 에 하드코딩돼 있던 것을 config 로 뺀 것뿐이다.
        # mark4.x 의 0.3 / 2.0 은 같은 샘플에 hard·soft 를 함께 거는 구조에서 나온 값이라 옮겨오지 않는다.
        # [변경 2026-08-22] 0.7 -> 0.3. 아래 use_pseudo_label_ce 로 unlabeled 신호가
        # "teacher 분포 따라하기(KL)" 에서 "teacher 가 고른 정답으로 CE" 로 바뀌면서
        # alpha 의 의미가 달라졌다. 이제 두 항이 모두 CE 이고, unlabeled 쪽 정답은
        # 사람 라벨이 아니라 정확도 0.894 짜리 pseudo-label 이다. 정확도가 낮은 쪽이
        # 사람 라벨보다 2.33 배 무거우면 안 되므로, mark4.x 와 같은 0.3 으로 내린다.
        self.distill_alpha = 0.3
        self.distill_temperature = 4.0
        # [추가 2026-08-18] teacher fusion 의 others 칸 조립 규칙: "min" | "mean" | "max".
        # fusion_score_mode 가 "raw" 일 때만 쓰인다("margin" 이면 others 칸이 -max(margin) 로
        # 정해져 이 값이 무시된다). 실측 근거는 vild/teacher_fusion.py docstring 참조.
        self.fusion_others_rule = "min"

        # === [신설 2026-08-22] teacher 조립 방식과 pseudo-label 학습 ===
        # 배경: 1차 학습 뒤 실측에서 (a) fused teacher 자신의 val 정확도가 0.7488 로
        # student(0.8395)보다 낮고, (b) T=4.0 에서 soft target 의 정규화 엔트로피가 0.9704 로
        # 거의 균등분포라 KL 이 나르는 정보가 0.0113 nats 뿐이고, (c) 그런데도 teacher 의
        # argmax 는 non-others 에서 0.8940 로 꽤 맞는다는 것이 확인됐다.
        # 결론: teacher 에게서 분포를 받지 말고 **정답만** 받는다. 그게 mark3.0.0 의 원래
        # hybrid(soft label 의 argmax 를 hard label 로 쓰는 방식)이기도 하다.

        # 담당 칸에 target 로짓 원본을 넣을지(raw) margin(target-others)을 넣을지(margin).
        # margin 이 전문가별 스케일 차이(평균 2.07, 표준편차 3.8배)를 상쇄한다.
        self.fusion_score_mode = "margin"
        # unlabeled 손실을 KL(분포 따라하기) 대신 pseudo-label CE 로 건다.
        self.use_pseudo_label_ce = True
        # pseudo-label 채택 임계값. 담당 전문가의 "내 클래스" 확률(specialist positive weight)
        # 최대값이 이 값 미만이면 그 샘플은 학습에서 통째로 뺀다. 8명 중 아무도 확신하지 못한
        # 클립에 억지로 정답을 붙이면 그 오답이 그대로 student 에게 주입되기 때문이다.
        # 0.5 는 "적어도 한 명은 자기 것이라고 말한다"는 뜻이다.
        self.pseudo_label_min_confidence = 0.5
        # 채택된 샘플에 per-sample 가중치를 준다. 가중치는 그 최대 확률값 자체.
        # 확신이 0.95 인 클립과 0.52 인 클립을 같은 무게로 배우지 않게 한다.
        self.pseudo_label_confidence_weighting = True
        # distill branch 에도 분류 손실(pseudo CE)을 건다.
        # 지금까지 이 갈래는 feature KD 만 받아 분류를 배운 적이 없는데, eval 에서는
        # distill_branch_eval_weight=0.5 로 최종 확률의 절반을 차지하고 있었다.
        self.distill_branch_classify = True
        # distill branch 마지막 ReLU. 기본은 뺀다(근거는 vild/vild_head.py docstring).
        self.distill_branch_final_relu = False
        self.text_loss_weight = 1.0
        self.image_loss_weight = 1.0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # === 데이터 경로 ===
        self.audio_dir = os.path.join("data_wav")
        self.prompt_bank_path = os.path.join(SHARED_DIR, "resources", "prompt_bank.json")

        # === 내부 캐시 ===
        self._text_emb = None
        self._eval_text_emb = None # 평가용 텍스트 임베딩 캐시 추가
        self._prompt_texts = None
        self.prompt_templates = get_prompt_templates(self.prompt_bank_path)
        self.class_synonyms = get_class_synonyms(self.prompt_bank_path)


    # ==============================================================================
    # 3. 클래스 및 텍스트 관련 메서드
    # ==============================================================================
    def get_class_index(self, class_name: str) -> int:
        """주어진 클래스 이름의 인덱스를 반환. Unlabeled의 경우 -1을 반환."""
        if class_name in self.labeled_classes:
            return self.labeled_classes.index(class_name)
        elif class_name == self.unlabeled_class_identifier:
            return -1
        else:
            raise ValueError(
                f"[Config Error] '{class_name}'는 mark_version '{self.mark_version}'에 등록되지 않은 클래스입니다.\n"
                f"=> 현재 사용 가능한 클래스: {self.labeled_classes}"
            )

    def get_classes_for_parser(self) -> list:
        """
        데이터 파싱 시 유효한 모든 레이블 목록을 반환. (dummy_label 포함)
        학습 데이터셋 구성 시 사용됨.
        """
        return self.labeled_classes

    def get_classes_for_text_prompts(self) -> list:
        """
        [기존 호환성 유지] 텍스트 프롬프트 생성에 사용될 클래스 목록을 반환.
        기본적으로 파서용 클래스 목록과 동일하게 동작함.
        """
        return self.labeled_classes
    
    def get_classes_for_evaluation(self) -> list:
        """
        [추가된 메서드] 모델 성능 평가에 사용될 실제 타겟 클래스 목록을 반환.
        'dummy_label'과 같이 평가에 사용되지 않는 레이블은 제외됨.
        """
        # self.classes 리스트에서 'dummy_label'을 필터링하여 반환
        return [cls for cls in self.classes if cls != 'dummy_label']

    def get_target_label_map(self) -> dict:
        """
        [수정] 평가용 클래스 목록을 기준으로 라벨-인덱스 맵을 생성함.
        모델의 최종 출력과 매칭시킬 때 사용됨.
        """
        # 평가용 클래스 목록을 사용하도록 변경
        return {class_name: i for i, class_name in enumerate(self.get_classes_for_evaluation())}

    @property
    def run_tag(self) -> str:
        """[신설 2026-08-21 / 확장 2026-08-22] 산출물 파일명에 붙일 실행 태그.

        서로 다른 학습 설정이 같은 이름으로 저장돼 앞선 결과를 덮는 것을 막는다.
        붙는 순서는 아래 코드 그대로이고, 조합되면 이어붙는다(예: mark5.0_PL_DKD).

            (없음) : 1차 학습과 완전히 같은 설정 — vanilla KD, raw 조립
            _PL    : use_pseudo_label_ce=True  (unlabeled 를 KL 대신 pseudo-label CE 로)
            _RAW   : fusion_score_mode="raw"   (PL 을 쓰면서 옛 조립으로 되돌린 경우만)
            _DKD   : use_dkd=True

        태그가 갈리는 산출물: student 체크포인트 · 손실곡선 PNG · loss_history CSV ·
        혼동행렬 · ROC · performance_summary · calibration_details ·
        prediction_details · 설명 PNG 폴더.

        일부러 태그를 안 붙이는 것: cache/{mark_version}/ 의 teacher feature 캐시,
        resources/ 의 teacher .pth, dataset_*.csv 인덱스.

        ⚠ eval.py 도 이 태그로 체크포인트를 찾으므로, 학습과 평가의 config 가 같아야 한다.
        """
        tag = self.mark_version
        if getattr(self, "use_pseudo_label_ce", False):
            tag += "_PL"
            if getattr(self, "fusion_score_mode", "margin") == "raw":
                tag += "_RAW"
        if self.use_dkd:
            tag += "_DKD"
        return tag

    @property
    def others_entropy_threshold(self) -> float:
        """
        [추가 2026-07-11] others_confidence_threshold와 "같은 엄격도(같은 top_conf 지점)"가
        되도록 클래스 수 기반으로 자동 역산한다. top_conf=confidence_threshold이고 나머지
        확률이 (클래스수-1)개에 균등분산되는 최악의 경우를 가정해 그때의 정규화 entropy를
        threshold로 삼는다. 클래스 수가 다른 mark_version 사이에서 entropy 조건이
        confidence 조건보다 부당하게 엄격/느슨해지는 것을 방지한다.
        """
        p = self.others_confidence_threshold
        n = self.num_distinct_labeled_classes
        if n <= 1:
            return 1.0
        rest = 1.0 - p
        probs = [p] + [rest / (n - 1)] * (n - 1)
        entropy = -sum(x * math.log(x, 2) for x in probs if x > 1e-12)
        return entropy / math.log2(n)

    @property
    def num_input_channels(self) -> int:
        if self.visual_view_type == "mel":
            return 1
        if self.visual_view_type == "mel_delta":
            return 3
        if self.visual_view_type == "mel_energy":
            return 2
        return 1

    def get_prompt_texts_for_class(self, class_name: str) -> list:
        return get_prompt_texts_for_class(class_name, self.prompt_bank_path)

    def get_class_text_embeddings(self, for_evaluation: bool = False) -> torch.Tensor:
        """
        클래스 이름에 대한 텍스트 임베딩을 생성하여 반환함.
        :param for_evaluation: True일 경우, 평가용 클래스 목록을 사용하여 임베딩을 생성함.
        """
        if for_evaluation:
            # 평가용 임베딩 생성 및 캐싱
            if self._eval_text_emb is None:
                classes = self.get_classes_for_evaluation()
                model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
                aggregated = []
                for class_name in classes:
                    prompts = self.get_prompt_texts_for_class(class_name)
                    emb = model.encode(prompts, convert_to_tensor=True).to(self.device)
                    aggregated.append(emb.mean(dim=0))
                self._eval_text_emb = torch.stack(aggregated, dim=0)
            return self._eval_text_emb
        else:
            # 학습용(기존) 임베딩 생성 및 캐싱
            if self._text_emb is None:
                classes = self.get_classes_for_text_prompts()
                model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
                aggregated = []
                for class_name in classes:
                    prompts = self.get_prompt_texts_for_class(class_name)
                    emb = model.encode(prompts, convert_to_tensor=True).to(self.device)
                    aggregated.append(emb.mean(dim=0))
                self._text_emb = torch.stack(aggregated, dim=0)
            return self._text_emb
