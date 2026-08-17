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
        self.class_pair_margin_overrides = {
            ("water_toilet", "water_shower"): 0.03,
            ("construction", "machine_noise"): 0.03,
        }
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
        self.distill_alpha = 0.7
        self.distill_temperature = 4.0
        # [추가 2026-08-18] teacher fusion 의 others 칸 조립 규칙: "min" | "mean" | "max".
        # 기본 min. 옛 동작(다수결 편향이 있는 평균)으로 되돌리려면 "mean" 으로 두면 된다.
        # 실측 근거는 vild/teacher_fusion.py 의 WeightedTeacherFusion docstring 참조.
        self.fusion_others_rule = "min"
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
