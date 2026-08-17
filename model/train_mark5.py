import os
import sys
import csv
import argparse
import hashlib

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader, Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
VILD_DIR = os.path.join(PROJECT_ROOT, "vild")
RESOURCES_DIR = os.path.join(PROJECT_ROOT, "resources")
for p in (PROJECT_ROOT, UTILS_DIR, VILD_DIR):
    if p not in sys.path:
        sys.path.append(p)

from feature_cache import get_feature_cache_path, get_metadata_cache_path
from vild_config import AudioViLDConfig
from vild_model import (
    LearnableBackgroundEmbedding,
    ViLDTextHead,
    build_audio_encoder,
    build_teacher_encoder,
)
from vild_head import DualBranchStudentHead
from vild_parser_teacher import AudioParser
from teacher_fusion import WeightedTeacherFusion
from seed_utils import set_seed
SHARED_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "shared_vild"))
if SHARED_DIR not in sys.path:
    sys.path.append(SHARED_DIR)
from checkpoint_utils import apply_config_metadata, load_checkpoint, resolve_state_dict, save_checkpoint


def _resolve_csv_path(filename, required=True):
    """CSV 를 찾아 경로를 돌려준다.

    [변경 2026-08-09] required=False 를 추가했다. dataset_val.csv 는 있으면 쓰고 없으면
    옛 방식으로 폴백해야 해서, 없을 때 예외 대신 None 이 필요하다.
    """
    candidates = [
        os.path.join(PROJECT_ROOT, filename),
        os.path.join(BASE_DIR, filename),
        os.path.join(PROJECT_ROOT, "preprocessing", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    if not required:
        return None
    raise FileNotFoundError(f"[ERROR] CSV 파일을 찾을 수 없습니다: {candidates}")


def _resolve_resource_path(filename):
    candidates = [
        os.path.join(RESOURCES_DIR, filename),
        os.path.join(PROJECT_ROOT, filename),
        os.path.join(BASE_DIR, filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"[ERROR] 리소스 파일을 찾을 수 없습니다: {candidates}")


def _align_teacher_arch(config, state_dict, class_name, ckpt_name):
    """[추가 2026-08-17] teacher .pth 안의 키 이름을 보고 어느 인코더 구조인지 판정해
    config.use_large_teacher_encoder 를 그것에 맞춘다.

    이렇게 하는 이유: 구조가 안 맞으면 load_state_dict 가 Missing/Unexpected 키 수십 줄만
    토하고 죽어서, 무엇이 문제인지 알아내는 데 시간이 걸린다(2026-08-17 실제로 그랬다).
    설정 플래그보다 파일에 든 실물이 사실이므로 파일 기준으로 맞춘다.

      conv_block1.*      -> SimpleAudioEncoder (student 인코더, 3블록·약 14.3만 파라미터.
                            2026-08-17 이전에는 2블록 4.5만이었다)
      model.3.0.weight   -> TeacherAudioEncoder (4블록, 약 49만 파라미터)
    """
    keys = set(state_dict.keys())
    if any(k.startswith("conv_block1.") for k in keys):
        detected_large = False
    elif "model.3.0.weight" in keys:
        detected_large = True
    else:
        raise RuntimeError(
            f"[ERROR] teacher 체크포인트의 인코더 구조를 판정할 수 없습니다: {ckpt_name} "
            f"(class={class_name}). 아는 구조는 SimpleAudioEncoder(conv_block1.*)와 "
            f"TeacherAudioEncoder(model.3.0.weight) 두 가지입니다. "
            f"실제 키 예시: {sorted(keys)[:5]}"
        )

    if getattr(config, "use_large_teacher_encoder", False) != detected_large:
        arch = "TeacherAudioEncoder(4블록)" if detected_large else "SimpleAudioEncoder(student 인코더)"
        print(
            f"[INFO] teacher 인코더 구조를 체크포인트에 맞춰 조정합니다 — "
            f"{class_name}: {arch} ({ckpt_name})"
        )
        config.use_large_teacher_encoder = detected_large
    return detected_large


class EnsembleTeacher:
    def __init__(self, specialist_config, device, others_rule="min"):
        self.device = device
        self.specialists = {}
        self.embedding_dim = None
        self.fusion = None  # WeightedTeacherFusion은 배치마다 재생성하지 않고 1회 생성 후 재사용
        self.others_rule = others_rule  # [추가 2026-08-18] fusion 의 others 칸 조립 규칙
        for class_name, paths in specialist_config.items():
            encoder_ckpt = load_checkpoint(_resolve_resource_path(paths["encoder_path"]), map_location=device)
            config = AudioViLDConfig(mark_version=paths["mark_version"])
            apply_config_metadata(config, encoder_ckpt)
            # [수정 2026-08-17] student 용 build_audio_encoder 가 아니라 teacher 용
            # build_teacher_encoder 를 쓴다. mark4.x teacher 8개는 2026-07-13 teacher 강화 이후
            # 4블록 TeacherAudioEncoder 로 학습됐는데, 여기서 student 용 SimpleAudioEncoder 를 만들어
            # 넣으려다 state_dict 가 안 맞아 죽었다. 작년 11월 더미 teacher(186KB)가 2블록이었기
            # 때문에 이 코드가 그때 기준으로 남아 있었던 것이다.
            encoder_state = resolve_state_dict(encoder_ckpt, "model_state_dict", "encoder_state_dict", "model")
            _align_teacher_arch(config, encoder_state, class_name, paths["encoder_path"])
            encoder = build_teacher_encoder(config).to(self.device)
            encoder.load_state_dict(encoder_state)
            encoder.eval()

            classifier_ckpt = load_checkpoint(_resolve_resource_path(paths["classifier_path"]), map_location=device)
            apply_config_metadata(config, classifier_ckpt)
            classifier = ViLDTextHead(config).to(self.device)
            classifier.load_state_dict(resolve_state_dict(classifier_ckpt, "classifier_state_dict", "head_state_dict", "head"))
            classifier.eval()
            if "text_embeddings" in classifier_ckpt:
                text_emb = classifier_ckpt["text_embeddings"].to(device)
            else:
                text_emb = config.get_class_text_embeddings().to(device)
            self.specialists[class_name] = {
                "encoder": encoder,
                "classifier": classifier,
                "text_emb": text_emb,
            }
            if self.embedding_dim is None:
                self.embedding_dim = config.embedding_dim

    @torch.no_grad()
    def __call__(self, unlabeled_audio_batch, student_class_map):
        fusion_inputs = {}
        for class_name, models in self.specialists.items():
            region_emb = models["encoder"](unlabeled_audio_batch)
            logits = models["classifier"](region_emb, models["text_emb"])
            fusion_inputs[class_name] = {
                "logits": logits,
                "features": region_emb,
            }
        if self.fusion is None:
            # [변경 2026-08-18] others 조립 규칙을 config 에서 받는다. 기본은 min
            # (평균은 무관한 전문가 7명의 "내 거 아님" 표까지 섞여 others 가 유리해진다.
            #  근거 실측은 vild/teacher_fusion.py 의 docstring 참조).
            self.fusion = WeightedTeacherFusion(
                student_class_map, self.embedding_dim, self.device,
                others_rule=self.others_rule,
            )
        return self.fusion.fuse(fusion_inputs)


class SemiSupervisedDataset(Dataset):
    def __init__(self, file_path_list, parser, config, is_labeled=True):
        self.samples = []
        self.parser = parser
        self.config = config
        self.is_labeled = is_labeled
        valid_labels = set(config.get_classes_for_evaluation()) if is_labeled else None

        skipped_label_counter = {}
        for item in file_path_list:
            path = item["path"]
            label = item.get("label")
            if is_labeled and label not in valid_labels:
                # [추가] 9-class 밖 라벨은 조용히 넘기지 않고 집계
                skipped_label_counter[label] = skipped_label_counter.get(label, 0) + 1
                continue
            try:
                segment_records = parser.load_and_segment_with_metadata(path)
                for record in segment_records:
                    seg = record["tensor"]
                    if seg is not None:
                        metadata = {
                            "path": path,
                            "label": label if is_labeled else "unlabeled",
                            "segment_index": record["segment_index"],
                            "start_frame": record["start_frame"],
                            "end_frame": record["end_frame"],
                            "saliency": record["saliency"],
                        }
                        self.samples.append((seg, label if is_labeled else -1, metadata))
            except Exception as e:
                print(f"[ERROR] Failed to parse {path}: {e}")

        # [추가] 조용한 탈락 방지: 9-class 밖 라벨이 있으면 개수와 함께 보고하고,
        # labeled인데 유효 샘플이 0개면 즉시 중단(옛 6-class 데이터로 돌리는 사고 방지).
        if skipped_label_counter:
            print(
                f"[WARN] 9-class 밖 라벨로 건너뛴 labeled 파일: {skipped_label_counter} "
                f"(허용 클래스: {sorted(valid_labels)})"
            )
        if is_labeled and len(self.samples) == 0:
            raise ValueError(
                f"[ERROR] 유효한 labeled 세그먼트가 0개입니다. 입력 {len(file_path_list)}개가 모두 탈락했습니다. "
                f"건너뛴 라벨: {skipped_label_counter}. "
                "데이터 폴더/CSV 라벨이 9-class와 일치하는지 확인하세요."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class SampleListDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def custom_collate(batch):
    mels, labels, metadata = zip(*batch)
    return torch.stack(mels, dim=0), list(labels), list(metadata)


def dkd_loss(student_logits, teacher_logits, target_idx, T, dkd_alpha, dkd_beta):
    """DKD(Decoupled Knowledge Distillation, Zhao et al. 2022) — KD 를 TCKD + NCKD 로 분해.

        TCKD  타겟 클래스 vs 나머지 전체 의 **이진** 분포에 대한 KL
        NCKD  타겟을 뺀 나머지 클래스들끼리 재정규화한 분포에 대한 KL
        DKD   = dkd_alpha · TCKD + dkd_beta · NCKD

    vanilla KD 를 분해하면 KD = TCKD + (1 - p_t)·NCKD 가 되어, teacher 가 확신할수록
    NCKD(= dark knowledge 의 본체)가 눌린다. DKD 는 그 계수를 떼어내 독립적인 beta 로 세운다.

    target_idx 는 호출부에서 **teacher 앙상블의 argmax** 로 넘긴다(unlabeled 라 정답이 없음).
    """
    n_cls = student_logits.size(1)
    mask = F.one_hot(target_idx, num_classes=n_cls).bool()

    ps = F.softmax(student_logits / T, dim=1)
    pt = F.softmax(teacher_logits / T, dim=1)

    # --- TCKD: [타겟, 비타겟합] 2차원 분포 ---
    ps_t = ps[mask].unsqueeze(1)
    pt_t = pt[mask].unsqueeze(1)
    bs = torch.cat([ps_t, 1.0 - ps_t], dim=1).clamp_min(1e-8)
    bt = torch.cat([pt_t, 1.0 - pt_t], dim=1).clamp_min(1e-8)
    tckd = (bt * (bt.log() - bs.log())).sum(1).mean() * (T ** 2)

    # --- NCKD: 타겟을 제외하고 재정규화한 분포 ---
    # 타겟 위치를 -inf 로 눌러 softmax 에서 빠지게 한다. 그 자리는 log 가 -inf 가 되므로
    # 합산 전에 0 으로 덮어 nan(0 * -inf)을 막는다.
    s_nt = student_logits.masked_fill(mask, float("-inf"))
    t_nt = teacher_logits.masked_fill(mask, float("-inf"))
    log_ps_nt = F.log_softmax(s_nt / T, dim=1)
    log_pt_nt = F.log_softmax(t_nt / T, dim=1)
    pt_nt = log_pt_nt.exp()
    kl = (pt_nt * (log_pt_nt - log_ps_nt)).masked_fill(mask, 0.0)
    nckd = kl.sum(1).mean() * (T ** 2)

    return dkd_alpha * tckd + dkd_beta * nckd, tckd, nckd


class DistillationLoss(nn.Module):
    """[변경 2026-08-09] use_dkd 스위치 추가.

    False 면 기존 경로(단일 KL divergence)를 그대로 타므로 이전 학습과 동일하게 동작한다.
    True 면 soft 항만 DKD 로 바뀌고, hard/feature KD 와 alpha(hard vs soft 균형)는 건드리지 않는다.
    alpha 와 dkd_alpha/dkd_beta 는 **층이 다르다** — alpha 는 hard vs KD 의 균형이고
    dkd_alpha/beta 는 KD 내부에서 TCKD vs NCKD 의 균형이다.
    """

    def __init__(self, temperature, alpha, feature_kd_weight=0.0,
                 use_dkd=False, dkd_alpha=1.0, dkd_beta=1.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.feature_kd_weight = feature_kd_weight
        self.use_dkd = use_dkd
        self.dkd_alpha = dkd_alpha
        self.dkd_beta = dkd_beta
        self.hard = nn.CrossEntropyLoss()
        self.soft = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, hard_targets=None, teacher_logits=None, student_features=None, teacher_features=None):
        total = torch.tensor(0.0, device=student_logits.device)
        hard_loss = torch.tensor(0.0, device=student_logits.device)
        soft_loss = torch.tensor(0.0, device=student_logits.device)
        feat_loss = torch.tensor(0.0, device=student_logits.device)

        if hard_targets is not None and len(hard_targets) > 0:
            hard_loss = self.hard(student_logits, hard_targets)
            total = total + (1 - self.alpha) * hard_loss

        if teacher_logits is not None and len(teacher_logits) > 0:
            if self.use_dkd:
                # 타겟 축은 teacher 앙상블의 argmax. unlabeled 라 정답 라벨이 없다.
                target_idx = teacher_logits.argmax(dim=1)
                soft_loss, _, _ = dkd_loss(
                    student_logits, teacher_logits, target_idx,
                    self.temperature, self.dkd_alpha, self.dkd_beta)
            else:
                soft_loss = self.soft(
                    F.log_softmax(student_logits / self.temperature, dim=1),
                    F.softmax(teacher_logits / self.temperature, dim=1),
                ) * (self.temperature ** 2)
            total = total + self.alpha * soft_loss

        if (
            self.feature_kd_weight > 0
            and student_features is not None
            and teacher_features is not None
            and len(teacher_features) > 0
        ):
            sf = F.normalize(student_features, dim=1)
            tf = F.normalize(teacher_features, dim=1)
            feat_loss = 0.5 * F.l1_loss(sf, tf) + 0.5 * (1 - F.cosine_similarity(sf, tf, dim=1).mean())
            total = total + self.feature_kd_weight * feat_loss

        return total, hard_loss, soft_loss, feat_loss


def _stable_file_split(path, val_ratio=0.1):
    """[폴백 전용 / 2026-08-09] dataset_val.csv 가 없을 때만 쓰는 옛 방식.

    파일명 해시로 10% 를 val 로 뗀다. 재현은 되지만 **수집 단계에서 세션 단위로 갈라 둔
    split 을 무시한다** — 같은 세션(같은 녹음 시간대)의 클립이 train 과 val 에 함께 들어가
    val 성능이 낙관 편향된다. mark4.x 는 data/{split} 폴더로 나뉘어 있어 이 문제가 없었고,
    mark5 도 2026-08-09 부터 data_source_val -> data_val -> dataset_val.csv 를 쓴다.
    """
    key = os.path.basename(path).encode("utf-8")
    digest = hashlib.md5(key).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"


def train_mark5(seed_value=42, mark_version="mark5.0"):
    set_seed(seed_value)
    config = AudioViLDConfig(mark_version=mark_version)
    device = config.device
    parser = AudioParser(config, segment_mode=True)

    labeled_files = list(csv.DictReader(open(_resolve_csv_path("dataset_labeled.csv"), newline="", encoding="utf-8")))

    # [변경 2026-08-09] dataset_unlabeled.csv 가 없어도 터지지 않게 하되, **크게 경고한다.**
    # 이 파일이 없으면 unlabeled 샘플이 0개가 되고, 아래 학습 루프의
    #     if unlabeled_indices:  ...  ensemble_teacher(...)
    # 블록이 한 번도 실행되지 않는다. 즉 **8개 teacher 가 한 번도 불리지 않고 soft loss·feature KD 가
    # 전혀 없는 순수 supervised 학습**이 된다. mark5 의 핵심(Dynamic Online Ensemble KD)이
    # 통째로 빠지므로 조용히 넘어가면 안 된다.
    unlabeled_csv = _resolve_csv_path("dataset_unlabeled.csv", required=False)
    unlabeled_files = (list(csv.DictReader(open(unlabeled_csv, newline="", encoding="utf-8")))
                       if unlabeled_csv else [])
    if not unlabeled_files:
        print("[WARN] " + "=" * 74)
        print("[WARN] unlabeled 데이터가 0개입니다 — 지식 증류가 작동하지 않습니다.")
        print("[WARN]   · 8개 teacher(ensemble_teacher)가 한 번도 호출되지 않습니다.")
        print("[WARN]   · soft loss / feature KD 가 전부 0 이고 hard loss 만 남습니다.")
        print("[WARN]   · 결과적으로 순수 supervised 학습이 됩니다(mark5 의 novelty 상실).")
        print("[WARN] data_source_unlabeled 를 채우고 run_all.py 의 unlabeled 단계를 돌리십시오.")
        print("[WARN] " + "=" * 74)

    labeled_dataset = SemiSupervisedDataset(labeled_files, parser, config, is_labeled=True)
    unlabeled_dataset = SemiSupervisedDataset(unlabeled_files, parser, config, is_labeled=False)

    # [변경 2026-08-09] val 을 dataset_val.csv 에서 읽는다.
    # 수집 단계(mark5_slicer.py)에서 SINS·CHiME 은 세션 단위로, AI Hub 는 클립 단위로
    # train/val/test 를 갈라 두었다. 그런데 이 함수가 dataset_labeled.csv 를 파일명 해시로
    # 10% 떼어 val 을 만들고 있어서 그 분리가 통째로 무시되고 있었다(같은 세션 클립이
    # train 과 val 에 섞임 -> val 성능이 낙관 편향). dataset_val.csv 가 있으면 그것을 쓰고,
    # 없을 때만 옛 해시 분할로 폴백하되 눈에 띄게 경고한다.
    val_csv = _resolve_csv_path("dataset_val.csv", required=False)
    if val_csv and os.path.exists(val_csv):
        val_files = list(csv.DictReader(open(val_csv, newline="", encoding="utf-8")))
        val_dataset_src = SemiSupervisedDataset(val_files, parser, config, is_labeled=True)
        train_samples = list(labeled_dataset.samples)
        val_samples = list(val_dataset_src.samples)
        print(f"[INFO] val 을 dataset_val.csv 에서 읽었습니다 — "
              f"train {len(train_samples)}개 / val {len(val_samples)}개 (폴더 기반 분리)")
    else:
        print("[WARN] " + "=" * 70)
        print("[WARN] dataset_val.csv 가 없어 파일명 해시로 train 의 10% 를 val 로 뗍니다(옛 방식).")
        print("[WARN] 이러면 수집 단계의 세션 단위 split 이 무시되어 val 성능이 낙관 편향됩니다.")
        print("[WARN] preprocessing/fix_audio_length.py 로 data_source_val -> data_val 을 만들고")
        print("[WARN] generate_dataset_index_mark5.py --mode val 을 먼저 돌리십시오.")
        print("[WARN] " + "=" * 70)
        train_samples = []
        val_samples = []
        for sample in labeled_dataset.samples:
            _, label, metadata = sample
            split_name = _stable_file_split(metadata["path"], val_ratio=0.1)
            if split_name == "val":
                val_samples.append(sample)
            else:
                train_samples.append(sample)

    train_dataset = SampleListDataset(train_samples)
    val_dataset = SampleListDataset(val_samples)

    final_train_dataset = ConcatDataset([train_dataset, unlabeled_dataset])

    train_loader = DataLoader(final_train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate)

    student_encoder = build_audio_encoder(config).to(device)
    student_branch = DualBranchStudentHead(config.embedding_dim).to(device)
    student_classifier = ViLDTextHead(config).to(device)
    student_text_emb = config.get_class_text_embeddings(for_evaluation=True).to(device)
    student_label_map = config.get_target_label_map()

    # [추가] dummy_label 등이 student 차원에 새지 않았는지 즉시 검증(누수 시 학습 전에 터지게).
    eval_classes = config.get_classes_for_evaluation()
    assert student_text_emb.shape[0] == len(student_label_map) == len(eval_classes), (
        f"[ERROR] student 클래스 차원 불일치: text_emb={student_text_emb.shape[0]}, "
        f"label_map={len(student_label_map)}, eval_classes={len(eval_classes)} (현재 기대값 9). "
        "for_evaluation=True 누락 또는 dummy_label 누수/ config 불일치를 의심하세요."
    )

    # [추가] 학습형 background(others) 임베딩. use_background_embedding=False면 사용하지 않음(None).
    background_embedding = None
    if config.use_background_embedding:
        background_embedding = LearnableBackgroundEmbedding(config.embedding_dim).to(device)

    specialist_config = {
        "heavy_impact": {"mark_version": "mark4.1", "encoder_path": "best_teacher_encoder_mark4.1.pth", "classifier_path": "best_teacher_classifier_mark4.1.pth"},
        "dragging": {"mark_version": "mark4.2", "encoder_path": "best_teacher_encoder_mark4.2.pth", "classifier_path": "best_teacher_classifier_mark4.2.pth"},
        "construction": {"mark_version": "mark4.3", "encoder_path": "best_teacher_encoder_mark4.3.pth", "classifier_path": "best_teacher_classifier_mark4.3.pth"},
        "machine_noise": {"mark_version": "mark4.4", "encoder_path": "best_teacher_encoder_mark4.4.pth", "classifier_path": "best_teacher_classifier_mark4.4.pth"},
        "media_talking": {"mark_version": "mark4.5", "encoder_path": "best_teacher_encoder_mark4.5.pth", "classifier_path": "best_teacher_classifier_mark4.5.pth"},
        "water_toilet": {"mark_version": "mark4.6", "encoder_path": "best_teacher_encoder_mark4.6.pth", "classifier_path": "best_teacher_classifier_mark4.6.pth"},
        "water_shower": {"mark_version": "mark4.7", "encoder_path": "best_teacher_encoder_mark4.7.pth", "classifier_path": "best_teacher_classifier_mark4.7.pth"},
        "dog_bark": {"mark_version": "mark4.8", "encoder_path": "best_teacher_encoder_mark4.8.pth", "classifier_path": "best_teacher_classifier_mark4.8.pth"},
    }
    _others_rule = getattr(config, "fusion_others_rule", "min")
    ensemble_teacher = EnsembleTeacher(specialist_config, device, others_rule=_others_rule)
    print(f"[INFO] teacher fusion: others 칸 조립 규칙 = {_others_rule}")
    teacher_feature_cache = []
    unlabeled_metadata_cache = []

    # [변경 2026-08-09] DKD 설정을 config 에서 읽는다. getattr 로 받아 옛 config 와도 호환.
    _use_dkd = getattr(config, "use_dkd", False)
    _dkd_a = getattr(config, "dkd_alpha", 1.0)
    _dkd_b = getattr(config, "dkd_beta", 1.0)
    # [변경 2026-08-17] alpha·T 를 하드코딩에서 config 로 뺐다. 값 자체는 그대로다
    # (0.7 / 4.0 은 컨퍼런스 코드 mark3.2 에서 검증한 값이고 mark5 는 그 손실 구조를 물려받았다).
    # mark4.x 가 쓰는 0.3 / 2.0 은 같은 샘플에 hard·soft 를 함께 거는 구조에서 soft 가 hard 를
    # 굶긴 문제를 잡느라 내린 값이므로, labeled→hard / unlabeled→soft 로 나뉜 mark5 에는
    # 그대로 옮겨오지 않는다. 다만 조정이 필요할 때 코드를 고치지 않고 config 만 바꾸도록 통로를 둔다.
    _alpha = getattr(config, "distill_alpha", 0.7)
    _temp = getattr(config, "distill_temperature", 4.0)
    criterion = DistillationLoss(
        temperature=_temp, alpha=_alpha,
        feature_kd_weight=config.feature_kd_weight if config.use_feature_kd else 0.0,
        use_dkd=_use_dkd, dkd_alpha=_dkd_a, dkd_beta=_dkd_b)
    print(f"[INFO] KD 모드: {'DKD' if _use_dkd else 'vanilla KD'}"
          + (f" (dkd_alpha={_dkd_a}, dkd_beta={_dkd_b}, 타겟축=teacher argmax)" if _use_dkd else "")
          + f" | alpha={_alpha}, T={_temp}, feature_kd={config.feature_kd_weight if config.use_feature_kd else 0.0}")
    optimizer_params = list(student_encoder.parameters()) + list(student_branch.parameters())
    if background_embedding is not None:
        optimizer_params += list(background_embedding.parameters())
    # [변경 2026-08-17] weight_decay 와 LR 스케줄러를 mark4.x 와 같게 맞췄다.
    # mark4.x 는 weight_decay 가 전무해 teacher 가 val best 를 epoch 5 에 찍고 곧장 과적합하던 것을
    # 실측하고 1e-4 를 넣었고, ReduceLROnPlateau(factor 0.5, patience 3)도 함께 쓴다.
    # mark5 에는 둘 다 없어서 학습 절차가 달랐고, 그러면 mark4.x 결과와 같은 표에 올릴 수 없다.
    # [변경 2026-08-18] 기본값을 config 값(1e-4)과 같게 둔다. 0.0 이면 드라이브에 옛 config 가
    # 올라갔을 때 정규화 없이 조용히 학습된다.
    _weight_decay = getattr(config, "weight_decay", 1e-4)
    optimizer = optim.Adam(optimizer_params, lr=config.learning_rate, weight_decay=_weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    _patience = getattr(config, "early_stopping_patience", 10)
    print(f"[INFO] 학습 절차: weight_decay={_weight_decay}, ReduceLROnPlateau(factor=0.5, patience=3), "
          f"early stopping patience={_patience}, best val 체크포인트 저장")

    train_hist, val_hist = [], []
    # [추가] train/val 직접 비교를 위해 컴포넌트별 손실 히스토리 분리 기록
    #  - train_hist        : Total(hard+soft+feat)  → 참고용
    #  - train_hard_hist   : raw hard CE            → val_hard_hist와 직접 비교 가능
    #  - train_soft/feat   : KD 컴포넌트            → 학습 진행 참고용
    #  - val_hard_hist     : val의 raw hard CE      → train_hard_hist와 직접 비교 가능
    train_hard_hist, train_soft_hist, train_feat_hist = [], [], []
    # [추가] background embedding 보조 loss(bg_loss) 히스토리
    train_bg_hist = []
    val_hard_hist = []
    print(f"[INFO] Student training ({mark_version}) started on {device}")
    # [추가 2026-08-17] best val 체크포인트 + early stopping 상태
    best_val = float("inf")
    best_epoch = -1
    no_improve = 0
    stopped_early = False

    for epoch in range(config.num_epochs):
        student_encoder.train()
        student_branch.train()
        total_loss = total_hard_loss = total_soft_loss = total_feat_loss = 0.0
        total_bg_loss = 0.0
        # [추가 2026-08-17] 캐시를 에폭마다 비운다. teacher 는 eval·no_grad·가중치 고정이라
        # 같은 입력에 늘 같은 값을 내므로 에폭마다 쌓아봐야 같은 값이 num_epochs 벌 중복될 뿐이다.
        # 그대로 두면 80 에폭에서 리스트가 1GB 넘게 불어나고 torch.cat 이 그만큼 사본을 또 만든다.
        teacher_feature_cache.clear()
        unlabeled_metadata_cache.clear()

        for mel_batch, label_batch, metadata_batch in train_loader:
            mel = mel_batch.to(device)
            optimizer.zero_grad()

            labeled_indices = [i for i, lbl in enumerate(label_batch) if isinstance(lbl, str)]
            unlabeled_indices = [i for i, lbl in enumerate(label_batch) if lbl == -1]
            loss = torch.tensor(0.0, device=device)

            if labeled_indices:
                labeled_mel = mel[labeled_indices]
                labeled_targets = torch.tensor([student_label_map[label_batch[i]] for i in labeled_indices], dtype=torch.long, device=device)
                base_features = student_encoder(labeled_mel)
                supervised_features, _ = student_branch(base_features)
                student_logits = student_classifier(supervised_features, student_text_emb)
                hard_total, hard_loss, _, _ = criterion(student_logits, hard_targets=labeled_targets)
                loss = loss + hard_total
                total_hard_loss += hard_loss.item()

                if config.use_background_embedding and background_embedding is not None:
                    others_idx = student_label_map["others"]
                    others_mask = (labeled_targets == others_idx)
                    if others_mask.any():
                        target_feat = F.normalize(supervised_features[others_mask], dim=1)
                        bg = F.normalize(background_embedding(), dim=0).unsqueeze(0).expand_as(target_feat)
                        bg_loss = (1 - F.cosine_similarity(target_feat, bg, dim=1)).mean()
                        loss = loss + config.background_embedding_weight * bg_loss
                        total_bg_loss += bg_loss.item()

            if unlabeled_indices:
                unlabeled_mel = mel[unlabeled_indices]
                fusion_output = ensemble_teacher(unlabeled_mel, student_label_map)
                base_features = student_encoder(unlabeled_mel)
                supervised_features, distill_features = student_branch(base_features)
                student_logits = student_classifier(supervised_features, student_text_emb)
                soft_total, _, soft_loss, feat_loss = criterion(
                    student_logits,
                    teacher_logits=fusion_output.logits,
                    student_features=distill_features,
                    teacher_features=fusion_output.features,
                )
                loss = loss + soft_total
                total_soft_loss += soft_loss.item()
                total_feat_loss += feat_loss.item()
                teacher_feature_cache.append(fusion_output.features.detach().cpu())
                for local_pos, batch_idx in enumerate(unlabeled_indices):
                    meta = dict(metadata_batch[batch_idx])
                    meta["teacher_disagreement"] = float(fusion_output.disagreement[local_pos].item())
                    meta["teacher_uncertainty"] = float(fusion_output.uncertainty[local_pos].item())
                    meta["teacher_specialist_weights"] = {
                        name: float(weight[local_pos].item())
                        for name, weight in fusion_output.specialist_weights.items()
                    }
                    unlabeled_metadata_cache.append(meta)

            if loss.requires_grad and loss.item() > 0:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        avg_hard = total_hard_loss / max(1, len(train_loader))
        avg_soft = total_soft_loss / max(1, len(train_loader))
        avg_feat = total_feat_loss / max(1, len(train_loader))
        avg_bg = total_bg_loss / max(1, len(train_loader))
        train_hist.append(avg_loss)
        train_hard_hist.append(avg_hard)
        train_soft_hist.append(avg_soft)
        train_feat_hist.append(avg_feat)
        train_bg_hist.append(avg_bg)

        student_encoder.eval()
        student_branch.eval()
        val_total = 0.0
        val_hard_total = 0.0
        with torch.no_grad():
            for mel_batch, label_batch, _ in val_loader:
                mel = mel_batch.to(device)
                targets = torch.tensor([student_label_map[lbl] for lbl in label_batch], dtype=torch.long, device=device)
                base_features = student_encoder(mel)
                supervised_features, _ = student_branch(base_features)
                logits = student_classifier(supervised_features, student_text_emb)
                # [수정] total((1-α)·hard)뿐 아니라 raw hard CE도 따로 기록해 train_hard와 동일 척도로 비교
                val_loss, val_hard, _, _ = criterion(logits, hard_targets=targets)
                val_total += val_loss.item()
                val_hard_total += val_hard.item()
        avg_val = val_total / max(1, len(val_loader))
        avg_val_hard = val_hard_total / max(1, len(val_loader))
        val_hist.append(avg_val)
        val_hard_hist.append(avg_val_hard)

        print(f"[Epoch {epoch+1}] Total {avg_loss:.4f} | Hard {avg_hard:.4f} | Soft {avg_soft:.4f} | Feat {avg_feat:.4f} | BG {avg_bg:.4f} | Val(hard*) {avg_val_hard:.4f}")

        # [추가 2026-08-17] LR 스케줄 · best 저장 · early stopping (mark4.x 와 같은 절차)
        scheduler.step(avg_val)
        if avg_val < best_val:
            print(f"[EarlyStopping] Val loss {best_val:.6f} -> {avg_val:.6f}. Saving...")
            best_val = avg_val
            best_epoch = epoch + 1
            no_improve = 0
            save_checkpoint(
                os.path.join(BASE_DIR, f"student_model_{mark_version}.pth"),
                model_type="student_full",
                mark_version=mark_version,
                model_state=student_encoder.state_dict(),
                branch_state=student_branch.state_dict(),
                classifier_state=student_classifier.state_dict(),
                background_state=background_embedding.state_dict() if background_embedding is not None else None,
                # [추가 2026-08-17] config 를 넘겨 config_metadata 를 남긴다. 전에는 인자가 빠져 있어
                # mark5 산출물만 provenance 가 빈 dict 였다(teacher 파일에는 10개가 들어 있다).
                config=config,
            )
        else:
            no_improve += 1
            print(f"[EarlyStopping] counter: {no_improve}/{_patience}")
            if no_improve >= _patience:
                stopped_early = True
                print(f"[INFO] Early stopping. best epoch {best_epoch} (val {best_val:.6f})")
                break

    print("Training finished."
          + (f" (early stop, best epoch {best_epoch}/{config.num_epochs}, val {best_val:.6f})" if stopped_early
             else f" (best epoch {best_epoch}/{config.num_epochs}, val {best_val:.6f})"))
    # [변경 2026-08-18] 저장된 체크포인트가 없는 경우의 처리를 손실곡선 저장 뒤로 미룬다.
    # 여기서 바로 raise 하면(예: val loss 가 첫 에폭부터 NaN 이라 한 번도 개선되지 않은 경우)
    # 이번 수정으로 지키려던 손실 히스토리를 또 잃는다. 원인 분석에 필요한 것이 바로 그 곡선이다.
    no_checkpoint = best_epoch < 0
    if no_checkpoint:
        print("[ERROR] val 이 한 번도 개선되지 않아 저장된 체크포인트가 없습니다. "
              "손실 곡선만 저장하고 종료합니다(val loss 가 NaN 인지 확인하십시오).")

    plot_dir = os.path.join(PROJECT_ROOT, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    # [수정] train 총손실(hard+soft+feat)과 val(hard만)을 한 축에서 직접 비교하면
    #        측정 대상이 달라 과적합을 오판할 수 있음.
    #   -> 직접 비교 가능한 'Train Hard(raw CE) vs Val Hard(raw CE)'를 주 곡선(굵게)으로,
    #      Train Total/Soft/Feat은 학습 진행 참고용(점/파선)으로 함께 표기.
    plt.figure(figsize=(9, 6))
    plt.plot(train_hist, label="Train Total (hard+soft+feat, 참고)", linestyle="--", alpha=0.55)
    plt.plot(train_soft_hist, label="Train Soft KD (참고)", linestyle=":", alpha=0.55)
    plt.plot(train_feat_hist, label="Train Feat KD (참고)", linestyle=":", alpha=0.55)
    plt.plot(train_hard_hist, label="Train Hard (raw CE)", linewidth=2)
    plt.plot(val_hard_hist, label="Val Hard (raw CE)", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Student Loss ({mark_version})\n* 직접 비교는 Train Hard vs Val Hard (동일 raw CE) 기준")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"loss_curve_student_{mark_version}.png"))

    # [추가] 손실곡선 raw 숫자를 CSV로도 저장. PNG만 있으면 다른 모델(CED-Tiny 등)과
    # 겹쳐 그리는 비교 그래프를 다시 만들 수 없어서, 비교 실험용으로 숫자 그대로 남긴다.
    loss_history_csv = os.path.join(plot_dir, f"loss_history_{mark_version}.csv")
    with open(loss_history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_total", "train_hard", "train_soft", "train_feat", "train_bg", "val_total", "val_hard"])
        for i in range(len(train_hist)):
            writer.writerow([
                i + 1, train_hist[i], train_hard_hist[i], train_soft_hist[i],
                train_feat_hist[i], train_bg_hist[i], val_hist[i], val_hard_hist[i],
            ])
    print(f"[INFO] 손실곡선 CSV 저장: {loss_history_csv}")

    # [이동·수정 2026-08-17] teacher 특징·메타데이터 캐시 저장.
    # 전에는 이 블록이 손실곡선 저장보다 앞에 있었고 cache 디렉터리를 아무도 만들지 않아서,
    # 80 에폭을 다 돌고 여기서 "Parent directory ... does not exist" 로 죽으면서
    # 손실곡선 PNG·CSV 를 통째로 잃었다(학습·체크포인트는 이미 끝난 뒤라 더 아깝다).
    # 그래서 (1) 디렉터리를 만들고 (2) 진단용인 이 저장을 맨 뒤로 돌리고 (3) 실패해도
    # 학습 결과를 잃지 않도록 경고만 남기고 넘어간다.
    try:
        cache_dir = os.path.join(PROJECT_ROOT, "cache", mark_version)
        os.makedirs(cache_dir, exist_ok=True)
        if teacher_feature_cache:
            torch.save(torch.cat(teacher_feature_cache, dim=0), get_feature_cache_path(PROJECT_ROOT, mark_version, "unlabeled_train"))
        if unlabeled_metadata_cache:
            torch.save(unlabeled_metadata_cache, get_metadata_cache_path(PROJECT_ROOT, mark_version, "unlabeled_train"))
        if teacher_feature_cache or unlabeled_metadata_cache:
            print(f"[INFO] teacher 캐시 저장: {cache_dir} (마지막 에폭분)")
    except Exception as e:
        print(f"[WARN] teacher 캐시 저장 실패(학습 결과에는 영향 없음): {e}")

    if no_checkpoint:
        raise RuntimeError(
            "[ERROR] val 이 한 번도 개선되지 않아 student_model 체크포인트가 없습니다. "
            f"손실 곡선은 {plot_dir} 에 저장했습니다."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mark5.0 Student 모델을 학습합니다.")
    parser.add_argument("--mark_version", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_mark5(seed_value=args.seed, mark_version=args.mark_version)
