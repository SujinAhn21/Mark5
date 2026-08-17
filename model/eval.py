import os
import sys
import csv
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
VILD_DIR = os.path.join(PROJECT_ROOT, "vild")
for p in (PROJECT_ROOT, UTILS_DIR, VILD_DIR):
    if p not in sys.path:
        sys.path.append(p)

from postprocess_utils import (
    aggregate_segment_probs,
    apply_abstention,
    apply_class_pair_calibration,
    apply_others_calibration,
    apply_temporal_smoothing,
    save_visual_explanation,
)
from vild_config import AudioViLDConfig
from vild_model import LearnableBackgroundEmbedding, ViLDTextHead, build_audio_encoder
from vild_head import DualBranchStudentHead
from vild_parser_teacher import AudioParser
SHARED_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "shared_vild"))
if SHARED_DIR not in sys.path:
    sys.path.append(SHARED_DIR)
from checkpoint_utils import load_checkpoint, resolve_state_dict


def _resolve_dataset_csv_path(split: str):
    """[변경 2026-08-17] test 고정이었던 것을 split 인자로 일반화했다.

    mark4.x 는 eval.py --split val 로 val 을 따로 평가해 운영점(threshold)을 고르고
    test 는 마지막에 한 번만 썼는데, mark5 에는 그 경로가 없어서 val 을 평가할 수단이
    아예 없었다(dataset_test.csv 만 읽었음).
    """
    filename = f"dataset_{split}.csv"
    candidates = [
        os.path.join(PROJECT_ROOT, filename),
        os.path.join(BASE_DIR, filename),
        os.path.join(PROJECT_ROOT, "preprocessing", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"[ERROR] {split} 인덱스 파일을 찾을 수 없습니다: {candidates}")


def evaluate(mark_version: str, split: str = "test"):
    config = AudioViLDConfig(mark_version=mark_version)
    device = config.device
    parser = AudioParser(config, segment_mode=True)

    # [추가 2026-08-17] split 이 test 가 아니면 결과 파일명에 _{split} 을 붙여
    # test 결과를 덮어쓰지 않게 한다(mark4.x eval.py 의 out_tag 규칙과 동일).
    out_tag = mark_version if split == "test" else f"{mark_version}_{split}"

    csv_path = _resolve_dataset_csv_path(split)
    print(f"[INFO] 평가 대상: split={split} · 인덱스={csv_path}")
    test_files = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    model_path = os.path.join(BASE_DIR, f"student_model_{mark_version}.pth")
    checkpoint = load_checkpoint(model_path, map_location=device)

    encoder = build_audio_encoder(config).to(device)
    encoder.load_state_dict(resolve_state_dict(checkpoint, "model_state_dict", "encoder_state_dict", "model"))
    branch_head = DualBranchStudentHead(config.embedding_dim).to(device)
    # [수정 2026-08-17] resolve_state_dict 는 아는 키가 하나도 없으면 KeyError 를 던지므로,
    # 아래 else 의 경고는 도달할 수 없는 코드였다(구버전 .pth 를 넣으면 경고 대신 죽었다).
    # 직접 get 으로 받아 경고 분기가 실제로 살아나게 한다.
    branch_state = (
        checkpoint.get("branch_state_dict")
        or checkpoint.get("head_state_dict")
        or checkpoint.get("head")
    )
    if branch_state is not None:
        branch_head.load_state_dict(branch_state, strict=False)
    else:
        print("[WARN] branch_state_dict가 없어 기본 branch head로 평가합니다. 새 모델 재학습이 권장됩니다.")
    text_head = ViLDTextHead(config).to(device)
    text_emb = config.get_class_text_embeddings(for_evaluation=True).to(device)

    # [추가] 학습형 background(others) 임베딩 로드. use_background_embedding=False거나
    # 체크포인트에 없으면(구버전 호환) None으로 두고 아래 max-override 로직을 건너뜀.
    background_embedding = None
    if config.use_background_embedding:
        bg_state = checkpoint.get("background_state_dict")
        if bg_state is not None:
            background_embedding = LearnableBackgroundEmbedding(config.embedding_dim).to(device)
            background_embedding.load_state_dict(bg_state)
            background_embedding.eval()
        else:
            print("[WARN] background_state_dict가 없어 background embedding 없이 평가합니다. 새 모델 재학습이 권장됩니다.")

    encoder.eval()
    branch_head.eval()
    text_head.eval()

    class_names = config.get_classes_for_evaluation()
    label_map = config.get_target_label_map()
    plot_dir = os.path.join(PROJECT_ROOT, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    all_labels, all_preds, all_probs = [], [], []
    calibration_rows = []
    prediction_rows = []   # [추가 2026-08-17] 샘플별 확률까지 남기는 행(운영점 재계산용)
    skipped_label_counter = {}
    for row in test_files:
        path = row["path"]
        label = row["label"]
        if label not in label_map:
            # [추가] 9-class 밖 라벨은 조용히 넘기지 않고 집계
            skipped_label_counter[label] = skipped_label_counter.get(label, 0) + 1
            continue
        segment_records = parser.load_and_segment_with_metadata(path)
        if not segment_records:
            continue

        segment_probs = []
        saliency_scores = [record["saliency"] for record in segment_records]
        with torch.no_grad():
            for record in segment_records:
                seg = record["tensor"]
                if seg.ndim == 3:
                    seg = seg.unsqueeze(0)
                seg = seg.to(device)
                base_features = encoder(seg)
                supervised_features, distill_features = branch_head(base_features)
                sup_logits = text_head(supervised_features, text_emb)
                distill_logits = text_head(distill_features, text_emb)

                if background_embedding is not None:
                    others_idx = class_names.index("others")
                    bg_norm = F.normalize(background_embedding(), dim=0)

                    sup_norm = F.normalize(supervised_features, dim=1)
                    sup_bg_logit = (sup_norm @ bg_norm) / text_head.temperature
                    sup_logits = sup_logits.clone()
                    sup_logits[:, others_idx] = torch.maximum(sup_logits[:, others_idx], sup_bg_logit)

                    distill_norm = F.normalize(distill_features, dim=1)
                    distill_bg_logit = (distill_norm @ bg_norm) / text_head.temperature
                    distill_logits = distill_logits.clone()
                    distill_logits[:, others_idx] = torch.maximum(distill_logits[:, others_idx], distill_bg_logit)

                w = getattr(config, "distill_branch_eval_weight", 0.5)
                prob = (
                    (1 - w) * torch.softmax(sup_logits, dim=-1)
                    + w * torch.softmax(distill_logits, dim=-1)
                ).squeeze(0).cpu().numpy()
                segment_probs.append(prob)

        if config.enable_temporal_smoothing:
            segment_probs = apply_temporal_smoothing(segment_probs, config.temporal_smoothing_alpha)

        aggregated, seg_weights = aggregate_segment_probs(segment_probs, saliency_scores, config)
        calibrated_prob = apply_class_pair_calibration(aggregated, class_names, config)
        # [추가 2026-08-17] others 보정 전 확률과 그 argmax 를 남긴다.
        # 이 두 가지가 있어야 others_confidence/margin/entropy 를 바꿨을 때의 결과를
        # 모델 재실행 없이 CSV 만으로 다시 계산할 수 있다(mark4.x 의 threshold 스윕과 같은 원리).
        raw_prob = calibrated_prob.copy()
        raw_pred = int(np.argmax(raw_prob))
        calibrated_prob, pred, calib_meta = apply_others_calibration(calibrated_prob, class_names, config)
        calibrated_prob, pred, abstained = apply_abstention(calibrated_prob, class_names, config)
        all_labels.append(label_map[label])
        all_preds.append(pred)
        all_probs.append(calibrated_prob)
        prediction_rows.append(
            {
                "Filename": os.path.basename(path),
                "True Label": label,
                "Raw Predicted Label": class_names[raw_pred],
                "Predicted Label": class_names[pred],
                "Forced To Others": calib_meta["forced_to_others"],
                "Abstained": abstained,
                "Raw Top Confidence": calib_meta["raw_top_conf"],
                "Raw Margin": calib_meta["raw_margin"],
                "Entropy": calib_meta["entropy"],
                **{f"Prob_{n}": float(raw_prob[i]) for i, n in enumerate(class_names)},
            }
        )
        calibration_rows.append({
            "path": path,
            "true_label": label,
            "pred_label": class_names[pred],
            "forced_to_others": calib_meta["forced_to_others"],
            "abstained": abstained,
            "raw_top_conf": calib_meta["raw_top_conf"],
            "raw_margin": calib_meta["raw_margin"],
            "entropy": calib_meta["entropy"],
        })
        save_visual_explanation(path, segment_records, segment_probs, seg_weights, class_names, calibrated_prob, pred, config, plot_dir)

    # [추가] 조용한 탈락 방지: 9-class 밖 라벨 보고 + 평가 샘플 0이면 즉시 중단
    if skipped_label_counter:
        print(
            f"[WARN] 9-class 밖 라벨로 건너뛴 {split} 파일: {skipped_label_counter} "
            f"(허용 클래스: {sorted(label_map.keys())})"
        )
    if len(all_labels) == 0:
        raise ValueError(
            f"[ERROR] 평가 가능한 {split} 샘플이 0개입니다. 건너뛴 라벨: {skipped_label_counter}. "
            f"dataset_{split}.csv 라벨이 9-class와 일치하는지 확인하세요."
        )
    print(f"[INFO] 유효 {split} 샘플: {len(all_labels)}개")

    report = classification_report(
        all_labels,
        all_preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print(report)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 12})
    plt.title(f"Confusion Matrix - {out_tag}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"confusion_matrix_{out_tag}.png"))
    plt.close()

    # [추가] mark4 eval.py와 동일한 형태로 Accuracy/P/R/F1/ROC AUC를 CSV로 저장(기존엔 콘솔 print만 하고 파일에 안 남았음).
    all_probs_arr = np.array(all_probs)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=list(range(len(class_names))), zero_division=0
    )
    # [수정] labels 명시 + try/except로 방어. 테스트셋에 9-class 중 일부가 아예 없으면
    # (1) labels 없이는 "열 개수 불일치"로 죽고, (2) labels를 줘도 그 클래스의 OvR 이진화가
    # "양성 샘플 0개"가 되어 AUC 자체가 수학적으로 정의 불가능해 여전히 ValueError가 난다.
    # 작은 test set에서는 흔히 벌어질 수 있는 상황이라, 죽이지 않고 NaN 처리 후 경고만 남긴다.
    try:
        roc_auc_macro = roc_auc_score(
            all_labels, all_probs_arr, multi_class="ovr", average="macro",
            labels=list(range(len(class_names))),
        )
        print(f"ROC AUC (macro, one-vs-rest): {roc_auc_macro:.4f}")
    except ValueError as e:
        roc_auc_macro = float("nan")
        print(f"[WARN] ROC AUC 계산 불가(테스트셋에 없는 클래스가 있을 수 있음): {e}")

    # [추가] others FPR: 실제 라벨이 "others"인데 8개 타겟 클래스 중 하나로 오분류된 비율.
    # = 1 - Recall(others). "조용한 배경음을 소음으로 잘못 경보하는 비율"이라는 실용적 의미를 명시하기 위해 별도 계산.
    others_idx = class_names.index("others")
    others_row = cm[others_idx]
    others_row_sum = int(others_row.sum())
    others_fpr = float((others_row_sum - others_row[others_idx]) / others_row_sum) if others_row_sum > 0 else float("nan")
    print(f"[others FPR] 실제 others인데 타겟 클래스로 오분류된 비율: {others_fpr:.4f}")

    # [추가] mark4 eval.py의 roc_curve 저장과 동일한 형태(9-class는 one-vs-rest로 클래스별 곡선).
    plt.figure(figsize=(7, 6))
    all_labels_arr = np.array(all_labels)
    for i, cname in enumerate(class_names):
        fpr_curve, tpr_curve, _ = roc_curve((all_labels_arr == i).astype(int), all_probs_arr[:, i])
        auc_i = auc(fpr_curve, tpr_curve)
        plt.plot(fpr_curve, tpr_curve, label=f"{cname} AUC={auc_i:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC ({out_tag})")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"roc_curve_{out_tag}.png"))
    plt.close()

    summary_csv = os.path.join(plot_dir, f"performance_summary_{out_tag}.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        f.write(f"# Performance Summary for {out_tag}\n\n")
        pd.DataFrame({
            "Metric": ["Accuracy", "ROC AUC (Macro, OvR)", "Others FPR"],
            "Score": [accuracy, roc_auc_macro, others_fpr],
        }).to_csv(f, index=False)
        f.write("\n# Class-wise Metrics\n\n")
        pd.DataFrame({
            "Class": class_names, "Precision": precision, "Recall": recall, "F1-Score": f1,
        }).to_csv(f, index=False)
    print(f"[INFO] 성능 요약 CSV 저장: {summary_csv}")

    if calibration_rows:
        pd.DataFrame(calibration_rows).to_csv(
            os.path.join(plot_dir, f"calibration_details_{out_tag}.csv"),
            index=False,
            encoding="utf-8",
        )

    # [추가 2026-08-17] 샘플별 예측 CSV. 클래스별 확률(others 보정 전)을 담고 있어
    # 이 파일 하나로 운영점(others_confidence/margin/entropy)을 바꿔가며 재계산할 수 있다.
    if prediction_rows:
        pred_csv = os.path.join(plot_dir, f"prediction_details_{out_tag}.csv")
        pd.DataFrame(prediction_rows).to_csv(pred_csv, index=False, encoding="utf-8")
        print(f"[INFO] 예측 상세 CSV 저장: {pred_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="학습된 모델의 성능을 평가합니다.")
    parser.add_argument("--mark_version", type=str, required=True)
    # [추가 2026-08-17] 평가 split 선택. 기본은 test(기존 동작 그대로).
    # 운영점은 val 로 고르고 test 는 마지막 확인 한 번만 쓴다(mark4.x 와 같은 원칙).
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    args = parser.parse_args()
    evaluate(mark_version=args.mark_version, split=args.split)
