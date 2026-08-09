# 파일명: generate_dataset_index_mark5.py (하나로 통합된 최종 버전)

import os
import sys
import glob
import csv
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt

# === 경로 설정 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
VILD_DIR = os.path.join(PROJECT_ROOT, "vild")
for p in (PROJECT_ROOT, UTILS_DIR, VILD_DIR):
    if p not in sys.path:
        sys.path.append(p)

from vild_config import AudioViLDConfig

def save_csv(entries, output_path, header):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(entries)
    print(f"[완료] CSV 파일 저장됨: {output_path}")


def mirror_csv(entries, filename, header):
    targets = [
        os.path.join(PROJECT_ROOT, filename),
        os.path.join(BASE_DIR, filename),
        os.path.join(PROJECT_ROOT, "model", filename),
    ]
    for target in targets:
        save_csv(entries, target, header)

def plot_label_distribution(label_count: dict, title: str, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    labels = list(label_count.keys())
    counts = [label_count[label] for label in labels]
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color='skyblue')
    plt.title(title)
    plt.xlabel("Class Label")
    plt.ylabel("File Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[완료] 라벨 분포 시각화 저장됨: {save_path}")

def generate_index(mark_version: str, mode: str):
    """
    --mode 인자에 따라 학습용/검증용/테스트용 데이터 인덱스를 생성합니다.
    - 'train': data_labeled, data_unlabeled 폴더를 스캔하여 학습용 CSV 생성
    - 'val'  : data_val 폴더를 스캔하여 검증용 CSV 생성
    - 'test' : data_test 폴더를 스캔하여 평가용 CSV 생성
    """
    config = AudioViLDConfig(mark_version=mark_version)

    if mode == 'train':
        generate_train_index(config)
    elif mode == 'val':
        generate_val_index(config)
    elif mode == 'test':
        generate_test_index(config)
    else:
        print(f"[ERROR] 잘못된 모드입니다: '{mode}'. 'train', 'val', 'test' 중 하나를 사용하세요.")

def generate_train_index(config: AudioViLDConfig):
    """학습용 Labeled/Unlabeled 데이터 인덱스를 생성합니다."""
    # --- 1. Labeled 데이터 처리 ---
    print("\n" + "="*50 + "\n1. Labeled 데이터 인덱스 생성 (학습용)\n" + "="*50)
    data_dir = os.path.join(PROJECT_ROOT, "data_labeled")
    output_csv = os.path.join(PROJECT_ROOT, "dataset_labeled.csv")
    plot_path = os.path.join(PROJECT_ROOT, "plots", "label_dist_labeled.png")
    process_labeled_folder(data_dir, output_csv, plot_path, config)

    # --- 2. Unlabeled 데이터 처리 ---
    print("\n" + "="*50 + "\n2. Unlabeled 데이터 인덱스 생성 (학습용)\n" + "="*50)
    unlabeled_data_dir = os.path.join(PROJECT_ROOT, "data_unlabeled")
    if not os.path.isdir(unlabeled_data_dir):
        print(f"[경고] Unlabeled 데이터 폴더를 찾을 수 없습니다: {unlabeled_data_dir}. Unlabeled 인덱스를 건너뜁니다.")
        return

    unlabeled_audio_files = sorted(glob.glob(os.path.join(unlabeled_data_dir, "*.wav")))
    unlabeled_entries = [[path.replace("\\", "/")] for path in unlabeled_audio_files]
    output_unlabeled_csv = os.path.join(PROJECT_ROOT, "dataset_unlabeled.csv")
    mirror_csv(unlabeled_entries, "dataset_unlabeled.csv", header=["path"])
    print(f"\n--- Unlabeled 데이터 요약 ---\n총 {len(unlabeled_entries)}개의 인덱스를 생성했습니다.")

def generate_val_index(config: AudioViLDConfig):
    """검증용 Val 데이터 인덱스를 생성합니다.

    [신설 2026-08-09] 전에는 val 이 따로 없어서 train_mark5.py 가 dataset_labeled.csv 를
    파일명 해시로 10% 갈라 val 로 썼다. 그러면 val 이 train 과 같은 원본 풀에서 나오고,
    수집 단계에서 세션 단위로 갈라 둔 split 이 무의미해진다(같은 세션의 클립이 train 과
    val 에 섞임). mark4.x 는 data/{split} 폴더로 나뉘어 있어 이 문제가 없었다.

    mark5 는 클래스 폴더가 한 겹 더 있어(process_labeled_folder 가 1단계 폴더명을 클래스명으로
    읽는다) split 을 클래스 폴더 아래에 둘 수 없다. 그래서 data_source_test 가 이미 별도
    폴더인 패턴을 따라 data_source_val -> data_val 을 두고 여기서 인덱스를 만든다.
    """
    print("\n" + "="*50 + "\n1. Val 데이터 인덱스 생성 (검증용)\n" + "="*50)
    data_dir = os.path.join(PROJECT_ROOT, "data_val")
    output_csv = os.path.join(PROJECT_ROOT, "dataset_val.csv")
    plot_path = os.path.join(PROJECT_ROOT, "plots", "label_dist_val.png")
    process_labeled_folder(data_dir, output_csv, plot_path, config)


def generate_test_index(config: AudioViLDConfig):
    """평가용 Test 데이터 인덱스를 생성합니다."""
    print("\n" + "="*50 + "\n1. Test 데이터 인덱스 생성 (평가용)\n" + "="*50)
    data_dir = os.path.join(PROJECT_ROOT, "data_test")
    output_csv = os.path.join(PROJECT_ROOT, "dataset_test.csv")
    plot_path = os.path.join(PROJECT_ROOT, "plots", "label_dist_test.png")
    process_labeled_folder(data_dir, output_csv, plot_path, config)

def process_labeled_folder(data_dir, output_csv, plot_path, config):
    """(공통 로직) 폴더 구조 기반의 Labeled 데이터 처리 함수"""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"[ERROR] 데이터 폴더를 찾을 수 없습니다: {data_dir}")
    
    print(f"[INFO] 데이터 검색 경로: {data_dir}")
    
    entries = []
    label_count = defaultdict(int)
    found_labels = set()
    # [변경] dummy_label을 포함하던 get_classes_for_text_prompts() 대신 평가용 9-class 사용.
    expected_labels = set(config.get_classes_for_evaluation())

    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            found_labels.add(class_name)
            audio_files = glob.glob(os.path.join(class_dir, "*.wav"))
            for path in audio_files:
                entries.append((path.replace("\\", "/"), class_name))
                label_count[class_name] += 1

    # [추가] 9-class 밖 폴더명은 옛 6-class(Mark3.2) 잔재일 수 있음.
    # 조용히 통과시키면 학습/평가에서 무음 탈락하므로 CSV를 쓰기 전에 즉시 중단한다.
    unexpected = found_labels - expected_labels
    if unexpected:
        raise ValueError(
            f"[ERROR] 9-class 체계에 없는 폴더명이 발견되었습니다: {sorted(unexpected)}\n"
            f"  - 허용된 클래스(9): {sorted(expected_labels)}\n"
            f"  - 검사 경로: {data_dir}\n"
            f"  => 옛 6-class(Mark3.2) 잔재일 수 있습니다. "
            f"폴더명을 9-class로 정리한 뒤 다시 실행하세요."
        )

    entries.sort(key=lambda x: x[0])
    mirror_csv(entries, os.path.basename(output_csv), header=["path", "label"])

    print("\n--- 데이터 요약 ---")
    print(f"총 {len(entries)}개의 인덱스를 생성했습니다.")

    missing = expected_labels - found_labels
    if missing:
        print(f"[경고] 설정에 있으나 폴더가 없는 클래스: {sorted(list(missing))}")

    print("\n[라벨 분포]")
    for label, count in sorted(label_count.items()):
        print(f"  - {label}: {count}개 파일")

    plot_label_distribution(label_count, f"Dataset Distribution ({os.path.basename(data_dir)})", plot_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="학습/테스트용 데이터 인덱스 파일을 생성합니다.")
    parser.add_argument('--mark_version', type=str, required=True,
                        help="설정을 불러올 모델 버전")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'val', 'test'],
                        help="'train', 'val', 'test' 중 하나를 선택하세요.")
    args = parser.parse_args()

    generate_index(mark_version=args.mark_version, mode=args.mode)    
