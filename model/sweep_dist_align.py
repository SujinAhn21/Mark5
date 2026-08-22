"""분포 정렬(distribution alignment)의 tau 를 val 로 정하는 스윕 스크립트.

[신설 2026-08-22]

왜 필요한가
-----------
pseudo-label 의 클래스 분포가 균등에서 벗어나 있다. unlabeled 300클립 1500세그먼트 실측에서
construction 19.2% / others 2.4% 였다(균등이면 11.1%). 이 편향이 student 에 주입되어
test 혼동행렬에서 construction FP 21건(precision 0.687) · others recall 0.660 으로 나타난다.

사후 보정(클래스 가중치를 val 로 학습)은 2026-08-22 에 실패했다 — val 0.8814->0.9023 인데
test 0.8581->0.8581 로 전혀 전이되지 않았다(파라미터 10개를 430샘플로 맞춰 과적합).
그래서 학습 단계에서 고치기로 했고, 그 강도 tau 를 여기서 정한다.

무엇을 재는가
-------------
unlabeled 2,000개에는 정답이 없어 pseudo-label 이 맞는지 알 수 없다. 그래서 정답이 있는
val 430클립을 대리물로 쓴다. val 을 teacher 8개에 통과시키면 "pseudo-label 이라면 이렇게
붙었을 것"이 나오고, 정답과 대조해 품질을 직접 잴 수 있다.

tau 를 바꿔가며 다음을 본다.
  - teacher argmax 정확도 (pseudo-label 이 몇 % 맞나)
  - 클래스별 recall (특히 others 와 construction)
  - 예측 분포가 균등(1/9)에 얼마나 가까워지나

⚠ 여기서 재는 것은 **teacher 자신의 품질**이지 student 성능이 아니다. teacher pseudo-label 이
좋아지면 student 도 좋아질 것이라는 가정 위에서 tau 를 고르는 것이고, 그 가정의 검증은
재학습 1회로만 가능하다.

⚠ 학습 때는 EMA 가 배치를 따라가며 갱신되지만, 여기서는 val 을 한 번 훑으므로 EMA 가
학습만큼 수렴하지 않는다. 그래서 --two_pass 를 기본으로 둔다(1회차로 분포를 재고,
2회차에서 그 분포로 보정). 학습에서의 정상 상태에 더 가깝다.

실행
----
    python model/sweep_dist_align.py
    python model/sweep_dist_align.py --taus 0,0.25,0.5,0.75,1.0 --limit 200
"""

import argparse
import csv
import os
import sys
from collections import Counter

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
for _p in ("vild", "utils", "shared_vild", "model"):
    _full = os.path.join(PROJECT_ROOT, _p)
    if _full not in sys.path:
        sys.path.append(_full)

from vild_config import AudioViLDConfig
from vild_parser_teacher import AudioParser
from train_mark5 import EnsembleTeacher, _resolve_csv_path

SPECIALISTS = {
    "heavy_impact": "mark4.1", "dragging": "mark4.2", "construction": "mark4.3",
    "machine_noise": "mark4.4", "media_talking": "mark4.5", "water_toilet": "mark4.6",
    "water_shower": "mark4.7", "dog_bark": "mark4.8",
}


def build_specialist_config():
    return {
        cls: {
            "mark_version": mv,
            "encoder_path": f"best_teacher_encoder_{mv}.pth",
            "classifier_path": f"best_teacher_classifier_{mv}.pth",
        }
        for cls, mv in SPECIALISTS.items()
    }


def collect_segments(rows, parser, device):
    """클립마다 세그먼트 텐서와 정답 라벨을 모은다. 파싱은 한 번만 하고 재사용한다."""
    data = []
    for i, row in enumerate(rows):
        try:
            recs = parser.load_and_segment_with_metadata(row["path"])
        except Exception as e:
            print(f"  [skip] {row['path']}: {e}")
            continue
        segs = [r["tensor"] for r in recs if r.get("tensor") is not None]
        if not segs:
            continue
        data.append((torch.stack(segs).to(device), row["label"]))
        if (i + 1) % 100 == 0:
            print(f"  파싱 {i+1}/{len(rows)}")
    return data


def evaluate(data, teacher, class_map, classes, two_pass=True):
    """teacher 를 통과시켜 클립 단위 argmax 예측을 낸다.

    two_pass=True 면 1회차로 EMA 분포만 채우고 2회차에서 보정된 값을 쓴다.
    """
    passes = 2 if (two_pass and teacher.use_distribution_alignment) else 1
    preds, truths = [], []
    for p in range(passes):
        preds, truths = [], []
        for mel, label in data:
            out = teacher(mel, class_map)
            prob = torch.softmax(out.logits, dim=1).mean(dim=0)
            preds.append(classes[int(prob.argmax())])
            truths.append(label)
    return preds, truths


def report(tag, preds, truths, classes):
    n = len(truths)
    acc = sum(1 for p, t in zip(preds, truths) if p == t) / n
    dist = Counter(preds)
    print(f"\n  [{tag}]  클립 정확도 {acc:.4f}  ({sum(1 for p,t in zip(preds,truths) if p==t)}/{n})")
    print(f"  {'클래스':14s} {'recall':>8s} {'예측비율':>10s}")
    rec = {}
    for c in classes:
        tot = sum(1 for t in truths if t == c)
        hit = sum(1 for p, t in zip(preds, truths) if t == c and p == c)
        r = hit / tot if tot else 0.0
        rec[c] = r
        print(f"  {c:14s} {r:8.3f} {dist[c]/n:10.4f}")
    # 예측 분포가 균등에서 얼마나 벗어났나 (L1 거리)
    u = 1.0 / len(classes)
    l1 = sum(abs(dist[c] / n - u) for c in classes)
    print(f"  균등분포와의 L1 거리 : {l1:.4f}  (0 이면 완전 균등)")
    return acc, rec, l1


def main():
    ap = argparse.ArgumentParser(description="분포 정렬 tau 스윕 (val 기준)")
    ap.add_argument("--taus", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--limit", type=int, default=None, help="클립 수 제한(빠른 확인용)")
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="기본 val. test 는 최종 확인용이므로 함부로 쓰지 말 것")
    ap.add_argument("--momentum", type=float, default=None,
                    help="미지정이면 config 값. val 은 27배치뿐이라 0.9 정도가 적절할 수 있음")
    ap.add_argument("--single_pass", action="store_true", help="2-pass 대신 1-pass")
    args = ap.parse_args()

    config = AudioViLDConfig(mark_version="mark5.0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = config.get_classes_for_evaluation()
    class_map = config.get_target_label_map()

    rows = list(csv.DictReader(open(_resolve_csv_path(f"dataset_{args.split}.csv"),
                                    newline="", encoding="utf-8")))
    if args.limit:
        rows = rows[:args.limit]

    print(f"device={device} · split={args.split} · 클립 {len(rows)}개")
    print(f"클래스 순서: {classes}")
    if args.split == "test":
        print("⚠ test 로 tau 를 고르면 안 된다. 확인 목적일 때만 쓸 것.")
    print()
    print("[1/2] 오디오 파싱 (한 번만, 이후 tau 마다 재사용)")
    parser = AudioParser(config, segment_mode=True)
    data = collect_segments(rows, parser, device)
    print(f"  유효 클립 {len(data)}개")

    taus = [float(t) for t in args.taus.split(",") if t.strip()]
    mom = args.momentum if args.momentum is not None else getattr(config, "dist_align_momentum", 0.999)
    spec = build_specialist_config()

    print()
    print(f"[2/2] tau 스윕 — momentum={mom}, {'1-pass' if args.single_pass else '2-pass'}")
    results = []

    # 기준선: 정렬 없음
    base_teacher = EnsembleTeacher(
        spec, device,
        others_rule=getattr(config, "fusion_others_rule", "min"),
        score_mode=getattr(config, "fusion_score_mode", "margin"),
        use_distribution_alignment=False)
    preds, truths = evaluate(data, base_teacher, class_map, classes, two_pass=False)
    acc, rec, l1 = report("정렬 없음 (현재)", preds, truths, classes)
    results.append(("off", acc, rec, l1))

    for tau in taus:
        t = EnsembleTeacher(
            spec, device,
            others_rule=getattr(config, "fusion_others_rule", "min"),
            score_mode=getattr(config, "fusion_score_mode", "margin"),
            use_distribution_alignment=True, dist_align_tau=tau, dist_align_momentum=mom)
        preds, truths = evaluate(data, t, class_map, classes, two_pass=not args.single_pass)
        acc, rec, l1 = report(f"tau = {tau:g}", preds, truths, classes)
        results.append((f"{tau:g}", acc, rec, l1))

    print()
    print("=" * 78)
    print("요약")
    print("=" * 78)
    print(f"  {'tau':>6s} {'정확도':>9s} {'others R':>10s} {'const R':>9s} {'machine R':>10s} {'균등L1':>8s}")
    for tag, acc, rec, l1 in results:
        print(f"  {tag:>6s} {acc:9.4f} {rec['others']:10.3f} {rec['construction']:9.3f} "
              f"{rec['machine_noise']:10.3f} {l1:8.4f}")
    best = max(results, key=lambda r: r[1])
    print()
    print(f"  정확도 기준 최적 : tau = {best[0]}  (정확도 {best[1]:.4f})")
    print()
    print("  ⚠ 이것은 teacher pseudo-label 의 품질이지 student 성능이 아니다.")
    print("    student 개선 여부는 이 tau 로 재학습해야 확인된다.")


if __name__ == "__main__":
    main()
