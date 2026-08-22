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

EMA 를 쓰지 않는 이유 (2026-08-22 수정)
--------------------------------------
처음에는 학습과 똑같이 EMA 로 분포를 누적했는데, 그 측정이 학습 조건을 전혀 재현하지 못했다.

  - dataset_val.csv 는 **클래스별로 완전히 정렬돼 있다**(construction 48개 연속, dog_bark 47개
    연속, ...). 이 스크립트는 클립을 CSV 순서대로 하나씩(배치 = 5세그먼트) 통과시키므로,
    momentum 을 낮추면 EMA 가 "지금 construction 이 100%" 라고 추정하고 정렬이 **정답 클래스를
    억누른다.** 실제로 momentum 0.9 에서 tau 1.0 이면 정확도가 0.8744 -> 0.5186 으로 무너졌다.
  - 반대로 momentum 0.999 로 두면 54배치로는 EMA 가 초기값(균등)에서 3% 밖에 안 움직여
    보정이 사실상 안 걸린다(others 보정 비율 1.03배, 필요한 값은 2.51배).

학습은 shuffle=True 로 배치 16개가 무작위이고 1250배치 x 49epoch = 6만 배치를 돌아 EMA 가
전역 분포에 수렴한다. 그 정상 상태를 재현하려면 EMA 를 흉내낼 게 아니라 **전체 분포를 한 번에
계산해 고정 보정**으로 재는 것이 맞다. 부수적으로 teacher forward 가 1회면 되고 tau 스윕은
계산만으로 끝난다.

실행
----
    python model/sweep_dist_align.py
    python model/sweep_dist_align.py --taus 0,0.1,0.25,0.5,0.75,1.0 --limit 200
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


def collect_probs(data, teacher, class_map):
    """teacher 를 한 번만 통과시켜 세그먼트 확률을 전부 모은다.

    돌려주는 것:
      seg_probs : 클립마다 [세그먼트 수, 9] 확률 텐서의 리스트 (정렬 **전** 값)
      truths    : 클립별 정답 라벨
    이후 tau 스윕은 이 확률 위에서 계산만으로 한다.
    """
    seg_probs, truths = [], []
    for mel, label in data:
        out = teacher(mel, class_map)
        seg_probs.append(torch.softmax(out.logits, dim=1).detach())
        truths.append(label)
    return seg_probs, truths


def align_and_predict(seg_probs, classes, tau):
    """전역 분포로 고정 보정한 뒤 클립 단위 argmax.

    p_hat 은 전체 세그먼트의 평균 확률(= 학습에서 EMA 가 수렴하는 값),
    p_target 은 균등이다. aligned ∝ p x (p_target / p_hat)^tau.
    """
    allp = torch.cat(seg_probs, dim=0)
    p_hat = allp.mean(dim=0)
    p_hat = p_hat / p_hat.sum().clamp_min(1e-12)
    n = len(classes)
    p_target = torch.full_like(p_hat, 1.0 / n)

    preds = []
    if tau == 0:
        ratio = torch.ones_like(p_hat)
    else:
        ratio = (p_target / p_hat.clamp_min(1e-8)) ** tau
    for p in seg_probs:
        q = p * ratio.unsqueeze(0)
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)
        preds.append(classes[int(q.mean(dim=0).argmax())])
    return preds, p_hat


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
    spec = build_specialist_config()

    print()
    print("[2/2] teacher 통과 (1회) — 이후 tau 스윕은 계산만")
    # 정렬은 끄고 원본 확률을 받는다. 보정은 아래에서 전역 분포로 직접 건다.
    base_teacher = EnsembleTeacher(
        spec, device,
        others_rule=getattr(config, "fusion_others_rule", "min"),
        score_mode=getattr(config, "fusion_score_mode", "margin"),
        use_distribution_alignment=False)
    seg_probs, truths = collect_probs(data, base_teacher, class_map)
    total_segs = sum(p.shape[0] for p in seg_probs)
    print(f"  세그먼트 {total_segs}개 확보")

    _, p_hat = align_and_predict(seg_probs, classes, 0.0)
    print()
    print("  teacher 관측 분포 p_hat (학습에서 EMA 가 수렴하는 값)")
    for i, c in enumerate(classes):
        print(f"    {c:14s} {p_hat[i]:.4f}   보정배율(tau=1) x{(1.0/len(classes))/max(float(p_hat[i]),1e-8):.2f}")

    results = []
    preds, _ = align_and_predict(seg_probs, classes, 0.0)
    acc, rec, l1 = report("정렬 없음 (현재)", preds, truths, classes)
    results.append(("off", acc, rec, l1))

    for tau in taus:
        if tau == 0:
            continue  # off 와 동일하므로 건너뛴다
        preds, _ = align_and_predict(seg_probs, classes, tau)
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
