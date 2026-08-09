"""
mark5_slicer.py — mark5.0(9-class 다중분류)의 학습·검증·평가 데이터를 한 번에 수집하는 스크립트.

Mark4.x 는 버전마다 "타겟 1개 + others" 이진 구성이라 aihub_slicer / sins_slicer 가 각각
그 전제로 짜여 있다. mark5.0 은 9개 클래스를 동시에 만들어야 하고 저장 구조도 다르므로
(클래스별 폴더) 두 슬라이서의 창 선택·품질 게이트·provenance 기록 방식만 그대로 가져와
새로 만든다.

■ 클래스별 출처 (2026-08-08 실측 + 2026-08-09 CHiME-Home 추가)

provenance 와 AI Hub 71296 원본 38개 카테고리를 대조한 결과, 9개 클래스 중 8개는 AI Hub 안에서
전부 나오고 media_talking 만 없다. 가장 가까운 것이 '옥외설치확성기의소음'인데 실외 상가
확성기이고 AI Hub 자신도 이것을 '사업장 소음'으로 분류했다. 게다가 VS_ 볼륨에 95개뿐이다.
채널 단서를 줄이려고 클래스 정의를 왜곡하는 것은 본말전도라 media_talking 에 넣지 않는다.

    heavy_impact   AI Hub  중량충격음 3종(어른발걸음·아이들발걸음·망치질)
    dragging       AI Hub  경량충격음 가구끄는소리
    construction   AI Hub  공사장 12종(건설장비 7 + 차량 5)
    machine_noise  AI Hub  생활소음 4종(드럼세탁기·통돌이세탁기·진공청소기·식기세척기)
    media_talking  SINS watching_tv + CHiME-Home v 를 절반씩   <- AI Hub 에 없음
    water_toilet   AI Hub  생활소음 화장실물내리는소리
    water_shower   AI Hub  생활소음 샤워할때물소리
    dog_bark       AI Hub  애완동물 강아지짓는소리
    others         (SINS 비TV + CHiME v없음) 절반 + AI Hub 15종 절반

■ media_talking 을 두 집에서 절반씩 뽑는 이유 (2026-08-09 신설)

2026-08-08 에 mark4.5 가 val accuracy 1.0000 / ROC AUC 1.0000 을 낸 것이 성능이 아니라 채널
지름길이었다(내용이 쓰는 비율 1.7%, 정상 버전은 38~52%). SINS 단일 출처로 재구축해 그것은
잡았으나(val 0.9953) 환경이 한 채뿐이고 SINS 는 한 세션이 한 활동이라 타겟과 others 가 세션을
공유할 수 없다는 한계가 남았다. 2026-08-09 에 CHiME-Home(WASPAA 2015, 영국 단독주택 lounge,
CC BY-NC-SA 3.0)을 절반 섞어 mark4.5 를 다시 만들면서 두 문제를 해결했다. CHiME-Home 은
**같은 세션 안에서 TV 가 켜졌다 꺼졌다 하므로** 한 세션에서 두 클래스를 다 뽑을 수 있다.

mark5.0 도 같은 구성을 쓴다. 그래야 4.5 와 mark5 의 media_talking 이 같은 성질이 되고,
4.5 만 특별한 버전이 되지 않는다.

■ ⚠ 9-class 에서는 완전한 직교가 불가능하다 — 반드시 논문에 밝힐 것

mark4.5(2-class)에서는 두 출처를 두 클래스에 50:50 으로 넣어 "출처를 알아도 클래스를 맞힐 수
없는" 상태를 만들 수 있었다. mark5.0 에서는 안 된다. **AI Hub 에 TV 소리가 없어서** 나머지
7개 클래스에는 SINS·CHiME 소리가 들어갈 수 없기 때문이다. 어떻게 배합해도

    "SINS 나 CHiME 이 감지되면 media_talking 아니면 others"

라는 단서가 남고, 이것만으로 7개 클래스를 배제할 수 있다. 지금 배합에서 채널을 감지했을 때
media_talking 일 확률은 약 67%(=318/(318+158))이며 9-class 기저확률 11% 보다 높다.

others 의 AI Hub 몫을 줄이면 이 숫자를 60% 나 50% 까지 낮출 수 있지만 그렇게 하지 않았다.
others 는 "8개 타겟이 아닌 층간소음 도메인 소리"(문여닫기·런닝머신·골프퍼팅·악기 2종·고양이·
사업장 5종·교통 4종)이고, AI Hub 몫이 빠지면 실제 운영에서 그 소리들이 8개 클래스 중 하나로
오분류된다. 게다가 val·test 는 others 가 47개뿐이라 AI Hub 몫이 23개든 16개든 통계적으로
비슷하다 — 비율을 바꿔도 val·test 에서 얻는 것이 없다.

대신 수집 뒤 진단으로 정량화한다. media_talking 이 채널로만 맞고 있다면 (1) media_talking 의
recall 이 유독 높고 (2) others 중 SINS·CHiME 출신이 media_talking 으로 몰려 오분류되며
(3) media_talking 과 나머지 7개 클래스 사이의 혼동이 거의 없다. others 에 두 집을 절반 넣어
두었기 때문에 (2) 가 탐지기 역할을 한다.

■ 규모 (2026-08-08 교수님 지시: Mark4 와 같은 2000/430/430)

split 전체를 9개 클래스가 나눠 가진다. 끝수는 최대잉여법으로 붙여 합이 정확히 맞게 한다.

    train 2000 = 222 x 9 + 2   -> 앞 2개 클래스만 223
    val    430 =  47 x 9 + 7   -> 앞 7개 클래스만 48
    test   430 =  47 x 9 + 7   -> 앞 7개 클래스만 48

■ 저장 구조

generate_dataset_index_mark5.py 의 process_labeled_folder 가 폴더 바로 아래 1단계 폴더명을
클래스명으로 읽고, 9-class 밖 이름이 있으면 중단시킨다. 그래서 split 을 그 아래에 두면 안 된다.
data_source_test 가 이미 별도 폴더인 패턴을 따라 val 도 폴더를 나눈다.

    data_source_labeled/{class}/{class}_train_NNN.wav    (train 2000)
    data_source_val/{class}/{class}_val_NNN.wav          (val 430)
    data_source_test/{class}/{class}_test_NNN.wav        (test 430)

■ 창 선택 — 출처마다 다르다(원본 구조가 다르기 때문)

  AI Hub : 48kHz 15초 클립 + annotation json. aihub_slicer 와 같게 가장 긴 annotation 중앙을
           먼저 쓰고, 창 RMS 가 미달이면 다음 annotation, 전부 미달이면 클립 중앙.
           16kHz 로 리샘플한다.
  SINS   : 16kHz 10초 4채널. sins_slicer 와 같게 가운데 고정(3.5~6.5초), 떨어지면 앞(0~3) ->
           뒤(7~10). ch0 만 쓰고 노드별 DC 오프셋을 뺀다.
  CHiME  : 16kHz 4초 mono. chime_slicer 와 같게 가운데 고정(0.5~3.5초), 떨어지면 앞(0.0) ->
           뒤(1.0). 여유가 1초뿐이라 대체 창은 겹침이 커서 사실상 다음 파일로 넘어간다.
           이미 16kHz mono 라 리샘플·채널선택이 필요 없고, DC 는 |1e-5| 수준으로 사실상 0 이지만
           SINS 와 같은 처리를 통과시키려고 창 평균 빼기는 그대로 적용한다.

■ 누수 방지

  - 한 원본 클립에서 세그먼트 1개만 뽑는다(mark5 안에서 클립이 두 split 에 걸치지 않는다).
  - SINS·CHiME 은 세션 단위 split 을 그대로 따른다(매니페스트의 split 열).
    SINS 는 같은 시각을 노드 4개가 동시에 녹음하므로 노드로 나누면 train 과 test 에 같은 순간이
    들어간다. CHiME 은 연속 녹음이라 같은 세션의 앞뒤 청크가 같은 상황을 담고 있다.
  - mark4.x 가 쓴 원본 클립과는 겹쳐도 된다. provenance 중복 판정은 2026-08-01 변경으로
    같은 mark_version 안에서만 적용된다(버전이 늘수록 others 풀이 마르는 것을 막으려고 그렇게 정함).
    mark4 와 mark5 는 별개 실험이라 원본을 공유해도 한 모델 안의 누수가 아니다.

■ ★세션 라운드로빈 (2026-08-09 신설)

활동·조합별 쿼터만으로는 세션이 고르게 안 섞인다. media_talking 은 SINS 활동이 watching_tv
하나, CHiME 조합이 v 하나뿐이라 **세션 배분이 아예 일어나지 않는다.** 2026-08-09 mark4.5
1차 수집에서 test 세션 하나가 타겟 0 / others 55 가 되어 그 세션이 곧 others 가 되는 일이
실제로 벌어졌다. 그래서 활동·조합 안에서 세션을 번갈아 꺼낸다.

■ 사용 예

  python preprocessing/mark5_slicer.py --dry_run
  python preprocessing/mark5_slicer.py
"""
import os
import re
import sys
import csv
import json
import time
import shutil
import hashlib
import argparse
from datetime import date
from collections import defaultdict

import numpy as np
import pandas as pd
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from validate_dataset import check_waveform                      # noqa: E402

TARGET_SR = 16000
SEG_SEC = 3.0
MIN_SEG_RMS = 1e-3           # aihub_slicer 와 같은 값. 이보다 낮으면 무음 창으로 보고 다음 후보로
SINS_WINDOW_STARTS = (3.5, 0.0, 7.0)      # 10초 원본
CHIME_WINDOW_STARTS = (0.5, 0.0, 1.0)     # 4초 원본
MARK_VERSION = "mark5.0"

# 클래스 -> AI Hub 카테고리 폴더명에 들어 있는 키워드. 부분일치로 찾는다.
AIHUB_MAP = {
    "heavy_impact":  ["1.중량충격음_a", "1.중량충격음_b", "1.중량충격음_c"],
    "dragging":      ["2.경량충격음_a"],
    "construction":  ["B.공사장"],
    "machine_noise": ["3.생활소음_c", "3.생활소음_d", "3.생활소음_e", "3.생활소음_f"],
    "water_toilet":  ["3.생활소음_a"],
    "water_shower":  ["3.생활소음_b"],
    "dog_bark":      ["5.애완동물_a"],
}
# others 로 쓸 AI Hub 카테고리(타겟 7개 클래스에 안 쓰인 15종).
OTHERS_AIHUB = ["2.경량충격음_b", "2.경량충격음_c", "2.경량충격음_d",
                "4.악기_a", "4.악기_b", "5.애완동물_b", "C.사업장", "D.교통소음"]

SPLIT_DIRS = {
    "train": "data_source_labeled",
    "val":   "data_source_val",
    "test":  "data_source_test",
}
SPLIT_TOTALS = {"train": 2000, "val": 430, "test": 430}


def classes_from_config():
    """vild_config.py 에서 mark5.0 의 9개 클래스를 순서 그대로 읽는다.

    손으로 적지 않는 이유는 generate_dataset_index_mark5.py 가 같은 목록으로 폴더명을
    검사하기 때문이다. 어긋나면 인덱스 생성이 중단된다. torch 를 끌고 오지 않으려고
    소스를 텍스트로 읽어 정규식으로 뽑는다(/mnt/c 의 venv 에서 torch import 는 76초).
    """
    cfg_path = os.path.join(PROJECT_ROOT, "vild", "vild_config.py")
    src = open(cfg_path, encoding="utf-8").read()
    m = re.search(r'mark_version\s*==\s*["\']mark5\.0["\'].*?self\.classes\s*=\s*\[(.*?)\]',
                  src, re.S)
    if not m:
        raise SystemExit(f"[ERROR] {cfg_path} 에서 mark5.0 클래스 정의를 못 찾았습니다.")
    return re.findall(r'["\']([A-Za-z_]+)["\']', m.group(1))


def split_quota(total, classes):
    """split 총량을 클래스 수로 나눈다. 끝수는 최대잉여법 — 앞쪽 클래스부터 +1.

    9개 클래스의 소수부가 전부 같으므로(2000/9 = 222.22) 순서가 곧 우선순위가 된다.
    config 의 클래스 순서를 그대로 쓰기 때문에 몇 번을 돌려도 같은 배분이 나온다.
    """
    n = len(classes)
    base, rest = divmod(total, n)
    return {c: base + (1 if i < rest else 0) for i, c in enumerate(classes)}


def scan_aihub(root):
    """해제 폴더를 스캔해 카테고리 목록을 만든다. aihub_slicer.scan_aihub 와 같은 규칙.

    폴더명은 TS_/VS_ 접두어 + 카테고리명이고, 라벨은 TL_/VL_ ..._label 에 있다.
    """
    cats = []
    for d in sorted(os.listdir(root)):
        full = os.path.join(root, d)
        if not os.path.isdir(full) or d.endswith("_label"):
            continue
        vol = d[:2]
        if vol not in ("TS", "VS"):
            continue
        name = d[3:]
        label_dir = os.path.join(root, f"{'TL' if vol == 'TS' else 'VL'}_{name}_label")
        wavs = sorted(f for f in os.listdir(full) if f.endswith(".wav"))
        if not wavs:
            continue
        if not os.path.isdir(label_dir):
            label_dir = None
        cats.append({"volume": vol, "category": name, "dir": full,
                     "label_dir": label_dir, "wavs": wavs})
    return cats


def belongs(category, keywords):
    return any(k in category for k in keywords)


_resampler = None


def slice_resample(x, sr, start, end):
    """창을 잘라 16kHz 로 맞춘다. torchaudio 는 여기서만 필요해서 늦게 import 한다."""
    global _resampler
    seg = x[int(start * sr): int(end * sr)]
    if sr != TARGET_SR:
        import torch
        import torchaudio.functional as AF
        if _resampler is None:
            _resampler = (torch, AF)
        torch, AF = _resampler
        t = torch.from_numpy(seg).unsqueeze(0)
        seg = AF.resample(t, sr, TARGET_SR).squeeze(0).numpy()
    return seg.astype(np.float32)


def pick_window_aihub(wav_path, label_path):
    """AI Hub 클립에서 3초 창을 고른다. aihub_slicer.pick_window 와 같은 규칙.

    가장 긴 annotation 의 중앙을 먼저 쓰고, 그 창의 RMS 가 MIN_SEG_RMS 미만이면 다음
    annotation 으로 넘어간다. 전부 미달이면 클립 중앙을 쓰되 품질 게이트가 다시 거른다.
    """
    info = sf.info(wav_path)
    dur = info.frames / info.samplerate
    anns = []
    if label_path and os.path.exists(label_path):
        try:
            with open(label_path, encoding="utf-8") as f:
                anns = json.load(f).get("annotation", [])
            anns = sorted(anns, key=lambda a: a["endTime"] - a["startTime"], reverse=True)
        except Exception:
            anns = []

    data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    x = data.mean(axis=1)

    def window_at(center):
        start = min(max(center - SEG_SEC / 2, 0.0), max(dur - SEG_SEC, 0.0))
        return start, min(start + SEG_SEC, dur)

    candidates = [window_at((a["startTime"] + a["endTime"]) / 2) for a in anns]
    candidates.append(window_at(dur / 2))
    texts = [a.get("labelText", "") for a in anns] + [""]

    for (start, end), text in zip(candidates, texts):
        seg = x[int(start * sr): int(end * sr)]
        if len(seg) and float(np.sqrt(np.mean(seg ** 2))) >= MIN_SEG_RMS:
            return x, sr, start, end, text
    start, end = candidates[0]
    return x, sr, start, end, texts[0]


def read_window_sins(path, start_sec, remove_dc=True):
    """SINS wav 에서 start_sec 부터 3초를 ch0 만 읽는다. sins_slicer.read_window 와 같다.

    노드마다 고유한 DC 바이어스가 있어(DevNode1 -0.0297 / 2 +0.0156 / 3 -0.0040 / 4 +0.0427)
    그대로 두면 멜 최저 대역에 큰 에너지로 들어간다. 클래스와는 무관하므로 지름길은 아니지만
    소리가 아닌 성분이라 뺀다.
    """
    info = sf.info(path)
    if info.samplerate != TARGET_SR:
        raise ValueError(f"샘플레이트 {info.samplerate} != {TARGET_SR}")
    beg = int(round(start_sec * TARGET_SR))
    end = beg + int(round(SEG_SEC * TARGET_SR))
    if end > info.frames:
        raise ValueError(f"길이부족 frames={info.frames} < {end}")
    x, _ = sf.read(path, start=beg, stop=end, dtype="float32", always_2d=True)
    seg = np.ascontiguousarray(x[:, 0])
    if remove_dc:
        seg = seg - np.float32(seg.mean())
    return seg


def read_window_chime(path, start_sec, remove_dc=True):
    """CHiME-Home 청크에서 start_sec 부터 3초를 읽는다. chime_slicer.read_window 와 같다.

    16kHz mono 판을 전제한다. 48kHz stereo 판을 잘못 넘기면 예외로 막는다(리샘플하지 않는다 —
    다른 출처와 전처리를 같게 두려면 원본이 16kHz 여야 한다).
    """
    info = sf.info(path)
    if info.samplerate != TARGET_SR:
        raise ValueError(f"샘플레이트 {info.samplerate} != {TARGET_SR} (16kHz 판을 쓰십시오)")
    beg = int(round(start_sec * TARGET_SR))
    end = beg + int(round(SEG_SEC * TARGET_SR))
    if end > info.frames:
        raise ValueError(f"길이부족 frames={info.frames} < {end}")
    x, _ = sf.read(path, start=beg, stop=end, dtype="float32", always_2d=True)
    seg = np.ascontiguousarray(x[:, 0])
    if remove_dc:
        seg = seg - np.float32(seg.mean())
    return seg


def load_manifest(path, what):
    """계획 CSV -> (split, role) 별 목록. 열: split,role,wav,activity,sess,idx,node

    sins_manifest.csv 와 chime_manifest.csv 가 같은 열 구조라 한 함수로 읽는다.
    """
    need = {"split", "role", "wav", "activity", "sess", "idx", "node"}
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        raise SystemExit(f"[ERROR] {what} manifest 가 비어 있습니다: {path}")
    missing = need - set(rows[0])
    if missing:
        raise SystemExit(f"[ERROR] {what} manifest 에 없는 열: {sorted(missing)}")
    out = defaultdict(list)
    for r in rows:
        out[(r["split"], r["role"])].append(r)
    return out


def build_queue(items, want):
    """활동·조합별 쿼터를 정한 뒤, 각 묶음 안에서 **세션을 번갈아** 꺼내 대기열을 만든다.

    [2026-08-09 신설] 쿼터만으로는 세션이 안 섞인다. media_talking 은 SINS 활동이
    watching_tv 하나, CHiME 조합이 v 하나뿐이라 세션 배분이 아예 일어나지 않아, 목록 앞
    세션에서 필요량이 다 채워지고 뒤 세션은 한 클래스도 못 낸다. mark4.5 1차 수집에서
    test 세션 하나가 타겟 0 / others 55 가 되어 그 세션이 곧 others 가 되는 일이 실제로
    벌어졌다(고친 뒤 타겟 52 / others 74).

    돌려주는 값은 (queue, leftover) 이고, queue 를 다 쓰면 leftover 로 넘어간다.
    """
    q = activity_quota(items, want)
    by_act = defaultdict(list)
    for r in items:
        by_act[r["activity"]].append(r)
    queue, leftover, rest_by_act = [], [], []
    for a in sorted(by_act):
        want_a = q.get(a, 0)
        by_sess = defaultdict(list)
        for r in by_act[a]:
            by_sess[r["sess"]].append(r)
        picked = []
        while len(picked) < want_a and any(by_sess.values()):
            for s in sorted(by_sess):
                if by_sess[s] and len(picked) < want_a:
                    picked.append(by_sess[s].pop(0))
        queue.extend(picked)
        rest_by_act.append([r for s in sorted(by_sess) for r in by_sess[s]])
    # [2026-08-09 수정] 예비 목록도 활동을 번갈아 꺼낸다. 전에는 활동 순서대로 이어붙여서
    # pop(0) 이 사전순 첫 활동만 소진했다(take_aihub 에서 카테고리로 같은 문제를 고쳤던 것).
    while any(rest_by_act):
        for lst in rest_by_act:
            if lst:
                leftover.append(lst.pop(0))
    return queue, leftover


def activity_quota(items, want):
    """SINS 칸 하나를 활동별로 쪼갠다. 비율은 manifest 의 활동별 개수 비율. 끝수는 최대잉여법.

    sins_slicer 와 같은 함수다. 앞에서부터 꺼내 쓰면 매니페스트가 활동 순서로 이어붙은
    목록이라 앞쪽 활동에서 쿼터가 다 차 버린다(2026-08-08 에 vacuum_cleaner 4개 / eating 0개가
    나왔던 버그).
    """
    counts = defaultdict(int)
    for r in items:
        counts[r["activity"]] += 1
    total = sum(counts.values())
    if total == 0:
        return {}
    raw = {a: want * c / total for a, c in counts.items()}
    quota = {a: int(v) for a, v in raw.items()}
    rest = want - sum(quota.values())
    for a in sorted(raw, key=lambda k: (-(raw[k] - quota[k]), k))[:rest]:
        quota[a] += 1
    return quota


def distribute_evenly(total, keys):
    """total 을 카테고리들에 균등 배분. 끝수는 앞쪽부터 +1(무작위를 쓰지 않아 재현된다)."""
    n = len(keys)
    if n == 0:
        return {}
    base, rest = divmod(total, n)
    return {k: base + (1 if i < rest else 0) for i, k in enumerate(sorted(keys))}


def out_dir(split, cls):
    return os.path.join(PROJECT_ROOT, SPLIT_DIRS[split], cls)


# ── unlabeled 모드 ────────────────────────────────────────────────────────────
# [신설 2026-08-09] mark5 는 반지도 구조다. 학습 루프가 배치를 둘로 갈라
#     labeled   -> student 예측 vs 정답 라벨          -> hard loss
#     unlabeled -> student 예측 vs 8개 teacher 판정   -> soft loss + feature KD
# 로 쓰는데, **unlabeled 가 0개면 `if unlabeled_indices:` 블록이 한 번도 실행되지 않아
# ensemble_teacher 가 아예 호출되지 않는다.** 즉 Dynamic Online Ensemble KD 가 통째로
# 빠지고 순수 supervised 학습이 된다(2026-08-09 실측으로 확인).
#
# 그래서 labeled 와 별개로 unlabeled 를 수집한다. 핵심 규칙 세 가지:
#   (1) **라벨을 붙이지 않는다.** 클래스 폴더 없이 평평하게 저장한다
#       (data_source_unlabeled/unlabeled_NNNN.wav). teacher 가 판정할 재료이므로 정답이 없다.
#   (2) **labeled train 과 같은 클래스 분배로 뽑는다.** 8개 teacher 가 각자 전문 영역의 재료를
#       고르게 받아야 증류 신호가 한쪽으로 쏠리지 않는다. AI Hub 만 뽑으면 media_talking
#       전문가가 놀고, SINS·CHiME 만 뽑으면 나머지 7명이 논다.
#   (3) **val·test 에는 넣지 않는다.** 정답이 없어 채점에 쓸 수 없다. 그래서 labeled 의
#       70:15:15(2000/430/430) 비율은 그대로 유지된다 — unlabeled 는 별개 축이다.
UNLABELED_DIR = "data_source_unlabeled"


def unlabeled_out_path(idx):
    return os.path.join(PROJECT_ROOT, UNLABELED_DIR, f"unlabeled_{idx:04d}.wav")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aihub_root", type=str,
                    default=os.path.expanduser("~/workspace/aihub_raw/71296/extracted"))
    ap.add_argument("--sins_root", type=str,
                    default=os.path.expanduser("~/workspace/sins_raw"))
    ap.add_argument("--sins_manifest", type=str,
                    default=os.path.join(PROJECT_ROOT, "..", "Mark4.5",
                                         "preprocessing", "sins_manifest.csv"))
    ap.add_argument("--chime_chunk_dir", type=str,
                    default=os.path.expanduser("~/workspace/chime_home_raw/chime_home/chunks"))
    ap.add_argument("--chime_manifest", type=str,
                    default=os.path.join(PROJECT_ROOT, "..", "Mark4.5",
                                         "preprocessing", "chime_manifest.csv"))
    ap.add_argument("--provenance_path", type=str,
                    default=os.path.join(PROJECT_ROOT, "..", "data_provenance.xlsx"))
    ap.add_argument("--others_house_ratio", type=float, default=0.5,
                    help="others 안에서 집 녹음(SINS+CHiME)이 차지할 비율(나머지는 AI Hub 15종). "
                         "높이면 채널 단서가 줄지만 others 의 AI Hub 다양성이 준다")
    ap.add_argument("--mode", type=str, default="labeled", choices=["labeled", "unlabeled"],
                    help="labeled: train/val/test 를 클래스 폴더에 수집(기본). "
                         "unlabeled: 라벨 없이 data_source_unlabeled 에 평평하게 수집(증류용)")
    ap.add_argument("--unlabeled_total", type=int, default=2000,
                    help="--mode unlabeled 일 때 수집할 개수. 기본 2000 = labeled train 과 1:1. "
                         "val·test 에는 넣지 않으므로 labeled 의 70:15:15 비율은 그대로 유지된다")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--no_quality_check", action="store_true")
    ap.add_argument("--keep_dc", action="store_true",
                    help="SINS·CHiME 의 DC 오프셋을 빼지 않는다(권장하지 않음)")
    args = ap.parse_args()

    classes = classes_from_config()
    if len(classes) != 9:
        raise SystemExit(f"[ERROR] mark5.0 클래스가 9개가 아닙니다: {classes}")
    prov_path = os.path.abspath(args.provenance_path)
    aihub_root = os.path.abspath(os.path.expanduser(args.aihub_root))
    sins_root = os.path.abspath(os.path.expanduser(args.sins_root))

    quota = {sp: split_quota(SPLIT_TOTALS[sp], classes) for sp in SPLIT_TOTALS}
    print(f"[INFO] mark_version={MARK_VERSION}  클래스 {len(classes)}개: {classes}")
    print("[INFO] split 배분(최대잉여법):")
    for sp in ("train", "val", "test"):
        print(f"   {sp:5s} 합계 {SPLIT_TOTALS[sp]:4d} = "
              + ", ".join(f"{c} {quota[sp][c]}" for c in classes))

    # media_talking 을 SINS / CHiME 으로 반씩 나눈다(두 집에서 절반씩).
    media_split = {}
    for sp in ("train", "val", "test"):
        want = quota[sp]["media_talking"]
        n_sins = want // 2
        media_split[sp] = (n_sins, want - n_sins)
    print("[INFO] media_talking 내부 배분: "
          + ", ".join(f"{sp} SINS {a} + CHiME {b}" for sp, (a, b) in media_split.items()))

    # others 를 (SINS + CHiME) / AI Hub 로 쪼개고, 집 몫을 다시 반씩 나눈다.
    others_split = {}
    for sp in ("train", "val", "test"):
        want = quota[sp]["others"]
        n_house = int(round(want * args.others_house_ratio))
        n_sins = n_house // 2
        others_split[sp] = (n_sins, n_house - n_sins, want - n_house)
    print("[INFO] others 내부 배분: "
          + ", ".join(f"{sp} SINS {a} + CHiME {b} + AI Hub {c}"
                      for sp, (a, b, c) in others_split.items()))

    # 채널 단서가 얼마나 남는지 미리 계산해 보여준다(9-class 기저확률은 1/9 = 11.1%).
    n_media_house = sum(sum(media_split[sp]) for sp in SPLIT_TOTALS)
    n_others_house = sum(others_split[sp][0] + others_split[sp][1] for sp in SPLIT_TOTALS)
    print(f"[INFO] ⚠집 녹음(SINS·CHiME) 총 {n_media_house + n_others_house}개 중 "
          f"media_talking {n_media_house} / others {n_others_house} — "
          f"채널을 감지했을 때 media_talking 일 확률 "
          f"{n_media_house / (n_media_house + n_others_house) * 100:.1f}% "
          f"(9-class 기저 11.1%). AI Hub 에 TV 소리가 없어 구조적으로 남는 단서이며 "
          f"수집 후 진단으로 정량화할 것.")

    # 1) provenance 대조 — 이 버전이 이미 쓴 원본만 본다(2026-08-01 규칙).
    prov = pd.read_excel(prov_path)
    mine = prov[prov["mark_version"] == MARK_VERSION] if "mark_version" in prov.columns else prov
    used_src = set()
    if "aihub_src_file" in mine.columns:
        used_src = set(mine.loc[mine["aihub_src_file"].notna(), "aihub_src_file"].astype(str))
    print(f"[INFO] provenance 전체 {len(prov)}행 중 {MARK_VERSION} {len(mine)}행, "
          f"이미 쓴 원본 {len(used_src)}개")

    # 2) 원본 풀 구성
    cats = scan_aihub(aihub_root)
    pool_aihub = defaultdict(list)      # 클래스 -> [(카테고리, dir, label_dir, wav)]
    for c in cats:
        wavs = [w for w in c["wavs"] if w not in used_src]
        for cls, keys in AIHUB_MAP.items():
            if belongs(c["category"], keys):
                pool_aihub[cls] += [(c["category"], c["dir"], c["label_dir"], w) for w in wavs]
        if belongs(c["category"], OTHERS_AIHUB):
            pool_aihub["others"] += [(c["category"], c["dir"], c["label_dir"], w) for w in wavs]

    manifest = load_manifest(os.path.abspath(os.path.expanduser(args.sins_manifest)), "SINS")
    pool_sins = {}
    for (sp, role), items in manifest.items():
        d = os.path.join(sins_root, sp, role)
        pool_sins[(sp, role)] = [r for r in items
                                 if os.path.exists(os.path.join(d, r["wav"]))]

    chime_dir = os.path.abspath(os.path.expanduser(args.chime_chunk_dir))
    manifest_ch = load_manifest(os.path.abspath(os.path.expanduser(args.chime_manifest)), "CHiME")
    pool_chime = {}
    for (sp, role), items in manifest_ch.items():
        pool_chime[(sp, role)] = [r for r in items
                                  if os.path.exists(os.path.join(chime_dir, r["wav"]))]

    # 3) 계획 점검 — 부족하면 여기서 멈춘다.
    print("\n[계획] 클래스별 필요량 대 원본 가용량")
    shortage = []
    for cls in classes:
        need_total = sum(quota[sp][cls] for sp in ("train", "val", "test"))
        if cls == "media_talking":
            det = []
            for sp in ("train", "val", "test"):
                n_s, n_c = media_split[sp]
                a_s = len(pool_sins.get((sp, "target"), []))
                a_c = len(pool_chime.get((sp, "target"), []))
                det.append(f"{sp} SINS {n_s}<={a_s} / CHiME {n_c}<={a_c}")
                if a_s < n_s:
                    shortage.append((cls + "(SINS)", sp, n_s, a_s))
                if a_c < n_c:
                    shortage.append((cls + "(CHiME)", sp, n_c, a_c))
            print(f"   {cls:14s} SINS + CHiME 절반씩      필요 {need_total:4d}")
            for line in det:
                print(f"                    {line}")
        elif cls == "others":
            ok = True
            for sp in SPLIT_TOTALS:
                n_sins, n_chime, _ = others_split[sp]
                a_s = len(pool_sins.get((sp, "others"), []))
                a_c = len(pool_chime.get((sp, "others"), []))
                if a_s < n_sins:
                    shortage.append((cls + "(SINS)", sp, n_sins, a_s)); ok = False
                if a_c < n_chime:
                    shortage.append((cls + "(CHiME)", sp, n_chime, a_c)); ok = False
            n_aihub_total = sum(others_split[sp][2] for sp in SPLIT_TOTALS)
            if len(pool_aihub["others"]) < n_aihub_total:
                shortage.append((cls + "(AI Hub)", "전체", n_aihub_total, len(pool_aihub["others"])))
                ok = False
            print(f"   {cls:14s} 집 녹음 + AI Hub 15종    필요 {need_total:4d}  "
                  f"[SINS {sum(others_split[sp][0] for sp in SPLIT_TOTALS)} / "
                  f"CHiME {sum(others_split[sp][1] for sp in SPLIT_TOTALS)} / "
                  f"AI Hub {n_aihub_total}<={len(pool_aihub['others'])}]  {'OK' if ok else '부족'}")
        else:
            avail = len(pool_aihub[cls])
            if avail < need_total:
                shortage.append((cls, "전체", need_total, avail))
            n_cat = len({c for c, _, _, _ in pool_aihub[cls]})
            print(f"   {cls:14s} AI Hub {n_cat:2d}종            필요 {need_total:4d}  "
                  f"가용 {avail:5d}  {'OK' if avail >= need_total else '부족'}")

    if shortage:
        print("\n[ERROR] 원본이 부족합니다:")
        for cls, sp, want, have in shortage:
            print(f"  - {cls} / {sp}: 필요 {want}, 있음 {have}")
        sys.exit(1)

    if args.mode == "unlabeled":
        un_q = split_quota(args.unlabeled_total, classes)
        print(f"\n[계획/unlabeled] 총 {args.unlabeled_total}개 — 라벨 없이 "
              f"{UNLABELED_DIR}/unlabeled_NNNN.wav 로 평평하게 저장")
        print("  labeled train 과 같은 클래스 분배로 뽑되 라벨은 붙이지 않는다"
              "(teacher 8명이 각자 전문 영역의 재료를 고르게 받도록).")
        print("  SINS·CHiME 은 train split 에서만 꺼낸다 — val·test 세션에서 뽑으면 라벨이 없어도"
              " 그 세션 소리가 학습에 들어가 누수가 된다.")
        for cls in classes:
            want = un_q[cls]
            if cls == "media_talking":
                a = want // 2
                av_s = len(pool_sins.get(("train", "target"), []))
                av_c = len(pool_chime.get(("train", "target"), []))
                ok = "OK" if (av_s >= a and av_c >= want - a) else "부족"
                print(f"   {cls:14s} {want:4d} = SINS {a}<={av_s} + CHiME {want-a}<={av_c}   {ok}")
            elif cls == "others":
                n_house = int(round(want * args.others_house_ratio))
                a = n_house // 2
                av_s = len(pool_sins.get(("train", "others"), []))
                av_c = len(pool_chime.get(("train", "others"), []))
                av_a = len(pool_aihub["others"])
                ok = ("OK" if (av_s >= a and av_c >= n_house - a and av_a >= want - n_house)
                      else "부족")
                print(f"   {cls:14s} {want:4d} = SINS {a}<={av_s} + CHiME {n_house-a}<={av_c}"
                      f" + AI Hub {want-n_house}<={av_a}   {ok}")
            else:
                av = len(pool_aihub[cls])
                print(f"   {cls:14s} {want:4d} = AI Hub {want}<={av}   "
                      f"{'OK' if av >= want else '부족'}")
        print("  ⚠val·test 에는 넣지 않으므로 labeled 의 70:15:15(2000/430/430) 는 그대로 유지된다.")

    if args.dry_run:
        print("\n[DRY RUN] 추출은 하지 않고 종료합니다.")
        return

    # 4) 추출
    backup = prov_path + f".bak_before_mark5_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(prov_path, backup)
    print(f"\n[INFO] provenance 백업: {os.path.basename(backup)}")
    if "source" not in prov.columns:
        prov["source"] = "fsd50k"

    new_rows, rejected, unfilled = [], [], []
    unlabeled_counter = [0]     # unlabeled 파일 일련번호(리스트로 둬 중첩 함수에서 갱신)
    used_clip = set(used_src)          # mark5 안에서 한 원본 클립은 한 번만
    cat_used, act_used = defaultdict(int), defaultdict(int)
    t0 = time.time()
    todo = sum(SPLIT_TOTALS.values())
    done = 0

    def save(seg, cls, sp, idx, rec):
        """3초 파형을 저장하고 provenance 행을 만든다.

        [변경 2026-08-09] unlabeled 모드(sp == "unlabeled")에서는 클래스 폴더를 만들지 않고
        평평하게 저장한다. 라벨이 없는 데이터라 클래스별로 나눌 근거가 없고, 학습 코드도
        dataset_unlabeled.csv 를 path 한 열로만 읽는다(generate_train_index 의 unlabeled 분기).
        어느 카테고리에서 뽑았는지는 provenance 의 aihub_category 에 남으므로 추적은 된다.
        """
        if sp == "unlabeled":
            unlabeled_counter[0] += 1
            path = unlabeled_out_path(unlabeled_counter[0])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            name = os.path.basename(path)
            sf.write(path, seg, TARGET_SR, subtype="FLOAT")
            new_rows.append({
                "local_filename": name,
                "original_labels": rec["category"],
                # target_class 는 '뽑아온 영역'을 기록해 둔다(라벨이 아니라 추적용).
                # 학습에서는 dataset_unlabeled.csv 에 path 만 들어가므로 이 값이 쓰이지 않는다.
                "target_class": f"unlabeled:{cls}", "assigned_split": "unlabeled",
                "mark_version": MARK_VERSION,
                "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
                "size_bytes": os.path.getsize(path),
                "download_date": date.today().isoformat(), "source_type": "original",
                "removed_20260715": "active",
                "source": rec["source"],
                "aihub_src_file": rec["wav"],
                "aihub_category": rec["category"],
                "aihub_volume": rec["volume"],
                "seg_start_sec": round(rec["start"], 3),
                "seg_end_sec": round(rec["start"] + SEG_SEC, 3),
            })
            return
        d = out_dir(sp, cls)
        os.makedirs(d, exist_ok=True)
        name = f"{cls}_{sp}_{idx:03d}.wav"
        path = os.path.join(d, name)
        sf.write(path, seg, TARGET_SR, subtype="FLOAT")
        new_rows.append({
            "local_filename": name,
            "original_labels": rec["category"],
            "target_class": cls, "assigned_split": sp,
            "mark_version": MARK_VERSION,
            "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
            "size_bytes": os.path.getsize(path),
            "download_date": date.today().isoformat(), "source_type": "original",
            "removed_20260715": "active",
            "source": rec["source"],
            "aihub_src_file": rec["wav"],
            "aihub_category": rec["category"],
            "aihub_volume": rec["volume"],
            "seg_start_sec": round(rec["start"], 3),
            "seg_end_sec": round(rec["start"] + SEG_SEC, 3),
        })

    def take_aihub(cls, sp, want, pool, idx0, save_sp=None):
        """AI Hub 풀에서 want 개를 뽑아 저장한다. 카테고리 균등 배분 + 예비도 카테고리 라운드로빈.

        [2026-08-09 수정] 두 가지를 고쳤다.

        (1) **쿼터를 계산하기 전에 이미 쓴 클립을 거른다.** 전에는 원본 풀을 그대로 넘겨
            카테고리별 앞부분을 겨냥했는데, split 을 val -> test -> train 순으로 처리하므로
            test·train 에서는 그 앞부분이 전부 used_clip 이라 통째로 skip 되고 예비 목록으로
            넘어갔다.
        (2) **예비 목록도 카테고리를 번갈아 꺼낸다.** 전에는 카테고리 순서대로 이어붙여서
            pop(0) 이 앞 카테고리만 소진했다.

        둘이 겹쳐서 실제로 이렇게 쏠렸다(2026-08-09 1차 수집 실측):
            heavy_impact  319 = 어른발걸음 171 / 아이들발걸음 74 / 망치질 74   (균등이면 106씩)
            construction  318 = 항타기 95 / 항발기 41 / 착암기 19 / ...        (균등이면 26씩)
        어른발걸음 171 = 16(val) + 48(test) + 58 + 49(train) 로 계산이 정확히 맞았다.
        클래스 안의 대표성이 무너지면 클래스별 성능이 "그 클래스"가 아니라 "그 클래스의 우세
        카테고리" 성능이 된다.
        """
        nonlocal done
        pool = [item for item in pool if item[3] not in used_clip]
        by_cat = defaultdict(list)
        for item in pool:
            by_cat[item[0]].append(item)
        q = distribute_evenly(want, list(by_cat))
        queue, rest_by_cat = [], []
        for c in sorted(by_cat):
            queue.extend(by_cat[c][:q.get(c, 0)])
            rest_by_cat.append(by_cat[c][q.get(c, 0):])
        leftover = []
        while any(rest_by_cat):
            for lst in rest_by_cat:
                if lst:
                    leftover.append(lst.pop(0))
        idx, made = idx0, 0
        while made < want and (queue or leftover):
            cat, d, ld, w = queue.pop(0) if queue else leftover.pop(0)
            if w in used_clip:
                continue
            wav_path = os.path.join(d, w)
            label_path = os.path.join(ld, os.path.splitext(w)[0] + ".json") if ld else None
            try:
                x, sr, start, end, _ = pick_window_aihub(wav_path, label_path)
                seg = slice_resample(x, sr, start, end)
            except Exception as e:
                rejected.append((w, cat, f"읽기실패 {e}"))
                continue
            bad = (None if args.no_quality_check
                   else check_waveform(seg, TARGET_SR, min_duration=SEG_SEC))
            if bad is not None:
                rejected.append((w, cat, bad))
                continue
            save(seg, cls, save_sp or sp, idx,
                 {"category": cat, "wav": w, "volume": os.path.basename(d)[:2],
                  "start": start, "source": "aihub"})
            used_clip.add(w)
            cat_used[cat] += 1
            idx += 1
            made += 1
            done += 1
        if made < want:
            unfilled.append((sp, cls, "aihub", want, made))
        return idx

    def take_chime(cls, sp, role, want, idx0, save_sp=None):
        """CHiME 풀에서 want 개를 뽑아 저장한다. 조합별 쿼터 + 세션 라운드로빈 + 창 3순위."""
        nonlocal done
        # [2026-08-09 수정] take_sins 와 같은 이유로 이미 쓴 클립을 먼저 거른다.
        items = [r for r in pool_chime.get((sp, role), []) if r["wav"] not in used_clip]
        queue, leftover = build_queue(items, want)
        idx, made = idx0, 0
        while made < want and (queue or leftover):
            rec = queue.pop(0) if queue else leftover.pop(0)
            if rec["wav"] in used_clip:
                continue
            got = None
            for start in CHIME_WINDOW_STARTS:
                try:
                    seg = read_window_chime(os.path.join(chime_dir, rec["wav"]), start,
                                            remove_dc=not args.keep_dc)
                except Exception as e:
                    rejected.append((rec["wav"], rec["activity"], f"읽기실패 {e}"))
                    continue
                bad = (None if args.no_quality_check
                       else check_waveform(seg, TARGET_SR, min_duration=SEG_SEC))
                if bad is None:
                    got = (start, seg)
                    break
                rejected.append((rec["wav"], rec["activity"], bad))
            if got is None:
                continue
            start, seg = got
            save(seg, cls, save_sp or sp, idx,
                 {"category": rec["activity"], "wav": rec["wav"],
                  "volume": rec["sess"], "start": start, "source": "chime_home"})
            used_clip.add(rec["wav"])
            act_used[f"chime:{rec['activity']}"] += 1
            idx += 1
            made += 1
            done += 1
        if made < want:
            unfilled.append((sp, cls, "chime", want, made))
        return idx

    def take_sins(cls, sp, role, want, idx0, save_sp=None):
        """SINS 풀에서 want 개를 뽑아 저장한다. 활동별 쿼터 + 세션 라운드로빈 + 창 3순위 재시도."""
        nonlocal done
        # [2026-08-09 수정] 이미 쓴 클립을 **쿼터 계산 전에** 거른다. take_aihub 와 같은 버그가
        # 여기 남아 있었다 — queue 항목이 전부 used_clip 이면 통째로 skip 되고 예비 목록(활동
        # 사전순으로 이어붙인 것)의 첫 활동만 채워진다. labeled 수집은 provenance 를 비우고
        # 시작해 드러나지 않았고, unlabeled(labeled 2,860개 위에 얹음)에서 SINS others 55개가
        # 전부 cooking 으로 쏠려 발견됐다.
        items = [r for r in pool_sins.get((sp, role), []) if r["wav"] not in used_clip]
        d = os.path.join(sins_root, sp, role)
        queue, leftover = build_queue(items, want)
        idx, made = idx0, 0
        while made < want and (queue or leftover):
            rec = queue.pop(0) if queue else leftover.pop(0)
            if rec["wav"] in used_clip:
                continue
            got = None
            for start in SINS_WINDOW_STARTS:
                try:
                    seg = read_window_sins(os.path.join(d, rec["wav"]), start,
                                           remove_dc=not args.keep_dc)
                except Exception as e:
                    rejected.append((rec["wav"], rec["activity"], f"읽기실패 {e}"))
                    continue
                bad = (None if args.no_quality_check
                       else check_waveform(seg, TARGET_SR, min_duration=SEG_SEC))
                if bad is None:
                    got = (start, seg)
                    break
                rejected.append((rec["wav"], rec["activity"], bad))
            if got is None:
                continue
            start, seg = got
            save(seg, cls, save_sp or sp, idx,
                 {"category": rec["activity"], "wav": rec["wav"],
                  "volume": f"DevNode{rec['node']}", "start": start, "source": "sins"})
            used_clip.add(rec["wav"])
            act_used[rec["activity"]] += 1
            idx += 1
            made += 1
            done += 1
        if made < want:
            unfilled.append((sp, cls, "sins", want, made))
        return idx

    if args.mode == "unlabeled":
        # labeled train 과 같은 클래스 분배로 뽑되 라벨은 붙이지 않는다.
        # SINS·CHiME 은 **train split 에서만** 꺼낸다 — val·test 세션에서 뽑으면 라벨이 없어도
        # 그 세션의 소리가 학습에 들어가 누수가 된다(연속 녹음이라 같은 상황을 담고 있음).
        # AI Hub 는 세션 개념이 없고 클립 단위 중복 방지가 걸려 있어 그대로 쓴다.
        un_q = split_quota(args.unlabeled_total, classes)
        n_s = args.unlabeled_total // len(classes) // 2
        print(f"\n##### unlabeled ({args.unlabeled_total}개) #####", flush=True)
        for cls in classes:
            want = un_q[cls]
            if cls == "media_talking":
                a = want // 2
                nxt = take_sins(cls, "train", "target", a, 1, save_sp="unlabeled")
                take_chime(cls, "train", "target", want - a, nxt, save_sp="unlabeled")
            elif cls == "others":
                n_house = int(round(want * args.others_house_ratio))
                a = n_house // 2
                nxt = take_sins(cls, "train", "others", a, 1, save_sp="unlabeled")
                nxt = take_chime(cls, "train", "others", n_house - a, nxt, save_sp="unlabeled")
                take_aihub(cls, "train", want - n_house, pool_aihub["others"], nxt,
                           save_sp="unlabeled")
            else:
                take_aihub(cls, "train", want, pool_aihub[cls], 1, save_sp="unlabeled")
            print(f"  {cls:14s} {want:3d}개  (누적 {done}/{args.unlabeled_total}, "
                  f"{(time.time()-t0)/60:.1f}분)", flush=True)
    else:
        for sp in ("val", "test", "train"):
            print(f"\n##### {sp} #####", flush=True)
            for cls in classes:
                want = quota[sp][cls]
                if cls == "media_talking":
                    n_sins, n_chime = media_split[sp]
                    nxt = take_sins(cls, sp, "target", n_sins, 1)
                    take_chime(cls, sp, "target", n_chime, nxt)
                elif cls == "others":
                    n_sins, n_chime, n_aihub = others_split[sp]
                    nxt = take_sins(cls, sp, "others", n_sins, 1)
                    nxt = take_chime(cls, sp, "others", n_chime, nxt)
                    take_aihub(cls, sp, n_aihub, pool_aihub["others"], nxt)
                else:
                    take_aihub(cls, sp, want, pool_aihub[cls], 1)
                print(f"  {cls:14s} {want:3d}개  (누적 {done}/{todo}, "
                      f"{(time.time()-t0)/60:.1f}분)", flush=True)

    if new_rows:
        merged = pd.concat([prov, pd.DataFrame(new_rows)], ignore_index=True)
        merged.to_excel(prov_path, index=False)

    print(f"\n[완료] 신규 저장 {len(new_rows)}개 ({(time.time()-t0)/60:.1f}분)")
    made = defaultdict(int)
    for r in new_rows:
        made[(r["assigned_split"], r["target_class"])] += 1
    for sp in ("train", "val", "test"):
        print(f"[{sp}] " + ", ".join(f"{c} {made[(sp, c)]}" for c in classes)
              + f"  합계 {sum(made[(sp, c)] for c in classes)}")
    print("[AI Hub 카테고리별] " + ", ".join(f"{k} {v}" for k, v in
                                       sorted(cat_used.items(), key=lambda kv: -kv[1])))
    print("[SINS 활동별] " + ", ".join(f"{k} {v}" for k, v in
                                    sorted(act_used.items(), key=lambda kv: -kv[1])))
    if rejected:
        by_reason = defaultdict(int)
        for _, _, why in rejected:
            by_reason[str(why)[:30]] += 1
        print(f"[품질탈락] {len(rejected)}회 — "
              + ", ".join(f"{k} {v}" for k, v in sorted(by_reason.items(), key=lambda kv: -kv[1])))
    if unfilled:
        print("[WARN] 못 채운 칸:")
        for sp, cls, src, want, got in unfilled:
            print(f"  - {sp}/{cls}({src}): 필요 {want}, 채움 {got}")

    print("\n다음 단계: generate_dataset_index_mark5.py 로 인덱스를 만들기 전에 "
          "validate_dataset.py 로 전수 검증할 것.")


if __name__ == "__main__":
    main()
