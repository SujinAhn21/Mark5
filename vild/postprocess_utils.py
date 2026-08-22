import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def aggregate_segment_probs(segment_probs, saliency_scores, config):
    probs = np.asarray(segment_probs, dtype=np.float32)
    saliency = np.asarray(saliency_scores, dtype=np.float32)
    if probs.ndim != 2 or len(probs) == 0:
        raise ValueError("segment_probs must be a non-empty [K, C] array")

    if config.segment_aggregation_mode == "mean":
        weights = np.ones(len(probs), dtype=np.float32)
    else:
        confidence = probs.max(axis=1)
        conf_weights = np.power(np.clip(confidence, 1e-6, 1.0), config.segment_confidence_power)
        saliency_norm = saliency / max(float(saliency.max()), 1e-6)
        saliency_weights = np.power(np.clip(saliency_norm, 1e-6, 1.0), config.segment_saliency_power)
        weights = conf_weights * saliency_weights
    weights = weights / max(float(weights.sum()), 1e-6)
    aggregated = (probs * weights[:, None]).sum(axis=0)
    return aggregated, weights


def apply_others_calibration(prob_vec, class_names, config):
    """others 를 최종 결정하는 규칙.

    [변경 2026-08-22] 기본 규칙을 "확신이 낮으면 others"(confidence/margin/entropy 3조건)에서
    "Prob_others 자체가 임계값을 넘으면 others"(direct)로 바꿨다.

    왜 바꿨나. 옛 규칙은 모델이 아무것도 확신하지 못할 때 others 로 보내는 방식이라,
    모델의 전반적인 확신도에 결과가 휘둘린다. 실제로 pseudo-label CE 로 바꾼 뒤 student 의
    Raw 최고확률 평균이 0.4384 에서 0.8047 로 올라가자, 같은 임계값(0.45)에서 강제 건수가
    204건에서 22건으로 줄며 others recall 이 0.574 -> 0.191 로 무너졌다.

    그런데 others 를 알아보는 능력 자체는 오히려 좋아져 있었다. val 430클립 실측:
        others 단독 OvR AUC   기존 KD 0.9754  ->  pseudo-label CE 0.9871
        Prob_others 격차       +0.1917        ->  +0.2224
    즉 정보는 모델 안에 있는데 옛 규칙이 그것을 꺼내 쓰지 못하고 있었다.

    direct 규칙으로 두 모델을 같은 자로 비교하면(val 430클립):
        기존 KD          thr 0.16 -> acc 0.8488 / macroF1 0.8478 / others recall 0.702
        pseudo-label CE  thr 0.14 -> acc 0.8814 / macroF1 0.8807 / others recall 0.787
    thr 0.12~0.16 구간이 평평해(0.8744 / 0.8814 / 0.8721) 값에 예민하지 않다.

    ⚠ 이 임계값은 val 에서 고른 값이라 낙관 편향이 있다. test 로 확인해야 하고,
      재학습하면 확률 분포가 이동하므로 그때마다 다시 스윕해야 한다.
    옛 규칙으로 되돌리려면 config 의 others_decision_mode 를 "confidence" 로 두면 된다.
    """
    calibrated = prob_vec.copy()
    others_idx = class_names.index("others")
    top_idx = int(np.argmax(calibrated))
    sorted_idx = np.argsort(calibrated)[::-1]
    top_conf = float(calibrated[top_idx])
    second_conf = float(calibrated[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
    margin = top_conf - second_conf
    entropy = float(-(calibrated * np.log(np.clip(calibrated, 1e-8, 1.0))).sum() / np.log(len(class_names)))
    prob_others = float(calibrated[others_idx])

    mode = getattr(config, "others_decision_mode", "direct")
    if mode == "direct":
        thr = getattr(config, "others_direct_threshold", 0.14)
        if prob_others >= thr:
            # others 로 판정. 확률 벡터는 others 가 1등이 되도록만 최소 조정한다.
            if top_idx != others_idx:
                calibrated[others_idx] = max(calibrated[others_idx], top_conf + 1e-3)
                calibrated = calibrated / calibrated.sum()
            return calibrated, others_idx, {
                "forced_to_others": top_idx != others_idx,
                "raw_top_conf": top_conf,
                "raw_margin": margin,
                "entropy": entropy,
                "prob_others": prob_others,
            }
        # others 가 아니다. others 열을 빼고 다시 고른다.
        # (Prob_others 가 임계값 미만인데도 argmax 였다면 그 클립은 others 가 아니라고 본 것이므로
        #  나머지 8개 중에서 정해야 앞뒤가 맞는다.)
        masked = calibrated.copy()
        masked[others_idx] = -1.0
        return calibrated, int(np.argmax(masked)), {
            "forced_to_others": False,
            "raw_top_conf": top_conf,
            "raw_margin": margin,
            "entropy": entropy,
            "prob_others": prob_others,
        }

    forced = (
        top_idx != others_idx
        and (
            top_conf < config.others_confidence_threshold
            or margin < config.others_margin_threshold
            or entropy > config.others_entropy_threshold
        )
    )
    if forced:
        calibrated[others_idx] = max(calibrated[others_idx], top_conf + 1e-3)
        calibrated = calibrated / calibrated.sum()
        return calibrated, others_idx, {
            "forced_to_others": True,
            "raw_top_conf": top_conf,
            "raw_margin": margin,
            "entropy": entropy,
            "prob_others": prob_others,
        }

    return calibrated, top_idx, {
        "forced_to_others": False,
        "raw_top_conf": top_conf,
        "raw_margin": margin,
        "entropy": entropy,
        "prob_others": prob_others,
    }


def apply_class_pair_calibration(prob_vec, class_names, config):
    calibrated = prob_vec.copy()
    for pair_key, margin_threshold in getattr(config, "class_pair_margin_overrides", {}).items():
        left_name, right_name = pair_key
        if left_name not in class_names or right_name not in class_names:
            continue
        left_idx = class_names.index(left_name)
        right_idx = class_names.index(right_name)
        margin = abs(float(calibrated[left_idx] - calibrated[right_idx]))
        if margin < margin_threshold:
            avg = (calibrated[left_idx] + calibrated[right_idx]) / 2.0
            calibrated[left_idx] = avg
            calibrated[right_idx] = avg
    calibrated = calibrated / max(float(calibrated.sum()), 1e-6)
    return calibrated


def apply_temporal_smoothing(segment_probs, alpha):
    if len(segment_probs) <= 1:
        return segment_probs
    smoothed = []
    prev = np.asarray(segment_probs[0], dtype=np.float32)
    smoothed.append(prev)
    for cur in segment_probs[1:]:
        cur = np.asarray(cur, dtype=np.float32)
        prev = alpha * prev + (1.0 - alpha) * cur
        prev = prev / max(float(prev.sum()), 1e-6)
        smoothed.append(prev.copy())
    return smoothed


def apply_abstention(prob_vec, class_names, config):
    top_idx = int(np.argmax(prob_vec))
    top_conf = float(prob_vec[top_idx])
    if not getattr(config, "enable_abstention", False):
        return prob_vec, top_idx, False
    if top_conf < config.abstention_confidence_threshold:
        others_idx = class_names.index("others")
        abstained = prob_vec.copy()
        abstained[others_idx] = max(abstained[others_idx], top_conf + 1e-3)
        abstained = abstained / abstained.sum()
        return abstained, others_idx, True
    return prob_vec, top_idx, False


def save_visual_explanation(path, segment_records, segment_probs, segment_weights, class_names, final_prob, final_pred, config, plot_dir):
    if not config.save_visual_explanations:
        return

    explanation_dir = os.path.join(plot_dir, f"explanations_{config.run_tag}")
    os.makedirs(explanation_dir, exist_ok=True)
    order = np.argsort(segment_weights)[::-1][:config.explain_topk_segments]
    fig, axes = plt.subplots(len(order), 1, figsize=(10, 3 * len(order)))
    if len(order) == 1:
        axes = [axes]

    for ax, idx in zip(axes, order):
        record = segment_records[idx]
        seg = record["tensor"]
        if seg.ndim == 4:
            seg = seg.squeeze(0)
        base = seg[0].cpu().numpy() if seg.ndim == 3 else seg.cpu().numpy()
        sns.heatmap(base, ax=ax, cmap="magma", cbar=True)
        pred_idx = int(np.argmax(segment_probs[idx]))
        ax.set_title(
            f"seg#{record['segment_index']} weight={segment_weights[idx]:.3f} "
            f"pred={class_names[pred_idx]} conf={segment_probs[idx][pred_idx]:.3f} "
            f"time={record['start_frame']}:{record['end_frame']}"
        )

    fig.suptitle(
        f"{os.path.basename(path)} | final={class_names[final_pred]} "
        f"| probs={np.array2string(final_prob, precision=3, suppress_small=True)}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(explanation_dir, f"{os.path.splitext(os.path.basename(path))[0]}.png"))
    plt.close(fig)
