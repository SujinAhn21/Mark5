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
    calibrated = prob_vec.copy()
    others_idx = class_names.index("others")
    top_idx = int(np.argmax(calibrated))
    sorted_idx = np.argsort(calibrated)[::-1]
    top_conf = float(calibrated[top_idx])
    second_conf = float(calibrated[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
    margin = top_conf - second_conf
    entropy = float(-(calibrated * np.log(np.clip(calibrated, 1e-8, 1.0))).sum() / np.log(len(class_names)))

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
        }

    return calibrated, top_idx, {
        "forced_to_others": False,
        "raw_top_conf": top_conf,
        "raw_margin": margin,
        "entropy": entropy,
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
