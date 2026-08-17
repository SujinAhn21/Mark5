"""
shared_vild/checkpoint_utils.py
Mark4.x / Mark5.0 공통 Checkpoint 저장/로드 유틸리티.

저장 스키마:
  schema_version    : int
  model_type        : str  ("teacher_encoder", "teacher_classifier", "student_full", ...)
  mark_version      : str  ("mark4.1", "mark5.0", ...)
  config_metadata   : dict (encoder_type, visual_view_type, num_input_channels, segment 설정 등)
  model_state_dict  : OrderedDict
  branch_state_dict : OrderedDict | None
  classifier_state_dict : OrderedDict | None
  background_state_dict : OrderedDict | None
  text_embeddings   : Tensor | None
"""

import os
import torch

SCHEMA_VERSION = 1

_CONFIG_KEYS = [
    "encoder_type",
    "visual_view_type",
    "num_input_channels",
    "embedding_dim",
    "segment_length",
    "segment_hop",
    "max_segments",
    "n_mels",
    "sample_rate",
    "fft_size",
    "hop_length",
]


def extract_config_metadata(config) -> dict:
    meta = {}
    for key in _CONFIG_KEYS:
        val = getattr(config, key, None)
        if val is not None:
            meta[key] = val
    return meta


def apply_config_metadata(config, ckpt: dict):
    """체크포인트에 저장된 config 값을 config 객체에 되돌려 적용한다.

    [수정 2026-08-17] setter 가 없는 @property 는 건너뛴다.
    num_input_channels 는 AudioViLDConfig 에서 visual_view_type 으로부터 계산되는
    읽기 전용 property 인데, extract_config_metadata 가 _CONFIG_KEYS 에 그것을 포함해
    저장해 두기 때문에 setattr 에서 AttributeError 로 죽었다
    ("property 'num_input_channels' ... has no setter").
    이 함수는 mark5 의 EnsembleTeacher 에서만 호출되므로 mark4.x 에서는 드러나지 않았고,
    mark5.0 학습을 처음 돌린 2026-08-17 에 teacher 8개를 로드하는 첫 줄에서 터졌다.

    파생값을 건너뛰어도 잃는 정보는 없다. 계산의 근거인 visual_view_type 이 같은
    metadata 에 들어 있어 함께 적용되고, 그러면 property 가 같은 값을 다시 계산한다
    (teacher 8개 전부 visual_view_type='mel_delta' / num_input_channels=3 으로 확인).
    """
    meta = ckpt.get("config_metadata", {})
    for key, val in meta.items():
        if not hasattr(config, key):
            continue
        class_attr = getattr(type(config), key, None)
        if isinstance(class_attr, property) and class_attr.fset is None:
            continue
        setattr(config, key, val)


def save_checkpoint(
    path: str,
    model_type: str,
    mark_version: str,
    model_state=None,
    branch_state=None,
    classifier_state=None,
    background_state=None,
    text_embeddings=None,
    config=None,
):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_type": model_type,
        "mark_version": mark_version,
        "config_metadata": extract_config_metadata(config) if config is not None else {},
        "model_state_dict": model_state,
        "branch_state_dict": branch_state,
        "classifier_state_dict": classifier_state,
        "background_state_dict": background_state,
        "text_embeddings": text_embeddings,
    }
    torch.save(payload, path)


def load_checkpoint(path: str, map_location=None) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[checkpoint_utils] Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=map_location)
    if not isinstance(ckpt, dict):
        return {"model_state_dict": ckpt}
    return ckpt


def resolve_state_dict(ckpt: dict, *keys):
    for key in keys:
        if key in ckpt and ckpt[key] is not None:
            return ckpt[key]
    raise KeyError(
        f"[checkpoint_utils] None of the keys {keys} found in checkpoint. "
        f"Available keys: {list(ckpt.keys())}"
    )
