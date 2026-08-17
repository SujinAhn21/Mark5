"""
shared_vild/prompt_bank.py
Mark4.x / Mark5.0 공통 Prompt Bank 모듈.
prompt_bank.json에서 템플릿과 클래스 동의어를 로드하고,
각 클래스에 대한 prompt 텍스트 목록을 반환한다.
"""

import json
import os

_DEFAULT_TEMPLATES = [
    "a sound of {label} in the room",
    "the audio of {label}",
    "an indoor sound that resembles {label}",
    "a recording of {label}",
]

_DEFAULT_SYNONYMS = {
    "heavy_impact": ["heavy impact", "strong thud", "impact on floor"],
    "dragging": ["dragging", "scraping drag", "object dragging on floor"],
    "construction": ["construction", "construction work", "renovation noise"],
    "machine_noise": ["machine noise", "mechanical humming", "appliance machine sound"],
    "media_talking": ["media talking", "tv speech", "speaker talking audio"],
    "water_toilet": ["toilet water", "toilet flush", "bathroom flush sound"],
    "water_shower": ["shower water", "shower running", "bathroom shower sound"],
    "dog_bark": ["dog bark", "barking dog", "canine bark"],
    "others": ["other sound", "background noise", "non target sound"],
}

_cache = {}


def _load_bank(path):
    """prompt_bank.json 을 읽는다. 없으면 이 파일 위쪽의 하드코딩 기본값으로 넘어간다.

    [수정 2026-08-17] 경로를 줬는데 파일이 없을 때 아무 말 없이 넘어가던 것을 경고하도록 바꿨다.
    하드코딩 기본값은 json 과 10개 클래스 전부 내용이 다르다(특히 others 는
    json ["traffic noise","people talking","household appliance sound"] 대
    하드코딩 ["other sound","background noise","non target sound"] 로 의미가 아예 다르다).
    teacher 는 json 프롬프트로 학습돼 있으므로, 드라이브에 shared_vild/resources/ 가 안 올라간 채
    학습이 돌면 student 만 다른 텍스트 공간을 쓰면서 teacher 와 정렬이 깨진다. 그런데도 지금까지는
    조용히 진행돼서 로그만 봐서는 알 수 없었다.
    """
    if path in _cache:
        return _cache[path]
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _cache[path] = data
        return data
    if path:
        print(
            f"[WARN] prompt_bank.json 을 찾지 못했습니다: {path}\n"
            f"       하드코딩 기본 프롬프트로 대체합니다 — teacher 가 학습에 쓴 프롬프트와 다르므로\n"
            f"       텍스트 정렬이 깨집니다. shared_vild/resources/prompt_bank.json 이 있는지 확인하십시오."
        )
    _cache[path] = {}
    return {}


def get_prompt_templates(path=None):
    data = _load_bank(path)
    return data.get("prompt_templates", _DEFAULT_TEMPLATES)


def get_class_synonyms(path=None):
    data = _load_bank(path)
    return data.get("class_synonyms", _DEFAULT_SYNONYMS)


def get_prompt_texts_for_class(class_name, path=None):
    templates = get_prompt_templates(path)
    synonyms_map = get_class_synonyms(path)
    synonyms = synonyms_map.get(class_name, [class_name.replace("_", " ")])
    prompts = []
    for synonym in synonyms:
        for template in templates:
            prompts.append(template.format(label=synonym))
    return prompts
