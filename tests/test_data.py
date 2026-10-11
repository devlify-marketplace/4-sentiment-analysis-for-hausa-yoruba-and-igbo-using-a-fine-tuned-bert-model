import pandas as pd
import pytest

from naija_sentiment.data import LABELS, normalize_label, stratified_cap, validate_frame, validate_language


@pytest.mark.parametrize("raw,expected", [
    ("negative", "negative"), ("NEU", "neutral"), ("Positive sentiment", "positive"),
    (0, "negative"), (1, "neutral"), (2, "positive"),
])
def test_normalize_label(raw, expected):
    assert normalize_label(raw) == expected


def test_unknown_label_rejected():
    with pytest.raises(ValueError):
        normalize_label("mixed")


def test_validate_frame_removes_empty_text_and_normalizes():
    frame = pd.DataFrame({"tweet": ["  hello  ", "  "], "label": ["POS", "negative"]})
    result = validate_frame(frame)
    assert len(result) == 1
    assert result.iloc[0].tweet == "hello"
    assert result.iloc[0].label == "positive"


def test_validate_frame_requires_columns():
    with pytest.raises(ValueError, match="missing required"):
        validate_frame(pd.DataFrame({"text": ["hi"], "label": ["positive"]}))


def test_stratified_cap_is_deterministic_and_preserves_equal_distribution():
    frame = pd.DataFrame({"tweet": [f"t{i}" for i in range(60)], "label": list(LABELS) * 20})
    a = stratified_cap(frame, 12, seed=7)
    b = stratified_cap(frame, 12, seed=7)
    assert len(a) == 12
    assert a.equals(b)
    assert set(a.label) == set(LABELS)
    assert a.label.value_counts().max() - a.label.value_counts().min() <= 1


def test_stratified_cap_preserves_imbalanced_class_proportions():
    labels = ["negative"] * 90 + ["neutral"] * 5 + ["positive"] * 5
    frame = pd.DataFrame({"tweet": [f"t{i}" for i in range(100)], "label": labels})
    sampled = stratified_cap(frame, 20, seed=12)
    assert len(sampled) == 20
    assert sampled.label.value_counts().to_dict() == {"negative": 18, "neutral": 1, "positive": 1}


@pytest.mark.parametrize("code", ["hau", "yor", "ibo"])
def test_supported_languages(code):
    assert validate_language(code) == code


def test_unsupported_language_rejected():
    with pytest.raises(ValueError):
        validate_language("pcm")
