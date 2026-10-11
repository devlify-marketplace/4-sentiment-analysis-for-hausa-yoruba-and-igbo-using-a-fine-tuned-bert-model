"""Data validation helpers shared by download, training, and tests."""
from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

LANGUAGES = {"hau": "Hausa", "yor": "Yoruba", "ibo": "Igbo"}
LABELS = ("negative", "neutral", "positive")
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}


def normalize_label(value: object) -> str:
    """Map common label spellings/IDs to canonical sentiment names."""
    if value is None or pd.isna(value):
        raise ValueError("Sentiment label cannot be empty")
    if isinstance(value, (int, float)) and float(value).is_integer():
        # Hugging Face ClassLabel IDs in NaijaSenti use negative/neutral/positive order.
        value = int(value)
        if value in (0, 1, 2):
            return LABELS[value]
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    aliases = {
        "neg": "negative", "negative": "negative", "negative sentiment": "negative",
        "neu": "neutral", "neutral": "neutral", "neutral sentiment": "neutral",
        "pos": "positive", "positive": "positive", "positive sentiment": "positive",
    }
    if text in aliases:
        return aliases[text]
    if text in {"0", "1", "2"}:
        return LABELS[int(text)]
    raise ValueError(f"Unrecognized sentiment label: {value!r}")


def validate_language(language: str) -> str:
    code = str(language).strip().lower()
    if code not in LANGUAGES:
        raise ValueError(f"Unsupported language {language!r}; choose one of {', '.join(LANGUAGES)}")
    return code


def validate_frame(frame: pd.DataFrame, source: str = "dataset") -> pd.DataFrame:
    """Validate and normalize a split while keeping original text unchanged."""
    required = {"tweet", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")
    out = frame.copy()
    out["tweet"] = out["tweet"].fillna("").astype(str).map(str.strip)
    out["label"] = out["label"].map(normalize_label)
    out = out[out["tweet"].str.len() > 0].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{source} contains no non-empty examples")
    if set(out["label"].unique()) - set(LABELS):
        raise ValueError(f"{source} contains labels outside {LABELS}")
    return out


def stratified_cap(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    """Sample deterministically within labels, approximately preserving class proportions."""
    frame = validate_frame(frame)
    if not limit or limit >= len(frame):
        return frame
    groups = list(frame.groupby("label", sort=True))
    total = len(frame)
    allocations = {label: (limit * len(group)) // total for label, group in groups}
    remainders = {label: (limit * len(group)) % total for label, group in groups}
    remaining = limit - sum(allocations.values())
    for label in sorted(remainders, key=lambda item: (-remainders[item], item))[:remaining]:
        allocations[label] += 1
    pieces = []
    for index, (label, group) in enumerate(groups):
        n = allocations[label]
        if n:
            pieces.append(group.sample(n=n, random_state=seed + index))
    if not pieces:
        return frame.sample(n=limit, random_state=seed).reset_index(drop=True)
    sampled = pd.concat(pieces, ignore_index=True)
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_split(root: str | Path, language: str, split: str) -> pd.DataFrame:
    language = validate_language(language)
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    path = Path(root) / "data" / "raw" / f"{language}_{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run: python scripts/download_data.py --root .")
    return validate_frame(pd.read_csv(path), str(path))
