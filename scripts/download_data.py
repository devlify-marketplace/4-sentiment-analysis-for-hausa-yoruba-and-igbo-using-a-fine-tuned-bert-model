#!/usr/bin/env python3
"""Download the publisher's official NaijaSenti TSV splits (without remote code)."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

DATASET_ID = "HausaNLP/NaijaSenti-Twitter"
REPOSITORY = "https://github.com/hausanlp/NaijaSenti"
RAW_BASE = "https://raw.githubusercontent.com/hausanlp/NaijaSenti/main/data/annotated_tweets"
LANGUAGES = ("hau", "yor", "ibo")
SPLITS = {"train": "train", "validation": "dev", "test": "test"}


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "naija-sentiment-coursework/1.0"})
    with urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"Unexpected HTTP status {response.status} for {url}")
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--overwrite", action="store_true", help="replace previously downloaded split CSVs")
    args = parser.parse_args()
    raw_dir = args.root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for language in LANGUAGES:
        for split, source_split in SPLITS.items():
            target = raw_dir / f"{language}_{split}.csv"
            if target.exists() and not args.overwrite:
                raise FileExistsError(f"{target} exists; pass --overwrite to replace it")
            source_url = f"{RAW_BASE}/{language}/{source_split}.tsv"
            print(f"Downloading {language}/{source_split} ...", flush=True)
            source_bytes = download(source_url)
            table = pd.read_csv(io.BytesIO(source_bytes), sep="\t", dtype=str, keep_default_na=False)
            if not {"tweet", "label"}.issubset(table.columns):
                raise ValueError(f"Unexpected source schema for {language}/{source_split}: {list(table.columns)}")
            table = table[["tweet", "label"]].copy()
            if table.empty:
                raise ValueError(f"Empty source split at {source_url}")
            table["language"] = language
            table.to_csv(target, index=False, encoding="utf-8")
            files[target.name] = {
                "rows": int(len(table)),
                "sha256": sha256_file(target),
                "bytes": target.stat().st_size,
                "source_split": source_split,
                "source_url": source_url,
                "source_sha256": sha256_bytes(source_bytes),
            }
            print(f"  {target.name}: {len(table):,} rows", flush=True)
    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_version": "1.0.0 (as described in the dataset card)",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repository": REPOSITORY,
        "dataset_card": "https://huggingface.co/datasets/HausaNLP/NaijaSenti-Twitter",
        "download_method": "publisher's raw GitHub TSV files; no remote dataset code executed",
        "languages": list(LANGUAGES),
        "splits": files,
        "license_note": "Dataset card states CC BY-NC-SA 4.0; see docs/DATA_PROVENANCE.md.",
    }
    out = raw_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
