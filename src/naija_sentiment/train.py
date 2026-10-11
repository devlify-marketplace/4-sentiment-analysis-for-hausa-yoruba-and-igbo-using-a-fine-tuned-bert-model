"""Fine-tune AfroXLMR on NaijaSenti and evaluate official held-out splits."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .data import LABELS, LABEL_TO_ID, LANGUAGES, load_split, stratified_cap


class EncodedTextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int):
        self.labels = [LABEL_TO_ID[label] for label in frame["label"].tolist()]
        self.encodings = tokenizer(
            frame["tweet"].astype(str).tolist(),
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=True,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: values[index] for key, values in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def collator(tokenizer):
    def batch(rows):
        labels = torch.tensor([row.pop("labels") for row in rows], dtype=torch.long)
        padded = tokenizer.pad(rows, padding=True, return_tensors="pt")
        padded["labels"] = labels
        return padded
    return batch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metric_summary(y_true, y_pred) -> dict:
    ids = list(range(len(LABELS)))
    report = classification_report(
        y_true, y_pred, labels=ids, target_names=list(LABELS), output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=ids, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=ids, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=ids, average="weighted", zero_division=0)),
        "per_class": {label: {k: float(v) if isinstance(v, (int, float, np.floating)) else int(v)
                               for k, v in report[label].items()} for label in LABELS},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ids).tolist(),
        "support": int(len(y_true)),
    }


def evaluate(model, loader, device, language_values=None):
    model.eval()
    targets, predictions, confidences = [], [], []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels").numpy().tolist()
            inputs = {k: v.to(device) for k, v in batch.items()}
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            targets.extend(labels)
            predictions.extend(probs.argmax(axis=1).tolist())
            confidences.extend(probs.max(axis=1).tolist())
    result = metric_summary(targets, predictions)
    result["targets"] = targets
    result["predictions"] = predictions
    result["confidences"] = confidences
    return result


def write_results_markdown(path: Path, metrics: dict, metadata: dict) -> None:
    lines = [
        "# Topic 4 — Evaluation Results", "",
        "> Results are generated from the actual saved-model run. Check the run metadata below; subset runs are not comparable to full-split runs.", "",
        f"- **Model:** `{metadata['model_id']}` fine-tuned for {metadata['epochs_completed']} epoch(s).",
        f"- **Selection:** highest validation macro-F1 (epoch {metadata['best_epoch']}).",
        f"- **Training rows:** {metadata['train_rows']} total; {metadata['max_train_per_language'] or 'all'} cap per language.",
        f"- **Official test rows:** {metrics['overall']['support']} total; subset cap per language: {metadata['max_eval_per_language'] or 'none'}.",
        f"- **Seed:** {metadata['seed']}; **device:** {metadata['device']}; **max length:** {metadata['max_length']}.", "",
        "## Held-out test metrics", "",
        "| Scope | n | Accuracy | Macro precision | Macro recall | Macro-F1 | Weighted-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scope, values in [("All languages", metrics["overall"])] + list(metrics["per_language"].items()):
        name = "All languages" if scope == "All languages" else LANGUAGES[scope]
        lines.append(f"| {name} | {values['support']} | {values['accuracy']:.4f} | {values['macro_precision']:.4f} | {values['macro_recall']:.4f} | {values['macro_f1']:.4f} | {values['weighted_f1']:.4f} |")
    lines += ["", "## Per-class test scores", "", "| Language | Sentiment | Precision | Recall | F1 | Support |", "|---|---|---:|---:|---:|---:|"]
    for lang, values in metrics["per_language"].items():
        for label, row in values["per_class"].items():
            lines.append(f"| {LANGUAGES[lang]} | {label} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1-score']:.4f} | {int(row['support'])} |")
    lines += ["", "## Interpretation", "", "Use macro-F1 and per-language/per-class results to compare coverage; accuracy alone can hide class imbalance. These scores measure only the released NaijaSenti Twitter benchmark under the exact configuration above. They are not expected performance on current social media, private feedback, other domains or every speaker community. Refer to `DATA_PROVENANCE.md` and the project report for limitations.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_artifacts(artifacts: Path, metrics: dict, history: list[dict]) -> None:
    cm = np.asarray(metrics["overall"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS,
           xlabel="Predicted sentiment", ylabel="True sentiment", title="Held-out test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > threshold else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(artifacts / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    languages = list(metrics["per_language"])
    x = np.arange(len(languages))
    scores = [metrics["per_language"][lang]["macro_f1"] for lang in languages]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar([LANGUAGES[lang] for lang in languages], scores, color=["#0b6bcb", "#15a382", "#d99512"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Held-out macro-F1 by language")
    ax.grid(axis="y", alpha=.2)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + .02, f"{score:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(artifacts / "language_metrics.png", dpi=160)
    plt.close(fig)

    if history:
        hist = pd.DataFrame(history)
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ax1.plot(hist["epoch"], hist["train_loss"], marker="o", color="#0b6bcb", label="Train loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Training loss")
        ax2 = ax1.twinx()
        ax2.plot(hist["epoch"], hist["validation_macro_f1"], marker="s", color="#15a382", label="Validation macro-F1")
        ax2.set_ylabel("Validation macro-F1")
        ax1.set_title("Training history")
        fig.tight_layout()
        fig.savefig(artifacts / "training_history.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max-train-per-language", type=int, default=None)
    parser.add_argument("--max-eval-per-language", type=int, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config or (root / "configs" / "default.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    seed_everything(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    train_cap = config["max_train_per_language"] if args.max_train_per_language is None else args.max_train_per_language
    eval_cap = config["max_eval_per_language"] if args.max_eval_per_language is None else args.max_eval_per_language
    if train_cap < 0 or eval_cap < 0:
        raise ValueError("subset caps must be zero (all examples) or positive integers")
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (root / "docs").mkdir(exist_ok=True)
    started = time.time()

    frames = {}
    split_counts = {}
    for split in ("train", "validation", "test"):
        parts = []
        for index, language in enumerate(config["languages"]):
            frame = load_split(root, language, split)
            cap = train_cap if split == "train" else eval_cap
            if cap:
                frame = stratified_cap(frame, cap, seed + index + (0 if split == "train" else 10))
            frame = frame.copy()
            frame["language"] = language
            parts.append(frame)
            split_counts[f"{language}_{split}"] = len(frame)
        frames[split] = pd.concat(parts, ignore_index=True)
    train_frame = frames["train"]
    if train_frame["label"].nunique() != len(LABELS):
        raise ValueError("Combined training set must contain all three sentiment classes")

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_id"], num_labels=len(LABELS),
        id2label={i: label for i, label in enumerate(LABELS)},
        label2id=LABEL_TO_ID, ignore_mismatched_sizes=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    make_loader = lambda frame, shuffle: DataLoader(
        EncodedTextDataset(frame, tokenizer, int(config["max_length"])),
        batch_size=int(config["batch_size"]), shuffle=shuffle,
        num_workers=int(config.get("num_workers", 0)), collate_fn=collator(tokenizer),
        pin_memory=device.type == "cuda",
    )
    train_loader = make_loader(train_frame, True)
    validation_loader = make_loader(frames["validation"], False)
    total_steps = len(train_loader) * int(config["epochs"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    warmup_steps = int(total_steps * float(config["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    best_f1, best_epoch, history = -1.0, 0, []
    model_dir = artifacts / "model"
    print(f"Device={device}; train={len(train_frame):,}; validation={len(frames['validation']):,}; test={len(frames['test']):,}; steps={total_steps:,}", flush=True)

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        losses = []
        for step, batch in enumerate(train_loader, 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            result = model(**batch)
            loss = result.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            if step % 100 == 0 or step == len(train_loader):
                print(f"epoch {epoch}/{config['epochs']} step {step}/{len(train_loader)} loss={np.mean(losses[-100:]):.4f}", flush=True)
        validation = evaluate(model, validation_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_macro_f1": validation["macro_f1"], "validation_accuracy": validation["accuracy"]}
        history.append(row)
        print(f"epoch {epoch}: train_loss={row['train_loss']:.4f}, val_macro_f1={row['validation_macro_f1']:.4f}", flush=True)
        if validation["macro_f1"] > best_f1:
            best_f1, best_epoch = validation["macro_f1"], epoch
            model.save_pretrained(model_dir, safe_serialization=True)
            tokenizer.save_pretrained(model_dir)

    pd.DataFrame(history).to_csv(artifacts / "training_history.csv", index=False)
    del model
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    test_frame = frames["test"]
    test_loader = make_loader(test_frame, False)
    evaluated = evaluate(model, test_loader, device)
    y_true, y_pred, conf = evaluated.pop("targets"), evaluated.pop("predictions"), evaluated.pop("confidences")
    test_frame = test_frame.reset_index(drop=True)
    results = {"overall": metric_summary(y_true, y_pred), "per_language": {}}
    for language in config["languages"]:
        mask = test_frame["language"].eq(language).to_numpy()
        y_lang = np.asarray(y_true)[mask]
        pred_lang = np.asarray(y_pred)[mask]
        results["per_language"][language] = metric_summary(y_lang, pred_lang)
    results["labels"] = list(LABELS)
    results["split_counts"] = split_counts
    results["protocol"] = "official train/validation/test splits; checkpoint selected only on validation macro-F1; test evaluated once"
    (artifacts / "test_metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output = pd.DataFrame({
        "language": test_frame["language"],
        "gold_label": [LABELS[i] for i in y_true],
        "predicted_label": [LABELS[i] for i in y_pred],
        "confidence": conf,
    })
    output.to_csv(artifacts / "test_predictions.csv", index=False)
    data_manifest = root / "data" / "raw" / "manifest.json"
    metadata = {
        "project": "Topic 4 — Hausa, Yoruba and Igbo sentiment analysis",
        "model_id": config["model_id"], "dataset_id": config["dataset_id"],
        "dataset_version": config["dataset_version"],
        "labels": list(LABELS), "languages": config["languages"],
        "seed": seed, "epochs_requested": int(config["epochs"]), "epochs_completed": len(history),
        "best_epoch": best_epoch, "best_validation_macro_f1": best_f1,
        "max_length": int(config["max_length"]), "batch_size": int(config["batch_size"]),
        "learning_rate": float(config["learning_rate"]), "weight_decay": float(config["weight_decay"]),
        "max_train_per_language": int(train_cap), "max_eval_per_language": int(eval_cap),
        "train_rows": int(len(train_frame)), "split_counts": split_counts,
        "device": str(device), "duration_seconds": round(time.time() - started, 2),
        "protocol": results["protocol"],
        "dataset_manifest": str(data_manifest.relative_to(root)) if data_manifest.exists() else None,
        "model_license": "MIT (pretrained checkpoint); see model card",
        "data_license_note": "NaijaSenti dataset card states CC BY-NC-SA 4.0; see docs/DATA_PROVENANCE.md",
    }
    (artifacts / "model_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plot_artifacts(artifacts, results, history)
    write_results_markdown(root / "docs" / "RESULTS.md", results, metadata)
    print(json.dumps({"best_epoch": best_epoch, "best_validation_macro_f1": best_f1,
                      "test_macro_f1": results["overall"]["macro_f1"], "per_language_macro_f1": {
                          lang: values["macro_f1"] for lang, values in results["per_language"].items()},
                      "artifacts": str(artifacts)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
