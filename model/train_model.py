"""
Дообучение ruBERT-base на размеченных данных ArgumentMining.
Запуск из корня проекта: python model/train_model.py
"""

import json
import random
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# ── Константы ──────────────────────────────────────────────────────────────────

LABELS   = ["CLAIM", "PREMISE", "EVIDENCE", "NON_ARG"]
LABEL2ID = {lbl: i for i, lbl in enumerate(LABELS)}
ID2LABEL = {i: lbl for i, lbl in enumerate(LABELS)}

ANNOTATED_DIR = Path("dataset/annotations.jsonl")
MODEL_NAME    = "DeepPavlov/rubert-base-cased"
OUTPUT_DIR    = Path("model/checkpoints")
MAX_LENGTH    = 256
BATCH_SIZE    = 16
EPOCHS        = 10
LR            = 2e-5
WARMUP_RATIO  = 0.1
VAL_RATIO     = 0.15
SEED          = 42


# ── Шаг 1: Загрузка и разбивка данных ─────────────────────────────────────────

def load_samples(annotations_file: Path) -> list[dict]:
    samples = []
    skipped = 0
    for line in annotations_file.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            skipped += 1
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        label = row.get("label", "").upper()
        text  = row.get("text", "").strip()
        if label in LABEL2ID and text:
            samples.append({"text": text, "label": label})
    if skipped:
        print(f"  Пропущено строк: {skipped}")
    return samples


def train_val_split(
    samples: list[dict],
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    data = samples[:]
    rng.shuffle(data)
    split = int(len(data) * (1 - val_ratio))
    return data[:split], data[split:]


def print_stats(name: str, samples: list[dict]) -> None:
    counts = Counter(s["label"] for s in samples)
    total  = len(samples)
    max_n  = max(counts.values(), default=1)
    print(f"\n{name}: {total} предложений")
    for lbl in LABELS:
        n   = counts.get(lbl, 0)
        bar = "█" * (n * 25 // max_n)
        print(f"  {lbl:<12} {n:>4}  {n/total*100:>5.1f}%  {bar}")


# ── Шаг 2: Токенизация и Dataset ──────────────────────────────────────────────

class ArgMiningDataset(Dataset):
    def __init__(self, samples: list[dict], tokenizer, max_length: int = MAX_LENGTH):
        self.encodings = tokenizer(
            [s["text"] for s in samples],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor([LABEL2ID[s["label"]] for s in samples])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "token_type_ids": self.encodings["token_type_ids"][idx],
            "labels":         self.labels[idx],
        }


# ── Шаг 3: Модель и цикл обучения ─────────────────────────────────────────────

def compute_class_weights(samples: list[dict]) -> torch.Tensor:
    counts = Counter(s["label"] for s in samples)
    total  = len(samples)
    # inversely proportional to frequency, normalised so mean weight = 1
    weights = torch.tensor(
        [total / (len(LABELS) * counts.get(lbl, 1)) for lbl in LABELS],
        dtype=torch.float,
    )
    return weights / weights.mean()


def train_epoch(model, loader, optimizer, scheduler, device,
                class_weights: torch.Tensor) -> float:
    model.train()
    loss_fn    = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    total_loss = 0.0
    for batch in loader:
        batch   = {k: v.to(device) for k, v in batch.items()}
        logits  = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
        loss    = loss_fn(logits, batch["labels"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, device) -> tuple[float, float, str]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch  = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds  = logits.argmax(dim=-1).cpu().tolist()
            all_preds  += preds
            all_labels += batch["labels"].cpu().tolist()

    f1     = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report = classification_report(
        all_labels, all_preds,
        target_names=LABELS,
        zero_division=0,
    )
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    acc = correct / len(all_labels)
    return acc, f1, report


def main() -> None:
    # ── данные ────────────────────────────────────────────────────────────────
    print("Загрузка данных...")
    all_samples = load_samples(ANNOTATED_DIR)
    if not all_samples:
        print(f"Данные не найдены: {ANNOTATED_DIR}")
        return

    train_data, val_data = train_val_split(all_samples)
    print_stats("Весь датасет", all_samples)
    print_stats("Train",        train_data)
    print_stats("Val",          val_data)

    # ── токенизация ───────────────────────────────────────────────────────────
    print(f"\nЗагрузка токенизатора {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_loader = DataLoader(
        ArgMiningDataset(train_data, tokenizer),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        ArgMiningDataset(val_data, tokenizer),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    # ── модель ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nУстройство: {device}")

    print(f"Загрузка модели {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    class_weights = compute_class_weights(train_data)
    print(f"\nВеса классов: " +
          ", ".join(f"{lbl}={class_weights[i]:.2f}" for i, lbl in enumerate(LABELS)))

    total_steps   = len(train_loader) * EPOCHS
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    optimizer     = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler     = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ── обучение ──────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_f1   = 0.0
    best_path = OUTPUT_DIR / "best"

    print(f"\nОбучение: {EPOCHS} эпох, {len(train_loader)} батчей/эпоха\n")
    for epoch in range(1, EPOCHS + 1):
        train_loss       = train_epoch(model, train_loader, optimizer, scheduler, device, class_weights)
        acc, f1, report  = eval_epoch(model, val_loader, device)

        marker = " ← лучший" if f1 > best_f1 else ""
        print(f"Эпоха {epoch}/{EPOCHS}  loss={train_loss:.4f}  acc={acc:.4f}  F1={f1:.4f}{marker}")

        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)

    print(f"\nЛучший macro F1: {best_f1:.4f}")
    print(f"Модель сохранена: {best_path}")

    # ── финальный отчёт по лучшей модели ─────────────────────────────────────
    print("\nОтчёт по val (лучшая эпоха):")
    best_model = AutoModelForSequenceClassification.from_pretrained(best_path).to(device)
    _, _, report = eval_epoch(best_model, val_loader, device)
    print(report)


if __name__ == "__main__":
    main()
