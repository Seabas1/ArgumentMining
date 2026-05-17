"""
Разметка документов своей обученной моделью ruBERT.
Запуск из корня проекта: python model/predict.py

По умолчанию берёт документы из data/rutar/raw/rutar_docs.jsonl
и сохраняет разметку в data/rutar/annotated/local_rubert/.
"""

import json
import sys
from pathlib import Path

# ── Константы ──────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = Path("model/checkpoints/best")
DOCS_FILE      = Path("data/rutar/raw/rutar_docs.jsonl")
OUT_DIR        = Path("data/rutar/annotated/local_rubert")
MAX_LENGTH     = 256
N_DOCS         = None  # None = все документы


# ── Загрузка модели ────────────────────────────────────────────────────────────

def load_model(checkpoint: Path):
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        print("Установи: pip install transformers torch")
        sys.exit(1)

    if not checkpoint.exists():
        print(f"Чекпоинт не найден: {checkpoint}")
        print("Сначала обучи модель: python model/train_model.py")
        sys.exit(1)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    model     = AutoModelForSequenceClassification.from_pretrained(str(checkpoint)).to(device)
    model.eval()
    print(f"Модель загружена: {checkpoint}  (device={device})")
    return model, tokenizer, device


# ── Инференс ───────────────────────────────────────────────────────────────────

def predict_sentence(model, tokenizer, device, sentence: str) -> dict:
    import torch
    inputs = tokenizer(
        sentence, return_tensors="pt",
        truncation=True, max_length=MAX_LENGTH, padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    idx   = probs.argmax().item()
    label = model.config.id2label[idx]
    conf  = probs[idx].item()
    return {
        "label":      label,
        "confidence": round(conf, 4),
        "reasoning":  f"local model p={conf:.3f}",
        "disputed":   conf < 0.65,
    }


# ── Разбивка на предложения ────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    import re
    try:
        from razdel import sentenize
        return [s.text.strip() for s in sentenize(text) if s.text.strip()]
    except ImportError:
        pass
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z«\d])", text)
    return [p.strip() for p in parts if p.strip()]


# ── Аннотация документа ────────────────────────────────────────────────────────

def annotate_document(model, tokenizer, device, text: str, doc_id: str) -> list[dict]:
    sentences = split_sentences(text)
    results   = []
    for i, sentence in enumerate(sentences):
        ann = predict_sentence(model, tokenizer, device, sentence)
        results.append({"doc_id": doc_id, "position": i, "text": sentence, **ann})
        print(f"  [{i + 1:3d}/{len(sentences)}] {ann['label']:12s} ({ann['confidence']:.2f}) | {sentence[:55]}...")
    return results


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    if not DOCS_FILE.exists():
        print(f"Файл с документами не найден: {DOCS_FILE}")
        print("Сначала запусти annotate.py хотя бы раз — он создаст кэш документов.")
        sys.exit(1)

    docs = []
    for line in DOCS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            docs.append(json.loads(line))

    if N_DOCS:
        docs = docs[:N_DOCS]

    print(f"Документов: {len(docs)}")
    model, tokenizer, device = load_model(CHECKPOINT_DIR)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0

    for doc in docs:
        doc_id   = doc["doc_id"]
        out_file = OUT_DIR / f"{doc_id}.jsonl"

        if out_file.exists():
            print(f"[{doc_id}] уже размечен, пропускаем")
            continue

        print(f"\n{'=' * 55}\nДокумент: {doc_id}\n{'=' * 55}")
        annotations = annotate_document(model, tokenizer, device, doc["text"], doc_id)

        with open(out_file, "w", encoding="utf-8") as f:
            for ann in annotations:
                f.write(json.dumps(ann, ensure_ascii=False) + "\n")

        total += len(annotations)
        print(f"  Сохранено: {out_file}")

    print(f"\nГотово! {total} предложений → {OUT_DIR}/")


if __name__ == "__main__":
    main()
