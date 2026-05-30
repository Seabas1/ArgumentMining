"""
Разметка документов своей обученной моделью ruBERT.
Запуск из корня проекта: python model/predict.py

Интерактивно спрашивает:
  1. путь к чекпоинту модели (по умолчанию model/checkpoints/best);
  2. файл для разметки — .txt (один документ) или .jsonl (много документов
     с полями doc_id/text, как в кэше annotate.py);
  3. куда сохранить результат.

Скриптовый режим:
  python model/predict.py --input doc.txt --output out.jsonl
  python model/predict.py --input data/rutar/raw/rutar_docs.jsonl --output preds/
"""

import argparse
import json
import sys
from pathlib import Path

# ── Константы ──────────────────────────────────────────────────────────────────

DEFAULT_CHECKPOINT = Path("model/checkpoints/best")
MAX_LENGTH         = 256


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


# ── Чтение входного файла ──────────────────────────────────────────────────────

def read_documents(input_path: Path) -> list[dict]:
    """
    Читает входной файл и возвращает список документов [{doc_id, text}].
    .jsonl — каждая строка JSON с полями doc_id/text (как в кэше annotate.py).
    .txt / прочее — весь файл = один документ, doc_id = имя файла.
    """
    if input_path.suffix.lower() == ".jsonl":
        docs = []
        for line in input_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = (row.get("text") or "").strip()
            if text:
                docs.append({"doc_id": str(row.get("doc_id", input_path.stem)), "text": text})
        return docs

    text = input_path.read_text(encoding="utf-8-sig").strip()
    return [{"doc_id": input_path.stem, "text": text}] if text else []


def write_annotations(annotations: list[dict], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for ann in annotations:
            f.write(json.dumps(ann, ensure_ascii=False) + "\n")


# ── Интерактивный ввод ──────────────────────────────────────────────────────────

def ask_path(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip().strip('"')
        if raw:
            return raw
        if default is not None:
            return default
        print("     Введите непустой путь.")


def resolve_output(out_arg: str, docs: list[dict], input_path: Path) -> dict[str, Path]:
    """
    Возвращает {doc_id: out_file}. Если документ один — out_arg трактуется как
    файл; если несколько — как директория (по файлу на документ).
    """
    out = Path(out_arg)
    is_dir = len(docs) > 1 or out_arg.endswith(("/", "\\")) or out.suffix == ""
    if is_dir:
        return {d["doc_id"]: out / f"{_safe(d['doc_id'])}.jsonl" for d in docs}
    return {docs[0]["doc_id"]: out}


def _safe(name: str) -> str:
    import re
    return re.sub(r"[^\w\-]", "_", name).strip("_") or "doc"


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Разметка документов обученной моделью ruBERT",
        epilog="Без аргументов открывает интерактивный режим.",
    )
    parser.add_argument("--checkpoint", default=None, help="Путь к чекпоинту модели")
    parser.add_argument("--input",      default=None, help="Файл для разметки (.txt или .jsonl)")
    parser.add_argument("--output",     default=None, help="Куда сохранить (файл или директория)")
    args = parser.parse_args()

    print()
    print("=" * 52)
    print("   Argument Mining — разметка обученной моделью")
    print("=" * 52)

    # Скриптовый режим: --input задан → чекпоинт по умолчанию без вопроса.
    if args.input:
        checkpoint = Path(args.checkpoint or DEFAULT_CHECKPOINT)
    else:
        checkpoint = Path(args.checkpoint or ask_path("\nПуть к чекпоинту", str(DEFAULT_CHECKPOINT)))

    input_arg  = args.input or ask_path("Файл для разметки (.txt или .jsonl)")
    input_path = Path(input_arg)
    if not input_path.exists():
        print(f"Файл не найден: {input_path}", file=sys.stderr)
        sys.exit(1)

    docs = read_documents(input_path)
    if not docs:
        print(f"В файле нет текста для разметки: {input_path}", file=sys.stderr)
        sys.exit(1)

    default_out = f"{input_path.stem}_annotated.jsonl" if len(docs) == 1 else f"{input_path.stem}_annotated"
    output_arg  = args.output or ask_path("Куда сохранить результат", default_out)
    out_map     = resolve_output(output_arg, docs, input_path)

    print(f"\n  Документов на вход: {len(docs)}")
    print(f"  Сохраним в:         {output_arg}")

    model, tokenizer, device = load_model(checkpoint)

    total = 0
    for doc in docs:
        doc_id   = doc["doc_id"]
        out_file = out_map[doc_id]
        print(f"\n{'=' * 55}\nДокумент: {doc_id}\n{'=' * 55}")
        annotations = annotate_document(model, tokenizer, device, doc["text"], doc_id)
        write_annotations(annotations, out_file)
        total += len(annotations)
        print(f"  Сохранено: {out_file}")

    print(f"\nГотово! {total} предложений из {len(docs)} документов.")


if __name__ == "__main__":
    main()
