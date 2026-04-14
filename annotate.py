"""
Разметка аргументов в юридических текстах через LLM
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

DEFAULT_MODEL = "qwen2.5:7b"
OUTPUT_DIR = Path("data/annotated")
RAW_DIR = Path("data/raw")
CACHE_FILE = RAW_DIR / "ruslawod_docs.jsonl"
LOCAL_PARQUET_PATTERN = "ruslawod*.parquet"
VALID_LABELS = {"CLAIM", "PREMISE", "EVIDENCE", "REBUTTAL", "NON_ARG"}
HF_DOWNLOAD_RETRIES = 3

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

SYSTEM_PROMPT = """Ты размечаешь предложения из российских судебных документов.

Метки:
- CLAIM    — тезис или требование стороны («полагает», «требует», «считает»)
- PREMISE  — довод, объясняющий почему claim верен («поскольку», «так как», «следовательно»)
- EVIDENCE — ссылка на закон, судпрактику, документ, дату, сумму (ст. X ГК РФ, накладная №...)
- REBUTTAL — опровержение оппонента («однако», «вместе с тем», «необоснованно»)
- NON_ARG  — только явный процедурный текст («заседание состоялось», «явился представитель»)

Правило: при малейшем сомнении ставь аргументативную метку, НЕ NON_ARG.
Пропустить аргумент хуже, чем поставить лишнюю метку.

Отвечай ТОЛЬКО JSON, без пояснений:
{"label": "МЕТКА", "confidence": 0.0, "reasoning": "1-2 предложения"}"""

FEW_SHOT = [
    ("Истец полагает, что ответчик нарушил условия договора поставки.",
     '{"label": "CLAIM", "confidence": 0.97, "reasoning": "Прямая позиция истца, слово «полагает»."}'),
    ("В соответствии со статьёй 309 ГК РФ обязательства должны исполняться надлежащим образом.",
     '{"label": "EVIDENCE", "confidence": 0.98, "reasoning": "Ссылка на норму закона — ст. 309 ГК РФ."}'),
    ("Поскольку ответчик допустил просрочку, истец вправе требовать неустойки.",
     '{"label": "PREMISE", "confidence": 0.93, "reasoning": "Логический вывод «поскольку → вправе», обосновывает требование."}'),
    ("Однако довод ответчика о форс-мажоре не подтверждён доказательствами.",
     '{"label": "REBUTTAL", "confidence": 0.96, "reasoning": "Опровержение позиции оппонента, слово «однако»."}'),
    ("Судебное заседание проводилось с участием представителя истца по доверенности.",
     '{"label": "NON_ARG", "confidence": 0.95, "reasoning": "Чисто процедурная фраза, нет аргументативной нагрузки."}'),
]


def split_sentences(text: str, min_len: int = 20) -> list[str]:
    try:
        from razdel import sentenize
        sentences = [s.text.strip() for s in sentenize(text)]
    except ImportError:
        text = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+(?=[А-ЯA-Z«])", text)
    return [s.strip() for s in sentences if len(s.strip()) >= min_len]


def annotate_sentence(client, model: str, sentence: str, ctx_before: list[str]) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_ex, assistant_ex in FEW_SHOT[:3]:
        messages.append({"role": "user", "content": user_ex})
        messages.append({"role": "assistant", "content": assistant_ex})

    user_content = ""
    if ctx_before:
        user_content += f"[Контекст]: {' '.join(ctx_before[-2:])}\n"
    user_content += f"[Размечай]: {sentence}"
    messages.append({"role": "user", "content": user_content})

    try:
        response = client.chat(
            model=model,
            messages=messages,
            format="json",
            options={"temperature": 0.0},
        )
        data = json.loads(response["message"]["content"].strip())
        label = data.get("label", "PREMISE").upper()
        if label not in VALID_LABELS:
            label = "PREMISE"
        confidence = float(data.get("confidence", 0.7))
        return {
            "label": label,
            "confidence": confidence,
            "reasoning": data.get("reasoning", ""),
            "disputed": confidence < 0.65,
        }
    except Exception as e:
        return {"label": "PREMISE", "confidence": 0.5,
                "reasoning": f"[ошибка] {e}", "disputed": True}


def annotate_document(client, model: str, text: str, doc_id: str) -> list[dict]:
    sentences = split_sentences(text)
    results = []
    for i, sentence in enumerate(sentences):
        ctx = [r["text"] for r in results[-2:]]
        ann = annotate_sentence(client, model, sentence, ctx)
        results.append({"doc_id": doc_id, "position": i, "text": sentence, **ann})
        print(f"  [{i+1:3d}/{len(sentences)}] {ann['label']:12s} ({ann['confidence']:.2f}) | {sentence[:55]}...")
    return results


def load_cached_docs() -> list[dict]:
    if not CACHE_FILE.exists():
        return []

    docs = []
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def save_cached_docs(docs: list[dict]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def extract_doc(item: dict, fallback_idx: int) -> dict | None:
    tokens = item.get("tokens") or item.get("words") or []
    text = (
        " ".join(str(t) for t in tokens) if tokens else
        item.get("text") or
        item.get("textIPS") or
        item.get("taggedtextIPS") or
        item.get("headingIPS") or
        ""
    )
    if len(text.strip()) < 100:
        return None

    doc_id = str(
        item.get("doc_id") or
        item.get("id") or
        item.get("pravogovruNd") or
        item.get("docNumberIPS") or
        f"doc_{fallback_idx:05d}"
    )
    return {"doc_id": doc_id, "text": text}


def load_local_parquet_docs(n: int) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("Установи: pip install datasets")
        sys.exit(1)

    parquet_files = sorted(RAW_DIR.glob(LOCAL_PARQUET_PATTERN))
    if not parquet_files:
        return []

    print(f"Найдены локальные parquet-файлы: {', '.join(str(p) for p in parquet_files)}")
    ds = load_dataset("parquet", data_files=[str(path) for path in parquet_files], split="train")

    docs, seen_ids = [], set()
    for idx, item in enumerate(ds):
        doc = extract_doc(item, idx)
        if doc is None:
            continue
        if doc["doc_id"] in seen_ids:
            continue
        seen_ids.add(doc["doc_id"])
        docs.append(doc)
        if len(docs) >= n:
            break

    print(f"Загружено из локального parquet: {len(docs)} документов")
    return docs


def fetch_ruslawod_docs(n: int) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("Установи: pip install datasets")
        sys.exit(1)

    print(f"Загружаем {n} документов из RusLawOD...")
    last_error = None
    for attempt in range(1, HF_DOWNLOAD_RETRIES + 1):
        try:
            ds = load_dataset("irlspbru/RusLawOD", split="train", streaming=True)
            break
        except Exception as e:
            last_error = e
            if attempt == HF_DOWNLOAD_RETRIES:
                print(f"Не удалось загрузить RusLawOD после {attempt} попыток: {e}")
                raise
            wait_seconds = attempt * 5
            print(f"Сбой загрузки RusLawOD (попытка {attempt}/{HF_DOWNLOAD_RETRIES}): {e}")
            print(f"Повтор через {wait_seconds} сек...")
            time.sleep(wait_seconds)
    else:
        raise last_error

    docs, seen_ids = [], set()
    for idx, item in enumerate(ds):
        if len(docs) >= n:
            break
        doc = extract_doc(item, idx)
        if doc is None:
            continue
        if doc["doc_id"] in seen_ids:
            continue
        seen_ids.add(doc["doc_id"])
        docs.append(doc)

    print(f"Загружено: {len(docs)} документов")
    return docs


def load_ruslawood(n: int) -> list[dict]:
    cached_docs = load_cached_docs()
    if len(cached_docs) >= n:
        print(f"Берём {n} документов из локального кэша: {CACHE_FILE}")
        return cached_docs[:n]

    local_docs = load_local_parquet_docs(n)
    if len(local_docs) >= n:
        save_cached_docs(local_docs)
        print(f"Кэш обновлён из локального parquet: {CACHE_FILE} ({len(local_docs)} документов)")
        return local_docs[:n]

    if cached_docs:
        print(f"В локальном кэше найдено {len(cached_docs)} документов, догружаем недостающие...")
    else:
        print("Локальный кэш пуст, загружаем документы из Hugging Face...")

    fetched_docs = fetch_ruslawod_docs(n)

    merged_docs = []
    seen_ids = set()
    for doc in cached_docs + fetched_docs:
        doc_id = doc["doc_id"]
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)
        merged_docs.append(doc)

    save_cached_docs(merged_docs)
    print(f"Кэш обновлён: {CACHE_FILE} ({len(merged_docs)} документов)")
    return merged_docs[:n]


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Argument Mining разметка через Ollama")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Модель Ollama")
    parser.add_argument("--n", type=int, default=50, help="Кол-во документов из RusLawOD")
    args = parser.parse_args()

    if os.getenv("HF_TOKEN"):
        print("HF_TOKEN найден в окружении")

    try:
        import ollama
        client = ollama.Client()
        client.list()
        print(f"Ollama подключена. Модель: {args.model}")
    except Exception as e:
        print(f"Ошибка подключения к Ollama: {e}")
        print("Запусти Ollama: ollama serve")
        print(f"Скачай модель:  ollama pull {args.model}")
        sys.exit(1)

    docs = load_ruslawood(args.n)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_annotated = 0

    for doc in docs:
        doc_id = doc["doc_id"]
        out_file = OUTPUT_DIR / f"{doc_id}.jsonl"
        if out_file.exists():
            print(f"[{doc_id}] уже размечен, пропускаем")
            continue

        print(f"\n{'='*55}\nДокумент: {doc_id}\n{'='*55}")
        annotations = annotate_document(client, args.model, doc["text"], doc_id)

        counts = Counter(a["label"] for a in annotations)
        disputed = sum(1 for a in annotations if a.get("disputed"))
        print(f"\n  Итого: {len(annotations)} | " +
              " ".join(f"{k}:{v}" for k, v in sorted(counts.items())) +
              f" | disputed:{disputed}")

        with open(out_file, "w", encoding="utf-8") as f:
            for ann in annotations:
                f.write(json.dumps(ann, ensure_ascii=False) + "\n")

        total_annotated += len(annotations)
        print(f"  Сохранено: {out_file}")

    print(f"\nГотово! {total_annotated} предложений, результаты: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
