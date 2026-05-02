import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from annotate import DATASETS

FIELDNAMES = ["doc_id", "position", "text", "label", "confidence",
              "reasoning", "disputed", "model", "source_file"]


def load_annotations(annotated_dir: Path, model_filter: str | None = None) -> list[dict]:
    rows = []
    subdirs = sorted(annotated_dir.iterdir()) if annotated_dir.exists() else []

    model_dirs = [d for d in subdirs if d.is_dir()]
    flat_files = list(annotated_dir.glob("*.jsonl")) if annotated_dir.exists() else []

    if model_dirs:
        for model_dir in model_dirs:
            if model_filter and model_dir.name != model_filter:
                continue
            for path in sorted(model_dir.glob("*.jsonl")):
                for row in _read_jsonl(path):
                    row["model"] = model_dir.name
                    row["source_file"] = path.name
                    rows.append(row)
    elif flat_files:
        for path in sorted(flat_files):
            for row in _read_jsonl(path):
                row.setdefault("model", "unknown")
                row["source_file"] = path.name
                rows.append(row)

    return rows


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in FIELDNAMES})


def list_models(annotated_dir: Path) -> list[str]:
    if not annotated_dir.exists():
        return []
    return sorted(d.name for d in annotated_dir.iterdir() if d.is_dir())


def interactive_dataset() -> str:
    print("\nВыберите датасет для объединения:")
    keys = list(DATASETS.keys())
    for i, key in enumerate(keys, 1):
        print(f"  {i}. {DATASETS[key]['label']}")
    while True:
        raw = input("Выбор [1]: ").strip() or "1"
        if raw.isdigit() and 1 <= int(raw) <= len(keys):
            return keys[int(raw) - 1]
        print(f"     Введите число от 1 до {len(keys)}")


def main():
    parser = argparse.ArgumentParser(
        description="Объединение размеченных JSONL в единый датасет",
        epilog="Без аргументов открывает интерактивное меню.",
    )
    parser.add_argument("--dataset",      choices=list(DATASETS.keys()), default=None)
    parser.add_argument("--model",        default=None, help="Только эта модель (имя подпапки)")
    parser.add_argument("--input-dir",    default=None)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--output-csv",   default=None)
    args = parser.parse_args()

    if args.input_dir and args.output_jsonl and args.output_csv:
        input_dir    = Path(args.input_dir)
        output_jsonl = Path(args.output_jsonl)
        output_csv   = Path(args.output_csv)
    else:
        dataset_key  = args.dataset or interactive_dataset()
        ds_dir       = DATASETS[dataset_key]["dir"]
        input_dir    = Path(args.input_dir)    if args.input_dir    else ds_dir / "annotated"
        output_jsonl = Path(args.output_jsonl) if args.output_jsonl else ds_dir / "final" / "annotations.jsonl"
        output_csv   = Path(args.output_csv)   if args.output_csv   else ds_dir / "final" / "annotations.csv"

    if not input_dir.exists():
        print(f"Папка не найдена: {input_dir}", file=sys.stderr)
        sys.exit(1)

    models = list_models(input_dir)
    if models:
        print(f"Найдены модели: {', '.join(models)}")
        if args.model and args.model not in models:
            print(f"Модель '{args.model}' не найдена. Доступные: {', '.join(models)}", file=sys.stderr)
            sys.exit(1)

    rows = load_annotations(input_dir, model_filter=args.model)
    if not rows:
        print(f"Нет данных в: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    save_jsonl(rows, output_jsonl)
    save_csv(rows, output_csv)

    label_counts = Counter(r["label"] for r in rows)
    model_counts: dict[str, int] = {}
    for r in rows:
        key = r.get("model", "unknown")
        model_counts[key] = model_counts.get(key, 0) + 1

    print(f"Объединено: {len(rows)} записей")
    print(f"Метки: {' '.join(f'{k}:{v}' for k, v in sorted(label_counts.items()))}")
    for m, c in sorted(model_counts.items()):
        print(f"  {m}: {c}")
    print(f"JSONL: {output_jsonl}")
    print(f"CSV:   {output_csv}")


if __name__ == "__main__":
    main()
