import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from annotate import DATASETS

FIELDNAMES = ["doc_id", "position", "text", "label", "confidence",
              "reasoning", "disputed", "model", "source_file"]


def load_annotations(annotated_dir: Path, model_filter: str | None = None,
                     exclude_models: set[str] | None = None) -> list[dict]:
    rows = []
    exclude_models = exclude_models or set()
    subdirs = sorted(annotated_dir.iterdir()) if annotated_dir.exists() else []

    model_dirs = [d for d in subdirs if d.is_dir()]
    flat_files = list(annotated_dir.glob("*.jsonl")) if annotated_dir.exists() else []

    if model_dirs:
        for model_dir in model_dirs:
            if model_filter and model_dir.name != model_filter:
                continue
            if model_dir.name in exclude_models:
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


GLOBAL_FINAL_JSONL = Path("dataset/annotations.jsonl")
GLOBAL_FINAL_CSV   = Path("dataset/annotations.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Объединение размеченных JSONL в единый датасет",
        epilog="Без аргументов собирает ВСЕ датасеты в dataset/.",
    )
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default=None,
                        help="Только один датасет (по умолчанию — все)")
    parser.add_argument("--model",   default=None, help="Только эта модель (имя подпапки)")
    parser.add_argument("--exclude-model", action="append", default=[], metavar="NAME",
                        help="Исключить разметку этой модели (можно указывать несколько раз)")
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--output-csv",   default=None)
    parser.add_argument("--skip-disputed", action="store_true",
                        help="Исключить записи с disputed=True (ошибки и низкая уверенность)")
    parser.add_argument("--include-non-ru", action="store_true",
                        help="Включить нероссийские датасеты (по умолчанию англоязычные, напр. peessays, пропускаются)")
    args = parser.parse_args()

    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else GLOBAL_FINAL_JSONL
    output_csv   = Path(args.output_csv)   if args.output_csv   else GLOBAL_FINAL_CSV

    # Определяем какие датасеты собирать
    if args.dataset:
        datasets_to_merge = [args.dataset]
    else:
        datasets_to_merge = list(DATASETS.keys())

    # В обучающий датасет идут только русскоязычные источники. Англоязычные
    # (например, peessays) служат лишь для проверки пайплайна и в обучающую
    # выборку не включаются. Явный --dataset уважается как есть.
    if not args.dataset and not args.include_non_ru:
        skipped = [d for d in datasets_to_merge if DATASETS[d].get("lang") != "ru"]
        datasets_to_merge = [d for d in datasets_to_merge if DATASETS[d].get("lang") == "ru"]
        if skipped:
            print(f"Пропущены нероссийские датасеты: {', '.join(skipped)} "
                  f"(--include-non-ru чтобы включить)")

    all_rows = []
    for dataset_key in datasets_to_merge:
        input_dir = DATASETS[dataset_key]["dir"] / "annotated"
        if not input_dir.exists():
            print(f"  [{dataset_key}] нет папки annotated — пропускаем")
            continue

        models = list_models(input_dir)
        rows = load_annotations(input_dir, model_filter=args.model,
                                exclude_models=set(args.exclude_model))
        if args.skip_disputed:
            before = len(rows)
            rows = [r for r in rows if not r.get("disputed")]
            if before != len(rows):
                print(f"  [{dataset_key}] отфильтровано disputed: {before - len(rows)}")
        if rows:
            print(f"  [{dataset_key}] {len(rows)} записей  (модели: {', '.join(models)})")
            all_rows.extend(rows)
        else:
            print(f"  [{dataset_key}] нет данных")

    if not all_rows:
        print("Нет данных для объединения.", file=sys.stderr)
        sys.exit(1)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    save_jsonl(all_rows, output_jsonl)
    save_csv(all_rows, output_csv)

    label_counts = Counter(r["label"] for r in all_rows)
    model_counts: dict[str, int] = {}
    for r in all_rows:
        key = r.get("model", "unknown")
        model_counts[key] = model_counts.get(key, 0) + 1

    print(f"\nИтого: {len(all_rows)} записей")
    print(f"Метки: {' '.join(f'{k}:{v}' for k, v in sorted(label_counts.items()))}")
    for m, c in sorted(model_counts.items()):
        print(f"  {m}: {c}")
    print(f"\nJSONL: {output_jsonl}")
    print(f"CSV:   {output_csv}")


if __name__ == "__main__":
    main()
