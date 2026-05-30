"""
Сборка датасета из разметки ОБУЧЕННОЙ модели (ruBERT) в dataset_model/.

В отличие от merge_annotations.py (который собирает обучающий датасет из
LLM-разметки и ручного эталона), этот скрипт берёт только предсказания
своей модели — папки data/<dataset>/annotated/<model-dir>/ — и сохраняет
их в отдельный каталог dataset_model/. Полезно для анализа того, как
модель размечает корпус, без смешивания с обучающими данными.

Запуск из корня проекта:
    python merge_model_annotations.py
    python merge_model_annotations.py --model-dir model_checkpoints_best
    python merge_model_annotations.py --dataset rutar --skip-disputed
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from annotate import DATASETS
from merge_annotations import load_annotations, save_jsonl, save_csv, list_models

DEFAULT_MODEL_DIR = "model_checkpoints_best"
OUTPUT_JSONL = Path("dataset_model/annotations.jsonl")
OUTPUT_CSV   = Path("dataset_model/annotations.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Сборка датасета из разметки обученной модели в dataset_model/",
        epilog="Без аргументов собирает разметку model_checkpoints_best по всем русским датасетам.",
    )
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default=None,
                        help="Только один датасет (по умолчанию — все русские)")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help=f"Имя подпапки с разметкой модели (по умолчанию {DEFAULT_MODEL_DIR})")
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--output-csv",   default=None)
    parser.add_argument("--skip-disputed", action="store_true",
                        help="Исключить записи с disputed=True (низкая уверенность модели)")
    parser.add_argument("--include-non-ru", action="store_true",
                        help="Включить нероссийские датасеты (по умолчанию пропускаются)")
    args = parser.parse_args()

    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else OUTPUT_JSONL
    output_csv   = Path(args.output_csv)   if args.output_csv   else OUTPUT_CSV

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = list(DATASETS.keys())
        if not args.include_non_ru:
            skipped = [d for d in datasets if DATASETS[d].get("lang") != "ru"]
            datasets = [d for d in datasets if DATASETS[d].get("lang") == "ru"]
            if skipped:
                print(f"Пропущены нероссийские датасеты: {', '.join(skipped)} "
                      f"(--include-non-ru чтобы включить)")

    all_rows = []
    for dataset_key in datasets:
        input_dir = DATASETS[dataset_key]["dir"] / "annotated"
        if not input_dir.exists():
            print(f"  [{dataset_key}] нет папки annotated — пропускаем")
            continue

        if args.model_dir not in list_models(input_dir):
            print(f"  [{dataset_key}] нет разметки модели '{args.model_dir}' — пропускаем")
            continue

        rows = load_annotations(input_dir, model_filter=args.model_dir)
        if args.skip_disputed:
            before = len(rows)
            rows = [r for r in rows if not r.get("disputed")]
            if before != len(rows):
                print(f"  [{dataset_key}] отфильтровано disputed: {before - len(rows)}")
        if rows:
            print(f"  [{dataset_key}] {len(rows)} записей")
            all_rows.extend(rows)
        else:
            print(f"  [{dataset_key}] нет данных")

    if not all_rows:
        print(f"Нет разметки модели '{args.model_dir}' для сборки.", file=sys.stderr)
        sys.exit(1)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(all_rows, output_jsonl)
    save_csv(all_rows, output_csv)

    label_counts = Counter(r["label"] for r in all_rows)
    print(f"\nИтого: {len(all_rows)} записей  (модель: {args.model_dir})")
    print(f"Метки: {' '.join(f'{k}:{v}' for k, v in sorted(label_counts.items()))}")
    print(f"\nJSONL: {output_jsonl}")
    print(f"CSV:   {output_csv}")


if __name__ == "__main__":
    main()
