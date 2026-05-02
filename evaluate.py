"""
Сравнение разметки модели с эталоном (папка `my`) по метрикам precision/recall/F1.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from annotate import DATASETS, VALID_LABELS

REFERENCE_DIR = "my"
LABELS = ["CLAIM", "PREMISE", "EVIDENCE", "REBUTTAL", "NON_ARG"]


def load_folder(folder: Path) -> dict[tuple, str]:
    """Загружает аннотации → {(doc_id, position): label}."""
    result = {}
    for f in sorted(folder.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["doc_id"], int(row["position"]))
            result[key] = row["label"]
    return result


def precision_recall_f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def evaluate(ref: dict, pred: dict) -> None:
    common = set(ref) & set(pred)
    if not common:
        print("Нет общих документов/позиций для сравнения.")
        return

    ref_labels  = [ref[k]  for k in sorted(common)]
    pred_labels = [pred[k] for k in sorted(common)]

    total   = len(common)
    correct = sum(r == p for r, p in zip(ref_labels, pred_labels))

    print(f"Общих позиций : {total}")
    print(f"Только в эталоне : {len(ref) - len(common)}")
    print(f"Только в модели  : {len(pred) - len(common)}")
    print(f"Accuracy         : {correct / total * 100:.1f}%  ({correct}/{total})")
    print()

    # Confusion matrix counts per label
    tp_map = defaultdict(int)
    fp_map = defaultdict(int)
    fn_map = defaultdict(int)

    for r, p in zip(ref_labels, pred_labels):
        if r == p:
            tp_map[r] += 1
        else:
            fn_map[r] += 1
            fp_map[p] += 1

    # Per-label metrics
    print(f"{'Метка':<12} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'Support':>8}")
    print("-" * 60)

    macro_p = macro_r = macro_f = 0.0
    active = 0

    for lbl in LABELS:
        support = sum(1 for r in ref_labels if r == lbl)
        if support == 0:
            continue
        active += 1
        p, r, f = precision_recall_f1(tp_map[lbl], fp_map[lbl], fn_map[lbl])
        macro_p += p
        macro_r += r
        macro_f += f
        print(f"{lbl:<12} {p:>6.3f} {r:>6.3f} {f:>6.3f} {tp_map[lbl]:>5} {fp_map[lbl]:>5} {fn_map[lbl]:>5} {support:>8}")

    print("-" * 60)
    if active:
        print(f"{'macro avg':<12} {macro_p/active:>6.3f} {macro_r/active:>6.3f} {macro_f/active:>6.3f}")

    # Confusion matrix
    print()
    print("Матрица ошибок (строки = эталон, столбцы = модель):")
    w = 10
    header = f"{'':12}" + "".join(f"{lbl[:6]:>{w}}" for lbl in LABELS)
    print(header)
    for r_lbl in LABELS:
        if not any(r == r_lbl for r in ref_labels):
            continue
        row_str = f"{r_lbl:<12}"
        for p_lbl in LABELS:
            count = sum(1 for r, p in zip(ref_labels, pred_labels) if r == r_lbl and p == p_lbl)
            row_str += f"{count:>{w}}"
        print(row_str)


def list_models(annotated_dir: Path) -> list[str]:
    return sorted(
        d.name for d in annotated_dir.iterdir()
        if d.is_dir() and d.name != REFERENCE_DIR
    )


def ask_dataset() -> str:
    keys = list(DATASETS.keys())
    print("\nДатасет:")
    for i, k in enumerate(keys, 1):
        print(f"  {i}. {DATASETS[k]['label']}")
    while True:
        raw = input("Выбор [1]: ").strip() or "1"
        if raw.isdigit() and 1 <= int(raw) <= len(keys):
            return keys[int(raw) - 1]
        print(f"  Введите число от 1 до {len(keys)}")


def ask_model(annotated: Path) -> str:
    models = list_models(annotated)
    if not models:
        print("Нет папок с моделями для сравнения.", file=sys.stderr)
        sys.exit(1)
    print(f"\nМодель (сравниваем с '{REFERENCE_DIR}'):")
    for i, m in enumerate(models, 1):
        print(f"  {i}. {m}")
    while True:
        raw = input("Выбор [1]: ").strip() or "1"
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            return models[int(raw) - 1]
        print(f"  Введите число от 1 до {len(models)}")


def main():
    parser = argparse.ArgumentParser(
        description="Оценка разметки модели относительно эталона (папка 'my')",
        epilog="Без аргументов открывает интерактивное меню.",
    )
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default=None)
    parser.add_argument("--model",   default=None, help="Имя подпапки модели")
    args = parser.parse_args()

    print()
    print("=" * 50)
    print("   Argument Mining — оценка качества разметки")
    print("=" * 50)

    dataset_key = args.dataset or ask_dataset()
    annotated   = DATASETS[dataset_key]["dir"] / "annotated"

    ref_dir = annotated / REFERENCE_DIR
    if not ref_dir.exists():
        print(f"Эталон не найден: {ref_dir}", file=sys.stderr)
        sys.exit(1)

    model_name = args.model or ask_model(annotated)
    model_dir  = annotated / model_name

    if not model_dir.exists():
        print(f"Папка модели не найдена: {model_dir}", file=sys.stderr)
        sys.exit(1)

    ref  = load_folder(ref_dir)
    pred = load_folder(model_dir)

    print(f"\nЭталон : {ref_dir}  ({len(ref)} позиций)")
    print(f"Модель : {model_dir}  ({len(pred)} позиций)")
    print()

    evaluate(ref, pred)


if __name__ == "__main__":
    main()
