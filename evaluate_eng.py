"""
Оценка качества LLM-разметки на английском датасете Persuasive Essays.

Gold-метки берутся из кэша annotate.py (поле _gold — список меток по позициям предложений).
Предсказания загружаются из data/peessays/annotated/<model>/.

Запуск:
    python evaluate_eng.py
    python evaluate_eng.py --model qwen3_5-9b_latest
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

DATASET_DIR = Path("data/peessays")
CACHE_FILE  = DATASET_DIR / "raw" / "peessays_docs.jsonl"
LABELS      = ["CLAIM", "PREMISE", "EVIDENCE", "REBUTTAL", "NON_ARG"]


def load_gold(cache_file: Path) -> dict[str, str]:
    """
    Загружает gold-метки из кэша → {doc_id: label}.
    _gold — строка с меткой для каждого discourse element.
    """
    if not cache_file.exists():
        print(f"Кэш не найден: {cache_file}", file=sys.stderr)
        print("Сначала запусти: python annotate.py --dataset peessays ...", file=sys.stderr)
        sys.exit(1)

    gold: dict[str, str] = {}
    for line in cache_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        doc_id = row.get("doc_id", "")
        label  = row.get("_gold", "")
        if doc_id and label:
            gold[doc_id] = label

    if not gold:
        print("Gold-метки не найдены. Убедись что annotate.py загрузил датасет с полем _gold.",
              file=sys.stderr)
        sys.exit(1)
    return gold


def load_predictions(model_dir: Path) -> dict[str, str]:
    """
    Загружает предсказания модели → {doc_id: label}.
    Каждый файл = один discourse element, берём метку position=0
    (если позиций несколько — берём наиболее частую).
    """
    pred: dict[str, str] = {}
    for f in sorted(model_dir.glob("*.jsonl")):
        labels: list[str] = []
        doc_id = None
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = row["doc_id"]
            labels.append(row["label"])
        if doc_id and labels:
            pred[doc_id] = max(set(labels), key=labels.count)
    return pred


def list_models(annotated_dir: Path) -> list[str]:
    if not annotated_dir.exists():
        return []
    return sorted(d.name for d in annotated_dir.iterdir() if d.is_dir())


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def evaluate(gold: dict[str, str], pred: dict[str, str]) -> None:
    common = set(gold) & set(pred)
    if not common:
        print("Нет общих doc_id для сравнения.")
        print("Убедись, что annotate.py обработал те же discourse elements, что есть в кэше.")
        return

    gold_seq = [gold[k] for k in sorted(common)]
    pred_seq = [pred[k] for k in sorted(common)]

    total   = len(common)
    correct = sum(g == p for g, p in zip(gold_seq, pred_seq))

    print(f"Предложений в gold   : {len(gold)}")
    print(f"Предложений в модели : {len(pred)}")
    print(f"Совпадающих          : {total}")
    print(f"Accuracy             : {correct / total * 100:.1f}%  ({correct}/{total})")
    print()

    tp_map: dict[str, int] = defaultdict(int)
    fp_map: dict[str, int] = defaultdict(int)
    fn_map: dict[str, int] = defaultdict(int)

    for g, p in zip(gold_seq, pred_seq):
        if g == p:
            tp_map[g] += 1
        else:
            fn_map[g] += 1
            fp_map[p] += 1

    print(f"{'Label':<12} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'Support':>8}")
    print("-" * 60)

    macro_p = macro_r = macro_f = 0.0
    active = 0

    for lbl in LABELS:
        support = sum(1 for g in gold_seq if g == lbl)
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

    # Матрица ошибок
    present = [lbl for lbl in LABELS if any(g == lbl for g in gold_seq)]
    print()
    print("Confusion matrix (rows = gold, cols = predicted):")
    w = 10
    print(f"{'':12}" + "".join(f"{lbl[:6]:>{w}}" for lbl in present))
    for g_lbl in present:
        row = f"{g_lbl:<12}"
        for p_lbl in present:
            count = sum(1 for g, p in zip(gold_seq, pred_seq) if g == g_lbl and p == p_lbl)
            row += f"{count:>{w}}"
        print(row)

    # Распределение gold vs predicted
    print()
    print(f"{'Label':<12} {'Gold':>8} {'Predicted':>10}")
    print("-" * 32)
    for lbl in LABELS:
        g_cnt = sum(1 for g in gold_seq if g == lbl)
        p_cnt = sum(1 for p in pred_seq if p == lbl)
        if g_cnt > 0 or p_cnt > 0:
            print(f"{lbl:<12} {g_cnt:>8} {p_cnt:>10}")


def ask_model(annotated_dir: Path) -> str:
    models = list_models(annotated_dir)
    if not models:
        print(f"Нет папок с моделями в {annotated_dir}", file=sys.stderr)
        print("Сначала запусти: python annotate.py --dataset peessays ...", file=sys.stderr)
        sys.exit(1)
    print("\nМодель для оценки:")
    for i, m in enumerate(models, 1):
        print(f"  {i}. {m}")
    while True:
        raw = input("Выбор [1]: ").strip() or "1"
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            return models[int(raw) - 1]
        print(f"  Введите число от 1 до {len(models)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Оценка LLM-разметки Persuasive Essays vs gold labels",
        epilog="Без аргументов открывает интерактивное меню.",
    )
    parser.add_argument("--model", default=None,
                        help="Имя подпапки модели в data/peessays/annotated/")
    args = parser.parse_args()

    print()
    print("=" * 55)
    print("   Argument Mining — оценка LLM на Persuasive Essays")
    print("=" * 55)

    annotated_dir = DATASET_DIR / "annotated"
    model_name    = args.model or ask_model(annotated_dir)
    model_dir     = annotated_dir / model_name

    if not model_dir.exists():
        print(f"Папка не найдена: {model_dir}", file=sys.stderr)
        sys.exit(1)

    gold = load_gold(CACHE_FILE)
    pred = load_predictions(model_dir)

    print(f"\nGold:    {CACHE_FILE}  ({len(gold)} предложений)")
    print(f"Модель:  {model_dir}  ({len(pred)} предложений)")

    label_dist: dict[str, int] = defaultdict(int)
    for lbl in gold.values():
        label_dist[lbl] += 1
    print("\nРаспределение gold-меток:")
    for lbl in LABELS:
        n = label_dist[lbl]
        if n:
            print(f"  {lbl:<12} {n:>5}  {n/len(gold)*100:>5.1f}%")

    print()
    evaluate(gold, pred)


if __name__ == "__main__":
    main()
