import argparse
import csv
import json
from pathlib import Path

ANNOTATED_DIR = Path("data/annotated")
FINAL_DIR = Path("data/final")
OUTPUT_JSONL = FINAL_DIR / "annotations.jsonl"
OUTPUT_CSV = FINAL_DIR / "annotations.csv"


def load_annotations(input_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(input_dir.glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["source_file"] = path.name
                rows.append(row)
    return rows


def save_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return

    fieldnames = [
        "doc_id",
        "position",
        "text",
        "label",
        "confidence",
        "reasoning",
        "disputed",
        "source_file",
    ]

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main():
    parser = argparse.ArgumentParser(description="Объединение размеченных JSONL в единый датасет")
    parser.add_argument("--input-dir", default=str(ANNOTATED_DIR), help="Папка с JSONL-файлами разметки")
    parser.add_argument("--output-jsonl", default=str(OUTPUT_JSONL), help="Путь для итогового JSONL")
    parser.add_argument("--output-csv", default=str(OUTPUT_CSV), help="Путь для итогового CSV")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_jsonl = Path(args.output_jsonl)
    output_csv = Path(args.output_csv)

    if not input_dir.exists():
        raise SystemExit(f"Папка не найдена: {input_dir}")

    rows = load_annotations(input_dir)
    if not rows:
        raise SystemExit(f"В папке нет данных для объединения: {input_dir}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    save_jsonl(rows, output_jsonl)
    save_csv(rows, output_csv)

    print(f"Объединено записей: {len(rows)}")
    print(f"JSONL: {output_jsonl}")
    print(f"CSV:   {output_csv}")


if __name__ == "__main__":
    main()
