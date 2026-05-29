# Argument Mining — разметка аргументов в русскоязычных правовых текстах

Инструмент автоматической разметки предложений с помощью LLM. Цель — создать датасет для обучения модели Argument Mining с оптимизацией на **полноту (Recall)**: лучше предложить лишний аргумент, чем пропустить важный довод.

---

## Структура проекта

```
ArgumentMining/
├── annotate.py           # Разметка документов через LLM
├── merge_annotations.py  # Объединение результатов в единый датасет
├── evaluate.py           # Сравнение разметки с эталоном
├── requirements.txt
├── .env                  # Секреты (не в git)
├── .env.example          # Шаблон для .env
├── model/
│   ├── train_model.py    # Дообучение ruBERT на dataset/annotations.jsonl
│   └── predict.py        # Инференс обученной модели на новых документах
├── dataset/                       # Готовый датасет для обучения (в git)
│   ├── annotations.jsonl
│   └── annotations.csv
└── data/                          # Сырые и размеченные данные (не в git)
    ├── ruslawod/                  # Русские правовые документы (RusLawOD)
    │   ├── raw/
    │   └── annotated/
    ├── rutar/                     # Письма Минфина/ФНС — налоговое обоснование (RU)
    │   ├── raw/
    │   └── annotated/
    └── sudresh/                   # Судебные решения РФ (sud-resh-benchmark)
        ├── raw/
        └── annotated/
```

---

## Датасеты

| Ключ | Источник | Язык | Назначение |
|------|----------|------|------------|
| `ruslawod` | [irlspbru/RusLawOD](https://huggingface.co/datasets/irlspbru/RusLawOD) | RU | Русские правовые документы |
| `rutar` | локальный файл `data/rutar/raw/rutar.xlsx` | RU | 166 писем Минфина/ФНС с налоговым обоснованием |
| `sudresh` | [lawful-good-project/sud-resh-benchmark](https://huggingface.co/datasets/lawful-good-project/sud-resh-benchmark) | RU | ~1000 уникальных судебных решений РФ (10 правовых областей) |

---

## Метки

| Метка | Описание |
|-------|----------|
| `CLAIM` | Позиция, требование или вывод стороны / суда / органа |
| `PREMISE` | Обоснование, почему CLAIM верен; причинно-следственная связь |
| `EVIDENCE` | Конкретная ссылка на норму, документ, дату, сумму, факт как доказательство |
| `REBUTTAL` | Опровержение или несогласие с чужой позицией |
| `NON_ARG` | Чисто процедурный/технический текст без какой-либо позиции |

---

## Установка

```bash
python -m venv venv
source venv/Scripts/activate       # Windows bash
# или: venv\Scripts\Activate.ps1   # PowerShell

pip install -r requirements.txt
cp .env.example .env               # заполни ключи
```

---

## Использование

### Разметка (`annotate.py`)

**Интерактивный режим** — запуск без аргументов открывает меню:

```bash
python annotate.py
```

**Скриптовый режим:**

```bash
python annotate.py --dataset ruslawod --provider gemini --model gemini-2.5-flash --n 100
python annotate.py --dataset sudresh --provider ollama --model qwen2.5:7b --n all
```

#### Провайдеры

| Провайдер | Требования |
|-----------|------------|
| `ollama` | `ollama serve` + `ollama pull <model>` |
| `gemini` | `GEMINI_API_KEY` в `.env` |
| `local` | обученная модель в `model/checkpoints/best/` |

Скрипт автоматически **пропускает уже размеченные документы** — запуск можно прерывать и возобновлять.

---

### Объединение в датасет (`merge_annotations.py`)

Собирает все датасеты в единый `dataset/annotations.jsonl`:

```bash
python merge_annotations.py              # все датасеты
python merge_annotations.py --dataset rutar
```

---

### Обучение модели (`model/train_model.py`)

Дообучает `DeepPavlov/rubert-base-cased` на `data/final/annotations.jsonl`.

```bash
python model/train_model.py
```

Лучшая модель сохраняется в `model/checkpoints/best/`.

---

### Разметка своей моделью (`model/predict.py`)

```bash
python model/predict.py
```

Читает документы из `data/rutar/raw/rutar_docs.jsonl`, сохраняет в `data/rutar/annotated/local_rubert/`.

---

### Оценка качества (`evaluate.py`)

Сравнивает разметку модели с ручным эталоном (папка `my` внутри `annotated/`):

```bash
python evaluate.py
python evaluate.py --dataset rutar --model gemini-2_5-flash
```

---

## Формат данных

Каждая строка в `annotated/*.jsonl` и итоговом JSONL:

```json
{
  "doc_id": "102013222",
  "position": 0,
  "text": "Истец полагает, что ответчик нарушил условия договора.",
  "label": "CLAIM",
  "confidence": 0.97,
  "reasoning": "Прямая позиция истца, слово «полагает».",
  "disputed": false,
  "source_file": "102013222.jsonl"
}
```

`disputed: true` — эвристика: `confidence < 0.65` по самооценке модели. Используется как сигнал для приоритизации ручной проверки.

---

## Переменные окружения (`.env`)

```bash
HF_TOKEN=hf_...           # Hugging Face — для загрузки датасетов
GEMINI_API_KEY=AIza...    # Google Gemini API
```
