# Argument Mining — разметка аргументов в русскоязычных правовых текстах

Инструмент автоматической разметки предложений с помощью LLM и дообученной модели **ruBERT**. Цель — создать датасет для задачи Argument Mining и обучить классификатор, оптимизированный на **полноту (Recall)** по редким классам.

Пайплайн состоит из четырёх шагов: **разметка** документов (LLM или своя модель) → **сборка** единого датасета → **дообучение** ruBERT → **инференс** на новых документах. Качество на каждом шаге проверяется скриптами оценки.

---

## Структура проекта

```
ArgumentMining/
├── annotate.py           # Разметка документов через LLM / свою модель
├── merge_annotations.py  # Объединение результатов в единый датасет
├── evaluate.py           # Сравнение разметки с ручным эталоном (RU)
├── evaluate_eng.py       # Оценка LLM на английском бенчмарке Persuasive Essays
├── requirements.txt
├── .env                  # Секреты (не в git)
├── .env.example          # Шаблон для .env
├── model/
│   ├── train_model.py    # Дообучение ruBERT на dataset/annotations.jsonl
│   ├── predict.py        # Инференс обученной модели на произвольном файле
│   └── checkpoints/best/ # Лучший чекпоинт (по macro F1 на валидации)
├── dataset/              # Готовый датасет для обучения (в git)
│   ├── annotations.jsonl
│   ├── annotations.csv
│   └── README.md         # Карточка датасета (HuggingFace Dataset Card)
└── data/                 # Сырые и размеченные данные (не в git)
    ├── ruslawod/         # Русские правовые документы (RusLawOD)
    │   ├── raw/          #   кэш документов + parquet
    │   └── annotated/    #   <model>/  и  my/ (ручной эталон)
    ├── rutar/            # Письма Минфина/ФНС — налоговое обоснование (RU)
    ├── sudresh/          # Судебные решения РФ (sud-resh-benchmark)
    └── peessays/         # Persuasive Essays — английский AM-бенчмарк
```

---

## Датасеты
| Ключ | Источник | Язык | Назначение |
|------|----------|------|------------|
| `ruslawod` | [irlspbru/RusLawOD](https://huggingface.co/datasets/irlspbru/RusLawOD) | RU | Русские правовые документы |
| `rutar` | локальный файл `data/rutar/raw/rutar.xlsx` | RU | 166 писем Минфина/ФНС с налоговым обоснованием |
| `sudresh` | [lawful-good-project/sud-resh-benchmark](https://huggingface.co/datasets/lawful-good-project/sud-resh-benchmark) | RU | ~1000 уникальных судебных решений РФ (10 правовых областей) |
| `peessays` | PERSUADE corpus ([Kaggle Feedback Prize 2021](https://www.kaggle.com/competitions/feedback-prize-2021/data)) | EN | Английский бенчмарк для проверки кросс-языкового пайплайна |

Для русских датасетов промпт автоматически берётся на русском, для `peessays` — на английском (поле `lang` в словаре `DATASETS`).

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

Дообучает `DeepPavlov/rubert-base-cased` на `dataset/annotations.jsonl`
(5 классов, взвешенная кросс-энтропия для редких меток).

```bash
python model/train_model.py
```

Гиперпараметры заданы константами в начале файла (`LR`, `EPOCHS`, `BATCH_SIZE`).
Лучший чекпоинт по macro F1 на валидации сохраняется в `model/checkpoints/best/`.

---

### Разметка своей моделью (`model/predict.py`)

Интерактивно спрашивает чекпоинт, входной файл и куда сохранить результат:

```bash
python model/predict.py
```

Скриптовый режим (без вопросов):

```bash
# Один документ из .txt → один .jsonl
python model/predict.py --input doc.txt --output doc_annotated.jsonl

# Много документов из .jsonl (поля doc_id/text) → папка с файлами
python model/predict.py --input data/rutar/raw/rutar_docs.jsonl --output preds/
```

Входной файл: `.txt` (весь файл = один документ) или `.jsonl` (по документу
на строку, как в кэше `annotate.py`). Модель локальная — никаких API-ключей
и лимитов.

---

### Оценка качества (`evaluate.py`, `evaluate_eng.py`)

`evaluate.py` сравнивает разметку модели с ручным эталоном (папка `my`
внутри `annotated/`) — precision/recall/F1 по классам + матрица ошибок:

```bash
python evaluate.py
python evaluate.py --dataset ruslawod --model model_checkpoints_best
python evaluate.py --dataset rutar --model gemini-2_5-flash
```

`evaluate_eng.py` оценивает LLM-разметку на английском бенчмарке Persuasive
Essays относительно gold-меток (проверка кросс-языкового пайплайна):

```bash
python evaluate_eng.py --model qwen3_5-9b_latest
```

---

## Результаты

**Датасет** (`dataset/annotations.jsonl`): 11 865 предложений, 427 документов.

| Метка | Доля |
|-------|------|
| `CLAIM` | 49.7% |
| `EVIDENCE` | 32.3% |
| `NON_ARG` | 14.9% |
| `PREMISE` | 1.9% |
| `REBUTTAL` | 1.2% |

**Дообученная модель ruBERT vs LLM** — честное сравнение на отложенной
выборке (held-out, эталон — человек; ни одна система не видела эти
предложения при обучении):

| Разметчик | Accuracy | macro F1 | Скорость |
|-----------|----------|----------|----------|
| ruBERT (своя модель) | **77.8%** | **0.691** | ~19 предл./с (CPU) |
| Qwen3.5-9b (Ollama) | 66.1% | 0.566 | ~1 документ/мин |

Дообученная модель **обгоняет LLM-аннотатор по качеству** на правовом домене
и работает на 1–2 порядка быстрее без GPU, API-ключей и лимитов.

> Замечание: сравнение на _всём_ ручном эталоне было бы некорректным —
> модель обучалась на нём. Поэтому воспроизводится точный train/val-сплит
> (`seed=42`, 15%) и обе системы оцениваются только на отложенных
> предложениях через одинаковый человеческий эталон.

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
