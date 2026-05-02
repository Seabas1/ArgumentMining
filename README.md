# Argument Mining — разметка аргументов в русскоязычных правовых текстах

Инструмент автоматической разметки предложений с помощью LLM. Цель — создать датасет для обучения модели Argument Mining с оптимизацией на **полноту (Recall)**: лучше предложить лишний аргумент, чем пропустить важный довод.

---

## Структура проекта

```
ArgumentMining/
├── annotate.py           # Разметка документов через LLM
├── merge_annotations.py  # Объединение результатов в единый датасет
├── requirements.txt
├── .env                  # Секреты (не в git)
├── .env.example          # Шаблон для .env
└── data/
    ├── ruslawod/                  # Русские правовые документы (RusLawOD)
    │   ├── raw/                   # parquet-файлы + JSONL-кэш
    │   ├── annotated/             # По одному JSONL на документ
    │   └── final/                 # annotations.jsonl, annotations.csv
    ├── mixed_legal/               # Судебные решения + юридические новости (RU)
    │   ├── raw/
    │   ├── annotated/
    │   └── final/
    └── rutar/                     # Письма Минфина/ФНС — налоговое обоснование (RU)
        ├── raw/                   # rutar.xlsx + sources_dataset_for_rutar.json
        ├── annotated/
        └── final/
```

---

## Датасеты

| Ключ | Источник | Язык | Назначение |
|------|----------|------|------------|
| `ruslawod` | [irlspbru/RusLawOD](https://huggingface.co/datasets/irlspbru/RusLawOD) | RU | Основной корпус правовых текстов |
| `mixed_legal` | [RussianNLP/Mixed-Summarization-Dataset](https://huggingface.co/datasets/RussianNLP/Mixed-Summarization-Dataset) | RU | Судебные решения + юридические новости — дополнительный правовой корпус |
| `rutar` | локальный файл `data/rutar/raw/rutar.xlsx` | RU | 199 писем Минфина/ФНС с налоговым обоснованием — валидация качества разметки |

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

```
====================================================
   Argument Mining — разметка аргументов
====================================================

Датасет:
  >1. RusLawOD — русские правовые документы
   2. Mixed Legal RU — судебные решения и юридические новости (RU)
   3. RuTaR — письма Минфина/ФНС с налоговым обоснованием (RU)
Выбор [1]:

Провайдер LLM:
  >1. Ollama (локально)
   2. Gemini API
Выбор [1]:

Модели Ollama (скачаны локально):
  >1. gemma4:e4b
   2. llama3:latest
   3. qwen2.5:7b
Выбор [1]:

Количество документов [50]:
```

**Скриптовый режим** — все параметры через флаги:

```bash
python annotate.py --dataset ruslawod --provider gemini --model gemini-2.5-flash --n 100
python annotate.py --dataset mixed_legal --provider ollama --model qwen2.5:7b --n 30
```

#### Провайдеры

| Провайдер | Команда | Требования |
|-----------|---------|------------|
| Ollama | `--provider ollama` | `ollama serve` + `ollama pull <model>` |
| Gemini | `--provider gemini` | `GEMINI_API_KEY` в `.env` |

#### Локальные модели (Ollama)

Список доступных моделей определяется автоматически через `ollama list`. Примеры:
- `qwen2.5:7b`, `qwen3.5-9b`, `gemma4:e4b`, `llama3:latest`

> Ollama доступна через Python-пакет (`pip install ollama`) или напрямую через REST API — пакет необязателен, достаточно запущенного процесса `ollama serve`.

Скрипт автоматически **пропускает уже размеченные документы** — запуск можно прерывать и возобновлять.

---

### Объединение в датасет (`merge_annotations.py`)

```bash
python merge_annotations.py              # интерактивный выбор датасета
python merge_annotations.py --dataset ruslawod
```

Результат — `data/<dataset>/final/annotations.jsonl` и `annotations.csv`.

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

`disputed: true` — эвристика: `confidence < 0.65` по самооценке модели. Используется как сигнал для приоритизации ручной проверки, не как калиброванная метрика качества.

---

## Переменные окружения (`.env`)

```bash
HF_TOKEN=hf_...           # Hugging Face — для загрузки датасетов
GEMINI_API_KEY=AIza...    # Google Gemini API
```
