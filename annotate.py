"""
Разметка аргументов в юридических / аргументативных текстах через LLM.
Поддерживаемые провайдеры: Ollama, Gemini API.
Поддерживаемые датасеты: RusLawOD, Mixed Legal RU, RuTaR.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

# Конфигурация датасетов 

DATASETS = {
    "ruslawod": {
        "label": "RusLawOD — русские правовые документы",
        "dir": Path("data/ruslawod"),
        "lang": "ru",
        "hf_id": "irlspbru/RusLawOD",
    },
    "rutar": {
        "label": "RuTaR — письма Минфина/ФНС с налоговым обоснованием (RU)",
        "dir": Path("data/rutar"),
        "lang": "ru",
        "hf_id": None,
    },
    "sudresh": {
        "label": "sud-resh-benchmark — судебные решения РФ (10 правовых областей)",
        "dir": Path("data/sudresh"),
        "lang": "ru",
        "hf_id": "lawful-good-project/sud-resh-benchmark",
    },
}

PROVIDERS = {
    "ollama":    "Ollama (локально)",
    "gemini":    "Gemini API",
    "local":     "Своя модель (ruBERT fine-tuned)",
}

HF_DOWNLOAD_RETRIES   = 3
GEMINI_MAX_RETRIES    = 6
GEMINI_BASE_DELAY     = 3.0
GEMINI_RPM_LIMIT      = 10
GEMINI_MIN_INTERVAL   = 60.0 / GEMINI_RPM_LIMIT  # 6 seconds on free tier
_gemini_last_call     = [0.0]  # mutable container for rate-limiter state

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

# ── Схемы ответа Gemini ───────────────────────────────────────────────────────

_ANNOTATION_ITEM_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "label":      {"type": "STRING", "enum": ["CLAIM", "PREMISE", "EVIDENCE", "REBUTTAL", "NON_ARG"]},
        "confidence": {"type": "NUMBER"},
        "reasoning":  {"type": "STRING"},
    },
    "required": ["label", "confidence", "reasoning"],
}

GEMINI_SINGLE_GEN_CONFIG = {
    "temperature": 0.0,
    "response_mime_type": "application/json",
    "response_schema": _ANNOTATION_ITEM_SCHEMA,
}

GEMINI_BATCH_GEN_CONFIG = {
    "temperature": 0.0,
    "response_mime_type": "application/json",
    "response_schema": {
        "type": "ARRAY",
        "items": _ANNOTATION_ITEM_SCHEMA,
    },
}

# ── Системные промпты ─────────────────────────────────────────────────────────

SYSTEM_PROMPT_RU = """Ты размечаешь предложения из российских правовых документов (судебные решения, постановления, законы).

МЕТКИ — выбирай ОДНУ:

CLAIM    — позиция, требование или вывод стороны / суда / органа.
           Маркеры: «полагает», «требует», «считает», «просит», «суд приходит к выводу»,
           «подлежит/не подлежит удовлетворению», «необходимо», «следует», «обязать».
           Пункты постановлений («Обязать X», «Министерству Z обеспечить...») — всегда CLAIM.
           ТЕСТ: уберите аргументацию — осталась ли позиция? Если да → CLAIM.
           «Суд пришёл к выводу, что требования обоснованы, поскольку...» → CLAIM
           (вывод суда важнее сопутствующего «поскольку»).

PREMISE  — логическое обоснование без опоры на внешний источник; причинно-следственная цепочка.
           Маркеры: «поскольку», «так как», «в связи с тем что», «следовательно», «учитывая что».
           ТЕСТ: есть логика «X → Y», но НЕТ ссылки на норму/документ/факт → PREMISE.
           Если в предложении есть и «поскольку», и ссылка на норму — ставь EVIDENCE (шаг 2 алгоритма).

EVIDENCE — предложение существует РАДИ ссылки на внешний источник как доказательство.
           Маркеры: «ст. X», «п. Y», «согласно», «на основании», «накладная №», «договор от»,
           «как установлено судом», «что подтверждается», «из материалов дела следует».
           ТЕСТ: уберите ссылку — теряется ли смысл предложения? Если да → EVIDENCE.
           «Согласно ст. 309 ГК РФ, обязательства исполняются надлежащим образом» → EVIDENCE.
           «Суд считает требования обоснованными (со ссылкой на ст. 309)» → CLAIM,
           потому что главная функция — вывод суда, а не ссылка на норму.

REBUTTAL — прямое опровержение конкретной чужой позиции.
           Маркеры: «однако», «вместе с тем», «между тем», «необоснованно», «не может быть принято»,
           «отклоняется», «суд не соглашается».
           ВАЖНО: одного слова «однако» недостаточно — нужна явная чужая позиция для опровержения.
           «Ответчик утверждал X, однако суд установил Y» → REBUTTAL.
           «Однако требование было частично удовлетворено» → CLAIM (нет чужого тезиса).

NON_ARG  — ТОЛЬКО чисто процедурный текст: даты заседаний, явка сторон, подписи, реквизиты,
           нумерация разделов, технические пометки без какой-либо позиции.
           ПРАВИЛО СОМНЕНИЯ: если не уверен — НЕ ставь NON_ARG. Лучше ошибиться в сторону CLAIM.

АЛГОРИТМ (строго по порядку):
1. Есть явное опровержение чужой позиции? → REBUTTAL
2. Предложение существует ради ссылки на норму/документ/факт? → EVIDENCE
3. Есть позиция, требование, вывод суда/стороны? → CLAIM
4. Есть логическая связь «поскольку/так как → следовательно»? → PREMISE
5. Только реквизиты/подписи/явка — ноль аргументации? → NON_ARG

Отвечай ТОЛЬКО JSON, без пояснений:
{"label": "МЕТКА", "confidence": 0.0-1.0, "reasoning": "1-2 предложения"}"""

SYSTEM_PROMPT_RU_BATCH = """Ты размечаешь пронумерованный список предложений из российских правовых документов (судебные решения, постановления, законы).

ВАЖНО: предложения идут ПОДРЯД в исходном тексте. Учитывай их связь:
— если предложение N опровергает позицию из предложений 1..N-1 → REBUTTAL;
— если предложение N обосновывает вывод из предложений 1..N-1 → PREMISE или EVIDENCE;
— «однако» в предложении 3 может быть REBUTTAL к CLAIM из предложения 1 того же батча.

МЕТКИ — выбирай ОДНУ для каждого предложения:

CLAIM    — позиция, требование или вывод стороны / суда / органа.
           Пункты постановлений («Обязать X», «Обеспечить Y») — всегда CLAIM.
           ТЕСТ: уберите аргументацию — осталась ли позиция? Если да → CLAIM.

PREMISE  — логическое обоснование БЕЗ опоры на внешний источник («поскольку X → Y»).
           Если есть и «поскольку», и ссылка на норму — ставь EVIDENCE.

EVIDENCE — предложение существует РАДИ ссылки на норму/документ/факт.
           Маркеры: «ст. X», «согласно», «на основании», «что подтверждается», «из материалов дела».
           ТЕСТ: уберите ссылку — теряется ли смысл? Если да → EVIDENCE.
           «Суд считает требования обоснованными на основании ст. 309» → CLAIM (вывод важнее ссылки).

REBUTTAL — прямое опровержение конкретной чужой позиции (одного «однако» недостаточно).
           Чужая позиция может быть в [Контексте] или в предыдущих предложениях этого же батча.

NON_ARG  — ТОЛЬКО реквизиты/явка/подписи без аргументации.
           Правило сомнения: не уверен — НЕ ставь NON_ARG.

АЛГОРИТМ: REBUTTAL > EVIDENCE > CLAIM > PREMISE > NON_ARG

Отвечай ТОЛЬКО JSON-массивом — ровно столько объектов, сколько пронумерованных предложений:
[{"label": "МЕТКА", "confidence": 0.0-1.0, "reasoning": "1-2 предложения"}, ...]"""

SYSTEM_PROMPT_EN = """You annotate sentences from argumentative essays and opinion texts.

LABELS — choose ONE:

CLAIM    — a position, assertion, or conclusion put forward by the author.
           Markers: "should", "must", "believe", "argue", "conclude", "therefore".
           The main thesis and supporting claims are both CLAIM.

PREMISE  — reasoning that justifies why a CLAIM is true; cause-effect or logical support.
           Markers: "because", "since", "as", "given that", "due to", "consequently".

EVIDENCE — a specific reference to a source, statistic, study, law, or concrete fact used as proof.
           Markers: "according to", "a study shows", "data indicates", "as reported by", "in 2023".
           Do NOT confuse with CLAIM: EVIDENCE supports a claim, it is not the claim itself.

REBUTTAL — counter-argument or disagreement with an opposing position.
           Markers: "however", "nevertheless", "on the other hand", "critics argue", "opponents claim".
           Must contain an element of opposition to another viewpoint.

NON_ARG  — purely structural or transitional text with NO argumentative content.
           Examples: introductory phrases, topic sentences without a stance, essay headings.
           Do NOT label as NON_ARG if the sentence expresses any position or stance.

DECISION ALGORITHM:
1. Does it counter another viewpoint? → REBUTTAL
2. Does it cite a source, statistic, or concrete fact as proof? → EVIDENCE
3. Does it assert a position, conclusion, or stance? → CLAIM
4. Does it provide reasoning ("because", "since")? → PREMISE
5. Only structural/transitional text? → NON_ARG

Reply with ONLY JSON, no explanation:
{"label": "LABEL", "confidence": 0.0-1.0, "reasoning": "1-2 sentences"}"""

SYSTEM_PROMPT_EN_BATCH = """You annotate a numbered list of sentences from argumentative essays.

Same label rules — choose ONE per sentence:
CLAIM / PREMISE / EVIDENCE / REBUTTAL / NON_ARG

Decision order: REBUTTAL > EVIDENCE > CLAIM > PREMISE > NON_ARG

Reply with ONLY a JSON array — exactly as many objects as numbered sentences:
[{"label": "LABEL", "confidence": 0.0-1.0, "reasoning": "1-2 sentences"}, ...]"""

VALID_LABELS = {"CLAIM", "PREMISE", "EVIDENCE", "REBUTTAL", "NON_ARG"}

# ── Few-shot примеры ──────────────────────────────────────────────────────────

FEW_SHOT_RU = [
    ("Истец полагает, что ответчик нарушил условия договора поставки.",
     '{"label": "CLAIM", "confidence": 0.97, "reasoning": "Прямая позиция истца, слово «полагает». Тест: убираем аргументацию — позиция остаётся."}'),
    ("В соответствии со статьёй 309 ГК РФ обязательства должны исполняться надлежащим образом.",
     '{"label": "EVIDENCE", "confidence": 0.98, "reasoning": "Предложение существует ради ссылки на ст. 309 ГК РФ. Тест: уберите норму — смысл теряется → EVIDENCE."}'),
    ("Суд считает исковые требования подлежащими удовлетворению на основании ст. 309 ГК РФ.",
     '{"label": "CLAIM", "confidence": 0.96, "reasoning": "Главная функция — вывод суда, ст. 309 лишь усиливает его. Тест: без ссылки позиция остаётся → CLAIM, не EVIDENCE."}'),
    ("Поскольку ответчик допустил просрочку, истец вправе требовать неустойки.",
     '{"label": "PREMISE", "confidence": 0.93, "reasoning": "Логическая цепочка «поскольку X → вправе Y» без ссылки на внешний источник → PREMISE."}'),
    ("Поскольку в рассматриваемой ситуации расходы непосредственно связаны с деятельностью, направленной на получение дохода, они отвечают критерию экономической обоснованности.",
     '{"label": "PREMISE", "confidence": 0.94, "reasoning": "Логика «поскольку связаны с доходом → обоснованны» без ссылки на конкретную норму НК. Тест: убери рассуждение — вывод теряет основание → PREMISE."}'),
    ("Учитывая, что организация не является налоговым агентом в данной ситуации, обязанности по удержанию и перечислению НДФЛ у неё не возникает.",
     '{"label": "PREMISE", "confidence": 0.93, "reasoning": "«Учитывая, что X → Y» — логический вывод из факта (статус агента), конкретная норма НК не цитируется → PREMISE, не EVIDENCE."}'),
    ("В связи с тем что операции по реализации ценных бумаг освобождены от налогообложения, входной НДС по расходам на их приобретение к вычету не принимается.",
     '{"label": "PREMISE", "confidence": 0.92, "reasoning": "Причинно-следственная цепочка «освобождены → не принимается к вычету» без ссылки на статью НК → PREMISE. Сравни: если бы было «согласно пп. 12 п. 2 ст. 149 НК РФ» — стало бы EVIDENCE."}'),
    ("Поскольку согласно п. 1 ст. 330 ГК РФ неустойкой признаётся определённая законом денежная сумма, требование истца правомерно.",
     '{"label": "EVIDENCE", "confidence": 0.95, "reasoning": "Есть и «поскольку», и ссылка на норму — алгоритм: EVIDENCE > PREMISE. Предложение существует ради нормы."}'),
    ("Поскольку согласно пункту 2 статьи 346.11 НК РФ организации на УСН не признаются плательщиками НДС, выставление счёта-фактуры не требуется.",
     '{"label": "EVIDENCE", "confidence": 0.96, "reasoning": "«Поскольку» есть, но главная функция — норма п. 2 ст. 346.11 НК РФ. Убери норму — смысл теряется → EVIDENCE, не PREMISE."}'),
    ("Ответчик утверждал об отсутствии своей вины, однако материалами дела это не подтверждается.",
     '{"label": "REBUTTAL", "confidence": 0.96, "reasoning": "Явная чужая позиция ответчика опровергается судом — REBUTTAL, не просто «однако»."}'),
    ("Однако суд частично удовлетворил требования истца.",
     '{"label": "CLAIM", "confidence": 0.88, "reasoning": "Слово «однако» есть, но нет чужого тезиса для опровержения — это вывод суда → CLAIM."}'),
    ("Судебное заседание проводилось с участием представителя истца по доверенности.",
     '{"label": "NON_ARG", "confidence": 0.95, "reasoning": "Чисто процедурная фраза о составе участников, никакой позиции нет."}'),
    ("2. Министерству финансов Российской Федерации обеспечить финансирование мероприятий в срок до 1 марта.",
     '{"label": "CLAIM", "confidence": 0.93, "reasoning": "Властное поручение органу — это CLAIM. Тест: убираем дату — требование остаётся."}'),
]

FEW_SHOT_RU_BATCH = [(
    """[Контекст]: Дело рассматривалось в Арбитражном суде г. Москвы.
1. Истец полагает, что ответчик нарушил условия договора поставки.
2. В соответствии со статьёй 309 ГК РФ обязательства должны исполняться надлежащим образом.
3. Суд считает требования обоснованными на основании ст. 309 ГК РФ.
4. Поскольку ответчик допустил просрочку, истец вправе требовать неустойки.
5. Ответчик ссылался на форс-мажор, однако доказательств этому не представлено.
6. Судебное заседание проводилось с участием представителя истца по доверенности.""",
    """[{"label": "CLAIM", "confidence": 0.97, "reasoning": "Позиция истца, слово «полагает» — тест: позиция остаётся без аргументации."},
{"label": "EVIDENCE", "confidence": 0.98, "reasoning": "Предложение существует ради нормы ст. 309 ГК РФ. Убери норму — смысл теряется."},
{"label": "CLAIM", "confidence": 0.95, "reasoning": "Главная функция — вывод суда. Ссылка на ст. 309 лишь усиливает его, но тест: без ссылки вывод остаётся → CLAIM, не EVIDENCE."},
{"label": "PREMISE", "confidence": 0.93, "reasoning": "Логика «поскольку X → вправе Y», нет ссылки на внешний источник → PREMISE."},
{"label": "REBUTTAL", "confidence": 0.96, "reasoning": "Чужая позиция ответчика (форс-мажор) явно опровергается — полноценный REBUTTAL."},
{"label": "NON_ARG", "confidence": 0.95, "reasoning": "Чисто процедурная фраза о составе участников, никакой позиции."}]"""
),(
    """[Контекст]: Департамент налоговой политики рассмотрел обращение и сообщает следующее.
1. Согласно подпункту 1 пункта 1 статьи 146 НК РФ объектом налогообложения НДС признаётся реализация товаров на территории Российской Федерации.
2. Учитывая, что местом реализации услуг является территория иностранного государства, данные операции объектом налогообложения НДС не признаются.
3. Поскольку согласно пункту 2 статьи 346.11 НК РФ организации на УСН не являются плательщиками НДС, выставление счёта-фактуры не требуется.
4. Учитывая изложенное, налогоплательщик не обязан уплачивать НДС по указанным операциям.
5. Одновременно сообщаем, что настоящее письмо не содержит правовых норм и не является нормативным правовым актом.""",
    """[{"label": "EVIDENCE", "confidence": 0.98, "reasoning": "Предложение существует ради нормы пп. 1 п. 1 ст. 146 НК РФ — убери норму, смысл теряется → EVIDENCE."},
{"label": "PREMISE", "confidence": 0.94, "reasoning": "«Учитывая, что место реализации — иностранное государство → не объект НДС». Нет ссылки на конкретную статью НК, только логический вывод из факта → PREMISE, не EVIDENCE."},
{"label": "EVIDENCE", "confidence": 0.96, "reasoning": "«Поскольку» есть, но главная функция — норма п. 2 ст. 346.11 НК РФ. Убери норму — смысл теряется → EVIDENCE, не PREMISE."},
{"label": "CLAIM", "confidence": 0.97, "reasoning": "«Учитывая изложенное» — итоговый вывод Департамента. Тест: уберите аргументацию — позиция остаётся → CLAIM."},
{"label": "NON_ARG", "confidence": 0.93, "reasoning": "Процедурная оговорка о статусе письма, никакой аргументации."}]"""
)]

FEW_SHOT_EN = [
    ("Students should be required to learn a second language in school.",
     '{"label": "CLAIM", "confidence": 0.96, "reasoning": "Direct assertion of a position using the modal should."}'),
    ("Because bilingualism enhances cognitive flexibility, students who learn two languages outperform peers on problem-solving tasks.",
     '{"label": "PREMISE", "confidence": 0.94, "reasoning": "Causal reasoning introduced by because, justifying a claim about bilingual students."}'),
    ("According to a 2022 study by the University of Edinburgh, bilingual children show a 12% improvement in executive function.",
     '{"label": "EVIDENCE", "confidence": 0.97, "reasoning": "Specific citation of a study with statistics used as proof."}'),
    ("However, critics argue that mandatory language classes reduce time for core subjects like mathematics.",
     '{"label": "REBUTTAL", "confidence": 0.95, "reasoning": "Counter-argument to the main position, introduced by however."}'),
    ("This essay will discuss the benefits and drawbacks of mandatory language education.",
     '{"label": "NON_ARG", "confidence": 0.93, "reasoning": "Structural introduction that announces the essay topic without taking a stance."}'),
    ("In conclusion, the evidence strongly supports making second language learning compulsory.",
     '{"label": "CLAIM", "confidence": 0.95, "reasoning": "Concluding assertion of the main position."}'),
]

FEW_SHOT_EN_BATCH = [(
    """1. Students should be required to learn a second language in school.
2. According to a 2022 study by the University of Edinburgh, bilingual children show a 12% improvement in executive function.
3. Because bilingualism enhances cognitive flexibility, students outperform peers on problem-solving tasks.
4. However, critics argue that mandatory language classes reduce time for core subjects.
5. This essay will discuss the benefits and drawbacks of mandatory language education.""",
    """[{"label": "CLAIM", "confidence": 0.96, "reasoning": "Direct assertion of a position using the modal should."},
{"label": "EVIDENCE", "confidence": 0.97, "reasoning": "Specific citation of a study with statistics used as proof."},
{"label": "PREMISE", "confidence": 0.94, "reasoning": "Causal reasoning introduced by because."},
{"label": "REBUTTAL", "confidence": 0.95, "reasoning": "Counter-argument introduced by however."},
{"label": "NON_ARG", "confidence": 0.93, "reasoning": "Structural introduction without a stance."}]"""
)]


# ── Утилиты ───────────────────────────────────────────────────────────────────

def _to_safe_id(raw: str) -> str:
    return re.sub(r'[^\w\-]', "_", raw).strip("_")


# Fallback: юридические сокращения, после которых точка не является концом предложения.
_ABBREV_RE = re.compile(
    r'\b(ст|пп?|ч|разд|подп|абз|рис|табл|прим|г|обл|руб?|коп|тыс|млн|млрд|т|стр?|с|др|проч)\.'
    r'(?=[\s\d«])',
    re.IGNORECASE,
)

try:
    from razdel import sentenize as _razdel_sentenize
    _RAZDEL_OK = True
except ImportError:
    _RAZDEL_OK = False


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if _RAZDEL_OK:
        sents = [s.text.strip() for s in _razdel_sentenize(text)]
    else:
        # Fallback: защищаем сокращения, затем сплиттим по границам предложений
        protected = _ABBREV_RE.sub(lambda m: m.group().replace(".", "\x00"), text)
        parts = re.split(r"\n{2,}", protected)
        sents = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            chunks = re.split(r"(?<=[.!?;])\s+(?=[А-ЯЁA-Z«\d])", part)
            sents.extend(chunks)
        sents = [s.replace("\x00", ".") for s in sents]

    return [s.strip() for s in sents if s.strip()]


def _parse_label_response(data: dict) -> dict:
    label = data.get("label", "").upper()
    if label not in VALID_LABELS:
        return {
            "label": "CLAIM",
            "confidence": 0.3,
            "reasoning": f"[неизвестная метка: {data.get('label')!r}] {data.get('reasoning', '')}",
            "disputed": True,
        }
    confidence = float(data.get("confidence", 0.7))
    return {
        "label": label,
        "confidence": confidence,
        "reasoning": data.get("reasoning", ""),
        "disputed": confidence < 0.65,
    }


class DailyQuotaExhausted(Exception):
    pass


def _is_garbage_file(path: Path, threshold: float = 0.9) -> bool:
    """Возвращает True если ≥90% записей — fallback (CLAIM confidence=0.30)."""
    try:
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not rows:
            return True
        fallbacks = sum(1 for r in rows if r.get("confidence", 1) <= 0.30 and r.get("disputed"))
        return fallbacks / len(rows) >= threshold
    except Exception:
        return True


def _is_daily_quota(e: Exception) -> bool:
    msg = str(e)
    return "PerDay" in msg or "GenerateRequestsPerDay" in msg


def _call_with_retry(fn, max_retries: int = GEMINI_MAX_RETRIES, base_delay: float = GEMINI_BASE_DELAY,
                     rate_delay: float = 60.0):
    """Повторяет вызов fn при сетевых сбоях и rate-limit с экспоненциальной задержкой."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            name = type(e).__name__
            msg = str(e)
            if _is_daily_quota(e):
                raise DailyQuotaExhausted(msg) from e
            is_rate = "ResourceExhausted" in name or "429" in msg or "quota" in msg.lower()
            is_retryable = is_rate or any(k in name for k in (
                "ServiceUnavailable", "InternalServerError", "DeadlineExceeded",
                "Aborted", "ConnectionError", "Timeout", "RemoteDisconnected",
                "ConnectTimeout", "ReadTimeout",
            )) or "503" in msg or "500" in msg
            if not is_retryable or attempt >= max_retries - 1:
                raise
            delay = rate_delay if is_rate else base_delay * (2 ** attempt)
            print(f"  [retry {attempt + 1}/{max_retries}] {name} — ждём {delay:.0f}с...")
            time.sleep(delay)


def _gemini_rate_limit():
    """Выдерживает минимальный интервал между запросами к Gemini (10 RPM)."""
    elapsed = time.time() - _gemini_last_call[0]
    if elapsed < GEMINI_MIN_INTERVAL:
        time.sleep(GEMINI_MIN_INTERVAL - elapsed)
    _gemini_last_call[0] = time.time()



# ── Минимальный HTTP-клиент Ollama (fallback если пакет ollama не установлен) ──

try:
    import requests as _requests
except ImportError:
    _requests = None


class _OllamaHTTPClient:
    """Вызывает Ollama REST API напрямую через requests — без пакета ollama."""
    BASE = "http://localhost:11434"

    def list(self) -> dict:
        return _requests.get(f"{self.BASE}/api/tags", timeout=5).json()

    def chat(self, model: str, messages: list, format: str = None,
             options: dict = None) -> dict:
        payload = {"model": model, "messages": messages, "stream": False}
        if format:
            payload["format"] = format
        if options:
            payload["options"] = options
        resp = _requests.post(f"{self.BASE}/api/chat", json=payload, timeout=(10, 120))
        resp.raise_for_status()
        data = resp.json()
        return {"message": {"content": data["message"]["content"]}}


def _make_ollama_client():
    """Возвращает ollama.Client() или _OllamaHTTPClient() как fallback."""
    try:
        import ollama
        return ollama.Client(timeout=120)
    except ImportError:
        if _requests is None:
            print("Ошибка: не установлен ни пакет 'ollama', ни 'requests'.")
            print("Установи хотя бы один: pip install ollama  или  pip install requests")
            sys.exit(1)
        return _OllamaHTTPClient()


# ── Аннотаторы ────────────────────────────────────────────────────────────────

def _build_messages(system_prompt: str, few_shot: list[tuple],
                    sentence: str, ctx_before: list[str],
                    ctx_label: str = "Контекст", annotate_label: str = "Размечай") -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    for user_ex, asst_ex in few_shot:
        messages.append({"role": "user", "content": user_ex})
        messages.append({"role": "assistant", "content": asst_ex})
    user_content = ""
    if ctx_before:
        user_content += f"[{ctx_label}]: {' '.join(ctx_before[-5:])}\n"
    user_content += f"[{annotate_label}]: {sentence}"
    messages.append({"role": "user", "content": user_content})
    return messages


def annotate_sentence_ollama(client, model: str, sentence: str,
                              ctx_before: list[str], lang: str = "ru") -> dict:
    if lang == "en":
        msgs = _build_messages(SYSTEM_PROMPT_EN, FEW_SHOT_EN, sentence, ctx_before,
                                "Context", "Annotate")
    else:
        msgs = _build_messages(SYSTEM_PROMPT_RU, FEW_SHOT_RU, sentence, ctx_before)
    is_qwen3 = "qwen3" in model.lower()
    options = {"temperature": 0.0}
    if is_qwen3:
        options["think"] = False
    try:
        response = client.chat(
            model=model, messages=msgs,
            format="json", options=options,
        )
        raw = response["message"]["content"].strip()
        data = _extract_json_from_response(raw)
        return _parse_label_response(data)
    except Exception as e:
        print(f"  [ollama error] {e}")
        return {"label": "CLAIM", "confidence": 0.3,
                "reasoning": f"[ошибка] {e}", "disputed": True}


def _extract_json_from_response(raw: str):
    """Извлекает JSON из ответа модели, обрабатывая думающие блоки и markdown."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Ищем JSON-массив
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    # Ищем JSON-объект
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    # Fallback: модель сломала структуру JSON — извлекаем label и confidence регекспами
    label_m = re.search(r'"label"\s*:\s*"([A-Z_]+)"', text)
    conf_m  = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
    if label_m and conf_m:
        return {
            "label":      label_m.group(1),
            "confidence": float(conf_m.group(1)),
            "reasoning":  "[неполный JSON]",
        }
    raise ValueError(f"JSON не найден в ответе: {text[:200]!r}")


def annotate_batch_gemini(gemini_model, sentences: list[str],
                           ctx_before: list[str], lang: str = "ru") -> list[dict]:
    """Батч-аннотация: 5 предложений за 1 запрос к Gemini."""
    ctx_label  = "Context" if lang == "en" else "Контекст"
    few_shot   = FEW_SHOT_EN_BATCH if lang == "en" else FEW_SHOT_RU_BATCH

    user_content = ""
    if ctx_before:
        user_content += f"[{ctx_label}]: {' '.join(ctx_before[-5:])}\n"
    for i, sent in enumerate(sentences, 1):
        user_content += f"{i}. {sent}\n"

    contents = []
    for user_ex, asst_ex in few_shot:
        contents.append({"role": "user",  "parts": [user_ex]})
        contents.append({"role": "model", "parts": [asst_ex]})
    contents.append({"role": "user", "parts": [user_content]})

    _gemini_rate_limit()

    def _call():
        return gemini_model.generate_content(
            contents, generation_config=GEMINI_BATCH_GEN_CONFIG,
        ).text.strip()

    try:
        raw = _call_with_retry(_call)
        items = _extract_json_from_response(raw)
        if not isinstance(items, list):
            items = [items]
        result = []
        for i, sent in enumerate(sentences):
            if i < len(items):
                result.append(_parse_label_response(items[i]))
            else:
                result.append({"label": "CLAIM", "confidence": 0.3,
                               "reasoning": "[нет ответа в батче]", "disputed": True})
        return result
    except DailyQuotaExhausted:
        raise
    except Exception as e:
        print(f"  [batch error] {e}")
        return [{"label": "CLAIM", "confidence": 0.3,
                "reasoning": f"[ошибка батча] {e}", "disputed": True}
               for _ in sentences]


def annotate_document(annotate_fn, text: str, doc_id: str) -> list[dict]:
    sentences = split_sentences(text)
    results = []
    for i, sentence in enumerate(sentences):
        ctx = [r["text"] for r in results[-5:]]
        ann = annotate_fn(sentence, ctx)
        results.append({"doc_id": doc_id, "position": i, "text": sentence, **ann})
        print(f"  [{i + 1:3d}/{len(sentences)}] {ann['label']:12s} ({ann['confidence']:.2f}) | {sentence[:55]}...")
    return results


def annotate_document_batch(batch_fn, text: str, doc_id: str,
                             batch_size: int | None = None) -> list[dict]:
    """Аннотирует документ батчами. batch_size=None → весь документ за 1 запрос."""
    sentences = split_sentences(text)
    size = batch_size or len(sentences)
    if size >= len(sentences):
        print(f"  Режим: весь документ за 1 запрос ({len(sentences)} предложений)")
    results = []
    for i in range(0, len(sentences), size):
        batch = sentences[i:i + size]
        ctx = [f"[{r['label']}] {r['text']}" for r in results[-5:]]
        anns = batch_fn(batch, ctx)
        for j, ann in enumerate(anns):
            pos = i + j
            results.append({"doc_id": doc_id, "position": pos, "text": batch[j], **ann})
            print(f"  [{pos + 1:3d}/{len(sentences)}] {ann['label']:12s} ({ann['confidence']:.2f}) | {batch[j][:55]}...")
    return results


# ── Интерактивное меню ────────────────────────────────────────────────────────

def ask(prompt: str, options: list[str], default: int = 1) -> int:
    for i, opt in enumerate(options, 1):
        marker = ">" if i == default else " "
        print(f"  {marker}{i}. {opt}")
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw)
        print(f"     Введите число от 1 до {len(options)}")


def ask_int(prompt: str, default: int) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print("     Введите положительное целое число")


def get_ollama_models() -> list[str]:
    try:
        import ollama
        client = ollama.Client()
        models = client.list()
        names = [m["name"] for m in models.get("models", [])]
        if names:
            return sorted(names)
    except Exception:
        pass
    # Fallback: вызываем ollama list через CLI (работает даже без Python-пакета)
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        lines = result.stdout.strip().splitlines()[1:]  # пропускаем заголовок NAME/ID/SIZE...
        names = [line.split()[0] for line in lines if line.strip()]
        return sorted(names) if names else []
    except Exception:
        return []



def interactive_setup(args) -> dict:
    """Спрашивает пользователя о пропущенных параметрах и возвращает полный конфиг."""
    print()
    print("=" * 52)
    print("   Argument Mining — разметка аргументов")
    print("=" * 52)

    # ── Датасет ───────────────────────────────────────
    if args.dataset:
        dataset_key = args.dataset
    else:
        print("\nДатасет:")
        idx = ask("Выбор", [d["label"] for d in DATASETS.values()])
        dataset_key = list(DATASETS.keys())[idx - 1]

    # ── Провайдер ─────────────────────────────────────
    if args.provider:
        provider = args.provider
    else:
        print("\nПровайдер LLM:")
        idx = ask("Выбор", list(PROVIDERS.values()))
        provider = list(PROVIDERS.keys())[idx - 1]

    # ── Модель ────────────────────────────────────────
    model = args.model

    if provider == "ollama" and not model:
        ollama_models = get_ollama_models()
        if ollama_models:
            print("\nМодели Ollama (скачаны локально):")
            idx = ask("Выбор", ollama_models)
            model = ollama_models[idx - 1]
        else:
            model = input("\nМодель Ollama [qwen2.5:7b]: ").strip() or "qwen2.5:7b"

    elif provider == "gemini" and not model:
        gemini_models = [
            "gemini-2.5-flash-lite (15 RPM, ~1000 RPD)",
            "gemini-2.5-flash      (10 RPM,  500 RPD)",
            "gemini-2.5-pro         (5 RPM,  100 RPD)",
            "gemini-3-flash         (preview)",
            "gemini-3.1-flash-lite  (preview)",
        ]
        gemini_model_ids = [
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-3-flash",
            "gemini-3.1-flash-lite",
        ]
        print("\nМодели Gemini:")
        idx = ask("Выбор", gemini_models)
        model = gemini_model_ids[idx - 1]

    # ── Количество документов ─────────────────────────
    n = args.n if args.n else ask_int("\nКоличество документов", 50)

    print()
    print(f"  Датасет:  {DATASETS[dataset_key]['label']}")
    print(f"  Провайдер: {PROVIDERS[provider]}")
    print(f"  Модель:   {model}")
    print(f"  Документов: {n}")
    print()

    return {"dataset": dataset_key, "provider": provider, "model": model, "n": n}


# ── Загрузчики датасетов ──────────────────────────────────────────────────────

def _load_hf_streaming(hf_id: str, n: int, extract_fn) -> list[dict]:
    """Загружает n документов из HuggingFace streaming-датасета."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Установи: pip install datasets")
        sys.exit(1)

    for attempt in range(1, HF_DOWNLOAD_RETRIES + 1):
        try:
            ds = load_dataset(hf_id, split="train", streaming=True)
            break
        except Exception as e:
            if attempt == HF_DOWNLOAD_RETRIES:
                raise
            wait = attempt * 5
            print(f"  Сбой загрузки (попытка {attempt}/{HF_DOWNLOAD_RETRIES}): {e}, повтор через {wait}с...")
            time.sleep(wait)

    docs, seen_ids = [], set()
    for idx, item in enumerate(ds):
        if len(docs) >= n:
            break
        doc = extract_fn(item, idx)
        if doc is None or doc["doc_id"] in seen_ids:
            continue
        seen_ids.add(doc["doc_id"])
        docs.append(doc)
    return docs


def _extract_ruslawod(item: dict, idx: int) -> dict | None:
    tokens = item.get("tokens") or item.get("words") or []
    text = (
        " ".join(str(t) for t in tokens) if tokens else
        item.get("text") or item.get("textIPS") or
        item.get("taggedtextIPS") or item.get("headingIPS") or ""
    )
    if len(text.strip()) < 100:
        return None
    doc_id = str(
        item.get("doc_id") or item.get("id") or
        item.get("pravogovruNd") or item.get("docNumberIPS") or
        f"doc_{idx:05d}"
    )
    return {"doc_id": doc_id, "text": text}


def _extract_rutar(item: dict, idx: int) -> dict | None:
    # xlsx был сохранён pandas с индексной колонкой → реальные поля сдвинуты:
    # "letter_type" содержит тело ответного письма (самое длинное поле)
    # "question_for_llm" содержит source URL
    # "date_publication" содержит заголовок письма
    text = (
        str(item.get("letter_type") or "").strip()
        or str(item.get("full_text") or "").strip()
        or str(item.get("answer_letter") or "").strip()
    )
    if len(text) < 100:
        return None
    url = str(item.get("question_for_llm") or item.get("source_url") or "")
    raw_id = url.rstrip("/").split("/")[-1]
    if not raw_id or len(raw_id) < 3:
        raw_id = str(item.get("date_publication") or item.get("title") or "")[:60]
    raw_id = raw_id or f"rutar_{idx:04d}"
    doc_id = _to_safe_id(raw_id)[:80] or f"rutar_{idx:04d}"
    return {"doc_id": doc_id, "text": text}



def load_cached_docs(cache_file: Path) -> list[dict]:
    if not cache_file.exists():
        return []
    docs = []
    with open(cache_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def save_cached_docs(docs: list[dict], cache_file: Path) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def load_ruslawod(n: int, raw_dir: Path) -> list[dict]:
    cache_file = raw_dir / "ruslawod_docs.jsonl"
    cached = load_cached_docs(cache_file)
    if len(cached) >= n:
        print(f"Берём {n} документов из кэша: {cache_file}")
        return cached[:n]

    parquet_files = sorted(raw_dir.glob("ruslawod*.parquet"))
    if parquet_files:
        try:
            from datasets import load_dataset
            print(f"Загружаем из локальных parquet: {', '.join(p.name for p in parquet_files)}")
            ds = load_dataset("parquet", data_files=[str(p) for p in parquet_files], split="train")
            docs, seen_ids = [], set()
            for idx, item in enumerate(ds):
                doc = _extract_ruslawod(item, idx)
                if doc and doc["doc_id"] not in seen_ids:
                    seen_ids.add(doc["doc_id"])
                    docs.append(doc)
                if len(docs) >= n:
                    break
            if len(docs) >= n:
                save_cached_docs(docs, cache_file)
                return docs[:n]
        except Exception as e:
            print(f"  Ошибка при чтении parquet: {e}")

    print(f"Загружаем {n} документов из HuggingFace ({DATASETS['ruslawod']['hf_id']})...")
    fetched = _load_hf_streaming(DATASETS["ruslawod"]["hf_id"], n, _extract_ruslawod)

    merged = list({doc["doc_id"]: doc for doc in cached + fetched}.values())
    save_cached_docs(merged, cache_file)
    return merged[:n]



def _read_xlsx_stdlib(path: Path) -> list[list]:
    """Читает xlsx через стандартный zipfile+xml без openpyxl/numpy."""
    import zipfile
    from xml.etree import ElementTree as ET

    SS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()

        shared = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall(f".//{{{SS}}}si"):
                texts = si.findall(f".//{{{SS}}}t")
                shared.append("".join(t.text or "" for t in texts))

        sheet_file = next(
            (n for n in names if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")),
            None,
        )
        if not sheet_file:
            return []

        root = ET.fromstring(zf.read(sheet_file))
        rows = []
        for row_el in root.findall(f".//{{{SS}}}row"):
            row = []
            for c in row_el.findall(f"{{{SS}}}c"):
                t = c.get("t", "")
                v_el = c.find(f"{{{SS}}}v")
                if v_el is None or v_el.text is None:
                    row.append(None)
                elif t == "s":
                    idx = int(v_el.text)
                    row.append(shared[idx] if idx < len(shared) else "")
                elif t == "inlineStr":
                    is_el = c.find(f".//{{{SS}}}t")
                    row.append(is_el.text if is_el is not None else "")
                elif t == "b":
                    row.append(v_el.text == "1")
                else:
                    row.append(v_el.text)
            rows.append(row)
    return rows


def load_rutar(n: int, raw_dir: Path) -> list[dict]:
    cache_file = raw_dir / "rutar_docs.jsonl"
    cached = load_cached_docs(cache_file)
    if cached:
        print(f"Берём документы из кэша: {cache_file}")
        return cached[:n]

    xlsx_path = raw_dir / "rutar.xlsx"
    if not xlsx_path.exists():
        print(f"Файл не найден: {xlsx_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Читаем {xlsx_path}...")
    raw_rows = _read_xlsx_stdlib(xlsx_path)
    if not raw_rows:
        print("Файл пустой или не удалось прочитать.", file=sys.stderr)
        sys.exit(1)

    headers = [str(h) if h is not None else f"col_{i}" for i, h in enumerate(raw_rows[0])]
    docs = []
    for idx, row in enumerate(raw_rows[1:]):
        padded = list(row) + [None] * (len(headers) - len(row))
        doc = _extract_rutar(dict(zip(headers, padded)), idx)
        if doc:
            docs.append(doc)

    save_cached_docs(docs, cache_file)
    print(f"Загружено {len(docs)} документов из {xlsx_path}")
    return docs[:n]


def load_sudresh(n: int, raw_dir: Path) -> list[dict]:
    cache_file = raw_dir / "sudresh_docs.jsonl"
    cached = load_cached_docs(cache_file)
    if len(cached) >= n:
        print(f"Берём {n} документов из кэша: {cache_file}")
        return cached[:n]

    print(f"Загружаем sud-resh-benchmark с HuggingFace...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("Установи: pip install datasets")
        sys.exit(1)

    ds = load_dataset("lawful-good-project/sud-resh-benchmark", split="train")

    # Каждое решение повторяется 7 раз (по числу инструкций) — дедуплицируем по source
    seen_texts: set[str] = {doc["text"] for doc in cached}
    docs = list(cached)

    for item in ds:
        if len(docs) >= n:
            break
        # source — исходный текст судебного решения
        text = str(item.get("source") or item.get("text") or "").strip()
        if len(text) < 200 or text in seen_texts:
            continue
        seen_texts.add(text)

        # doc_id из хеша id записи + категория
        raw_id  = str(item.get("id", ""))[:16]
        category = str(item.get("category", "")).replace("sud_resh_", "")
        doc_id  = f"sudresh_{category}_{raw_id}" if raw_id else f"sudresh_{len(docs):05d}"

        docs.append({"doc_id": doc_id, "text": text})

    save_cached_docs(docs, cache_file)
    print(f"Загружено {len(docs)} уникальных решений")
    return docs[:n]


DATASET_LOADERS = {
    "ruslawod": load_ruslawod,
    "rutar":    load_rutar,
    "sudresh":  load_sudresh,
}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        description="Argument Mining — разметка текстов через LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Запуск без аргументов открывает интерактивное меню.",
    )
    parser.add_argument("--dataset",  choices=list(DATASETS.keys()), default=None,
                        help="Датасет для разметки")
    parser.add_argument("--provider", choices=list(PROVIDERS.keys()), default=None,
                        help="LLM-провайдер")
    parser.add_argument("--model",    default=None,
                        help="Название модели (зависит от провайдера)")
    parser.add_argument("--n",        type=int, default=None,
                        help="Количество документов для разметки")
    args = parser.parse_args()

    if not all([args.dataset, args.provider, args.model, args.n]):
        cfg = interactive_setup(args)
        dataset_key = cfg["dataset"]
        provider    = cfg["provider"]
        model       = cfg["model"]
        n           = cfg["n"]
    else:
        dataset_key = args.dataset
        provider    = args.provider
        model       = args.model
        n           = args.n

    ds_cfg    = DATASETS[dataset_key]
    lang      = ds_cfg["lang"]
    raw_dir   = ds_cfg["dir"] / "raw"
    model_tag = _to_safe_id(model)
    out_dir   = ds_cfg["dir"] / "annotated" / model_tag

    # ── Инициализация клиента ─────────────────────────────────────────────────
    use_batch = False

    if provider == "ollama":
        try:
            client = _make_ollama_client()
            client.list()
            print(f"Ollama подключена. Модель: {model}")
        except Exception as e:
            print(f"Ошибка подключения к Ollama: {e}")
            print("Запусти Ollama: ollama serve")
            print(f"Скачай модель:  ollama pull {model}")
            sys.exit(1)
        annotate_fn = lambda sentence, ctx: annotate_sentence_ollama(client, model, sentence, ctx, lang)


    else:  # gemini — batch mode with safety settings + response schema
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Ошибка: GEMINI_API_KEY не задан. Добавь в .env: GEMINI_API_KEY=your_key")
            sys.exit(1)
        try:
            import google.generativeai as genai
            from google.generativeai.types import HarmCategory, HarmBlockThreshold
        except ImportError:
            print("Установи: pip install google-generativeai")
            sys.exit(1)

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT:        HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH:       HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        system_prompt_batch = SYSTEM_PROMPT_EN_BATCH if lang == "en" else SYSTEM_PROMPT_RU_BATCH
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_prompt_batch,
            generation_config={"temperature": 0.0},
            safety_settings=safety_settings,
        )
        print(f"Gemini подключена. Модель: {model} | mode=full_doc | rate≤{GEMINI_RPM_LIMIT} RPM")
        use_batch = True
        batch_fn  = lambda sentences, ctx: annotate_batch_gemini(gemini_model, sentences, ctx, lang)

    if os.getenv("HF_TOKEN"):
        print("HF_TOKEN найден.")

    # ── Загрузка документов ───────────────────────────────────────────────────
    raw_dir.mkdir(parents=True, exist_ok=True)
    docs = DATASET_LOADERS[dataset_key](n, raw_dir)

    # ── Разметка ──────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    total_annotated = 0

    for doc in docs:
        doc_id    = doc["doc_id"]
        safe_name = _to_safe_id(doc_id) or "doc"
        out_file  = out_dir / f"{safe_name}.jsonl"
        if out_file.exists():
            if _is_garbage_file(out_file):
                print(f"[{doc_id}] файл повреждён (все метки — fallback), перемечаем")
                out_file.unlink()
            else:
                print(f"[{doc_id}] уже размечен, пропускаем")
                continue

        print(f"\n{'=' * 55}\nДокумент: {doc_id}\n{'=' * 55}")
        try:
            if use_batch:
                annotations = annotate_document_batch(batch_fn, doc["text"], doc_id,
                                                      batch_size=None)
            else:
                annotations = annotate_document(annotate_fn, doc["text"], doc_id)
        except DailyQuotaExhausted:
            print("\nДневной лимит запросов Gemini исчерпан (20 req/day на free tier).")
            print(f"Уже размечено: {total_annotated} предложений → {out_dir}/")
            print("Запусти скрипт завтра — уже обработанные файлы будут пропущены.")
            sys.exit(0)

        counts   = Counter(a["label"] for a in annotations)
        disputed = sum(1 for a in annotations if a.get("disputed"))
        print(f"\n  Итого: {len(annotations)} | " +
              " ".join(f"{k}:{v}" for k, v in sorted(counts.items())) +
              f" | disputed:{disputed}")

        with open(out_file, "w", encoding="utf-8") as f:
            for ann in annotations:
                f.write(json.dumps(ann, ensure_ascii=False) + "\n")

        total_annotated += len(annotations)
        print(f"  Сохранено: {out_file}")

    print(f"\nГотово! {total_annotated} предложений  →  {out_dir}/")


if __name__ == "__main__":
    main()
