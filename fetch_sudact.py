"""
Загрузчик решений арбитражных судов с sudact.ru.

Скачивает решения по налоговым спорам и сохраняет в формате,
совместимом с остальным пайплайном (jsonl, поля doc_id + text).

Запуск из корня проекта:
    python data/sudact/fetch_sudact.py --n 200 --query "налог НДС"

Результат: data/sudact/raw/sudact_docs.jsonl
"""

import argparse
import json
import re
import time
from pathlib import Path

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Установи зависимости: pip install requests beautifulsoup4")
    raise SystemExit(1)

# ── Константы ──────────────────────────────────────────────────────────────────

BASE_URL    = "https://sudact.ru"
SEARCH_URL  = f"{BASE_URL}/arbitral/"
OUT_FILE    = Path("data/sudact/raw/sudact_docs.jsonl")
DELAY       = 2.0   # секунд между запросами (не перегружаем сервер)
PAGE_SIZE   = 20

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9",
    "Referer": BASE_URL,
}

# Метки решения — в заголовке судебного акта
_DECISION_RE = re.compile(
    r"(решение|постановление|определение|приговор)",
    re.IGNORECASE,
)

# Разделы страницы которые не нужны в тексте
_NOISE_SELECTORS = [
    "script", "style", "nav", "header", "footer",
    ".sidebar", ".breadcrumb", ".doc-navigation",
    ".doc-info", ".act-meta", ".act-controls",
]


# ── HTTP ───────────────────────────────────────────────────────────────────────

session = requests.Session()
session.headers.update(HEADERS)


def _get(url: str, params: dict = None, retries: int = 3) -> requests.Response | None:
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=20)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            wait = DELAY * (attempt + 1)
            print(f"  [retry {attempt + 1}/{retries}] {e} — ждём {wait:.0f}с")
            time.sleep(wait)
    return None


# ── Парсинг ────────────────────────────────────────────────────────────────────

def parse_search_page(html: str) -> list[dict]:
    """Возвращает [{title, url}] с одной страницы поиска."""
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for a in soup.select("a.result__title, a.doc-title, h3 > a, .result-title a"):
        href  = a.get("href", "")
        title = a.get_text(strip=True)
        if not href or not title:
            continue
        if not href.startswith("http"):
            href = BASE_URL + href
        results.append({"title": title, "url": href})
    return results


def parse_document(html: str, url: str) -> dict | None:
    """Извлекает чистый текст из страницы судебного акта."""
    soup = BeautifulSoup(html, "html.parser")

    # Удаляем шум
    for sel in _NOISE_SELECTORS:
        for tag in soup.select(sel):
            tag.decompose()

    # Ищем основной блок с текстом
    body = (
        soup.select_one(".act-text")
        or soup.select_one("#documentText")
        or soup.select_one(".doc-text")
        or soup.select_one("article")
        or soup.select_one("main")
    )
    if body is None:
        return None

    text = body.get_text(separator=" ", strip=True)
    text = re.sub(r"\s{2,}", " ", text).strip()

    if len(text) < 300:
        return None

    # doc_id из URL: последний числовой сегмент
    parts = url.rstrip("/").split("/")
    doc_id = next((p for p in reversed(parts) if p.isdigit()), None)
    if not doc_id:
        doc_id = re.sub(r"[^\w]", "_", url[-40:])

    return {"doc_id": doc_id, "text": text, "url": url}


# ── Поиск ──────────────────────────────────────────────────────────────────────

def search_documents(query: str, n: int) -> list[dict]:
    """Возвращает список {title, url} найденных документов (до n штук)."""
    found = []
    page  = 1
    seen  = set()

    while len(found) < n:
        params = {
            "page":  page,
            "count": PAGE_SIZE,
            "txt":   query,
            "doc_type": "решение",
        }
        print(f"  Поиск: страница {page} (найдено {len(found)}/{n})...")
        resp = _get(SEARCH_URL, params=params)
        if resp is None:
            break

        items = parse_search_page(resp.text)
        if not items:
            print("  Результаты кончились.")
            break

        for item in items:
            if item["url"] not in seen:
                seen.add(item["url"])
                found.append(item)
            if len(found) >= n:
                break

        page += 1
        time.sleep(DELAY)

    return found[:n]


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Загрузчик решений с sudact.ru")
    parser.add_argument("--n",     type=int, default=100,
                        help="Сколько документов скачать (по умолчанию 100)")
    parser.add_argument("--query", default="налог НДС НДФЛ",
                        help="Поисковый запрос (по умолчанию 'налог НДС НДФЛ')")
    args = parser.parse_args()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Загружаем уже скачанные
    existing_ids: set[str] = set()
    existing_docs: list[dict] = []
    if OUT_FILE.exists():
        for line in OUT_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                doc = json.loads(line)
                existing_ids.add(doc["doc_id"])
                existing_docs.append(doc)
        print(f"Уже скачано: {len(existing_docs)} документов")

    need = args.n - len(existing_docs)
    if need <= 0:
        print(f"Уже есть {len(existing_docs)} ≥ {args.n} документов, ничего не делаем.")
        return

    print(f"\nИщем документы: запрос='{args.query}', нужно ещё {need}")
    candidates = search_documents(args.query, need * 3)  # с запасом — часть отсеется
    print(f"Найдено кандидатов: {len(candidates)}")

    new_docs = []
    for item in candidates:
        if len(new_docs) >= need:
            break

        resp = _get(item["url"])
        if resp is None:
            continue

        doc = parse_document(resp.text, item["url"])
        if doc is None or doc["doc_id"] in existing_ids:
            continue

        existing_ids.add(doc["doc_id"])
        new_docs.append(doc)
        print(f"  [{len(new_docs):3d}/{need}] {doc['doc_id']}  {len(doc['text'])} симв.")
        time.sleep(DELAY)

    # Дозаписываем новые
    with OUT_FILE.open("a", encoding="utf-8") as f:
        for doc in new_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\nГотово: скачано {len(new_docs)} новых документов → {OUT_FILE}")
    print(f"Итого в файле: {len(existing_docs) + len(new_docs)}")


if __name__ == "__main__":
    main()
